"""Draw the two paired Map09 mapping-only rounds from saved results. Never starts a simulator."""

import argparse
import io
import json
import math
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer


VERSIONS = ('RL11', 'RL17-R')
SEEDS = (2026111901, 2026111902)
PANEL_SIZE = (544, 352)
WALL, HIT, MISS = (123, 131, 138), (0, 125, 83), (79, 109, 159)
CHARGED, FREE_PRED, COVERED = (198, 46, 49), (204, 155, 57), (121, 204, 172)
TRACKS = ('#006aab', '#ba6109', '#00815f', '#ae3271', '#6b49bc')
INK, SECONDARY, ALERT = '#172d3c', '#49606b', '#a32e32'


def font(size, bold=False):
    for name in ('msyhbd.ttc' if bold else 'msyh.ttc', 'NotoSansCJK-Regular.ttc', 'simhei.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError('Please install Microsoft YaHei or Noto Sans CJK to render Chinese labels')


def load_run(batch, number, version):
    directory = batch / f'r{number}_{version}'
    result = {}
    panels = [None] * 4
    try:
        result = json.loads((directory / 'result.json').read_text(encoding='utf-8-sig'))
        if not isinstance(result, dict):
            raise ValueError('结果格式有误')
        if (result['version'], result['seed'], result['round']) != (version, SEEDS[number - 1], number):
            raise ValueError('结果版本、轮次或种子不匹配')
        if result.get('map', 'Map09') != 'Map09':
            raise ValueError('结果地图不匹配')
        if result['payload'] != 0 or result['placed_count'] != 0:
            raise ValueError('缺少全程零载荷、零落点记录')
        if result.get('status') != 'ok':
            reason = result.get('error') or result.get('stop_reason') or result.get('status') or '状态未知'
            return result, panels, f'运行异常 · {str(reason)[:35]}'
        with np.load(directory / 'flight.npz', allow_pickle=False) as raw:
            truth = raw['truth'] > 0
            pred = raw['pred'] > 0
            visible = raw['visible'] > 0
            positions = raw['positions']
        if truth.ndim != 2 or not truth.any() or visible.shape != truth.shape or pred.shape != truth.shape:
            raise ValueError('地图掩膜形状或真值有误')
        if positions.shape != (result['steps'], 5, 2) or not len(positions):
            raise ValueError('轨迹与步数不匹配')
        if np.isinf(positions).any() or not np.array_equal(np.isnan(positions[:, :, 0]), np.isnan(positions[:, :, 1])):
            raise ValueError('轨迹坐标无效')
        if not isinstance(result['submitted'], bool):
            raise ValueError('提交状态无效')
        if result['submitted']:
            scorer = MapExplorationScorer()
            replayed_score = 100 * scorer.score(truth, pred, 0.0).s_map
            if not np.isclose(result['map_score'], replayed_score, rtol=0, atol=.0051):
                raise ValueError('提交图的分数不能复算')
            surface, _ = scorer.split_regions(truth)
            distance = scorer.distance_field(truth)
            diff = np.full((*truth.shape, 3), 255, dtype=np.uint8)
            diff[truth] = WALL
            diff[surface != 0] = MISS
            diff[(surface != 0) & pred] = HIT
            diff[pred & ~truth] = FREE_PRED
            diff[pred & (distance > scorer.config.dead_zone)] = CHARGED
            panels[0] = Image.fromarray(diff)
        background = np.full((*truth.shape, 3), 255, dtype=np.uint8)
        background[truth] = WALL
        panels[3] = Image.fromarray(background.copy())
        background[visible & ~truth] = COVERED
        panels[1] = Image.fromarray(background)
        path_map = panels[3].copy()
        draw = ImageDraw.Draw(path_map)
        h, w = truth.shape
        for slot, colour in enumerate(TRACKS):
            path = positions[:, slot]
            valid = np.isfinite(path).all(axis=1)
            transitions = np.diff(valid.astype(np.int8), prepend=0, append=0)
            for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                segment = path[start:end]
                points = list(zip(segment[:, 0] + w / 2, h / 2 - segment[:, 1]))
                if len(points) == 1:
                    draw.point(points[0], fill=colour)
                else:
                    draw.line(points, fill=colour, width=max(2, round(w / 400)))
        panels[2] = path_map
        return result, panels, ''
    except FileNotFoundError as err:
        return result, [None] * 4, f'缺少 {Path(err.filename).name}'
    except (KeyError, OSError, ValueError, TypeError, BadZipFile) as err:
        return result, [None] * 4, f'异常 · {str(err)[:42]}'


def metric(value, suffix=''):
    return f'{value:.2f}{suffix}' if isinstance(value, (float, int)) and math.isfinite(value) else '—'


def draw_round(batch, number):
    runs = [load_run(batch, number, version) for version in VERSIONS]
    canvas = Image.new('RGB', (1176, 1934), 'white')
    draw = ImageDraw.Draw(canvas)
    title, label, body, small = font(34, True), font(23, True), font(20), font(17)
    draw.text((25, 15), f'官方 Map09 · 无载荷建图 · 第 {number} 轮', font=title, fill=INK)
    caption = f'配对种子 {SEEDS[number - 1]}'
    if all(not error for _, _, error in runs):
        first, second = (r for r, _, _ in runs)
        for key, name, unit in [('map_score', '地图分', '分'), ('coverage_percent', '探索覆盖', '个百分点')]:
            if key == 'map_score' and not (first['submitted'] and second['submitted']):
                continue
            left, right = first.get(key), second.get(key)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                caption += f'    ·    RL17-R {name}较 RL11 {right - left:+.2f} {unit}'
    draw.text((26, 68), caption, font=body, fill=SECONDARY)
    left_edges = (26, 606)
    for x, (result, _, error), name in zip(left_edges, runs, ('RL11', '解压目录 RL17-R')):
        draw.text((x, 112), name, font=label, fill=INK)
        summary = error or f"{result['steps']:,} 步    {metric(result.get('wall_seconds'), ' 秒')}"
        if not error and (result.get('walltime_limited') or result.get('stop_reason') == 'wall_limit'):
            summary += '    达到墙钟上限'
        draw.text((x, 148), summary, font=body, fill=ALERT if error else SECONDARY)
    headings = ('提交图与真墙比对', '探索覆盖', '五机实际轨迹', '炸弹图 · 无载荷')
    for index, heading in enumerate(headings):
        top = 197 + index * 428
        draw.text((26, top), heading, font=label, fill=INK)
        if index == 0:
            x = 255
            for colour, name in ((HIT, '命中'), (MISS, '漏画'), (CHARGED, '受罚误画'), (FREE_PRED, '容差内免罚'), (WALL, '墙体内部')):
                draw.rectangle((x, top + 8, x + 14, top + 21), fill=colour)
                draw.text((x + 21, top + 1), name, font=small, fill=SECONDARY)
                x += int(draw.textlength(name, font=small)) + 45
        elif index == 1:
            draw.text((255, top + 3), '绿色为原生轨迹邻域覆盖，白色为未覆盖，灰色为真墙。', font=small, fill=SECONDARY)
        elif index == 2:
            for slot, colour in enumerate(TRACKS):
                x = 255 + slot * 97
                draw.line((x, top + 15, x + 21, top + 15), fill=colour, width=5)
                draw.text((x + 28, top + 1), f'{slot} 号', font=small, fill=SECONDARY)
        else:
            draw.text((255, top + 3), '灰色为真墙，无载荷场景中没有落点。', font=small, fill=SECONDARY)
        for col, (result, items, error) in enumerate(runs):
            x, y = left_edges[col], top + 68
            if error:
                detail = error
            elif index == 0:
                detail = f"地图分 {metric(result['map_score'])}" if result['submitted'] else '本局未提交地图'
            elif index == 1:
                detail = f"引擎探索覆盖 {metric(result.get('coverage_percent'), '%')}"
            elif index == 2:
                detail = '录像槽位 0–4，北向上'
            else:
                detail = '初始载荷 0 · 实际落点 0'
            draw.text((x, top + 36), detail, font=small, fill=ALERT if error or (index == 0 and not result.get('submitted')) else SECONDARY)
            panel = items[index]
            if panel is None:
                draw.rectangle((x, y, x + PANEL_SIZE[0], y + PANEL_SIZE[1]), fill='#f4f6f7')
                notice = '未提交 · 无提交图' if not error else '缺少可核验图件'
                draw.text((x + PANEL_SIZE[0] / 2, y + PANEL_SIZE[1] / 2), notice, font=body, fill=ALERT, anchor='mm')
            else:
                panel.thumbnail(PANEL_SIZE, Image.Resampling.LANCZOS)
                canvas.paste(panel, (x + (PANEL_SIZE[0] - panel.width) // 2, y + (PANEL_SIZE[1] - panel.height) // 2))
            draw.rectangle((x, y, x + PANEL_SIZE[0], y + PANEL_SIZE[1]), outline='#b4c0c5')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve(strict=True)
    if not batch.is_dir():
        parser.error('batch must be an existing directory')
    directory = batch / 'figures'
    directory.mkdir(exist_ok=True)
    files = [directory / f'Map09_r{number}_comparison.png' for number in (1, 2)]
    if any(path.exists() for path in files):
        raise FileExistsError('Existing comparison figure; select the intended new batch')
    for number, path in enumerate(files, 1):
        encoded = io.BytesIO()
        draw_round(batch, number).save(encoded, format='PNG')
        with path.open('xb') as stream:
            stream.write(encoded.getbuffer())
        print(path)


if __name__ == '__main__':
    main()
