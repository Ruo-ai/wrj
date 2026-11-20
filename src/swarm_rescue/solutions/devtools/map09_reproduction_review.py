"""Compare two completed Map09 simulator recordings without running a simulation."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from swarm_rescue.solutions.devtools.map09_extracted_mapping_figures import (
    CHARGED, FREE_PRED, HIT, INK, MISS, SECONDARY, TRACKS, WALL, font,
)
from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer


SEED = 2477227837
WIDTH, HEIGHT = 1700, 1100


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(paths):
    found = list(paths)
    if len(found) != 1:
        raise ValueError(f'Expected exactly one artifact, found {len(found)}')
    return found[0]


def load_run(attempt):
    result = read(attempt / 'result.json')
    request = read(attempt / 'request.json')
    resources = read(attempt / 'resources.json')
    telemetry_path = attempt / 'telemetry.json'
    telemetry = read(telemetry_path) if telemetry_path.is_file() else {}
    expected = dict(map='Map09', round=1, seed=SEED)
    if any(result.get(k) != value or request.get(k) != value for k, value in expected.items()):
        raise ValueError(f'Unexpected map, round or seed under {attempt}')
    if request.get('drones') != 5 or request.get('bombs') != 2 or request.get('smoke'):
        raise ValueError(f'Unexpected workload under {attempt}')
    png = one((attempt / 'artifacts').glob('*_exploration_diff.png'))
    recording = one((attempt / 'artifacts/recordings').glob('*.npz'))
    layout = one((attempt / 'artifacts').glob('*_bombs.json'))
    with Image.open(png) as image:
        if image.size != (3 * WIDTH + 4, HEIGHT + 128):
            raise ValueError('Unexpected official comparison PNG dimensions')
        rgb = np.array(image.convert('RGB'))
    grids = []
    for left in (0, WIDTH + 2):
        panel = rgb[24:24 + HEIGHT, left:left + WIDTH]
        if not np.all((panel == 40).all(axis=2) | (panel == 255).all(axis=2)):
            raise ValueError('Lossless truth or prediction panel expected')
        grids.append((panel[:, :, 0] == 40).astype(np.uint8))
    truth, pred = grids
    scorer = MapExplorationScorer()
    score = scorer.score(truth, pred, result['elapsed_walltime'])
    phase = result['phase1']
    if abs(score.s_map * 100 - phase['score_map']) > .0051:
        raise ValueError('Map score cannot be reconstructed from the original PNG')
    with np.load(recording, allow_pickle=False) as replay:
        states = replay['drone_states']
        stamps = replay['timestamps']
    if states.ndim != 2 or states.shape[1] != 5 or len(states) != len(stamps):
        raise ValueError('Unexpected replay array dimensions')
    layout_data = read(layout)
    bombs = layout_data['bombs']
    if layout_data['map_name'] != 'Map09' or len(bombs) != result['placed_count']:
        raise ValueError('Layout differs from recorded placement count')
    if not all(np.isfinite([b['x'], b['y']]).all() for b in bombs):
        raise ValueError('Nonfinite layout coordinates')
    issues = []
    if result.get('status') != 'ok':
        issues.append('result_status:' + str(result.get('status')))
    for key in ('walltime_limited', 'controller_crashed', 'engine_callback_errors'):
        if result.get(key):
            issues.append(key)
    if not result.get('submitted') or not result.get('valid_shape'):
        issues.append('invalid_submission')
    if resources.get('exit_code') != 0 or resources.get('status') != 'exited':
        issues.append('resource_exit')
    log = (attempt / 'run.log').read_text(encoding='utf-8', errors='replace')
    for marker in ('Traceback (most recent call last)', 'Exception ignored from cffi callback',
                   'Fatal Python error'):
        if marker in log:
            issues.append(marker)
    return dict(path=attempt, result=result, resources=resources, telemetry=telemetry,
                truth=truth, pred=pred, scorer=scorer, states=states, bombs=bombs,
                issues=issues, files={p.name: digest(p) for p in
                                     (png, recording, layout, attempt / 'result.json',
                                      attempt / 'request.json', attempt / 'resources.json')})


def map_panel(run):
    truth, pred = run['truth'].astype(bool), run['pred'].astype(bool)
    scorer = run['scorer']
    surface, _ = scorer.split_regions(truth)
    distance = scorer.distance_field(truth)
    canvas = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    canvas[truth] = WALL
    canvas[surface != 0] = MISS
    canvas[(surface != 0) & pred] = HIT
    canvas[pred & ~truth] = FREE_PRED
    canvas[pred & (distance > scorer.config.dead_zone)] = CHARGED
    return Image.fromarray(canvas)


def blank_panel(run):
    canvas = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    canvas[run['truth'] > 0] = (159, 165, 169)
    return Image.fromarray(canvas)


def track_panel(run):
    canvas = blank_panel(run)
    draw = ImageDraw.Draw(canvas)
    state = run['states']
    for slot, color in enumerate(TRACKS):
        xy = np.column_stack((state[:, slot]['x'] + WIDTH / 2,
                              HEIGHT / 2 - state[:, slot]['y']))
        valid = state[:, slot]['active'] & np.isfinite(xy).all(axis=1)
        transitions = np.diff(valid.astype(np.int8), prepend=0, append=0)
        for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
            points = [tuple(p) for p in xy[start:end]]
            if len(points) > 1:
                draw.line(points, fill=color, width=4)
            elif points:
                draw.point(points[0], fill=color)
    return canvas


def bomb_panel(run):
    canvas = blank_panel(run).convert('RGBA')
    draw = ImageDraw.Draw(canvas)
    for number, bomb in enumerate(run['bombs'], 1):
        x, y = bomb['x'] + WIDTH / 2, HEIGHT / 2 - bomb['y']
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill='#1766b5', outline='white', width=2)
        draw.text((x + 15, y - 19), str(number), font=font(27, True), fill='#124d84')
    return canvas.convert('RGB')


def draw_comparison(old, new, kind):
    title, painter = {'mapping': ('提交探索图与真墙比对', map_panel),
                      'trajectories': ('五机实际轨迹', track_panel),
                      'bombs': ('实际模拟落点', bomb_panel)}[kind]
    canvas = Image.new('RGB', (1210, 610), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 13), f'官方 Map09 · RL17-R · 第 1 轮 · 种子 {SEED}', font=font(28, True), fill=INK)
    draw.text((24, 56), title, font=font(23, True), fill=INK)
    if kind == 'mapping':
        cursor = 330
        for color, text in [(HIT, '命中'), (MISS, '漏画'), (CHARGED, '受罚误画'), (FREE_PRED, '容差内免罚')]:
            draw.rectangle((cursor, 65, cursor + 14, 79), fill=color)
            draw.text((cursor + 22, 58), text, font=font(17), fill=SECONDARY)
            cursor += int(draw.textlength(text, font=font(17))) + 50
    elif kind == 'trajectories':
        for slot, color in enumerate(TRACKS):
            left = 330 + slot * 110
            draw.line((left, 72, left + 23, 72), fill=color, width=5)
            draw.text((left + 31, 59), f'{slot} 号机', font=font(18), fill=SECONDARY)
    else:
        draw.text((330, 59), '蓝点为实际初始落点，编号按产物顺序；灰色为真实墙体。', font=font(18), fill=SECONDARY)
    for left, run, label in [(24, old, '历史原局 · 2026-11-16'), (626, new, '本次 · 解压目录 RL17-R')]:
        row = run['result']
        phase = row['phase1']
        draw.text((left, 105), label, font=font(24, True), fill=INK)
        details = f"地图分 {phase['score_map']:.5f} · {row['elapsed_steps']:,} 步 · {row['elapsed_walltime']:.2f} 秒"
        if kind == 'bombs':
            details = f"每机初始 2 个 · 实际落点 {len(run['bombs'])} 个 · 墙体分 {phase['score_wall']:.2f}"
        draw.text((left, 145), details, font=font(18), fill=SECONDARY)
        panel = painter(run)
        panel.thumbnail((560, 363), Image.Resampling.LANCZOS)
        canvas.paste(panel, (left, 187))
        draw.rectangle((left, 187, left + 560, 187 + 363), outline='#87969e')
        status = '正常完成' if not run['issues'] else '异常 · ' + ', '.join(run['issues'])[:46]
        draw.text((left, 568), status, font=font(18), fill=SECONDARY if not run['issues'] else '#a32e32')
    return canvas


def metrics(run):
    result, resources = run['result'], run['resources']
    phase = result['phase1']
    breakdown = phase['map_breakdown']
    return dict(source=str(run['path']), map='Map09', round=1, seed=SEED,
                controller=result.get('controller_class'), map_score=phase['score_map'],
                wall_score=phase['score_wall'], time_score=phase['score_time'],
                red_partial_score=phase['round_score'], score_status=phase['score_status'],
                wall_hits=breakdown['wall_hits'], wall_misses=breakdown['wall_pixels']-breakdown['wall_hits'],
                charged_pixels=breakdown['charged_pixels'], elapsed_steps=result['elapsed_steps'],
                elapsed_walltime=result['elapsed_walltime'], placed_count=result['placed_count'],
                expected_count=result['expected_count'], replay_frames=len(run['states']),
                last_bomb_step=run['telemetry'].get('last_bomb_step'),
                trajectory_exploration=result.get('trajectory_exploration'),
                resources=resources, issues=run['issues'], files=run['files'])


def summarize(old, new):
    if not np.array_equal(old['truth'], new['truth']):
        raise ValueError('The two scored truth maps differ')
    left, right = metrics(old), metrics(new)
    common = min(len(old['states']), len(new['states']))
    equal_frames = np.all(old['states'][:common] == new['states'][:common], axis=1)
    unequal = np.flatnonzero(~equal_frames)
    delta = {key: right[key] - left[key] for key in
             ('map_score', 'wall_score', 'time_score', 'red_partial_score', 'wall_hits',
              'wall_misses', 'charged_pixels', 'elapsed_steps', 'elapsed_walltime', 'placed_count')}
    return dict(old=left, new=right, new_minus_old=delta, same_truth=True,
                prediction_pixel_difference=int(np.count_nonzero(old['pred'] != new['pred'])),
                identical_common_state_frames=int(equal_frames.sum()),
                common_state_frames=common,
                first_different_state_frame=None if not len(unequal) else int(unequal[0]) + 1,
                exact_same_recorded_states=not len(unequal) and len(old['states']) == len(new['states']),
                exact_same_layout=old['bombs'] == new['bombs'],
                difficulty_score=None, competition_total=None,
                interpretation='Comparison of two recorded red phases; blue difficulty and full match total unevaluated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    old, new = load_run(args.old.resolve(strict=True)), load_run(args.new.resolve(strict=True))
    summary = summarize(old, new)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for kind in ('mapping', 'trajectories', 'bombs'):
        encoded = io.BytesIO()
        draw_comparison(old, new, kind).save(encoded, format='PNG')
        with (output / f'Map09_r1_{kind}.png').open('xb') as stream:
            stream.write(encoded.getbuffer())
    with (output / 'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(output)
    return 0 if not (old['issues'] or new['issues']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
