"""Audit submitted map pixels and recorded centre visits without running a controller."""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def one(items):
    item, = items
    return item


def region_bounds(width, height, module, enabled):
    yield 'all', (0, 0, width, height)
    for labels in [(('NW', 'NE'), ('SW', 'SE')),
                   (('NW9', 'N9', 'NE9'), ('W9', 'C9', 'E9'), ('SW9', 'S9', 'SE9'))]:
        n = len(labels)
        for row, names in enumerate(labels):
            for col, name in enumerate(names):
                yield name, (col * width / n, row * height / n,
                             (col + 1) * width / n, (row + 1) * height / n)
    if module is None:
        return
    tree = ast.parse(module.read_text(encoding='utf-8-sig'))
    spec = one(node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'ZONE_SPECS' for t in node.targets))
    for idx, (name, (w, h), (x, y)) in enumerate(ast.literal_eval(spec)):
        if name in enabled:
            cx, cy = width / 2 + x, height / 2 - y
            yield f'{name}_{idx}', (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def analyse(png, replay, expected_score, enabled, zone_module):
    # The engine uses two 2px dividers, a 24px heading, and a 104px footer.
    with Image.open(io.BytesIO(png)) as image:
        rgb = np.asarray(image.convert('RGB'))
    height, remainder = rgb.shape[0] - 128, (rgb.shape[1] - 4) % 3
    width = (rgb.shape[1] - 4) // 3
    if height < 1 or remainder or width < 1:
        raise ValueError('Unexpected comparison PNG dimensions')
    grids = []
    for start in (0, width + 2):
        panel = rgb[24:24 + height, start:start + width]
        if not np.all((panel == 40).all(axis=2) | (panel == 255).all(axis=2)):
            raise ValueError('Truth or submission panel is not a lossless binary panel')
        grids.append((panel[:, :, 0] == 40).astype(np.uint8))
    truth, submission = grids
    scorer = MapExplorationScorer()
    score = scorer.score(truth, submission, 0)
    if abs(score.s_map * 100 - expected_score) > 0.0051:
        raise ValueError('Recovered map score disagrees with receipt')
    wall, _ = scorer.split_regions(truth)
    hit = (wall > 0) & (submission > 0)
    distance = scorer.distance_field(truth)
    charged = (distance > scorer.config.dead_zone) & (submission > 0)
    far = (distance > 30) & (submission > 0)
    cost = scorer.penalty_curve(distance) * submission
    with np.load(io.BytesIO(replay), allow_pickle=False) as record:
        states = record['drone_states']
        if states.ndim != 2 or len(states) != len(record['timestamps']):
            raise ValueError('Inconsistent replay arrays')
    x = states['x'].astype(np.float64) + width / 2
    y = height / 2 - states['y'].astype(np.float64)
    active = states['active']
    if np.any(active & (~np.isfinite(x) | ~np.isfinite(y))):
        raise ValueError('Nonfinite live position')
    regions = {}
    for name, bounds in region_bounds(width, height, zone_module, enabled):
        left, top, right, bottom = bounds
        inside = active & (left <= x) & (x < right) & (top <= y) & (y < bottom)
        # A pixel is indexed by its integer column and row. Ceil expresses the
        # same half-open interval for integer pixel indices and float centres.
        crop = np.s_[max(0, int(np.ceil(top))):min(height, int(np.ceil(bottom))),
                     max(0, int(np.ceil(left))):min(width, int(np.ceil(right)))]
        regions[name] = {
            'bounds_ltrb': bounds,
            'active_frames_by_drone': inside.sum(axis=0).tolist(),
            'first_frame_by_drone': [int(np.argmax(m)) + 1 if m.any() else None
                                     for m in inside.T],
            'team_frames_any': int(inside.any(axis=1).sum()),
            'surface_pixels': int(np.count_nonzero(wall[crop])),
            'surface_hits': int(hit[crop].sum()),
            'charged_pixels': int(charged[crop].sum()),
            'wrong_over_30px': int(far[crop].sum()),
            'penalty_map_points': float(100 * scorer.config.penalty_weight *
                                        cost[crop].sum(dtype=np.float64) / score.wall.pixels),
        }
    if not np.isclose(regions['all']['penalty_map_points'], 100 * score.penalty, atol=0.0001):
        raise ValueError('Pixel penalty does not reconcile with score')
    return {
        'frames': len(states), 'end_active': active[-1].tolist(),
        'score_map': score.s_map * 100, 'dimensions': [width, height],
        'truth_sha256': digest(truth.tobytes()), 'states_sha256': digest(states.tobytes()),
        'png_sha256': digest(png), 'replay_sha256': digest(replay), 'regions': regions,
    }


def local_row(phase, zone_module):
    done = read_json(phase / 'done.json')
    attempt = phase / done['attempt']
    receipt = attempt / 'integrity.json'
    if digest(receipt.read_bytes()) != done['receipt_sha256']:
        raise ValueError(f'Invalid receipt: {phase}')
    frozen = read_json(receipt)
    plan = next(parent / 'plan.json' for parent in phase.parents if (parent / 'plan.json').is_file())
    if digest(plan.read_bytes()) != frozen['plan_sha256']:
        raise ValueError(f'Plan hash differs: {phase}')
    for path, expected in frozen['files'].items():
        if digest((attempt / path).read_bytes()) != expected:
            raise ValueError(f'File hash differs: {attempt / path}')
    run = read_json(attempt / 'result.json')
    if (run['status'] != 'ok' or run['walltime_limited'] or run['controller_crashed']
            or run['engine_callback_errors'] or not run['valid_shape'] or not run['submitted']):
        raise ValueError(f'Invalid red result: {phase}')
    png = one(k for k in frozen['files'] if k.endswith('_exploration_diff.png'))
    replay = one(k for k in frozen['files'] if k.startswith('artifacts/recordings/') and k.endswith('.npz'))
    zones = run['actual_conditions']['instantiated_zone_types']
    metrics = analyse((attempt / png).read_bytes(), (attempt / replay).read_bytes(),
                      run['phase1']['score_map'], zones, zone_module)
    return dict(metrics, map=run['map'], round=run['round'], seed=run['seed'],
                source=str(phase), controller=run['controller_class'], zones=zones,
                receipt_sha256=done['receipt_sha256'])


def platform_rows(path, zone_module):
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Platform archive CRC failure')
        index = one(n for n in archive.namelist() if n.endswith('/result.json'))
        report = json.loads(archive.read(index))
        for row in report['detail']:
            if row['崩溃'] != '否' or row['错误'] or row['分数状态'] != 'final':
                raise ValueError('Platform round has no valid final score')
            artifact = one(n for n in archive.namelist() if n.endswith('/' + row['炸弹文件']))
            settings = json.loads(archive.read(artifact))
            if row['地图'] != settings['map_name']:
                raise ValueError('Inconsistent platform map label')
            folder = artifact.rsplit('/', 1)[0]
            png = one(n for n in archive.namelist() if n.rpartition('/')[0] == folder
                      and n.endswith('_exploration_diff.png'))
            metrics = analyse(archive.read(png), archive.read(folder + '/run.npz'),
                              row['探测分'], settings['zones_config'], zone_module)
            yield dict(metrics, map=row['地图'], round=row['轮次'], seed=None,
                       source=folder, submission=report['submission_zip'], zones=settings['zones_config'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', type=Path, required=True)
    parser.add_argument('--local-red', type=Path, action='append', default=[])
    parser.add_argument('--zone-source', type=Path, help='Native walls module with literal ZONE_SPECS')
    parser.add_argument('--output', type=Path, required=True, help='New JSON file')
    args = parser.parse_args()
    rows = [local_row(path, args.zone_source) for path in args.local_red]
    rows.extend(platform_rows(args.platform, args.zone_source))
    if not rows or len({(r['map'], r['truth_sha256']) for r in rows}) != 1:
        raise ValueError('Map or truth differs across selected inputs')
    evidence = {'platform_sha256': digest(args.platform.read_bytes()),
                'zone_source_sha256': digest(args.zone_source.read_bytes()) if args.zone_source else None,
                'frames': '1-based; active body centre; zone rectangles may overlap',
                'points': 'Local map penalty normalized to the full scored surface', 'rows': rows}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
