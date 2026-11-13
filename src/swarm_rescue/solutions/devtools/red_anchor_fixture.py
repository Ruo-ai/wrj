"""Offline A5 fixtures with complete identical synthetic sensor inputs.

Truth here is a tiny artificial wall, never a competition map or online input.
Scores reuse the official map formula, but are NOT competition results.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np

from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper
from swarm_rescue.solutions.my_drone_red_mapping_anchor import StableAnchorMapper


def inputs(case):
    angles = np.linspace(-.65, .65, 48)
    position = np.zeros(2)
    heading = 0.
    rows = []
    for step in range(124):  # 32 initial, 80 outage, 12 recovery
        moving = case == 'turn_first_outlier' and step > 0
        odom = np.array([.12, 0., .004]) if moving else np.zeros(3)
        position = position + odom[0]*np.array([np.cos(heading), np.sin(heading)])
        heading += odom[2]
        absolute_angles = angles+heading
        distances = (90-position[0])/np.cos(absolute_angles)
        ys = position[1]+distances*np.sin(absolute_angles)
        distances[(distances <= 4) | (distances >= 226) | (np.abs(ys) > 70)] = 230.
        gps, compass = position.copy(), heading
        if 32 <= step < 112:
            gps, compass = None, None
        elif step >= 112:
            if case in ('first_outlier', 'turn_first_outlier') and step == 112:
                gps = gps+np.array([30., -12.]); compass += .2
            elif case == 'persistent_bias':
                gps = gps+np.array([15., 0.]); compass += .08
        rows.append((gps, compass, odom, distances.copy(), angles.copy()))
    return rows, position, heading


def input_hash(rows):
    h = hashlib.sha256()
    for row in rows:
        for value in row:
            if value is None:
                h.update(b'NONE')
            else:
                arr = np.asarray(value, dtype='<f8')
                h.update(str(arr.shape).encode('ascii')); h.update(arr.tobytes())
    return h.hexdigest()


def run_case(case, mapper_class):
    rows, position, heading = inputs(case)
    mapper = mapper_class((400, 400))
    mapper.RECORD_ANCHORS = True
    begin = time.perf_counter(); cpu = time.process_time()
    for row in rows:
        mapper.update(*row)
    elapsed, process = time.perf_counter()-begin, time.process_time()-cpu
    pred = mapper.to_pixels(mapper.mask())
    return mapper, pred, dict(input_sha256=input_hash(rows), steps=len(rows),
        update_wall_seconds=elapsed, update_process_seconds=process,
        endpoint_error_px=float(np.linalg.norm(mapper.position-position)),
        endpoint_heading_error_rad=float(abs(mapper.heading-heading)),
        stats=mapper.stats, anchor_events=list(getattr(mapper, 'anchor_events', [])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer
    scorer = MapExplorationScorer()
    truth = np.zeros((400, 400), np.uint8); truth[130:271, 290:296] = 1
    surface, _ = scorer.split_regions(truth)
    records = []
    for case in ('clean', 'first_outlier', 'turn_first_outlier', 'persistent_bias'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, cls, label in zip(axes, (SubmissionMapper, StableAnchorMapper), ('A3', 'A5')):
            mapper, pred, record = run_case(case, cls)
            score = scorer.score(truth, pred, 0.)
            record.update(case=case, controller=label, synthetic_map_formula=asdict(score))
            records.append(record)
            pixels = np.ones((400, 400, 3))*.96
            pixels[truth > 0] = (.3, .5, .9)
            pixels[(pred > 0) & (truth == 0)] = (.85, .2, .2)
            pixels[(pred > 0) & (surface > 0)] = (.15, .65, .3)
            ax.imshow(pixels); ax.set_xlim(250, 355); ax.set_ylim(295, 100)
            ax.set_title(f'{label}: hit {score.wall.hits}, charged {score.charged_pixels}\n'
                         f'endpoint error {record["endpoint_error_px"]:.2f}px')
        fig.suptitle(case+' / SYNTHETIC fixture, NOT competition scores')
        fig.tight_layout(); fig.savefig(args.output/(case+'.png'), dpi=120); plt.close(fig)
    # Separate allocation run so tracing overhead is not mixed into timings.
    tracemalloc.start()
    mapper, _, _ = run_case('first_outlier', StableAnchorMapper)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    result = dict(kind='synthetic_same_input_component_check', records=records,
                  allocation_check=dict(tracemalloc_peak_bytes=peak,
                     caveat='Python/NumPy traced allocation; not native engine RSS or official budget',
                     maximum_recovery_samples=mapper.stats['max_recovery_samples'],
                     maximum_pending_scans=mapper.stats['max_pending_scans']))
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
