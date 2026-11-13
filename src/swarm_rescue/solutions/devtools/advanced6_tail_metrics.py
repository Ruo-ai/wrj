"""Verify full-inventory frame and compute active-drone travel after it."""
import sys
from pathlib import Path

import numpy as np

from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev


def main(batch):
    results = []
    for run in sorted(batch.glob('*/red/Map*_r*')):
        row = ev.read(run/'result.json')
        with np.load(next((run/'artifacts/recordings').glob('*.npz')), allow_pickle=False) as data:
            full = np.flatnonzero(data['bomb_states']['active'].sum(axis=1)==row['expected_count'])
            first = int(full[0])
            states = data['drone_states'][first:]
            xy = np.stack((states['x'], states['y']), axis=-1)
            valid = states['active'][1:] & states['active'][:-1]
            distance = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
            results.append(dict(controller=row['controller_selector'], run=run.name, full_frame=first+1,
                                tail_frames=len(states)-1, tail_px=float(distance[valid].sum(dtype=np.float64)),
                                steps=row['elapsed_steps'], red_partial=row['phase1']['round_score']))
    ev.atomic_json(batch/'tail_metrics.json', results)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
