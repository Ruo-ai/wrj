"""Offline tracking of bomb motion. Recording columns can shift after disposal."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def audit(run):
    layout = json.loads((run/'layout.json').read_text(encoding='utf-8'))
    starts = np.array([(b['x'], b['y']) for b in layout['bombs']])
    ids = np.arange(len(starts))
    last = starts
    records = [dict(id=i+1, first_motion=None, disappearance=None, max_distance=0.)
               for i in range(len(starts))]
    recording, = (run/'artifacts/recordings').glob('*.npz')
    with np.load(recording, allow_pickle=False) as data:
        for step, frame in enumerate(data['bomb_states'], 1):
            active = frame[frame['active']]
            current = np.column_stack((active['x'], active['y']))
            cost = np.linalg.norm(last[:, None] - current, axis=2)
            rows, cols = linear_sum_assignment(cost)
            if len(cols) != len(current) or np.any(cost[rows, cols] > 30):
                raise ValueError(f'Ambiguous frame transition {run} {step}')
            for i in set(ids) - set(ids[rows]):
                records[i]['disappearance'] = step
            ids, last = ids[rows], current[cols]
            for i, distance in zip(ids, np.linalg.norm(last-starts[ids], axis=1)):
                row = records[i]
                row['max_distance'] = max(row['max_distance'], round(float(distance), 1))
                if distance > 35 and row['first_motion'] is None:
                    row['first_motion'] = step
    return records


if __name__ == '__main__':
    base = Path(sys.argv[1])
    records = {p.relative_to(base).as_posix(): audit(p.parent)
               for p in sorted(base.glob('*/*/layout.json'))}
    Path(sys.argv[2]).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
