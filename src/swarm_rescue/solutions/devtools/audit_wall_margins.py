"""Offline single-bomb contributions, using the actual exported positions."""
import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev


def audit(batch, name):
    ev.red.configure_ray_backend('cpu')
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreManager, walls_and_boundary_grid
    cls = getattr(importlib.import_module('swarm_rescue.maps.map_'+name[-2:]), name)
    world = cls(drone_type=DroneMotionless, number_drones=1)
    walls, boundary = walls_and_boundary_grid(world._playground)
    world._playground.window.close()
    rows = []
    for run in sorted(batch.glob(f'*/red/{name}_r*')):
        info = ev.read(run/'result.json')['phase1']['wall_breakdown']
        weight = np.where(boundary, info['boundary_wall_weight'], 1.)
        bombs = ev.read(ev.one_layout(run))['bombs']
        xy = np.array([(b['x'], b['y']) for b in bombs])
        with np.load(next((run/'artifacts/recordings').glob('*.npz')), allow_pickle=False) as z:
            states = z['bomb_states']
        last = states[-1]
        coords = np.column_stack((last['x'], last['y']))
        cost = np.linalg.norm(xy[:,None] - coords, axis=2)
        indices, slots = linear_sum_assignment(cost)
        if len(slots) != len(bombs) or not last['active'].all() or max(cost[indices, slots]) > 2:
            raise ValueError(f'Cannot identify recorded bombs {run}')
        masks = np.array([PlaceScoreManager.wall_destruction_masks(
            walls, [p.tolist()], info['blast_radius'], boundary)[2].astype(bool) for p in xy])
        count = masks.sum(axis=0)
        covered = np.zeros_like(walls, dtype=bool)
        times = np.argmax(states['active'][:,slots], axis=0)
        for i in np.argsort(times, kind='stable'):
            mask = masks[i]
            rows.append(dict(controller=run.parent.parent.name, run=run.name, bomb=int(i)+1,
                first_recorded_frame=int(times[i])+1, same_frame=int((times==times[i]).sum())>1,
                standalone=weight[mask].sum(), added=weight[mask & ~covered].sum(),
                exclusive=weight[mask & (count==1)].sum()))
            covered |= mask
        if weight[covered].sum() != info['weighted_destroyed']:
            raise ValueError(f'Wall union mismatch {run}')
    ev.atomic_json(batch/f'wall_margins_{name}.json', rows)


if __name__ == '__main__':
    audit(Path(sys.argv[1]), sys.argv[2])
