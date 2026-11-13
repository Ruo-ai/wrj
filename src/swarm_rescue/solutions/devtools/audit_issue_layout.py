"""Match the numbered screenshots against archived L3 runs and explain coverage."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev
from swarm_rescue.solutions.devtools.blind_layout_attribution import trace_audit
from swarm_rescue.solutions.devtools.report_advanced5 import events


def main(run):
    ev.red.configure_ray_backend('cpu')
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreManager, walls_and_boundary_grid
    result = ev.read(run/'result.json')
    name = result['map']
    cls = getattr(importlib.import_module('swarm_rescue.maps.map_'+name[-2:]),name)
    world = cls(drone_type=DroneMotionless,number_drones=1)
    walls, border = walls_and_boundary_grid(world._playground)
    world._playground.window.close()
    xy = np.array([(b['x'],b['y']) for b in ev.read(ev.one_layout(run))['bombs']])
    info = result['phase1']['wall_breakdown']
    masks = np.array([PlaceScoreManager.wall_destruction_masks(walls,[p.tolist()],
        info['blast_radius'],boundary_mask=border)[2].astype(bool) for p in xy])
    weight = np.where(border,info['boundary_wall_weight'],1.)
    with np.load(next((run/'artifacts/recordings').glob('*.npz')),allow_pickle=False) as replay:
        audit = trace_audit(replay,xy,masks,weight)
    assert audit['team']==info['weighted_destroyed']
    log = events(run)
    for row in audit['bombs']:
        i = row['bomb']-1
        before = [e for e in log if e['drone']==row['owner'] and e['step']<=row['frame']+1]
        decisions = [e for e in before if e['event']=='a6_selection' and e['selected']]
        decision = decisions[-1] if decisions else None
        row['seen_at_last_selection'] = [] if not decision else [
            j+1 for j in range(i) if any(np.linalg.norm(p-xy[j])<30 for p in decision['placed'])]
        row['selection_step'] = decision['step'] if decision else None
        row['release_check'] = any(e['event']=='placement_checked' and
                                   e['step']>=row['frame']-1 for e in before)
        row['exclusive'] = float(weight[masks[i] & (masks.sum(axis=0)==1)].sum())
        # Nearest actual wall pixel, in the scorer's grid coordinates.
        rr,cc = PlaceScoreManager._world_to_grid(*xy[i],walls.shape[1],walls.shape[0])
        y,x = np.nonzero(walls)
        row['wall_distance'] = float(np.hypot(y-rr,x-cc).min())
    ev.atomic_json(run.parents[2]/f'{run.parent.parent.name}_{name}_issue.json',audit)
    for row in audit['bombs']:
        print(name,row['bomb'],row['owner'],row['frame'],row['gross'],row['added'],
              row['exclusive'],round(row['wall_distance']),row['selection_step'],row['seen_at_last_selection'],row['release_check'])
    print(audit['overlap_pairs'])


if __name__=='__main__':
    main(Path(sys.argv[1]))
