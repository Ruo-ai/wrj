"""R20 read-only attribution of historical R19 navigation evidence.

Replays only pose filtering and local free/obstacle observations. It does NOT
replay task selection, communication, commands or a flight. Original sampled
goals and logged route releases remain the evidence for those behaviours.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import math

import cv2
import numpy as np

from swarm_rescue.solutions.devtools.red_pair19_batch import REPO, read, write, digest
from swarm_rescue.solutions.devtools.red_recovery18_audit import load_npz, semantic_objects, legal_arguments
from swarm_rescue.solutions.my_drone_red_advanced2 import _M01_PoseFilter, _M19_SurveyGrid
from swarm_rescue.solutions.devtools.red_pair19_results_review import truth_metric, only
from swarm_rescue.solutions.devtools.optimization.evaluation.runner import load_done, phase_valid


def log_events(run):
    result = []
    for line in (run/'run.log').read_text('utf-8').splitlines():
        if '_EVENT ' in line and line.startswith('V'):
            result.append(json.loads(line.split('_EVENT ',1)[1]))
    return result


def gates(spec):
    """Offline geometry labels only: gaps between collinear wall segments."""
    lines = defaultdict(list)
    start = next(e for e in spec['elements'] if e['type'] == 'start_area')
    home = (np.array(start['top_left'])+start['bottom_right'])/2
    for e in spec['elements']:
        if e['type'] != 'line_wall': continue
        (x1,y1),(x2,y2) = e['points']
        if x1 == x2: lines['v',x1].append((min(y1,y2),max(y1,y2),e['thickness']))
        elif y1 == y2: lines['h',y1].append((min(x1,x2),max(x1,x2),e['thickness']))
    result = []
    for (axis, fixed), spans in lines.items():
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            gap = right[0]-left[1]
            if not 24 <= gap <= 120: continue
            center = [fixed, (left[1]+right[0])/2] if axis == 'v' else [(left[1]+right[0])/2, fixed]
            result.append(dict(axis=axis, center_px=center, nominal_gap=gap,
                               endpoint_thickness=max(left[2],right[2]), home_distance=float(np.linalg.norm(np.array(center)-home))))
    result.sort(key=lambda g:g['home_distance'])
    for index, g in enumerate(result,1):g['id']=f'G{index:02d}'
    return result


def navigation_graph(grid, position):
    walls = cv2.dilate((grid.obstacle>=2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))).astype(bool)
    allowed = grid.free & ~walls; start=grid.cell(position);allowed[start]=True
    _, labels = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=4)
    reachable = (labels==labels[start]) & allowed
    return walls, allowed, reachable


def analyze(batch, name, out):
    run=batch/'runs'/f'{name}_r01'/'rl17_repair/red/attempt_001'
    original,checked=load_done(run.parent,digest(batch/'plan.json'))
    if checked!=run or not phase_valid(original,'red'):raise ValueError('Historical attempt invalid')
    dest=out/name;dest.mkdir(exist_ok=False)
    raw,truth,metric=truth_metric(run);h,w=truth.shape
    spec=read(read(run/'request.json')['map_file']);doors=gates(spec)
    inputs=load_npz(run/'mapper_legal_inputs.npz');objects=semantic_objects(inputs,read(run/'line_diagnostics.json'))
    motion=read(run/'motion_diagnostics.json');exploration=read(run/'exploration_diagnostics.json')
    local=read(run/'local13_diagnostics.json');events=log_events(run);summaries=[];arrays={}
    for ident, exp in exploration['drones'].items():
        owner=int(ident);data=inputs['drone_'+ident];angles=inputs['angles_'+ident]
        pose=_M01_PoseFilter();grid=_M19_SurveyGrid((w,h),lambda *a,**k:None)
        pose_samples={s['tick']:s for s in motion['drones'][ident]['samples']}
        exps={s['step']:s for s in exp['samples']};mine=[e for e in events if e['drone']==owner]
        holds=defaultdict(list)
        for e in mine:
            if e['event'] in ('hold_start','hold_end'):holds[e['step']-1].append(e['event']=='hold_start')
        holding=False;errors=[];rows=[];max_pose_error=0.;nav=np.zeros((len(data),3));matched=0
        for rowdata in data:
            tick=int(rowdata[0]);gps,compass,odom,d,a,semantic=legal_arguments(rowdata,angles,objects[owner,tick])
            answer=pose.update(gps,compass,odom)
            if answer is None:raise ValueError('Missing initialized pose')
            pos,heading=answer;nav[tick-1]=[*pos,heading]
            if tick in pose_samples:
                error=float(np.max(np.abs(pos-pose_samples[tick]['navigation']['position'])));max_pose_error=max(max_pose_error,error)
                if error>1e-9:errors.append(dict(step=tick,kind='pose',error=error))
            for value in holds[tick]:holding=value
            if not holding and d is not None and (tick==1 or tick%6==0):
                dynamic=np.zeros(len(d),bool)
                for obj in semantic:
                    if obj.entity_type.name in ('DRONE','BOMB'):
                        delta=np.abs((a-obj.angle+math.pi)%(2*math.pi)-math.pi)
                        dynamic|=(delta<.19)&(np.abs(d-obj.distance)<25)
                grid.observe(pos,heading,d,a,dynamic)
            if tick not in exps:continue
            sample=exps[tick];free=int(grid.free.sum());visits=int((grid.visits>0).sum())
            if (free,visits)!=(sample['free_cells'],sample['local_visits']):
                errors.append(dict(step=tick,kind='grid_counts',actual=[free,visits],recorded=[sample['free_cells'],sample['local_visits']]))
            else:matched+=1
            walls,allowed,reach=navigation_graph(grid,pos);gate_rows=[]
            for gate in doors:
                x,y=gate['center_px'];point=np.array([x-w/2,h/2-y]);cell=grid.cell(point)
                normal=np.array([1,0]) if gate['axis']=='v' else np.array([0,1])
                side_cells=[grid.cell(point+normal*offset) for offset in (-60,60)]
                gate_rows.append(dict(id=gate['id'],distance=float(np.linalg.norm(pos-point)),
                                      free=bool(grid.free[cell]),obstacle=float(grid.obstacle[cell]),inflated=bool(walls[cell]),
                                      reachable=bool(reach[cell]),side_free=[bool(grid.free[c]) for c in side_cells],
                                      side_reachable=[bool(reach[c]) for c in side_cells]))
            goal=None if sample['goal'] is None else grid.cell(sample['goal'])
            rows.append(dict(step=tick,role=sample['role'],position=pos.tolist(),truth=sample['truth'],
                             goal=sample['goal'],goal_reachable=None if goal is None else bool(reach[goal]),
                             free_cells=free,reachable_cells=int(reach.sum()),holding=holding,gates=gate_rows))
            arrays[f'd{owner}_t{tick}_free']=grid.free.copy();arrays[f'd{owner}_t{tick}_obstacle']=grid.obstacle.copy()
        arrays[f'nav_{owner}']=nav
        releases=Counter(e.get('reason') for e in mine if e['event']=='route_released')
        recovery=Counter(e.get('reason') for e in mine if e['event']=='recovery')
        stats=motion['drones'][ident];ls=local['drones'][ident]['samples']
        with np.load(only(run/'artifacts/recordings','*.npz'),allow_pickle=False) as recording:
            states=recording['drone_states'];t=states[:,owner]
            distances=np.hypot(t['x']-t['x'][0],t['y']-t['y'][0]);xy=np.c_[t['x'],t['y']]
            peer=np.stack([np.hypot(t['x']-states[:,i]['x'],t['y']-states[:,i]['y']) for i in range(states.shape[1]) if i!=owner])
        gate_summary=[]
        for g in doors:
            vals=[next(v for v in r['gates'] if v['id']==g['id']) for r in rows];x,y=g['center_px'];wp=np.array([x-w/2,h/2-y])
            gate_summary.append(dict(id=g['id'],min_truth_distance=float(np.linalg.norm(xy-wp,axis=1).min()),
                samples=len(vals),free_samples=sum(v['free'] for v in vals),inflated_samples=sum(v['inflated'] for v in vals),
                reachable_samples=sum(v['reachable'] for v in vals),both_sides_reachable_samples=sum(all(v['side_reachable']) for v in vals),
                near_goal_samples=int(sum(r['goal'] is not None and np.linalg.norm(np.asarray(r['goal'])-wp)<90 for r in rows))))
        record=dict(owner=owner,steps=len(data),grid_count_matches=matched,grid_sample_count=len(exps),pose_max_abs=max_pose_error,
                    reconstruction_valid=not errors,reconstruction_errors=errors,events=mine,samples=rows,gate_summary=gate_summary,
                    roles=exp['roles'],repeat_samples=exp['repeat_samples'],route_release_reasons=dict(releases),recovery_reasons=dict(recovery),
                    outage_steps=stats['outage_ticks'],distance=stats['distance'],max_displacement=float(distances.max()),
                    within180_fraction=float((distances<=180).mean()),peer_within40_fraction=float((peer.min(axis=0)<40).mean()),
                    comm_disabled_samples=sum(s['communication_disabled'] for s in ls),comm_total_samples=len(ls),
                    last_status=next((e for e in reversed(mine) if e['event']=='status'),None),
                    placements=[e for e in mine if e['event']=='placement_confirmed'])
        summaries.append(record)
        print(name,'d',owner,'grid',matched,'/',len(exps),'pose',max_pose_error,'gates',gate_summary[:2],flush=True)
    np.savez_compressed(dest/'navigation_reconstruction.npz',**arrays)
    report=dict(map=name,round=1,seed=read(run/'request.json')['seed'],controller='RL17-R',source=str(run),
                scope='historical input reconstruction; no task-selection or flight replay',gates=doors,drones=summaries,
                collision_contacts=motion['collision_contacts'],all_reconstructions_valid=all(r['reconstruction_valid'] for r in summaries),
                limitations=['Grid pixels were not recorded historically; reconstructed from exact inputs/code, checked against all saved counts and sampled poses.',
                             'Candidate sets, claims and complete routes were not saved; sampled final goals cannot prove every selection reason.',
                             'Gate geometry and true poses are offline labels only; nominal gap is not a demonstrated safe passage.'])
    write(dest/'analysis.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False)
    results=[analyze(args.batch.resolve(),name,out) for name in ('yh_map_fs_72','yh_map_fs_55')]
    write(out/'summary.json',dict(maps=[dict(map=r['map'],valid=r['all_reconstructions_valid'],collisions=r['collision_contacts']) for r in results]))


if __name__=='__main__':main()
