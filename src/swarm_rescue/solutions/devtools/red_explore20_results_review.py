"""Read-only R20 results audit and attribution; never starts a simulation."""
import argparse
from collections import Counter
from pathlib import Path
import re
import statistics
import zipfile

import numpy as np
from PIL import Image

from swarm_rescue.solutions.devtools.red_pair19_batch import read, write, digest
from swarm_rescue.solutions.devtools.red_pair19_results_review import truth_metric, only, compare_npz
from swarm_rescue.solutions.devtools.red_recovery19_weights import check_radio, metric_count
from swarm_rescue.solutions.devtools.red_explore20_audit import log_events, gates, navigation_graph
from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper
from swarm_rescue.solutions.my_drone_red_repair17 import RepairMapper17
from swarm_rescue.simulation.reporting.time_score import time_score_fraction


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values)) if values else {}


def run_detail(row):
    run = Path(row['attempt']); raw, truth, metric = truth_metric(run); h, w = truth.shape
    exp = read(run/'exploration_diagnostics.json'); trace = read(run/'explore20_diagnostics.json')
    radio = read(run/'p2l17_diagnostics.json'); motion = read(run/'motion_diagnostics.json')
    narrow = read(run/'narrow_diagnostics.json'); events = log_events(run)
    if len(exp['submissions']) != 1: raise ValueError('Expected exactly one actual submission')
    submission = exp['submissions'][0]; publications, accepted = check_radio(radio['radio'], submission)
    pred = np.asarray(read(only(run/'artifacts', '*_exploration.json'))['grid'], np.uint8)
    got = metric_count(metric, pred)
    for a,b in [('hits','wall_hits'),('missed','wall_missed'),('charged','charged_pixels'),('map_score','map_score')]:
        if abs(got[a]-row[b]) > 1e-6: raise ValueError('Map score mismatch: '+a)
    t = raw['phase1']['time_breakdown']; wb = raw['phase1']['wall_breakdown']
    if abs(100*time_score_fraction(t['elapsed'],t['time_limit'],t['time_best_ratio'])-row['time_score'])>1e-9:
        raise ValueError('Time score mismatch')
    if abs(100*min(1.,wb['weighted_destroyed']/wb['theoretical_optimum'])-row['wall_score'])>1e-9:
        raise ValueError('Wall breakdown formula mismatch')
    bombs = read(only(run/'artifacts','*_bombs.json'))['bombs']
    if len(bombs)!=row['bombs'] or raw['remaining_inventory']!=0: raise ValueError('Bomb inventory mismatch')
    mapper = (WallLineMapper if row['controller']=='rl11_lines' else RepairMapper17)((w,h))
    drones=[]
    with np.load(run/'exploration_stages.npz',allow_pickle=False) as maps, np.load(run/'local13_diagnostics.npz',allow_pickle=False) as visible:
        masks={i:maps[f"submitter_{submission['drone']}_source_{i}"] for i in accepted}
        merged=mapper.to_pixels(np.logical_or.reduce(list(masks.values())))
        if not np.array_equal(merged,pred):raise ValueError('Actual received-source union differs from submission')
        for ident in range(5):
            key=str(ident); final=trace['final'][key]; m=motion['drones'][key]; e=exp['drones'][key]
            mine=[x for x in events if x['drone']==ident]
            samples=trace['drones'][key]; ne=narrow['drones'][key]['events']
            own=maps[f'local_final_pixels_{ident}']; received=mapper.to_pixels(masks[ident])
            localmask=maps[f'local_final_{ident}']
            if not np.array_equal(mapper.to_pixels(localmask),own):raise ValueError('Final local pixel conversion mismatch')
            mask=visible['visible_'+key].astype(bool); mask=np.repeat(np.repeat(mask,4,0),4,1)[:h,:w]
            nav_errors=[s['navigation']['position_error'] for s in m['samples']]
            placements=[x['step'] for x in mine if x['event']=='placement_confirmed']
            without=mapper.to_pixels(np.logical_or.reduce([v for k,v in masks.items() if k!=ident]))
            without_score=metric_count(metric,without)
            drones.append(dict(owner=ident,roles=e['roles'],repeat_samples=e['repeat_samples'],
                sampled_visible_free_percent=100*int((mask&~truth.astype(bool)).sum())/int((~truth.astype(bool)).sum()),
                local=metric_count(metric,own),received=metric_count(metric,received),
                leave_one_source_out_delta={k:got[k]-without_score[k] for k in ('hits','missed','charged','map_score')},
                source_version=accepted[ident],source_age=submission['step']-accepted[ident],
                final_local_published_version=radio['drones'][key]['versions'][key],
                missing_local_cells=int((localmask&~masks[ident]).sum()),
                outdated_received_cells=int((masks[ident]&~localmask).sum()),
                outage_steps=m['outage_ticks'],distance=m['distance'],
                navigation_error_p95=float(np.percentile(nav_errors,95)),navigation_error_max=max(nav_errors),
                renewals=[x['tick'] for x in ne if x['reason']=='r20_budget_renewed'],
                placement_engine_steps=placements,rn7=final['rn7_stats'],
                passage_episodes=final['lifetime_episodes'],stalled_steps=final['stalled_steps'],yield_steps=final['yield_steps'],
                route_releases=dict(Counter(x.get('reason') for x in mine if x['event']=='route_released')),
                recovery_reasons=dict(Counter(x.get('reason') for x in mine if x['event']=='recovery')),
                communication_disabled_samples=sum(s['communication_disabled'] for s in samples),
                sample_count=len(samples)))
    placement_steps=[t for d in drones for t in d['placement_engine_steps']]
    if len(placement_steps)!=row['bombs']:raise ValueError('Confirmed placement log count differs')
    return dict(map=row['map'],round=row['round'],seed=row['seed'],controller=row['controller'],submission=submission,
        radio_publications=len(publications),radio_events=len(radio['radio']),
        received_union_matches_submission=True,actual_submission_metric=got,drones=drones,
        first_bomb_step=min(placement_steps),last_bomb_step=max(placement_steps),
        max_source_age=max(d['source_age'] for d in drones),mean_source_age=statistics.mean(d['source_age'] for d in drones),
        sum_outage_steps=sum(d['outage_steps'] for d in drones),
        sum_local_active_ticks=sum(d['rn7'].get('active_ticks',0) for d in drones),
        renewals=sum(len(d['renewals']) for d in drones),collision_contacts=motion['collision_contacts'],
        saved_plan_count=len(trace['plans']),score_checks=['map_from_pixels','time_curve','wall_breakdown_formula','bomb_count'],
        official_trajectory_exploration=raw['trajectory_exploration'],mean_health=raw['mean_health'])


def focus_plans(row):
    """Inspect actual saved pre-plan grids, without reconstructing a flight."""
    run=Path(row['attempt']);doc=read(run/'explore20_diagnostics.json')
    spec=read(read(run/'request.json')['map_file']);door=gates(spec)[0];rows=[]
    with np.load(run/'explore20_diagnostics.npz',allow_pickle=False) as arrays:
        for plan in doc['plans']:
            state=plan['state'];grid=_M19_SurveyGrid((state['width'],state['height']),lambda *a,**k:None)
            for k in ('free','obstacle'):setattr(grid,k,arrays[state[k]['array']])
            pos=np.asarray(plan['args'][0]);walls,_,reachable=navigation_graph(grid,pos)
            x,y=door['center_px'];point=np.array([x-grid.width/2,grid.height/2-y]);cell=grid.cell(point)
            route=plan['selected_route'];goal=grid.point(route[-1]) if route else None
            rows.append(dict(owner=plan['owner'],step=plan['step'],goal=None if goal is None else goal.tolist(),
                free=bool(grid.free[cell]),raw_obstacle=bool(grid.obstacle[cell]>=2),inflated=bool(walls[cell]),
                reachable=bool(reachable[cell]),near_gate_goal=goal is not None and float(np.linalg.norm(goal-point))<90,
                route_empty=not route))
    return dict(map=row['map'],round=row['round'],controller=row['controller'],gate=door,
        source='actual saved pre-plan grid snapshots; counts are planning instants, not all control steps',
        drones=[dict(owner=i,plans=len([x for x in rows if x['owner']==i]),
            **{k:sum(x[k] for x in rows if x['owner']==i) for k in ('free','raw_obstacle','inflated','reachable','near_gate_goal','route_empty')}) for i in range(5)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();batch=a.batch.resolve();out=a.output.resolve();summary=read(batch/'summary.json');plan=read(batch/'plan.json')
    details=[];failures=[];warnings=[];intervals=[];crc=0;png=0;pairs=[];focus=[]
    for index,row in enumerate(summary['rows'],1):
        run=Path(row['attempt'])
        try:
            rec=read(batch/f'run_{index:02d}.json');raw=read(run/'result.json')
            if rec['item']!=plan['schedule'][index-1] or rec['result']!=raw or rec['changed_during_run']:raise ValueError('Wrapper mismatch')
            request=read(run/'request.json')
            if digest(request['map_file'])!=row['map_sha256']:raise ValueError('Map hash mismatch')
            for path in run.rglob('*.npz'):
                with zipfile.ZipFile(path) as z:
                    if z.testzip():raise ValueError('NPZ CRC '+str(path))
                crc+=1
            for path in run.rglob('*.png'):
                with Image.open(path) as im:im.verify()
                png+=1
            detail=run_detail(row);details.append(detail)
            if row['map'] in ('yh_map_fs_72','yh_map_fs_55') and row['round']==1:focus.append(focus_plans(row))
            matches=[line for line in (run/'run.log').read_text('utf-8',errors='replace').splitlines()
                     if re.search(r'Warning:|WARNING:|Traceback|Exception ignored|AssertionError|ERROR:|RuntimeError:',line)]
            if matches:warnings.append(dict(map=row['map'],round=row['round'],controller=row['controller'],lines=matches))
            intervals.append(dict(index=index,start=(run/'request.json').stat().st_mtime,end=(run/'integrity.json').stat().st_mtime))
            print('AUDITED',row['map'],row['round'],row['controller'],flush=True)
        except Exception as exc:failures.append(dict(attempt=str(run),error=repr(exc)))
    for name,number in sorted({(r['map'],r['round']) for r in summary['rows']}):
        rows=[r for r in summary['rows'] if r['map']==name and r['round']==number];arms={r['controller']:Path(r['attempt']) for r in rows}
        x,y=arms['rl11_lines'],arms['rl17_repair']
        equality=compare_npz(only(x/'artifacts/recordings','*.npz'),only(y/'artifacts/recordings','*.npz'),['drone_states','bomb_states','timestamps'])
        coverage=compare_npz(x/'local13_diagnostics.npz',y/'local13_diagnostics.npz')
        conditions=[read(v/'result.json') for v in arms.values()]
        same_conditions=all(all(q[k]==conditions[0][k] for k in ('actual_conditions','engine_profile','score_configuration')) for q in conditions)
        if not same_conditions:failures.append(dict(map=name,round=number,error='Pair conditions mismatch'))
        pairs.append(dict(map=name,round=number,rl11_rl17_recording=equality,rl11_rl17_coverage=coverage,all_conditions_same=same_conditions))
    serial=all(x['end']<=y['start'] for x,y in zip(intervals,intervals[1:]))
    if not serial:failures.append(dict(error='Run file intervals overlap'))
    audit=dict(valid=not failures,failures=failures,log_warnings=warnings,checked_npz_crc=crc,checked_png=png,
        pair_checks=pairs,observed_serial=serial,intervals=intervals,
        external_cpu_contention='Cannot establish retrospectively from local receipts; no CPU-utilization history.',
        limitation='Wall score independently checked from saved breakdown formula; wall geometry scoring engine unchanged. No blue/full score.')
    write(out/'detailed_audit.json',audit);write(out/'run_details.json',details);write(out/'focus_planning.json',focus)
    print('AUDIT_VALID',audit['valid'],'failures',len(failures),'warnings',len(warnings),flush=True)
    if failures:raise ValueError(failures)


if __name__=='__main__':main()
