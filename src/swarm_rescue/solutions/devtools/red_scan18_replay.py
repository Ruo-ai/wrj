"""RL18-B fixed-tape shadow run; truth passed only to post-estimation metrics."""
import argparse
from collections import Counter
from pathlib import Path
import time

import cv2
import numpy as np

from swarm_rescue.solutions.devtools.red_recovery18_audit import (
    read, write, digest, load_npz, legal_arguments, semantic_objects, EVENTS,
)
from swarm_rescue.solutions.devtools.red_scan18_shadow import ShortWindow18, integrate
from swarm_rescue.solutions.devtools.red_p2l17_replay import ReplayProcess, summary


def wrap_array(a): return (a+np.pi)%(2*np.pi)-np.pi


def metric_arrays(poses,truth,data,angles,missing):
    position=np.linalg.norm(poses[:,:2]-truth[:,:2],axis=1)
    heading=np.abs(wrap_array(poses[:,2]-truth[:,2]))*180/np.pi
    rel=np.full(len(data),np.nan);trans=rel.copy(); drift=rel.copy()
    # Four steps match the estimator's scan cadence. Never span a recovery.
    valid=missing.copy();valid[:4]=False
    for lag in range(1,5):valid[lag:] &= missing[:-lag]
    rel[4:]=np.abs(wrap_array(wrap_array(poses[4:,2]-poses[:-4,2])-wrap_array(truth[4:,2]-truth[:-4,2])))*180/np.pi
    delta=poses[4:,:2]-poses[:-4,:2];td=truth[4:,:2]-truth[:-4,:2]
    c,s=np.cos(poses[:-4,2]),np.sin(poses[:-4,2]);ct,st=np.cos(truth[:-4,2]),np.sin(truth[:-4,2])
    body=np.column_stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]))
    tb=np.column_stack((ct*td[:,0]+st*td[:,1],-st*td[:,0]+ct*td[:,1]))
    trans[4:]=np.linalg.norm(body-tb,axis=1)
    rel[~valid]=np.nan;trans[~valid]=np.nan
    start=None
    for i,is_missing in enumerate(missing):
        if is_missing:
            if start is None:start=max(0,i-1)
            drift[i]=abs(wrap_array((poses[i,2]-poses[start,2])-(truth[i,2]-truth[start,2])))*180/np.pi
        else:start=None
    far=[];far_ticks=[]
    for i in np.flatnonzero(missing):
        d=data[i,11:]; good=bool(data[i,4]) & np.isfinite(d) & (d>=150) & (d<218) & np.isfinite(angles)
        # Identical measured ranges in both projections isolate pose amplification.
        # This is NOT distance to actual world hitpoint; range noise cancels.
        rr=d[good]; aa=angles[good]
        x=poses[i,0]-truth[i,0]+rr*(np.cos(aa+poses[i,2])-np.cos(aa+truth[i,2]))
        y=poses[i,1]-truth[i,1]+rr*(np.sin(aa+poses[i,2])-np.sin(aa+truth[i,2]))
        far.extend(np.hypot(x,y).tolist());far_ticks.extend([i+1]*len(rr))
    return dict(position=position,heading=heading,relative_turn=rel,relative_translation=trans,
                accumulated_heading=drift,far_endpoint=np.array(far),far_tick=np.array(far_ticks))


def summarize(a,mask):
    result={}
    for key,value in a.items():
        if key in ('far_endpoint','far_tick'):continue
        selected=value[mask];result[key]=summary(selected[np.isfinite(selected)])
    selected=a['far_endpoint'][mask[a['far_tick']-1]] if len(a['far_tick']) else np.array([])
    result['far_endpoint']=summary(selected)
    return result


def run_map(batch,baseline,out,map_name,process):
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    run=batch/f'runs/{map_name}_r01/rl11_lines/red/attempt_001'
    input_data=load_npz(run/'mapper_legal_inputs.npz');meta=read(run/'line_diagnostics.json')
    objects=semantic_objects(input_data,meta);recorded=load_npz(run/'p2l17_diagnostics.npz')
    references={c:load_npz(baseline/f'{map_name}_{c}/maps_poses.npz') for c in ('rl11_lines','rl17_repair')}
    arrays={};stats={};all_metrics={};latencies=[];matches=[];ranges=[]
    for owner in range(5):
        estimator=ShortWindow18();data=input_data[f'drone_{owner}'];angles=input_data[f'angles_{owner}']
        expected=recorded[f'pose_{owner}'];assert np.array_equal(expected[:,0],data[:,0])
        truth=expected[:,1:4];missing=~(data[:,1].astype(bool)&data[:,2].astype(bool))
        poses=[];odometry=[];odom=None;costs=[]
        for i,row in enumerate(data):
            tick=int(row[0]);args=legal_arguments(row,angles,objects[owner,tick])
            before=time.perf_counter();pose=estimator.update(*args);elapsed=time.perf_counter()-before
            costs.append(elapsed);latencies.append(elapsed);poses.append(pose)
            odom=pose.copy() if not missing[i] or odom is None else integrate(odom,args[2])
            odometry.append(odom.copy())
            if missing[i] and tick%4==0 and estimator.last['reason']!='odometry':
                matches.append(dict(owner=owner,step=tick,**estimator.last))
        arrays[f'shadow_{owner}']=np.array(poses);arrays[f'odometry_{owner}']=np.array(odometry)
        arrays[f'truth_{owner}']=truth;arrays[f'missing_{owner}']=missing;arrays[f'seconds_{owner}']=np.array(costs)
        methods={'shadow':arrays[f'shadow_{owner}'],'odometry':arrays[f'odometry_{owner}'],
                 **{c:r[f'poses_{owner}'] for c,r in references.items()}}
        for method,p in methods.items():
            a=metric_arrays(p,truth,data,angles,missing)
            all_metrics[owner,method]=a
            for key,value in a.items():arrays[f'{method}_{key}_{owner}']=value
        stats[str(owner)]=dict(estimator.stats)
        for event_owner,tick in EVENTS[map_name]:
            if event_owner!=owner:continue
            start_tick=int(np.flatnonzero(~missing[:tick-1])[-1]+1)
            mask=(data[:,0]>start_tick)&(data[:,0]<tick)
            ranges.append(dict(owner=owner,start=start_tick+1,end=tick-1,
                               metrics={m:summarize(all_metrics[owner,m],mask) for m in methods}))
    overall={}
    for method in ('rl11_lines','rl17_repair','odometry','shadow'):
        merged={}
        for key in ('position','heading','relative_turn','relative_translation','accumulated_heading','far_endpoint'):
            values=[]
            for owner in range(5):
                v=all_metrics[owner,method][key]
                if key!='far_endpoint':v=v[arrays[f'missing_{owner}']]
                values.extend(v[np.isfinite(v)])
            merged[key]=summary(values)
        overall[method]=merged
    # Offline failure labels are diagnostics only and cannot alter acceptance.
    failures=[];accepted=[e for e in matches if e['accepted']]
    for e in accepted:
        owner,i=e['owner'],e['step']-1;a=all_metrics[owner,'shadow']
        turn=a['relative_turn'][i];shift=a['relative_translation'][i]
        if np.isfinite(turn) and (turn>3 or shift>6):
            failures.append(dict(owner=owner,step=i+1,relative_turn_deg=float(turn),relative_translation_px=float(shift),decision=e))
    np.savez_compressed(out/'traces.npz',**arrays)
    write(out/'matches.json',matches);write(out/'failure_samples.json',failures)
    totals=Counter()
    for v in stats.values():totals.update(v)
    result=dict(map=map_name,source=str(run.resolve()),seed=read(run/'request.json')['seed'],round=1,
        input_sha256=digest(run/'mapper_legal_inputs.npz'),source_sha256=digest(Path(__file__).with_name('red_scan18_shadow.py')),
        runner_sha256=digest(__file__),metrics=overall,event_windows=ranges,stats=stats,total=dict(totals),
        accepted_four_step_failure_thresholds=dict(turn_deg=3,translation_px=6),
        failure_count=len(failures),accepted_count=len(accepted),
        resources=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                       update_seconds=summary(latencies),update_total_seconds=sum(latencies),
                       peak_rss_bytes=process.memory(),affinity=process.affinity),
        limitations=['Shadow poses are never written to mapper/navigation/radio/placement.',
                     'Four-step relative errors exclude all intervals crossing an absolute recovery.',
                     'Far endpoint error uses same measured 150..218 px ray at true and estimated pose: pose-induced projection error, not range-to-wall accuracy.',
                     'Local success, heuristic ranks and forward/reverse consistency do not prove global correctness. Entry anchors remain noisy.',
                     'Fixed two development seeds only; no tuning on holdout; no new official score.'])
    write(out/'results.json',result)
    print(__import__('json').dumps(dict(map=map_name,total=result['total'],metrics=overall,resources=result['resources'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert read(a.baseline/'summary.json')['valid'],'Stage A must pass first'
    a.output.mkdir(parents=True,exist_ok=False);process=ReplayProcess();cv2.setNumThreads(1)
    for name in EVENTS:run_map(a.batch,a.baseline,a.output/name,name,process)


if __name__=='__main__':main()
