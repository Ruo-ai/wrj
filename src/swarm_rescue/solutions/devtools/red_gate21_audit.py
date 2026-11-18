"""R21 historical portal audit. Sensor/visit replay only; never runs a flight.

Truth defines offline gate labels and geometry checks. Local candidate searches
use only recorded legal rays, semantic readings, reconstructed navigation pose
and visits verified against actual R20 pre-plan snapshots. The RN7 preferred
direction was not recorded: counterfactual rankings are labelled accordingly.
"""
import argparse
from collections import defaultdict, Counter
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from swarm_rescue.solutions.devtools.red_pair19_batch import read, write, digest
from swarm_rescue.solutions.devtools.red_pair19_results_review import only, truth_metric
from swarm_rescue.solutions.devtools.red_explore20_audit import gates, log_events, navigation_graph
from swarm_rescue.solutions.devtools.red_recovery18_audit import load_npz, semantic_objects, legal_arguments
from swarm_rescue.solutions.my_drone_red_advanced2 import _M01_PoseFilter, _M19_SurveyGrid
from swarm_rescue.solutions.my_drone_red_rn7 import LocalPassage, CoarseView, rotate
from swarm_rescue.solutions.my_drone_red_advanced5_narrow import NarrowLocalMap


def world_gate(gate,w,h):
    x,y=gate['center_px'];center=np.array([x-w/2,h/2-y])
    tangent=np.array([0.,1.]) if gate['axis']=='v' else np.array([1.,0.])
    normal=np.array([1.,0.]) if gate['axis']=='v' else np.array([0.,1.])
    return center,tangent,normal


def lookup(fine,points,field):
    points=np.atleast_2d(points);ix=fine.indices(points);x,y=ix[:,0],ix[:,1]
    inside=(x>=0)&(y>=0)&(x<fine.shape[1])&(y<fine.shape[0])
    out=np.zeros(len(points),field.dtype);out[inside]=field[y[inside],x[inside]]
    return out


def truth_clearance(points,dist,w,h):
    p=np.atleast_2d(points);x=np.rint(p[:,0]+w/2).astype(int);y=np.rint(h/2-p[:,1]).astype(int)
    ok=(x>=0)&(y>=0)&(x<w)&(y<h);out=np.zeros(len(p));out[ok]=dist[y[ok],x[ok]]
    return out


def pulled_path(fine,parents,end):
    cells=[end]
    while parents[cells[-1]] is not None:cells.append(parents[cells[-1]])
    cells.reverse()
    if not any(fine.squeeze[c] for c in cells[7:]):return []
    path=[np.zeros(2)]+[fine.point(c) for c in cells[1:]];result=[path[0]];i=0
    while i<len(path)-1:
        j=min(len(path)-1,i+20)
        while j>i+1 and not fine.line_clear(path[i],path[j]):j-=1
        if not fine.line_clear(path[i],path[j]):return []
        result.append(path[j]);i=j
    return result if len(result)>1 else []


def candidate_search(fine,coarse,position,heading,preferred=(1.,0.)):
    """Observe the unmodified route function's evaluated endpoint set."""
    candidates=[];returned={};base=CoarseView(coarse,position,heading)
    class Capture:
        visits=base.visits
        def cell(self,point):
            result=base.cell(point);loc=sys._getframe(1).f_locals
            if 'near_unknown' in loc and 'distance' in loc:
                c=loc['cell'];count=int(self.visits[result])
                candidates.append(dict(cell=c,point=np.asarray(point).copy(),count=count,
                    novelty=min(30.,float(loc['near_unknown'][c])),clearance=float(loc['clear'][c]),cost=float(loc['distance'])))
            return result
    previous=sys.getprofile();code=NarrowLocalMap.route.__code__
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is code:
            returned.update(parents=frame.f_locals.get('parents',{}),visited=frame.f_locals.get('visited',0))
    try:
        sys.setprofile(profile)
        path=fine.route(np.zeros(2),np.asarray(preferred,float),Capture(),limit=165)
    finally:sys.setprofile(previous)
    eligible=[x for x in candidates if x['count']<2]
    return dict(candidates=candidates,eligible=eligible,parents=returned.get('parents',{}),
                selected=path,reason=fine.last_reason,visited=returned.get('visited',0))


def section(gate,fine,grid,position,heading,true_position,true_heading,truth_dist):
    w,h=grid.width,grid.height;center,tangent,normal=world_gate(gate,w,h)
    offsets=np.arange(-gate['nominal_gap']/2,gate['nominal_gap']/2+.1,1.)
    side=1. if float((position-center)@normal)>=0 else -1.
    _,_,reach=navigation_graph(grid,position)
    raw=grid.free&~(grid.obstacle>=2);start=grid.cell(position);raw[start]=True
    _,labels=cv2.connectedComponents(raw.astype(np.uint8),connectivity=4);rawreach=(labels==labels[start])&raw
    walls=cv2.dilate((grid.obstacle>=2).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))).astype(bool)
    planes=[]
    for offset in (-24.,0.,24.):
        world=center+offset*side*normal+offsets[:,None]*tangent
        legal=rotate(world-true_position,-true_heading)
        estimated=rotate(world-position,-heading)
        free=lookup(fine,legal,fine.free);hit=lookup(fine,legal,fine.hits)>0
        cells=[grid.cell(p) for p in world]
        clearance=fine.sample(legal);truth=truth_clearance(world,truth_dist,w,h)
        planes.append(dict(normal_offset_from_plane=offset,samples=len(world),
            current_observed_free=int(free.sum()),current_endpoint=int(hit.sum()),current_unknown=int((~free&~hit).sum()),
            legal_disc_clear=int((clearance>=fine.PLAN_RADIUS).sum()),
            estimated_frame_disc_clear=int((fine.sample(estimated)>=fine.PLAN_RADIUS).sum()),
            physical_disc_clear=int((truth>=fine.TRACK_RADIUS).sum()),max_truth_clearance=float(truth.max()),
            coarse_free=sum(bool(grid.free[c]) for c in cells),coarse_raw_blocked=sum(bool(grid.obstacle[c]>=2) for c in cells),
            coarse_inflated_blocked=sum(bool(walls[c]) for c in cells),coarse_raw_reachable=sum(bool(rawreach[c]) for c in cells),
            coarse_reachable=sum(bool(reach[c]) for c in cells),distinct_coarse_cells=len(set(cells))))
    return planes


def candidate_gate(gate,search,fine,grid,position,heading,true_position,true_heading,truth_dist,observed_endpoint=None):
    center,tangent,normal=world_gate(gate,grid.width,grid.height)
    side=1. if float((true_position-center)@normal)>=0 else -1.
    crossing=[];tested=0;exhaustive=True
    for candidate in sorted(search['eligible'],key=lambda c:c['cost']):
        point=true_position+rotate(candidate['point'],true_heading)
        # Endpoint must be at least 24px beyond the gate plane and within 120px
        # along the aperture axis; crossing itself is checked against full gap.
        if side*float((point-center)@normal)>=-24 or abs(float((point-center)@tangent))>120:continue
        tested+=1;path=pulled_path(fine,search['parents'],candidate['cell'])
        if not path:continue
        world=true_position+rotate(np.array(path),true_heading);proper=False
        for a,b in zip(world,world[1:]):
            da=float((a-center)@normal);db=float((b-center)@normal)
            if da*db<=0 and abs(db-da)>1e-8:
                at=a+(b-a)*(-da/(db-da))
                if abs(float((at-center)@tangent))<=gate['nominal_gap']/2:proper=True
        if not proper:continue
        points=np.concatenate([np.linspace(a,b,max(2,int(np.linalg.norm(b-a))+1)) for a,b in zip(world,world[1:])])
        real_clear=float(truth_clearance(points,truth_dist,grid.width,grid.height).min())
        crossing.append(dict(endpoint=candidate['point'].tolist(),count=candidate['count'],novelty=candidate['novelty'],
            clearance=candidate['clearance'],cost=candidate['cost'],truth_min_clearance=real_clear,
            physical_disc_safe=real_clear>=fine.BODY_RADIUS,path=[p.tolist() for p in path]))
        # This is an existence audit, not a census of all equivalent endpoints.
        # Stop after the first physically clear, legally swept crossing path.
        if crossing[-1]['physical_disc_safe']:
            exhaustive=False;break
    recovered_match=None
    if observed_endpoint is not None:
        recovered_match=min((float(np.linalg.norm(c['point']-observed_endpoint)) for c in search['eligible']),default=None)
    safe=[c for c in crossing if c['physical_disc_safe']]
    return dict(legal_cross_candidates=len(crossing),physical_disc_safe_cross_candidates=len(safe),
                crossing_paths_tested=tested,crossing_counts_exhaustive=exhaustive,
                best_cross=min(safe or crossing,key=lambda c:c['cost'],default=None),recovered_endpoint_distance=recovered_match)


def replay_pose_visits(run,plans,inputs,objects,shape):
    """Replay only legal pose filtering and visit increments, verified exactly."""
    w,h=shape;events=log_events(run);motion=read(run/'motion_diagnostics.json');narrow=read(run/'narrow_diagnostics.json')
    rows={};visit_snaps={};checks=[]
    with np.load(run/'explore20_diagnostics.npz',allow_pickle=False) as saved:
        for owner in range(5):
            key=str(owner);data=inputs['drone_'+key];angles=inputs['angles_'+key]
            indexed={p['step']:p for p in plans if p['owner']==owner}
            starts={e['tick'] for e in narrow['drones'][key]['events'] if e['reason']=='entered'}
            holds={e['step']-1:e['event']=='hold_start' for e in events if e['drone']==owner and e['event'] in ('hold_start','hold_end')}
            pose=_M01_PoseFilter();grid=_M19_SurveyGrid((w,h),lambda *a,**k:None);holding=False;nav=[];matched=0;errors=[]
            for data_row in data:
                tick=int(data_row[0]);gps,compass,odom,d,a,_=legal_arguments(data_row,angles,objects[owner,tick])
                answer=pose.update(gps,compass,odom)
                if answer is None:raise ValueError('Uninitialized navigation pose')
                position,heading=answer;nav.append([*position,heading])
                if tick in holds:holding=holds[tick]
                if not holding and d is not None and (tick==1 or tick%6==0) and np.any(np.isfinite(d)&np.isfinite(a+heading)&(d>4)):
                    cell=grid.cell(position);grid.visits[cell]=min(65535,int(grid.visits[cell])+1)
                if tick in indexed:
                    p=indexed[tick]
                    if not np.allclose(position,p['args'][0],rtol=0,atol=1e-9):errors.append((tick,'position'))
                    elif not np.array_equal(grid.visits,saved[p['state']['visits']['array']]):errors.append((tick,'visits'))
                    else:matched+=1
                if tick in starts:visit_snaps[owner,tick]=grid.visits.copy()
            rows[owner]=np.array(nav)
            for s in motion['drones'][key]['samples']:
                if not np.allclose(rows[owner][s['tick']-1,:2],s['navigation']['position'],atol=1e-9,rtol=0):errors.append((s['tick'],'motion_position'))
            checks.append(dict(owner=owner,matched_plans=matched,total_plans=len(indexed),errors=errors))
    if any(c['errors'] for c in checks):raise ValueError('Navigation replay does not match recorded state: '+str(checks))
    return rows,visit_snaps,checks


def recorded_positions(run,motion):
    with np.load(only(run/'artifacts/recordings','*.npz'),allow_pickle=False) as z:record=z['drone_states']
    candidates=[]
    for offset in range(-3,4):
        errors=[]
        for owner in range(5):
            for sample in motion['drones'][str(owner)]['samples']:
                i=sample['tick']+offset
                if 0<=i<len(record):errors.append(float(np.linalg.norm(np.array([record[i,owner]['x'],record[i,owner]['y']])-sample['truth'])))
        candidates.append((max(errors),offset))
    error,offset=min(candidates)
    if error>.001:raise ValueError('Cannot align recording with controller ticks')
    return record,offset,error


RELEASES={'completed','mission_preempted','no_progress','task_timeout',
          'peer_wait_budget','fresh_scan_obstructed','inertial_risk','invalid_odometry'}


def task_state(owner,tick,events,samples):
    active=False;cooldown=75+owner*11;entered=0;last=None
    for e in events:
        if e['tick']>tick:break
        if e['reason']=='entered':active=True;entered+=1
        if e['reason'] in RELEASES:
            active=False;cooldown=max(cooldown,e['tick']+90+owner*7)
        last=e
    prior=[s for s in samples if s['step']<=tick]
    if prior:cooldown=max(cooldown,prior[-1]['cooldown'])
    return dict(active_by_events=active,entered=entered,cooldown_known_lower_bound=cooldown,
        certainly_in_cooldown=tick<cooldown,last_event=last,
        sampled_state_step=prior[-1]['step'] if prior else None,
        note='Unrecorded failed probes and mission gates can impose additional restrictions.')


def recovered_endpoint(owner,tick,events,samples,inputs):
    after=[e for e in events if e['tick']>=tick and e['reason']!='entered']
    stop=next((e['tick'] for e in after if e['reason'] in RELEASES or e['reason']=='local_replan'),None)
    sample=next((s for s in samples if s['step']>=tick and s['rn7_path'] and (stop is None or s['step']<stop)),None)
    if sample is None:return None,None,'No unchanged endpoint survived to a 24-tick sample'
    point=np.array(sample['rn7_path'][-1],float)
    for t in range(sample['step'],tick,-1):
        row=inputs['drone_'+str(owner)][t-1]
        if not row[3]:return None,None,'Missing odometry in inverse transport'
        distance,alpha,theta=row[8:11]
        point=rotate(point,theta)+distance*np.array([math.cos(alpha),math.sin(alpha)])
    return point,sample['step'],None


def entry_audit(doors,inputs,objects,nav,visits,narrow,trace,record,offset,td,w,h):
    rows=[]
    for owner in range(5):
        events=narrow['drones'][str(owner)]['events'];samples=trace['drones'][str(owner)]
        for e in events:
            if e['reason']!='entered':continue
            tick=e['tick'];position=nav[owner][tick-1,:2];heading=float(nav[owner][tick-1,2])
            state=record[tick+offset,owner];tp=np.array([float(state['x']),float(state['y'])]);th=float(state['angle'])
            nearby=[g for g in doors if np.linalg.norm(world_gate(g,w,h)[0]-tp)<=180]
            if not nearby:continue
            args=legal_arguments(inputs['drone_'+str(owner)][tick-1],inputs['angles_'+str(owner)],objects[owner,tick])
            fine=LocalPassage(args[3],args[4],args[5]);grid=_M19_SurveyGrid((w,h),lambda *a,**k:None)
            grid.visits=visits[owner,tick];search=candidate_search(fine,grid,position,heading)
            endpoint,sample_tick,error=recovered_endpoint(owner,tick,events,samples,inputs)
            matched=None;selected_path=[]
            if endpoint is not None and search['eligible']:
                near=min(search['eligible'],key=lambda c:np.linalg.norm(c['point']-endpoint))
                matched=float(np.linalg.norm(near['point']-endpoint))
                if matched<1e-6:selected_path=pulled_path(fine,search['parents'],near['cell'])
            for gate in nearby:
                cross=candidate_gate(gate,search,fine,grid,position,heading,tp,th,td,endpoint)
                center,tangent,normal=world_gate(gate,w,h);side=1 if (tp-center)@normal>=0 else -1
                chosen_beyond=None
                if endpoint is not None:
                    end=tp+rotate(endpoint,th)
                    chosen_beyond=bool(side*((end-center)@normal)<-24 and abs((end-center)@tangent)<120)
                rows.append(dict(owner=owner,step=tick,gate=gate['id'],navigation_error=float(np.linalg.norm(position-tp)),
                    recovered_endpoint=None if endpoint is None else endpoint.tolist(),sample_tick=sample_tick,
                    reconstruction_error=error,endpoint_match_error=matched,selected_path_reconstructed=bool(selected_path),
                    chosen_endpoint_beyond_gate=chosen_beyond,selected_path=[p.tolist() for p in selected_path],
                    true_position=tp.tolist(),true_heading=th,eligible_candidates=len(search['eligible']),**cross))
    return rows


def analyze(batch,name,number,controller,out,max_plans=None):
    run=batch/'runs'/f'{name}_r{number:02d}'/controller/'red/attempt_001'
    dest=out/f'{name}_r{number}_{controller}';dest.mkdir(exist_ok=False)
    trace=read(run/'explore20_diagnostics.json');motion=read(run/'motion_diagnostics.json');narrow=read(run/'narrow_diagnostics.json')
    inputs=load_npz(run/'mapper_legal_inputs.npz');objects=semantic_objects(inputs,read(run/'line_diagnostics.json'))
    raw,truth,_=truth_metric(run);h,w=truth.shape;spec=read(read(run/'request.json')['map_file']);doors=gates(spec)
    nav,visits,checks=replay_pose_visits(run,trace['plans'],inputs,objects,(w,h))
    record,record_offset,record_error=recorded_positions(run,motion)
    td=cv2.distanceTransform((~truth.astype(bool)).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    rows=[];examined=0
    with np.load(run/'explore20_diagnostics.npz',allow_pickle=False) as saved:
        for plan in trace['plans']:
            owner=plan['owner'];tick=plan['step'];position=nav[owner][tick-1,:2];heading=float(nav[owner][tick-1,2])
            if tick>=3672:continue
            nearby=[g for g in doors if np.linalg.norm(world_gate(g,w,h)[0]-position)<=150]
            if not nearby:continue
            if max_plans is not None and examined>=max_plans:break
            grid=_M19_SurveyGrid((w,h),lambda *a,**k:None)
            for k in ('free','obstacle','visits'):setattr(grid,k,saved[plan['state'][k]['array']])
            args=legal_arguments(inputs['drone_'+str(owner)][tick-1],inputs['angles_'+str(owner)],objects[owner,tick])
            if args[3] is None:continue
            fine=LocalPassage(args[3],args[4],args[5]);search=candidate_search(fine,grid,position,heading)
            state=record[tick+record_offset,owner];tp=np.array([float(state['x']),float(state['y'])]);th=float(state['angle'])
            entered=sum(e['reason']=='entered' and e['tick']<=tick for e in narrow['drones'][str(owner)]['events'])
            for gate in nearby:
                planes=section(gate,fine,grid,position,heading,tp,th,td)
                cross=candidate_gate(gate,search,fine,grid,position,heading,tp,th,td)
                rows.append(dict(owner=owner,step=tick,plan_index=plan['index'],gate=gate['id'],position=position.tolist(),heading=heading,
                    true_position=tp.tolist(),true_heading=th,navigation_error=float(np.linalg.norm(position-tp)),
                    episodes_already_entered=entered,task_state=task_state(owner,tick,narrow['drones'][str(owner)]['events'],trace['drones'][str(owner)]),
                    planes=planes,search_reason=search['reason'],
                    raw_candidates=len(search['candidates']),eligible_candidates=len(search['eligible']),**cross))
            examined+=1
            if examined%40==0:print(name,number,controller,'near-gate snapshots',examined,flush=True)
    entries=entry_audit(doors,inputs,objects,nav,visits,narrow,trace,record,record_offset,td,w,h) if max_plans is None else []
    result=dict(map=name,round=number,seed=read(run/'request.json')['seed'],controller=controller,source=str(run),
        source_hash=digest(run/'explore20_diagnostics.npz'),gates=doors,checks=checks,
        recording_tick_offset=record_offset,recording_max_error=record_error,examined_plans=examined,gate_cases=len(rows),
        snapshot_limit=max_plans,rows=rows,entries=entries,
        limitations=['Uses actual global-planning instants, not every RN7 probe. Original local preferred direction and full candidate scores absent.',
          'Candidate set is independent of preferred heading; reported selected +x route is a query, not the original controller choice.',
          'Legal candidates pass inherited observed-space swept-disc checks; actual truth circle check is offline and cannot guarantee noisy future execution.'])
    write(dest/'analysis.json',result)
    print('DONE',name,number,controller,'plans',examined,'gate cases',len(rows),'cross',sum(r['legal_cross_candidates']>0 for r in rows),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maps',nargs='+',default=['yh_map_fs_72','yh_map_fs_37','yh_map_fs_24'])
    p.add_argument('--rounds',nargs='+',type=int,default=[1,2]);p.add_argument('--controllers',nargs='+',default=['rl17_repair','r20_budget'])
    p.add_argument('--max-plans',type=int)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);rows=[]
    for name in a.maps:
        for number in a.rounds:
            for controller in a.controllers:
                r=analyze(a.batch.resolve(),name,number,controller,out,a.max_plans)
                rows.append({k:r[k] for k in ('map','round','seed','controller','examined_plans','gate_cases')})
    write(out/'summary.json',dict(runs=rows,full_flights=0))


if __name__=='__main__':main()
