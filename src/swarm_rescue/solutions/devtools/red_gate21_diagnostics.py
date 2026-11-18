"""Passive R21 probe snapshots. No alternate controller and no policy changes.

Legal state is copied inside planner calls. Truth, physical rays and contacts
are read only by the outer observer, outside the guarded controller call.
"""
from contextlib import contextmanager
from pathlib import Path
import math
import sys
import time

import cv2
import numpy as np

from swarm_rescue.solutions.devtools.red_pair19_batch import write
from swarm_rescue.solutions.devtools.red_explore20_diagnostics import plain


CANDIDATE_COLUMNS=['body_x','body_y','row','col','visits','novelty','clearance','cost','score']
CALL_COLUMNS=['owner','tick','allow','valid_odometry','episodes','cooldown','active_before',
    'active_after','desired','nav_x','nav_y','heading','velocity_x','velocity_y','returned_command']


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    from swarm_rescue.solutions.my_drone_red_rn7 import PassageMotion
    from swarm_rescue.solutions.my_drone_red_advanced5_narrow import NarrowLocalMap
    from swarm_rescue.simulation.utils.definitions import CollisionTypes
    old_control=gui_sr.run_control;old_command=PassageMotion.command
    old_route=NarrowLocalMap.route;old_update=gui_sr.GuiSR.on_update
    active=None;arrays={};probes=[];calls=[];executions=[];truth_rows=[];contacts=[]
    refs={};visible={};coverage=[];errors=[];handlers=[];seen=set();seconds=0.

    def route(fine,position,preferred,coarse,*args,**kwargs):
        nonlocal seconds
        if active is None:return old_route(fine,position,preferred,coarse,*args,**kwargs)
        started=time.perf_counter();index=len(probes);key=f'probe_{index}';candidates=[]
        record=dict(index=index,owner=active['owner'],step=active['step'],
            position=plain(position),preferred=plain(preferred),args=plain(args),kwargs=plain(kwargs),
            nav_position=plain(active['nav_position']),nav_heading=active['nav_heading'],
            desired=active['desired'],episodes_before=active['episodes'],cooldown_before=active['cooldown'],
            active_before=active['active_before'],cell_size=fine.CELL,shape=fine.shape)
        for name,value in [('free',fine.free),('hits',fine.hits),('clearance',fine.clearance),('squeeze',fine.squeeze),
                           ('coarse_free',coarse.coarse.free),('coarse_obstacle',coarse.coarse.obstacle),('visits',coarse.visits)]:
            arrays[key+'_'+name]=value.copy()
        record['arrays']=key
        class Capture:
            visits=coarse.visits
            def cell(self,point):
                # Delegate unchanged; neither array nor return cell is modified.
                nonlocal seconds
                started=time.perf_counter();cell=coarse.cell(point);loc=sys._getframe(1).f_locals
                if 'near_unknown' in loc and 'distance' in loc:
                    c=loc['cell'];count=int(self.visits[cell]);novelty=min(30.,float(loc['near_unknown'][c]))
                    clear=float(loc['clear'][c]);cost=float(loc['distance'])
                    score=.8*novelty+float((point-position)@loc['direction'])*.28+min(30.,clear)*.4-cost*.06-min(count,20)*3
                    candidates.append([*point,*c,count,novelty,clear,cost,score])
                seconds+=time.perf_counter()-started
                return cell
        seconds+=time.perf_counter()-started
        path=old_route(fine,position,preferred,Capture(),*args,**kwargs)
        started=time.perf_counter()
        arrays[key+'_candidates']=np.asarray(candidates,float).reshape(-1,len(CANDIDATE_COLUMNS))
        record.update(selected_path=plain(path),reason=fine.last_reason);probes.append(record)
        seconds+=time.perf_counter()-started
        return path

    def command(motion,d,a,coarse,position,heading,desired,objects=(),allow=True):
        nonlocal active,seconds
        started=time.perf_counter()
        before=dict(owner=motion.identifier,step=motion.tick,episodes=motion.episodes,cooldown=motion.cooldown,
                    active_before=bool(motion.path),nav_position=plain(position),nav_heading=heading,desired=desired)
        drone=refs.get(motion.identifier)
        if drone is not None:
            before['parent']=plain(dict(route=drone._route,
                goal=None if not drone._route else drone._explore.point(drone._route[-1]),
                bomb_target=drone._bomb_target,gathering=drone._gathering,holding=drone._holding,
                yield_until=drone._yield_until,attempt_step=drone._attempt_step,
                progress=drone._progress(),inventory=drone.carried_bombs_count(),
                submitted=drone._exploration_submitted))
        active=before;events=len(motion.events);probe_count=len(probes)
        before_path=plain(motion.path);before_velocity=plain(motion.velocity)
        seconds+=time.perf_counter()-started
        try:result=old_command(motion,d,a,coarse,position,heading,desired,objects,allow)
        finally:active=None
        started=time.perf_counter()
        p=[0.,0.] if position is None else position
        calls.append([motion.identifier,motion.tick,allow,motion.valid_odom,before['episodes'],before['cooldown'],
            before['active_before'],bool(motion.path),desired,*p,heading,*before_velocity,result is not None])
        if before['active_before'] or motion.path or len(probes)>probe_count or len(motion.events)>events:
            executions.append(dict(**before,allow=bool(allow),valid_odometry=motion.valid_odom,path_before=before_path,
                path_after=plain(motion.path),velocity=before_velocity,command=plain(result),
                events=[e for e in motion.events if e['tick']==motion.tick],
                peers=[dict(distance=float(o.distance),angle=float(o.angle)) for o in objects or ()
                       if getattr(o.entity_type,'name','')=='DRONE']))
        seconds+=time.perf_counter()-started
        return result

    def control(drone):
        nonlocal seconds
        refs[int(drone.identifier)]=drone
        result=old_control(drone);started=time.perf_counter()
        try:
            owner=int(drone.identifier);tick=drone._steps;position=np.asarray(drone.true_position());heading=float(drone.true_angle())
            nav=drone._position;mapper=drone._submission_mapper
            truth_rows.append([owner,tick,*position,heading,float(drone.drone_health),
                np.linalg.norm(position-nav) if nav is not None else -1.,
                np.linalg.norm(position-mapper.position) if mapper.position is not None else -1.])
            if tick%8==0 and drone.lidar_values() is not None:
                hp=drone.lidar()._hitpoints
                if hp is not None:
                    hp=np.asarray(hp);w,h=map(int,drone.size_area);angles=np.asarray(drone.lidar_rays_angles())+heading
                    endpoints=position+230*np.column_stack((np.cos(angles),np.sin(angles)))
                    hit=hp[:,8]!=0;endpoints[hit]=hp[hit,2:4]
                    delta=endpoints-position;length=np.linalg.norm(delta,axis=1)
                    endpoints=position+delta*np.maximum(0,1-4/np.maximum(length,1e-9))[:,None]
                    polygon=np.rint(np.column_stack((endpoints[:,0]+w/2,h/2-endpoints[:,1]))/4).astype(np.int32)
                    grid=visible.setdefault(owner,np.zeros(((h+3)//4,(w+3)//4),np.uint8));before=int(grid.sum())
                    if np.isfinite(endpoints).all():cv2.fillPoly(grid,[polygon],1)
                    coverage.append([owner,tick,int(grid.sum()),int(grid.sum())-before])
        except Exception as e:errors.append(repr(e))
        seconds+=time.perf_counter()-started
        return result

    def update(gui,delta):
        pg=gui._playground
        if id(pg) not in seen:
            seen.add(id(pg))
            for kind,label in ((CollisionTypes.WALL,'wall'),(CollisionTypes.DRONE,'drone')):
                handler=pg.space.add_collision_handler(CollisionTypes.DRONE,kind);previous=handler.begin
                handlers.append((handler,previous))
                def begin(arbiter,space,data,old=previous,name=label):
                    # Body identity only, never passed into online navigation.
                    owners=[]
                    for i,d in refs.items():
                        if any(s.body is d.base._pm_body for s in arbiter.shapes):owners.append(i)
                    contacts.append(dict(tick=max((d._steps for d in refs.values()),default=-1),kind=name,owners=owners,
                        points=[list(p.point_a) for p in arbiter.contact_point_set.points]))
                    return old(arbiter,space,data) if old else True
                handler.begin=begin
        return old_update(gui,delta)

    gui_sr.run_control=control;PassageMotion.command=command;NarrowLocalMap.route=route;gui_sr.GuiSR.on_update=update
    try:yield
    finally:
        gui_sr.run_control=old_control;PassageMotion.command=old_command;NarrowLocalMap.route=old_route;gui_sr.GuiSR.on_update=old_update
        for handler,previous in handlers:handler.begin=previous
        arrays.update(calls=np.asarray(calls,float).reshape(-1,len(CALL_COLUMNS)),
            truth=np.asarray(truth_rows,float).reshape(-1,8),coverage=np.asarray(coverage,int).reshape(-1,4))
        arrays.update({'visible_'+str(k):v for k,v in visible.items()})
        np.savez_compressed(Path(destination).with_suffix('.npz'),**arrays)
        write(destination,dict(schema='r21-passive-probes-v1',probes=probes,executions=executions,contacts=contacts,
            errors=errors,diagnostic_seconds=seconds,candidate_columns=CANDIDATE_COLUMNS,call_columns=CALL_COLUMNS,
            truth_columns=['owner','tick','x','y','heading','health','navigation_error','mapper_error'],
            coverage_columns=['owner','tick','sampled_visible_cells','new_cells'],
            limitations=['No policy change. Diagnostic overhead is inside engine wall clock.',
                'Physical coverage uses the established 8-tick/4px polygon estimate, not exact lidar coverage.',
                'Contact owner list can be empty and must be checked; begin contacts are not contact-duration counts.']))


def verify(path):
    from swarm_rescue.solutions.devtools.red_pair19_batch import read
    doc=read(path);bad=[]
    with np.load(Path(path).with_suffix('.npz'),allow_pickle=False) as z:
        for p in doc['probes']:
            c=z[p['arrays']+'_candidates'];path=p['selected_path']
            if path and p['kwargs'].get('target') is None:
                eligible=c[c[:,4]<2]
                if not len(eligible):bad.append([p['index'],'no_eligible_candidate']);continue
                end=np.asarray(path[-1]);matches=eligible[np.linalg.norm(eligible[:,:2]-end,axis=1)<1e-6]
                if not len(matches) or matches[:,8].max()<eligible[:,8].max()-1e-8:bad.append([p['index'],'rank_mismatch'])
        calls=len(z['calls']);truth=len(z['truth'])
    return dict(probes=len(doc['probes']),calls=calls,truth_rows=truth,rank_mismatches=bad,errors=doc['errors'],
        unattributed_contacts=sum(not c['owners'] for c in doc['contacts']))
