"""Passive RL13 entry, uncertainty and delivery trace, outside guarded control."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    old=gui_sr.run_control;rows={};errors=[];seconds=0.;visible={}
    def control(drone):
        nonlocal seconds
        command=old(drone);started=time.perf_counter()
        try:
            ident=str(drone.identifier);mapper=drone._submission_mapper;motion=drone._segment_motion
            row=rows.setdefault(ident,dict(entries=[],samples=[],last_missing=False,last_tick=-1))
            tick=int(drone._steps)
            if tick!=row['last_tick']:
                missing=drone.measured_gps_position() is None or drone.measured_compass_angle() is None
                entry=missing and not row['last_missing'];row['last_missing']=missing;row['last_tick']=tick
                if entry or tick%24==0:
                    truth=np.asarray(drone.true_position());angle=float(drone.true_angle())
                    record=dict(step=tick,missing=missing,communication_disabled=bool(drone.communicator_is_disabled()),
                        state=motion.state,reason=motion.reason,gathering=bool(drone._gathering),
                        progress=float(drone._progress()),pending=len(mapper.pending),
                        mapper_position_error=None if mapper.position is None else float(np.linalg.norm(mapper.position-truth)),
                        mapper_heading_error_degrees=abs(math.degrees((mapper.heading-angle+math.pi)%(2*math.pi)-math.pi)),
                        motion_position_sigma=math.sqrt(motion.pose.position_variance),
                        motion_heading_sigma=math.sqrt(motion.pose.heading_variance),
                        received_sources={str(k):v[1] for k,v in drone._radio_maps.snapshots.items()})
                    row['samples'].append(record)
                    if entry:row['entries'].append(record)
                row['local_stats']=dict(getattr(mapper,'local_stats',{}))
                row['local_events']=list(getattr(mapper,'local_events',[]))
                row['motion_stats']=dict(motion.stats)
                # Sparse physical lidar visibility, kept only in the passive
                # observer. Four-pixel cells + between-ray polygon interpolation
                # are an estimate, not a claim that every pixel was sensed.
                if tick%8==0 and drone.lidar_values() is not None:
                    hp=drone.lidar()._hitpoints
                    if hp is not None:
                        hp=np.asarray(hp);w,h=map(int,drone.size_area)
                        position=np.asarray(drone.true_position());heading=float(drone.true_angle())
                        angles=np.asarray(drone.lidar_rays_angles())+heading
                        endpoints=position+230*np.column_stack((np.cos(angles),np.sin(angles)))
                        hit=hp[:,8]!=0;endpoints[hit]=hp[hit,2:4]
                        delta=endpoints-position;length=np.linalg.norm(delta,axis=1)
                        endpoints=position+delta*np.maximum(0,1-4/np.maximum(length,1e-9))[:,None]
                        polygon=np.rint(np.column_stack((endpoints[:,0]+w/2,h/2-endpoints[:,1]))/4).astype(np.int32)
                        grid=visible.setdefault(ident,np.zeros(((h+3)//4,(w+3)//4),np.uint8))
                        if np.isfinite(endpoints).all():cv2.fillPoly(grid,[polygon],1)
        except Exception as exc:
            if len(errors)<12:errors.append(repr(exc))
        seconds+=time.perf_counter()-started
        return command
    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=old
        np.savez_compressed(Path(destination).with_suffix('.npz'),**{'visible_'+k:v for k,v in visible.items()})
        with Path(destination).open('x',encoding='utf-8') as f:
            json.dump(dict(schema='rl13-passive-v1',drones=rows,errors=errors,diagnostic_seconds=seconds,
                visibility='Physical lidar rays sampled every 8 ticks, rasterized at 4px with between-ray interpolation; approximate sampled visibility, not ground-truth full coverage.',
                delivery_policy='Unchanged RL11: special-motion cutoff .60, original bomb/gather/submit deadlines.'),f,indent=2)
