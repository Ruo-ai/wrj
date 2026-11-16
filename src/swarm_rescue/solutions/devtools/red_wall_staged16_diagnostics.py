"""Passive RL16 diagnostics. Truth read after guarded control, never fed back."""
from contextlib import contextmanager
from pathlib import Path
import json
import time
import numpy as np
from swarm_rescue.solutions.my_drone_red_wall16 import plain


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    original=gui_sr.run_control;refs={};rows={};hits={};blocks={};seconds=0.;errors=[];receipts=[]
    def control(drone):
        nonlocal seconds
        command=original(drone);started=time.perf_counter()
        try:
            k=str(drone.identifier);refs[k]=drone;tick=int(drone._steps)
            m=drone._submission_mapper;motion=drone._segment_motion;store=drone._radio_maps
            row=rows.setdefault(k,dict(samples=[],submitted=False))
            w,h=map(int,drone.size_area);hp=drone.lidar()._hitpoints
            if hp is not None:
                hp=np.asarray(hp);p=hp[hp[:,8]!=0,2:4]
                ix=np.rint(np.column_stack((p[:,0]+w/2,h/2-p[:,1]))).astype(int)
                good=(ix[:,0]>=0)&(ix[:,0]<w)&(ix[:,1]>=0)&(ix[:,1]<h)
                mask=hits.setdefault(k,np.zeros((h,w),np.uint8));mask[ix[good,1],ix[good,0]]=1
            # Retain references OUTSIDE the online mapper solely for audit. This
            # storage is charged to the diagnostic process, not online capacity.
            for b in getattr(m,'blocks',[])+([m.open_block] if getattr(m,'open_block',None) is not None else []):
                blocks[k,int(b['id'])]=b
            if tick%16==0:
                row['samples'].append(dict(step=tick,phase=getattr(motion,'align_state',None),
                    command=dict(command),position=None if drone._position is None else drone._position.tolist(),
                    heading=float(drone._heading),truth=np.asarray(drone.true_position()).tolist(),
                    true_heading=float(drone.true_angle()),velocity=motion.pose.velocity.tolist(),
                    route=[drone._explore.point(c).tolist() for c in drone._route],
                    nogps=drone.measured_gps_position() is None,nocompass=drone.measured_compass_angle() is None,
                    nocom=bool(drone.communicator_is_disabled()),stage_blocks=len(getattr(m,'blocks',[])),
                    version=int(getattr(m,'version',0)),health=float(drone.drone_health),
                    accepted_sources={str(s):int(v[1]) for s,v in store.snapshots.items()}))
            if drone._exploration_submitted and not row['submitted']:
                row['submitted']=True
                receipts.append(dict(submitter=int(k),step=tick,
                    received_sources={str(s):int(v[1]) for s,v in store.snapshots.items()},
                    receiver_conflicts=getattr(store,'conflict_cells',{}).copy()))
        except Exception as exc:
            if len(errors)<16:errors.append(repr(exc))
        seconds+=time.perf_counter()-started
        return command
    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=original;arrays={'hit_'+k:v for k,v in hits.items()};metadata=[]
        for k,d in refs.items():
            m=d._submission_mapper;motion=d._segment_motion
            rows[k].update(wall_stats=dict(getattr(motion,'wall_stats',{})),wall_events=list(getattr(motion,'wall_events',[])),
                stage_stats=dict(getattr(m,'stage_stats',{})),stage_events=list(getattr(m,'stage_events',[])),
                final_sources={str(s):int(v[1]) for s,v in d._radio_maps.snapshots.items()},
                pending_blocks=len(getattr(m,'blocks',[])),pending_open_scans=len(getattr(m,'open_block',None)['frames']) if getattr(m,'open_block',None) else 0)
            for name,mask in [('trusted',m.hits>=2.4),('local',m.mask()),
                    ('staged',getattr(m,'staged',np.zeros(m.shape,bool))),('validated',getattr(m,'validated',np.zeros(m.shape,bool)))]:
                arrays[name+'_'+k]=m.to_pixels(mask)
        for (k,bid),b in blocks.items():
            frames=[];points=[];free=[];pn=fn=0
            for t,wall,p,a,weight,ray in b['frames']:
                frames.append([t,*p,a,weight,pn,len(wall),fn,len(ray)])
                points.extend(wall);free.extend(ray);pn+=len(wall);fn+=len(ray)
            prefix=f'block_{k}_{bid}'
            arrays[prefix+'_frames']=np.asarray(frames,float).reshape(-1,9)
            arrays[prefix+'_wall']=np.asarray(points,np.float32).reshape(-1,2)
            arrays[prefix+'_free']=np.asarray(free,np.float32).reshape(-1,2)
            metadata.append(dict(drone=k,id=bid,**plain({n:v for n,v in b.items() if n not in ('id','frames')})))
        target=Path(destination);np.savez_compressed(target.with_suffix('.npz'),**arrays)
        with target.open('x',encoding='utf-8') as f:
            json.dump(dict(schema='rl16-passive-v1',seconds=seconds,errors=errors,drones=rows,
                submission_receipts=receipts,local_blocks=metadata,
                limits='16-tick control/pose sampling; true lidar hit endpoints every tick. Staged plot is cumulative entrance-projected sealed content, not accepted or currently pending content. Local frames archive is legal-only. Receipts are observed on submitter; controller has no ACK.'),f,indent=2)
