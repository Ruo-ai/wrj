"""Passive narrow audit after guarded controller returns; truth never fed back."""
from contextlib import contextmanager
from pathlib import Path
import json
import time
import numpy as np

@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    old=gui_sr.run_control;refs={};rows={};hits={};seconds=0.;errors=[]
    def control(drone):
        nonlocal seconds
        command=old(drone);start=time.perf_counter()
        try:
            key=str(drone.identifier);refs[key]=drone;tick=int(drone._steps)
            row=rows.setdefault(key,dict(samples=[]));w,h=map(int,drone.size_area)
            hp=drone.lidar()._hitpoints
            if hp is not None:
                hp=np.asarray(hp);p=hp[hp[:,8]!=0,2:4]
                ix=np.rint(np.column_stack((p[:,0]+w/2,h/2-p[:,1]))).astype(int)
                good=(ix[:,0]>=0)&(ix[:,0]<w)&(ix[:,1]>=0)&(ix[:,1]<h)
                mask=hits.setdefault(key,np.zeros((h,w),np.uint8));mask[ix[good,1],ix[good,0]]=1
            if tick%16==0 and drone._position is not None:
                n=drone._rn7;grid=drone._explore
                row['samples'].append(dict(step=tick,position=drone._position.tolist(),heading=float(drone._heading),
                    truth=np.asarray(drone.true_position()).tolist(),true_heading=float(drone.true_angle()),
                    velocity=n.velocity.tolist(),target=None if not drone._route else grid.point(drone._route[-1]).tolist(),
                    route=[grid.point(c).tolist() for c in drone._route],local_path=[p.tolist() for p in n.path],
                    context=getattr(n,'context',{}).copy(),command=dict(command),health=float(drone.drone_health),
                    replan_count=int(drone._replans),gathering=bool(drone._gathering),
                    sources={str(k):int(v[1]) for k,v in drone._radio_maps.snapshots.items()}))
        except Exception as exc:
            if len(errors)<16:errors.append(repr(exc))
        seconds+=time.perf_counter()-start
        return command
    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=old
        for k,d in refs.items():
            n=d._rn7;rows[k].update(stats=dict(n.stats),events=list(n.events),audit=getattr(n,'audit',[]),
                close_wall_ticks=getattr(n,'close_wall_ticks',None))
        target=Path(destination)
        with target.open('x',encoding='utf-8') as f:
            json.dump(dict(schema='rc15-narrow-v1',seconds=seconds,errors=errors,drones=rows,
                limits='Route/pose samples every 16 ticks; actual ray hit endpoints every tick. Physical positions are offline only. No inferred exact doorway width.'),f,indent=2)
        np.savez_compressed(target.with_suffix('.npz'),**{'hit_'+k:v for k,v in hits.items()})
