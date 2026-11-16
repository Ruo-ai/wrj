"""Passive full phase chain audit. Physical truth stays outside strategy."""
from contextlib import contextmanager
from pathlib import Path
import json
import time
import numpy as np
from swarm_rescue.solutions.my_drone_red_link14 import opening_candidates
from swarm_rescue.solutions.my_drone_red_rn7 import rotate


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    old=gui_sr.run_control;rows={};hits={};nav={};errors=[];seconds=0.;refs={}
    def control(drone):
        nonlocal seconds
        command=old(drone);start=time.perf_counter()
        try:
            ident=str(drone.identifier);tick=int(drone._steps);refs[ident]=drone
            row=rows.setdefault(ident,dict(samples=[],discoveries=[],events=[],event_cursor=0))
            w,h=map(int,drone.size_area);position=np.asarray(drone.true_position());heading=float(drone.true_angle())
            hp=drone.lidar()._hitpoints
            if hp is not None:
                hp=np.asarray(hp);p=hp[hp[:,8]!=0,2:4]
                ix=np.rint(np.column_stack((p[:,0]+w/2,h/2-p[:,1]))).astype(int)
                good=(ix[:,0]>=0)&(ix[:,0]<w)&(ix[:,1]>=0)&(ix[:,1]<h)
                grid=hits.setdefault(ident,np.zeros((h,w),np.uint8));grid[ix[good,1],ix[good,0]]=1
            for e in getattr(drone,'_link_events',()):
                if e['step']>=row['event_cursor']:
                    item=dict(e,truth=position.tolist(),true_heading=heading)
                    if 'body' in e:item['actual_point']=(position+rotate(e['body'],heading)).tolist()
                    row['events'].append(item)
            row['event_cursor']=tick+1
            if tick%24==0 and drone._position is not None:
                grid=drone._explore;key=f'n_{ident}_{tick}'
                nav[key+'_free']=np.packbits(grid.free);nav[key+'_walls']=np.packbits(grid.obstacle>=2)
                row['samples'].append(dict(step=tick,key=key,shape=grid.shape,cell=grid.CELL,
                    truth=position.tolist(),true_heading=heading,position=drone._position.tolist(),heading=float(drone._heading),
                    route=[list(map(int,c)) for c in drone._route],gathering=bool(drone._gathering),
                    bomb_target=None if drone._bomb_target is None else drone._bomb_target.tolist(),
                    return_state=getattr(drone,'_return_state',None),
                    opening=None if getattr(drone,'_opening',None) is None else int(drone._opening['id']),
                    communication_disabled=bool(drone.communicator_is_disabled()),
                    received_sources={str(k):v[1] for k,v in drone._radio_maps.snapshots.items()},
                    acknowledged_sources=dict(getattr(getattr(drone,'_receipt',None),'versions',{}))))
                if tick<3600:
                    candidates=opening_candidates(drone.lidar_values(),drone.lidar_rays_angles(),grid,
                        drone._position,drone._heading,drone.semantic_values())
                    for c in candidates:
                        row['discoveries'].append(dict(step=tick,body=c['body'].tolist(),estimated_point=c['world'].tolist(),
                            actual_point=(position+rotate(c['body'],heading)).tolist(),navigation=drone._position.tolist(),
                            truth=position.tolist(),gain=c['gain']))
        except Exception as exc:
            if len(errors)<16:errors.append(repr(exc))
        seconds+=time.perf_counter()-start
        return command
    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=old
        for k,drone in refs.items():
            rows[k]['stats']=dict(getattr(drone,'_link_stats',{}));rows[k].pop('event_cursor',None)
            rows[k]['final_receipt']=dict(getattr(getattr(drone,'_receipt',None),'versions',{}))
        target=Path(destination)
        with target.open('x',encoding='utf-8') as f:
            json.dump(dict(schema='rl14-chain-v1',drones=rows,errors=errors,seconds=seconds,
                limits='Actual ray endpoints every control tick; room masks offline. Candidate/route/grid audit every 24 ticks can miss brief decisions. Endpoints are not filled visible walls.'),f,indent=2)
        np.savez_compressed(target.with_suffix('.npz'),**{'hit_'+k:v for k,v in hits.items()},**nav)

