"""Passive RX10 evidence chain. Truth is read only after guarded control.

No replay claim: sparse raw endpoints, publication masks and final local maps
do not contain every fitted point or every free-space erasure operation.
"""
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    original=gui_sr.run_control;refs={};rows={};arrays={};ledger=[];errors=[]
    diagnostic_seconds=0.;submitted=[]
    def control(drone):
        nonlocal diagnostic_seconds
        command=original(drone);started=time.perf_counter()
        try:
            ident=str(drone.identifier);step=int(drone._steps);refs[ident]=drone
            row=rows.setdefault(ident,dict(samples=[],seen_cells={},repeat_samples=0,roles=Counter(),
                scan_samples=0,submissions=0,history=[],free_count=0))
            grid=drone._explore;mapper=getattr(drone,'_submission_mapper',None)
            role=('placement' if getattr(drone,'_bomb_target',None) is not None else
                  'hold' if getattr(drone,'_holding',False) else
                  'merge' if getattr(drone,'_gathering',False) and not getattr(drone,'_rx_continue',False) else 'explore')
            row['roles'][role]+=1
            if step%24==0:
                truth=np.asarray(drone.true_position(),float);cell=tuple(np.floor(truth/40).astype(int))
                count=int(grid.free.sum());seen=row['seen_cells'].get(cell,0)
                if role=='explore' and seen>=3 and count<=row['free_count']:row['repeat_samples']+=1
                row['seen_cells'][cell]=seen+1;row['free_count']=count
                row['samples'].append(dict(step=step,truth=truth.tolist(),role=role,
                    goal=grid.point(drone._route[-1]).tolist() if drone._route else None,
                    task=getattr(grid,'task',None)['id'] if getattr(grid,'task',None) else None,
                    free_cells=count,local_visits=int((grid.visits>0).sum()),
                    sources={str(k):v[1] for k,v in drone._radio_maps.snapshots.items()},
                    continue_exploring=bool(getattr(drone,'_rx_continue',False))))
            if step%40==0:
                points=drone.lidar()._hitpoints
                d=drone.lidar_values();a=drone.lidar_rays_angles()
                if points is not None and d is not None and a is not None:
                    points=np.asarray(points);d=np.asarray(d).ravel();a=np.asarray(a).ravel()
                    good=(points[:,8]!=0)&np.isfinite(d)&np.isfinite(a)&(d>4)&(d<218)
                    tp=points[good,2:4];length=d[good];angle=a[good]
                    pose=mapper.position if mapper is not None else drone._position
                    heading=mapper.heading if mapper is not None else drone._heading
                    if pose is not None:
                        estimate=pose+length[:,None]*np.column_stack((np.cos(angle+heading),np.sin(angle+heading)))
                        ledger.append(np.column_stack((np.full(len(tp),int(ident)),np.full(len(tp),step),tp,estimate)).astype(np.float32))
                        row['scan_samples']+=1
            store=getattr(drone._radio_maps,'mapping_masks',drone._radio_maps.masks)
            own=store.get(int(ident))
            if step%300==0 and own is not None:
                arrays[f'published_{ident}_{step}']=own.copy()
                row['history'].append(dict(step=step,published_sequence=drone._radio_maps.snapshots[int(ident)][1],
                    cells=int(own.sum()),pending=len(mapper.pending) if mapper is not None else None))
            if drone._exploration_submitted and not row['submissions']:
                row['submissions']=1
                submitted.append(dict(drone=int(ident),step=step,sources={str(k):v[1] for k,v in drone._radio_maps.snapshots.items()}))
                for source,mask in store.items():arrays[f'submitter_{ident}_source_{source}']=mask.copy()
        except Exception as exc:
            if len(errors)<12:errors.append(repr(exc))
        diagnostic_seconds+=time.perf_counter()-started
        return command
    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=original;root=Path(destination).parent
        for ident,drone in refs.items():
            row=rows[ident];row.pop('seen_cells',None)
            mapper=getattr(drone,'_submission_mapper',None);grid=drone._explore
            try:
                local=(mapper.mask() if mapper is not None else drone._map.mask()).copy()
                arrays[f'local_final_{ident}']=local
                arrays[f'local_final_pixels_{ident}']=(mapper if mapper is not None else drone._map).to_pixels(local)
                row['mapper_cell']=mapper.CELL if mapper is not None else drone._map.CELL
                row['mapper_stats']=dict(mapper.stats) if mapper is not None else {}
                row['mapper_pending']=len(mapper.pending) if mapper is not None else None
                row['branch_stats']=dict(getattr(grid,'stats',{}));row['branch_events']=list(getattr(grid,'events',[]))
                row['finish_stats']=dict(getattr(drone,'_rx_stats',{}))
                row['final_received_sources']={str(k):v[1] for k,v in drone._radio_maps.snapshots.items()}
            except Exception as exc:errors.append(repr(exc))
        arrays['endpoint_ledger']=np.concatenate(ledger) if ledger else np.empty((0,6),np.float32)
        np.savez_compressed(root/'exploration_stages.npz',**arrays)
        with Path(destination).open('x',encoding='utf-8') as stream:
            json.dump(dict(schema='rx10-passive-v1',drones=rows,submissions=submitted,
                diagnostic_seconds=diagnostic_seconds,errors=errors,complete_sensor_replay=False,
                ledger_columns=['drone','step','true_hit_x','true_hit_y','raw_estimated_hit_x','raw_estimated_hit_y']),stream,indent=2)
