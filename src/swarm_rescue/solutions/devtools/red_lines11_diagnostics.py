"""Passive legal mapper-input tape and every-step offline heading audit.

Truth exists only in diagnostics, never in a controller or replay input tape.
"""
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import numpy as np


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    original=gui_sr.run_control;rows={};tapes={};angles={};semantic=[];kinds={};errors=[];seconds=0.
    def control(drone):
        nonlocal seconds
        command=original(drone);started=time.perf_counter()
        try:
            ident=str(drone.identifier);step=int(drone._steps)
            gps=drone.measured_gps_position();compass=drone.measured_compass_angle();odom=drone.odometer_values()
            distances=drone.lidar_values();a=np.asarray(drone.lidar_rays_angles(),float).ravel()
            if ident not in angles:angles[ident]=a.copy()
            if not np.array_equal(a,angles[ident]):raise ValueError('Changing ray angles require new tape schema')
            header=[step,gps is not None,compass is not None,odom is not None,distances is not None,
                    *(np.asarray(gps).ravel() if gps is not None else [np.nan,np.nan]),
                    float(compass) if compass is not None else np.nan,
                    *(np.asarray(odom).ravel() if odom is not None else [np.nan]*3)]
            ranges=np.asarray(distances,float).ravel() if distances is not None else np.full(len(a),np.nan)
            tapes.setdefault(ident,[]).append(np.r_[header,ranges])
            for obj in drone.semantic_values() or ():
                name=getattr(obj.entity_type,'name','UNKNOWN')
                code=kinds.setdefault(name,len(kinds))
                semantic.append([int(ident),step,code,float(obj.distance),float(obj.angle)])
            mapper=drone._submission_mapper
            row=rows.setdefault(ident,dict(outage_heading=[],normal_heading=[],outage_position=[],
                speed_sum=0.,outage_steps=0,line_stats={},motion_guard={}))
            if mapper.position is not None:
                heading=float(drone.true_angle());position=np.asarray(drone.true_position())
                error=abs((mapper.heading-heading+math.pi)%(2*math.pi)-math.pi)*180/math.pi
                missing=gps is None or compass is None
                row['outage_heading' if missing else 'normal_heading'].append(error)
                if missing:
                    row['outage_position'].append(float(np.linalg.norm(mapper.position-position)))
                    row['outage_steps']+=1
                    if odom is not None:row['speed_sum']+=abs(float(odom[0]))
            row['line_stats']=dict(getattr(mapper.matcher,'stats',{}))
            row['motion_guard']=dict(getattr(drone,'_l11_stats',{}))
        except Exception as exc:
            if len(errors)<12:errors.append(repr(exc))
        seconds+=time.perf_counter()-started
        return command
    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=original;root=Path(destination).parent;arrays={}
        for ident,data in tapes.items():
            arrays['drone_'+ident]=np.asarray(data,dtype=np.float64)
            arrays['angles_'+ident]=angles[ident]
        arrays['semantic']=np.asarray(semantic,dtype=np.float64).reshape(-1,5)
        np.savez_compressed(root/'mapper_legal_inputs.npz',**arrays)
        for ident,row in rows.items():
            for key in ('outage_heading','normal_heading','outage_position'):
                data=row.pop(key);row[key+'_summary']=dict(count=len(data),mean=float(np.mean(data)) if data else None,
                    p95=float(np.percentile(data,95)) if data else None,maximum=max(data) if data else None)
        with Path(destination).open('x',encoding='utf-8') as f:
            json.dump(dict(schema='rl11-legal-mapper-tape-v1',drones=rows,diagnostic_seconds=seconds,errors=errors,
                kinds=kinds,mapper_inputs_complete=not errors,
                columns=['step','gps_present','compass_present','odom_present','lidar_present','gps_x','gps_y',
                         'compass','odom_distance','odom_alpha','odom_theta','lidar_ranges...']),f,indent=2)
