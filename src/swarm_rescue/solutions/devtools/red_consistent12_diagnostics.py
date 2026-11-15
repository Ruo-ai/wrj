"""Passive full navigation-grid snapshots for entrance diagnosis.

Reads all state after guarded control returns. No ground-truth ROI or diagnostic
data is supplied to the controller. 60-step grid snapshots are not a full replay.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    original=gui_sr.run_control;arrays={};rows=[];errors=[];seconds=0.
    def control(drone):
        nonlocal seconds
        command=original(drone);start=time.perf_counter()
        try:
            tick=int(drone._steps);ident=int(drone.identifier)
            if tick%60==0:
                grid=drone._explore;key=f'd{ident}_s{tick}'
                arrays[key+'_free']=grid.free.copy()
                arrays[key+'_obstacle']=grid.obstacle.copy()
                arrays[key+'_visited']=grid.visits.copy()
                arrays[key+'_goal_visits']=grid.goal_visits.copy()
                rows.append(dict(drone=ident,step=tick,key=key,cell=grid.CELL,
                    position=drone._position.tolist(),heading=float(drone._heading),
                    truth=np.asarray(drone.true_position()).tolist(),true_heading=float(drone.true_angle()),
                    gps_available=drone.measured_gps_position() is not None,
                    compass_available=drone.measured_compass_angle() is not None,
                    route=[list(c) for c in drone._route],blocked_goals=[list(c) for c in drone._blocked_goals],
                    gathering=bool(drone._gathering),holding=bool(getattr(drone,'_holding',False)),
                    bomb_target=None if drone._bomb_target is None else drone._bomb_target.tolist(),
                    narrow_active=bool(drone._rn7.path),narrow_stats=dict(drone._rn7.stats),
                    last_motion_angle=drone._last_motion_angle,
                    segment_state=drone._segment_motion.state,command=command.copy()))
                rows[-1]['navigation_pose_stats']=dict(getattr(drone._pose_filter,'stats',{}))
        except Exception as exc:
            if len(errors)<12:errors.append(repr(exc))
        seconds+=time.perf_counter()-start
        return command
    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=original
        np.savez_compressed(Path(destination).with_suffix('.npz'),**arrays)
        with Path(destination).open('x',encoding='utf-8') as f:
            json.dump(dict(schema='rl12-passive-nav-v1',rows=rows,errors=errors,
                           diagnostic_seconds=seconds,interval=60),f,indent=2)
