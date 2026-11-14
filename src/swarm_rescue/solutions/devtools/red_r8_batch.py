"""Independent R8 ablations; retained evaluator and RN7 diagnostics reused."""
import sys
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
from swarm_rescue.solutions.devtools import red_rn7_batch as driver

EXTRA={
    'r8_n':('my_drone_red_r8','MyDroneRedR8Narrow'),
    'r8_m':('my_drone_red_r8','MyDroneRedR8Mapping'),
    'r8_nm':('my_drone_red_r8','MyDroneRedR8Combined'),
    'r8_surface':('my_drone_red_r8','MyDroneRedR8Surface'),
    'r8_old_budget':('my_drone_red_r8','MyDroneRedR8OldBudget'),
    'r8_off':('my_drone_red_r8','MyDroneRedR8'),
}


@contextmanager
def endpoint_recording(destination):
    """Engine endpoint sampling AFTER guarded control, never policy input."""
    from swarm_rescue.simulation.gui_map import gui_sr
    original=gui_sr.run_control;observed={};rows={};errors=[]
    def control(drone):
        command=original(drone)
        try:
            ident=str(drone.identifier);w,h=map(int,drone.size_area)
            row=rows.setdefault(ident,dict(sampled_scans=0,mapper_stats={}))
            mapper=getattr(drone,'_submission_mapper',None)
            if mapper is not None:row['mapper_stats']=dict(mapper.stats)
            if getattr(drone,'_steps',0)%4==0:
                points=drone.lidar()._hitpoints
                if points is not None:
                    points=np.asarray(points);valid=points[:,8]!=0
                    xy=np.rint(np.column_stack((points[valid,2]+w/2,h/2-points[valid,3]))).astype(int)
                    xy=xy[(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)]
                    grid=observed.setdefault(ident,np.zeros((h,w),bool));grid[xy[:,1],xy[:,0]]=True
                    row['sampled_scans']+=1
        except Exception as error:
            if len(errors)<8:errors.append(repr(error))
        return command
    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=original;root=Path(destination).parent
        np.savez_compressed(root/'observed_endpoints.npz',**observed)
        with (root/'r8_evidence.json').open('x',encoding='utf-8') as f:
            json.dump(dict(drones=rows,diagnostic_errors=errors,sampling_stride=4,
                complete_sensor_replay=False,truth_access_outside_control=True),f,indent=2)


def main():
    old_module,old_extra,old_record=driver.MODULE,driver.EXTRA,driver.narrow_recording
    @contextmanager
    def recording(destination):
        with old_record(destination),endpoint_recording(destination):yield
    try:
        driver.MODULE='swarm_rescue.solutions.devtools.red_r8_batch';driver.EXTRA=EXTRA
        driver.narrow_recording=recording
        if '--worker' not in sys.argv and '--controllers' not in sys.argv:
            sys.argv+=['--controllers','r8_n','r8_m']
        driver.main()
    finally:driver.MODULE,driver.EXTRA,driver.narrow_recording=old_module,old_extra,old_record


if __name__=='__main__':main()
