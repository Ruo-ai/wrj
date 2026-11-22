"""Entry37 integration repair: planned settling is not a navigation stall.

Keep live navigation observations and radio publication at their original
cadence while a bounded entry owns motion. Do not run the competing planner,
loop detector or deployment scheduler during that explicitly owned interval.
The entry state machine and all calibration parameters are unchanged.
"""
import math
import numpy as np

from swarm_rescue.solutions.my_drone_red_entry37 import MyDroneRedEntry37
from swarm_rescue.solutions.my_drone_red_stopturn import body_motion


class MyDroneRedEntry37R1(MyDroneRedEntry37):
    def _navigate_motion(self):
        if not self._entry37.active:
            return super()._navigate_motion()
        command=self._empty_command()
        values=self.lidar_values()
        if values is None or self._position is None:
            return command
        d=np.asarray(values).ravel()
        angles=np.asarray(self.lidar_rays_angles())
        dynamic=np.zeros(len(d),bool)
        for obj in self.semantic_values() or ():
            if getattr(obj.entity_type,'name',None) in ('DRONE','BOMB'):
                delta=np.abs((angles-obj.angle+math.pi)%(2*math.pi)-math.pi)
                dynamic |= (delta<.19)&(np.abs(d-obj.distance)<25)
        measured=body_motion(self.odometer_values())
        if measured is not None:
            self._velocity=.4*self._velocity+.6*measured
        self._explore.step=self._steps
        if self._steps%6==0 or self._steps==1:
            self._explore.observe(self._position,self._heading,d,angles,dynamic)
        if self._steps%30==0:
            self._visited_snapshot=np.packbits(self._explore.shared_visited|(self._explore.visits>0)).tobytes()
            self._radio_maps.publish(int(self.identifier or 0),self._steps,self._map.mask())
        self._slow_ticks=0
        self._motion_anchor=None
        self._waypoint_progress.reset()
        self._loop_watch.samples.clear()
        return command
