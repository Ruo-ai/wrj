"""Entry37 owns the real navigation entry, including legacy static dispatch.

Advanced2._navigation calls super()._navigate_motion directly, bypassing a
derived _navigate_motion override. Active entry therefore bypasses that whole
decision entry explicitly, retaining R1's observation/publication routine.
"""
import numpy as np

from swarm_rescue.solutions.my_drone_red_entry37r1 import MyDroneRedEntry37R1
from swarm_rescue.solutions.my_drone_red_frontier34 import MyDroneRedCleanBoundary34


class MyDroneRedEntry37R2(MyDroneRedEntry37R1):
    def _navigation(self):
        guard=self._entry37
        near_bomb=(self._bomb_target is not None and self._position is not None and
                   np.linalg.norm(self._bomb_target-self._position)<55)
        eligible=(self.drone_health>10 and not self._gathering and
            not self._exploration_submitted and self._attempt_step is None and not near_bomb and
            self._steps>=self._escape_until and self._steps>=self._yield_until)
        was_active=guard.active
        command=guard.step(self.lidar_values(),self.lidar_rays_angles(),eligible,self._progress())
        if was_active and not guard.active:
            self._route=[]
            self._motion_anchor=None
            self._waypoint_progress.reset()
            self._loop_watch.samples.clear()
            self._segment_motion.release('entry37_finished',cooldown=1)
        if command is not None and guard.active:
            if not was_active:
                if self._rn7.path:self._rn7.release('entry37_retrace')
                self._segment_motion.release('entry37_retrace',cooldown=1)
            original=MyDroneRedEntry37R1._navigate_motion(self)
            original.update(command)
            guard.stats['owned_navigation_ticks']+=1
            return original
        original=MyDroneRedCleanBoundary34._navigation(self)
        if command is not None:original.update(command)
        return original
