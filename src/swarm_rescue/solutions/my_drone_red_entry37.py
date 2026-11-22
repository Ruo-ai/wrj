"""Clean + observed boundary, with one bounded recalibration per entry.

Submission mapper/renderer, sensor data, communication and deadlines stay
inherited. Physical settling lets both original pose filters reacquire; this
candidate does not reset either filter or reproject any occupancy grid.
"""
import numpy as np

from swarm_rescue.solutions.my_drone_red_frontier34 import MyDroneRedCleanBoundary34
from swarm_rescue.solutions.red_entry37 import Entry37


class MyDroneRedEntry37(MyDroneRedCleanBoundary34):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._entry37 = Entry37()

    def _update_observation(self):
        super()._update_observation()
        self._entry37.observe(self._steps, self._position, self._heading,
            self.measured_gps_position(), self.measured_compass_angle(), self.odometer_values())

    def _navigation(self):
        guard = self._entry37
        near_bomb = (self._bomb_target is not None and self._position is not None and
                     np.linalg.norm(self._bomb_target-self._position) < 55)
        eligible = (self.drone_health > 10 and not self._gathering and
                    not self._exploration_submitted and self._attempt_step is None and
                    not near_bomb and self._steps >= self._escape_until and
                    self._steps >= self._yield_until)
        was_active = guard.active
        command = guard.step(self.lidar_values(),self.lidar_rays_angles(),eligible,self._progress())
        if command is not None:
            # Keep the inherited observation, map publishing and planning
            # updates, but prevent competing local executors from advancing a
            # maneuver whose command is not actually being executed.
            if not was_active:
                if self._rn7.path:
                    self._rn7.release('entry37_retrace')
                self._segment_motion.release('entry37_retrace',cooldown=1)
            self._slow_ticks = 0
            self._waypoint_progress.last_advance = self._steps
            self._motion_anchor = self._position.copy()
            segmented, narrow = self.SEGMENTED_MOTION, self.NARROW_ENABLED
            try:
                self.SEGMENTED_MOTION = self.NARROW_ENABLED = False
                original = super()._navigation()
            finally:
                self.SEGMENTED_MOTION, self.NARROW_ENABLED = segmented, narrow
            original.update(command)
            if was_active and not guard.active:
                self._route = []
                self._segment_motion.release('entry37_finished',cooldown=1)
            return original
        if was_active:
            self._route = []
            self._segment_motion.release('entry37_finished',cooldown=1)
        return super()._navigation()

    def _deployment(self, command):
        if self._entry37.active:
            # No new deployment during the short entrance maneuver; existing
            # deployment/egress has priority and prevents starting the guard.
            return
        return super()._deployment(command)
