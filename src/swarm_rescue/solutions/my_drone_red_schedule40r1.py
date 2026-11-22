"""Schedule40R1: route normal navigation through the repaired virtual executor.

The inherited Advanced2 planner used super()._navigate_motion(), which bypassed
Schedule40's override. This isolated copy changes only that dispatch expression.
Historical files and the initial development flights remain frozen.
"""
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_schedule40 import MyDroneRedSchedule40
from swarm_rescue.solutions.my_drone_red_advanced2 import (
    MyDroneRedAdvanced2, _M17_choose_placement, _M11_safe_velocity, _M01_wrap_angle,
)


class ExplicitMotionDispatch40(MyDroneRedAdvanced2):
    def _navigation(self):
        grid = self._explore
        grid.step = self._steps
        if self._position is not None:
            for obj in self.semantic_values() or []:
                if getattr(obj.entity_type, 'name', None) == 'DISPOSAL_CENTER' and np.isfinite([obj.distance, obj.angle]).all() and (obj.distance > 0):
                    angle = self._heading + obj.angle
                    point = self._position + obj.distance * np.array([math.cos(angle), math.sin(angle)])
                    if self._disposal_observed is None:
                        self._disposal_observed = point
                    else:
                        self._disposal_observed += 0.15 * (point - self._disposal_observed)
        task = grid.survey
        if task is not None:
            matching = bool(self._route and self._route[-1] == task['cell'])
            resolved = task['unknown'] - grid.unknown_near(task['cell']) >= 3
            expired = self._steps - task['start'] >= 90
            if not matching or resolved or expired or (self._progress() >= 0.6):
                grid.finish_survey('observed' if resolved else 'budget' if expired else 'preempted')
                if matching and self._bomb_target is None:
                    self._route = []
        if self._position is not None and self.carried_bombs_count() > 0 and (self._progress() < 0.68):
            initial = self._initial_inventory or self.carried_bombs_count()
            placed = initial - self.carried_bombs_count()
            due = 0.1 + 0.36 * (placed + 1) / (initial + 1) + 0.008 * int(self.identifier or 0)
            if self._bomb_target is not None and self._steps - self._bomb_target_step > 180:
                self._drop_placement('target_age')
            if self._bomb_target is not None and self._steps - self._recheck_step >= 30 and (self._steps - self._bomb_target_step >= 30):
                field = self._coverage_field(force=True)
                self._recheck_step = self._steps
                if field.value(self._bomb_target) < max(1.0, 0.5 * self._bomb_gain):
                    self._drop_placement('marginal_gain_lost')
            committed = self._route and self._steps - self._goal_step < 100 and (self._progress() < 0.55)
            if not committed and self._progress() >= due and (self._bomb_target is None) and (self._steps - self._last_bomb_plan >= 45):
                plan = _M17_choose_placement(grid, self._position, self._coverage_field(force=True), self._disposal_observed, self._event)
                self._last_bomb_plan = self._steps
                if plan is not None:
                    if grid.survey is not None:
                        grid.finish_survey('placement_priority')
                    self._route = plan.route
                    self._goal_step = self._steps
                    self._bomb_target = plan.target
                    self._bomb_target_step = self._steps
                    self._bomb_gain = plan.gain
                    self._event('bomb_target', target=plan.target.tolist(), estimated_gain=plan.gain)
        grid.survey_enabled = 0.18 < self._progress() < 0.6 and self._bomb_target is None
        command = self._navigate_motion()
        if self._bomb_target is not None and self._position is not None and (self.lidar_values() is not None) and (self._progress() < 0.68) and (self._steps >= self._escape_until) and (self._steps >= self._yield_until):
            delta = self._bomb_target - self._position
            distance = float(np.linalg.norm(delta))
            if distance < 65 and grid.line_clear(self._position, self._bomb_target):
                if distance < 12:
                    move = np.clip(-0.7 * self._velocity, -1, 1)
                else:
                    angle = _M01_wrap_angle(math.atan2(delta[1], delta[0]) - self._heading)
                    (move, self._last_motion_angle) = _M11_safe_velocity(np.asarray(self.lidar_values()).ravel(), np.asarray(self.lidar_rays_angles()), angle, self._velocity, self._last_motion_angle, distance)
                command.update(forward=float(move[0]), lateral=float(move[1]))
        return command


class MyDroneRedSchedule40R1(MyDroneRedSchedule40, ExplicitMotionDispatch40):
    pass
