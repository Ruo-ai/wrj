"""Startup44 delivery R2 with an isolated observed-depth placement planner.

All upper startup, pose, mapping, boundary, communication and motion executors
stay inherited. The planner retains a later charge and keeps the existing
frontier/fine-passage tools enabled while carrying that reserve.
"""
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_startup44_delivery import MyDroneRedStartup44Delivery
from swarm_rescue.solutions.my_drone_red_schedule40r1 import ExplicitMotionDispatch40
from swarm_rescue.solutions.my_drone_red_advanced2 import _M11_safe_velocity, _M01_wrap_angle
from swarm_rescue.solutions.red_deep_bomb45 import placement_due, choose_deep_placement


class DeepBombDispatch45(ExplicitMotionDispatch40):
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
            due = placement_due(initial, self.carried_bombs_count(), int(self.identifier or 0))
            if self._bomb_target is not None and self._steps - self._bomb_target_step > 180:
                self._drop_placement('target_age')
            if self._bomb_target is not None and self._steps - self._recheck_step >= 30 and (self._steps - self._bomb_target_step >= 30):
                field = self._coverage_field(force=True)
                self._recheck_step = self._steps
                if field.value(self._bomb_target) < max(1.0, 0.5 * self._bomb_gain):
                    self._drop_placement('marginal_gain_lost')
            committed = self._route and self._steps - self._goal_step < 100 and (self._progress() < 0.55)
            if not committed and self._progress() >= due and (self._bomb_target is None) and (self._steps - self._last_bomb_plan >= 45):
                plan = choose_deep_placement(grid, self._position, self._coverage_field(force=True), self._disposal_observed, self._walk39, self._db45_first, self._steps, getattr(self._misc_data, "max_timestep_limit", 5400) or 5400, self._event)
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

class MyDroneRedStartup44DeepBomb(MyDroneRedStartup44Delivery, DeepBombDispatch45):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._db45_first = {}
        self._db45_events = []
        self._db45_reserve_ticks = 0

    def _update_observation(self):
        super()._update_observation()
        if self._position is not None:
            self._db45_first.setdefault(self._explore.cell(self._position), int(self._steps))

    def _reserve_exploring45(self):
        return (self.carried_bombs_count() > 0 and self._d39_mode == 'explore'
                and self._radio_maps.own_final is None and not self._gathering
                and self._bomb_target is None and not self._d39_budget.get('budget_due', False))

    def _navigate_motion(self):
        if self._reserve_exploring45():
            self._explore.extension_enabled = True
            self._db45_reserve_ticks += 1
        return super()._navigate_motion()

    def _release_stale_route(self, reason):
        # The same LP43 fine probe, also eligible while saving the late charge.
        eligible = (reason == 'new_wall' and self._route and self._reserve_exploring45()
            and self._steps >= self._fine43_retry and not self._entry37.active)
        if eligible:
            goal = self._explore.point(self._route[-1])
            if self._explore.unknown_near(self._route[-1]) >= 3 and self._start_fine43(goal):
                self._note43('coarse_rejection_rescued', goal=goal.tolist(), position=self._position.tolist())
                return
        return super()._release_stale_route(reason)

    def _event(self, kind, **data):
        if kind in ('deep_placement_selected', 'placement_requested', 'placement_confirmed',
                    'placement_retry', 'placement_reconsidered'):
            self._db45_events.append(dict(step=int(self._steps), kind=kind, **data))
        return super()._event(kind, **data)
