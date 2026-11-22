"""Schedule40: one exploration/delivery schedule, unchanged Continuous38 maps.

Independent candidate: historical source modules are not edited. The normal
motion body is the M14 executor with ONLY its legacy .72 gathering/route-clear
removed. The RN7 body retains all live safety checks and uses delivery mode
instead of the legacy .68 cutoff. Empty-inventory frontiers no longer wait
until .60. Return budget, source freeze, ACK, localization and rendering persist.
"""
from collections import Counter
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_delivery39r2 import MyDroneRedDelivery39R2
from swarm_rescue.solutions.my_drone_red_delivery39 import MyDroneRedDelivery39
from swarm_rescue.solutions.my_drone_red_rn7 import NarrowMixin
from swarm_rescue.solutions.my_drone_red_advanced2 import (
    _M13_obstructed_route, _M11_safe_velocity, _M01_wrap_angle,
)


class ScheduledNarrow40(NarrowMixin):
    # Insert before NarrowMixin in the existing MRO; skip only its old executor.
    def _navigation(self):
        narrow=self._rn7
        if narrow.path:
            self._slow_ticks=0;self._waypoint_progress.last_advance=self._steps
            if self._steps%60==0:self._motion_anchor=None
        original=super(NarrowMixin,self)._navigation()
        if not self.NARROW_ENABLED:return original
        near_bomb=(self._bomb_target is not None and self._position is not None
                   and np.linalg.norm(self._bomb_target-self._position)<55)
        allow=(self._position is not None and self.drone_health>10
               and self._d39_mode=='explore' and not self._gathering
               and not self._exploration_submitted and self._attempt_step is None
               and not near_bomb and self._steps>=self._yield_until)
        desired=self._last_motion_angle if self._last_motion_angle is not None else 0.
        command=narrow.command(self.lidar_values(),self.lidar_rays_angles(),self._explore,
                               self._position,self._heading,desired,self.semantic_values(),allow)
        if allow and self._progress() >= .68:
            self._schedule40_stats['late_narrow_eligible'] += 1
            if command is not None:self._schedule40_stats['late_narrow_command'] += 1
        if command is not None:
            motion=getattr(self,'_segment_motion',None)
            if motion is not None and motion.state not in ('NORMAL','COOLDOWN'):
                motion.release('rn7_local_passage')
            original.update(command)
        return original


class MyDroneRedSchedule40(MyDroneRedDelivery39R2, ScheduledNarrow40):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._schedule40_stats = Counter()

    def _navigation(self):
        if self._position is None:
            return super(MyDroneRedDelivery39,self)._navigation()
        if self._steps-self._d39_plan_step >= 24:
            self._path39()
        due = self._budget39()
        if self._d39_mode == 'explore' and due and self.carried_bombs_count()==0:
            self._d39_mode = 'return'; self._gathering = True; self._holding = False
            self._route = []; self._bomb_target = None
            self._d39_advance = self._steps
            if self._entry37.active:self._entry37.release('delivery39_return')
            self._segment_motion.release('delivery39_return',cooldown=1)
            if self._rn7.path:self._rn7.release('delivery39_return')
            self._note39('return_started',budget=self._d39_budget,
                contract_accepted=sorted(self._d39_accepts),position=self._position.tolist())
            self._f27_stats['gather_started_control'] = self._steps
        self._explore.extension_enabled = (self._d39_mode=='explore'
            and self.carried_bombs_count()==0 and not due)
        if self._explore.extension_enabled:
            self._schedule40_stats['frontier_enabled_ticks'] += 1
            if self._progress() < .60:self._schedule40_stats['early_frontier_enabled_ticks'] += 1
        if self._d39_mode == 'explore':
            self._gathering = False
            return super(MyDroneRedDelivery39,self)._navigation()
        return self._return_motion39()

    def _navigate_motion(self):
        if self._entry37.active:
            return super()._navigate_motion()
        command = self._empty_command()
        if self._position is None:
            return command
        values = self.lidar_values()
        if values is None:
            command.update(forward=float(-np.clip(self._velocity[0], -1, 1)), lateral=float(-np.clip(self._velocity[1], -1, 1)))
            return command
        d = np.asarray(values).ravel()
        angles = np.asarray(self.lidar_rays_angles())
        bomb_rays = np.zeros(len(d), bool)
        dynamic = np.zeros(len(d), bool)
        for obj in self.semantic_values() or []:
            kind = getattr(obj.entity_type, 'name', None)
            if kind in ('DRONE', 'BOMB'):
                delta = np.abs((angles - obj.angle + math.pi) % (2 * math.pi) - math.pi)
                matched = (delta < 0.19) & (np.abs(d - obj.distance) < 25)
                dynamic |= matched
                if kind == 'BOMB':
                    bomb_rays |= matched
        if self._launch_open is None:
            static = d[np.isfinite(d) & (d > 0) & ~dynamic]
            self._launch_open = bool(len(static) >= 12 and np.min(static) > 90)
        odom = self.odometer_values()
        if odom is not None and np.shape(odom) == (3,) and np.isfinite(odom).all():
            direction = odom[1] - odom[2]
            measured = odom[0] * np.array([math.cos(direction), math.sin(direction)])
            self._velocity = 0.4 * self._velocity + 0.6 * measured
        self._explore.step = self._steps
        if self._progress() > .72 and self._route:
            self._schedule40_stats['legacy_route_clear_prevented'] += 1
        if self._try_hold(d, dynamic, command):
            return command
        if not self._gathering and self._loop_watch.update(self._steps, self._position, float(np.linalg.norm(self._velocity)), int(self._explore.free.sum())):
            self._explore.avoid_region(self._position, self._steps + 480)
            self._escape_until = 0
            self._start_recovery(d, angles, 'local_loop')
            self._escape_until = self._steps + 55
            self._last_bomb_plan = self._steps + 80
            self._event('local_loop', position=list(map(float, self._position)))
        self._slow_ticks = self._slow_ticks + 1 if np.linalg.norm(self._velocity) < 0.8 else max(0, self._slow_ticks - 2)
        if self._slow_ticks >= 24 and self._steps - self._last_motion_recovery >= 40:
            self._start_recovery(d, angles, 'no_progress')
            self._last_motion_recovery = self._steps
            self._slow_ticks = 0
        if self._steps % 6 == 0 or self._steps == 1:
            self._explore.observe(self._position, self._heading, d, angles, dynamic)
        if self._steps % 6 == 0 and self._route and _M13_obstructed_route(self._explore, self._position, self._route):
            self._release_stale_route('new_wall')
        if self._steps % 30 == 0:
            self._visited_snapshot = np.packbits(self._explore.shared_visited | (self._explore.visits > 0)).tobytes()
            self._radio_maps.publish(int(self.identifier or 0), self._steps, self._map.mask())
        if self._direction_anchor is None:
            self._direction_anchor = self._position.copy()
        delta = self._position - self._direction_anchor
        if np.linalg.norm(delta) > 75:
            self._explore.forward_direction = delta / np.linalg.norm(delta)
            self._direction_anchor = self._position.copy()
        fresh = {i: p for (i, p) in self._peers.items() if self._steps - p[0] < 90}
        claims = [p[2] for (i, p) in fresh.items() if p[2] is not None and i != int(self.identifier or 0)]
        if self._steps % 60 == 0:
            if self._motion_anchor is not None and np.linalg.norm(self._position - self._motion_anchor) < 25:
                self._stalled_steps += 60
                self._start_recovery(d, angles, 'no_progress')
            self._motion_anchor = self._position.copy()
        self._blocked_goals = {g: t for (g, t) in self._blocked_goals.items() if t > self._steps}
        if self._launch_open and self._steps < 65 and (np.linalg.norm(self._position - self._home) < 180):
            self._explore.forward_direction = np.array([math.cos(self._launch_angle), math.sin(self._launch_angle)])
        if self._route and np.linalg.norm(self._position - self._explore.point(self._route[-1])) < 22:
            self._explore.retire_goal(self._route[-1])
            self._route = []
        if self._route and self._steps - self._goal_step > 600:
            self._explore.retire_goal(self._route[-1])
            self._route = []
        if not self._route and self._steps >= self._escape_until and (self._steps - self._last_route_step >= 12):
            self._route = self._explore.plan(self._position, self._home, int(self.identifier or 0), self._n, claims, self._blocked_goals, self._gathering)
            self._last_route_step = self._steps
            self._goal_step = self._steps
            self._replans += 1
        while len(self._route) > 1 and np.linalg.norm(self._position - self._explore.point(self._route[0])) < 22:
            self._route.pop(0)
        target_distance = 160.0
        if self._route:
            index = 0
            for (i, cell) in enumerate(self._route[:9]):
                point = self._explore.point(cell)
                if np.linalg.norm(point - self._position) > 130:
                    break
                if self._explore.line_clear(self._position, point):
                    index = i
                else:
                    break
            if index:
                self._route = self._route[index:]
            target = self._explore.point(self._route[0]) - self._position
            target_distance = float(np.linalg.norm(target))
            target_angle = math.atan2(target[1], target[0]) - self._heading
        else:
            if self._escape_heading is None or self._steps >= self._escape_until:
                self._start_recovery(d, angles, 'disconnected')
            target_angle = self._escape_heading - self._heading
        if self._steps < self._escape_until and self._escape_heading is not None:
            target_angle = self._escape_heading - self._heading
            target_distance = 160.0
        if self._launch_open and self._steps < 45:
            target_angle = self._launch_angle - self._heading
        if self._steps >= self._yield_until:
            for (i, (seq, position, goal)) in fresh.items():
                delta = position - self._position
                if self._steps - seq < 8 and i < int(self.identifier or 0) and (np.linalg.norm(delta) < 75):
                    angle = math.atan2(delta[1], delta[0]) - self._heading
                    if math.cos(angle - target_angle) > 0.4:
                        self._yield_heading = self._heading + angle - math.pi / 5
                        self._yield_until = self._steps + 18
                        break
        if self._steps < self._yield_until:
            target_angle = self._yield_heading - self._heading
            self._yield_steps += 1
        if self._route and self._steps >= self._escape_until and (self._steps >= self._yield_until):
            target_point = self._explore.point(self._route[0])
            if self._waypoint_progress.update(self._steps, self._position, target_point):
                self._release_stale_route('waypoint_stall')
                self._explore.avoid_region(self._position, self._steps + 240)
                self._start_recovery(d, angles, 'waypoint_stall')
                target_angle = self._escape_heading - self._heading
                target_distance = 160.0
        else:
            self._waypoint_progress.reset()
        steering_d = d.copy()
        steering_d[bomb_rays] = 80
        (move, self._last_motion_angle) = _M11_safe_velocity(steering_d, angles, _M01_wrap_angle(target_angle), self._velocity, self._last_motion_angle, target_distance)
        command.update(forward=float(move[0]), lateral=float(move[1]), rotation=0.0)
        return command
