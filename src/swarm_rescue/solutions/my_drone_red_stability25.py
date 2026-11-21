"""Stability25 draft: fixed-deadline legal startup orientation on original RL17-R.

All drones retain the real match clock and start inherited navigation at control
step 96. Only the two original launch windows use age since that common release.
The preparation period is real budget spent, including on safety fallback.
This does not promise identical noisy observations, trajectories or map scores.
"""
from collections import Counter
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_advanced2 import (
    _M01_wrap_angle, _M11_safe_velocity, _M13_obstructed_route, _M15_MyDroneRedV5,
)
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.my_drone_red_stopturn import body_motion, turn_clear


class StartupAlignment25:
    RELEASE_STEP = 96
    TARGET = 0.0
    ALIGN_TOLERANCE = .045
    HOLD_TOLERANCE = .09
    TURN_LIMIT = .5

    def __init__(self):
        self.ticks = 0
        self.stable = 0
        self.settled = False
        self.released = False
        self.release_step = None
        self.reason = None
        self.fallback_reason = None
        self.initial_heading = None
        self.final_heading = None
        self.final_error = None
        self.final_speed = None
        self.aligned_step = None
        self.stats = Counter()
        self.velocity = np.zeros(2)

    def command(self, step, heading, compass, odometry, distances, angles):
        if self.released:
            return None
        self.ticks += 1
        velocity = body_motion(odometry)
        if velocity is not None:
            theta = float(odometry[2])
            c, s = math.cos(theta), math.sin(theta)
            old = np.array([c*self.velocity[0]+s*self.velocity[1],
                            -s*self.velocity[0]+c*self.velocity[1]])
            self.velocity = .5*old + .5*velocity
            velocity = self.velocity
        valid_heading = (heading is not None and np.ndim(heading) == 0
                         and np.isfinite(heading))
        valid_compass = (compass is not None and np.ndim(compass) == 0
                         and np.isfinite(compass))
        valid = valid_heading and valid_compass and velocity is not None
        if valid_heading and self.initial_heading is None:
            self.initial_heading = float(heading)
        error = _M01_wrap_angle(self.TARGET - float(heading)) if valid_heading else None
        speed = float(np.linalg.norm(velocity)) if velocity is not None else None
        force = np.zeros(2) if velocity is None else np.clip(-.65 * velocity, -1., 1.)
        command = dict(forward=float(force[0]), lateral=float(force[1]), rotation=0.)
        good = (valid and abs(error) < self.ALIGN_TOLERANCE and speed < .45
                and abs(float(odometry[2])) < .045)
        self.stable = self.stable + 1 if good else 0
        if self.stable >= 4:
            self.settled = True
            if self.aligned_step is None:
                self.aligned_step = int(step)
                self.stats['first_alignment'] += 1
        if self.settled and valid and abs(error) >= self.HOLD_TOLERANCE:
            self.settled = False
            self.stats['alignment_rechecks'] += 1
        if step >= self.RELEASE_STEP:
            self.released = True
            self.release_step = int(step)
            self.final_heading = float(heading) if valid_heading else None
            self.final_error = float(error) if error is not None else None
            self.final_speed = speed
            aligned = (valid and self.settled and abs(error) < self.HOLD_TOLERANCE
                       and speed < .45 and abs(float(odometry[2])) < .045)
            self.reason = 'aligned' if aligned else (self.fallback_reason or 'budget_fallback')
            self.stats[self.reason] += 1
            return None
        self.stats['preparation_ticks'] += 1
        if not valid:
            self.fallback_reason = 'sensor_fallback'
            self.stats['fallback_hold_ticks'] += 1
            return command
        if self.fallback_reason is not None:
            self.stats['fresh_observation_rechecks'] += 1
        self.fallback_reason = None
        if self.settled:
            self.stats['aligned_hold_ticks'] += 1
            return command
        if abs(error) >= self.ALIGN_TOLERANCE:
            if not turn_clear(distances, angles):
                self.fallback_reason = 'clearance_fallback'
                self.stats['fallback_hold_ticks'] += 1
            elif speed < .45:
                command['rotation'] = float(np.clip(1.5 * error, -self.TURN_LIMIT, self.TURN_LIMIT))
                self.stats['turn_ticks'] += 1
            else:
                self.stats['brake_ticks'] += 1
        else:
            self.stats['settle_ticks'] += 1
        return command


class TargetMotion25(_M15_MyDroneRedV5):
    def _launch_age25(self):
        return self._steps - StartupAlignment25.RELEASE_STEP + 1

    def _navigate_motion(self):
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
        if self._progress() > 0.72 and (not self._gathering):
            self._gathering = True
            self._route = []
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
        if self._launch_open and self._launch_age25() < 65 and (np.linalg.norm(self._position - self._home) < 180):
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
        if self._launch_open and self._launch_age25() < 45:
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


class MyDroneRedStability25(MyDroneRedRepair17, TargetMotion25):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._startup25 = StartupAlignment25()
        self._d25_stats = self._startup25.stats
        self._startup25_reported = False

    def _navigation(self):
        startup = self._startup25
        command = startup.command(self._steps, self._heading, self.measured_compass_angle(),
                                  self.odometer_values(), self.lidar_values(), self.lidar_rays_angles())
        if startup.released and not self._startup25_reported:
            self._startup25_reported = True
            self._event('startup25_released', reason=startup.reason, release_step=startup.release_step,
                        aligned_step=startup.aligned_step, initial_heading=startup.initial_heading,
                        final_heading=startup.final_heading, final_error=startup.final_error,
                        final_speed=startup.final_speed)
        if command is not None:
            result = self._empty_command()
            result.update(command)
            return result
        return super()._navigation()
