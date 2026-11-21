"""Stability24: target-relative exploration steering on retained RL17-R.

The 48 alternative velocity directions follow the requested target angle instead
of the body's zero axis. For the same physical point cloud represented in rotated
body coordinates, valid-scan local steering is rotation-equivariant away from
numerical ties and threshold boundaries. The inherited insufficient-ray brake
is unchanged. Independently sampled scans and complete noisy runs can differ.

Only the exploration motion helper changes. The inherited near-bomb helper,
launch policy, mapping, radio, stop/turn, narrow passage and placement stay intact.
The navigation method below is copied verbatim from advanced2._M14, except for
its final helper call; an AST regression test enforces that boundary.
"""
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_advanced2 import (
    _SF_HIT_LIMIT, _M01_wrap_angle, _M13_obstructed_route,
    _M15_MyDroneRedV5,
)
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17


def safe_velocity24(distances, angles, target_angle, velocity, previous_angle=None, target_distance=float('inf')):
    """A swept disc plus braking distance; angles/velocity are body-relative.

    Commands are forces, so zero command is not a brake. Bound desired speed,
    then oppose measured residual velocity. No simulator state is accessed.
    """
    d = np.asarray(distances).ravel()
    a = np.asarray(angles).ravel()
    good = np.isfinite(d) & np.isfinite(a) & (d > 0)
    if good.sum() < 12:
        return (np.clip(-np.asarray(velocity) * 0.7, -1, 1), None)
    (d, a) = (d[good], a[good])
    headings = target_angle + np.linspace(-math.pi, math.pi, 48, endpoint=False)
    headings = np.append(headings, target_angle)
    (x, y) = (d * np.cos(a), d * np.sin(a))
    (cosine, sine) = (np.cos(headings)[:, None], np.sin(headings)[:, None])
    along = cosine * x + sine * y
    across = cosine * y - sine * x
    radius = 25.0
    intersects = (along > 0) & (np.abs(across) < radius) & (d[None, :] < _SF_HIT_LIMIT)
    entry = along - np.sqrt(np.maximum(0, radius ** 2 - across ** 2))
    clearance = np.maximum(0, np.min(np.where(intersects, entry, 220), axis=1))
    tight_intersects = (along > 0) & (np.abs(across) < 22) & (d[None, :] < _SF_HIT_LIMIT)
    tight_entry = along - np.sqrt(np.maximum(0, 22 ** 2 - across ** 2))
    tight_clearance = np.maximum(0, np.min(np.where(tight_intersects, tight_entry, 220), axis=1))
    passage = (np.cos(headings - target_angle) > 0.94) & (clearance < 30) & (tight_clearance > 65)
    clearance = np.where(passage, tight_clearance, clearance)
    speed = np.minimum(np.where(passage, 3.5, 6.5), np.sqrt(0.8 * np.maximum(0, clearance - 10)))
    speed = np.minimum(speed, math.sqrt(0.8 * max(0.0, target_distance - 6)))
    scores = speed * np.cos(headings - target_angle) + 0.008 * np.minimum(clearance, 120)
    if previous_angle is not None:
        scores += 0.35 * np.cos(headings - previous_angle)
    forward = np.cos(headings - target_angle) > 0.65
    if np.any(forward & (clearance > 35)):
        scores[~forward] -= 3.0
    best = int(np.argmax(scores))
    desired = speed[best] * np.array([math.cos(headings[best]), math.sin(headings[best])])
    command = 0.085 * desired + 0.65 * (desired - np.asarray(velocity))
    norm = np.linalg.norm(command)
    if norm > 1:
        command /= norm
    return (command, float(headings[best]))


class TargetMotion24(_M15_MyDroneRedV5):
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
        (move, self._last_motion_angle) = safe_velocity24(steering_d, angles, _M01_wrap_angle(target_angle), self._velocity, self._last_motion_angle, target_distance)
        command.update(forward=float(move[0]), lateral=float(move[1]), rotation=0.0)
        return command


class MyDroneRedStability24(MyDroneRedRepair17, TargetMotion24):
    """Insert only the motion override after Advanced2 in the existing MRO.

    Advanced2 calls super()._navigate_motion(), so a direct child override would
    be skipped. This insertion keeps every original wrapper in its original order.
    """
    pass
