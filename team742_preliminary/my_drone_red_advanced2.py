"""Standalone MyDroneRedAdvanced2: mechanically consolidated on 2026-10-28.

All red-strategy dependencies are included below. Only original project and
third-party libraries are imported. No strategy tuning is performed here.
Edit this file directly for future changes; historical module names in
section comments explain origin, not dependencies on deleted files.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import numpy as np
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData
import json
import cv2
from collections import deque
from types import SimpleNamespace
from dataclasses import dataclass

from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR as _SF_LIDAR_RANGE
# Current engine lidar Gaussian sigma=2.5 px. Four sigma separates range returns
# conservatively; repeated evidence remains necessary, not a hard noise guarantee.
_SF_LIDAR_SIGMA = 2.5
_SF_HIT_MARGIN = 4.0 * _SF_LIDAR_SIGMA
_SF_HIT_LIMIT = _SF_LIDAR_RANGE - _SF_HIT_MARGIN


# ============================================================================
# Integrated source: my_drone_place_example
# ============================================================================

_M00_WorldPoint = Tuple[float, float]
_M00_TeamMessage = Tuple[int, Any, Any, int]

class _M00_MyDronePlaceExample(DroneAbstract):
    """红队最小示例：巡航放弹 + 0 号机提交探索矩阵。"""
    _LEADER_DRONE_ID = 0
    _PLACE_BOMB_EVERY_STEPS = 60
    _INVENTORY_LOG_EVERY_STEPS = 60
    _EXPLORE_STRAIGHT_STEPS = 90
    _EXPLORE_TURN_STEPS = 28
    _EXPLORE_FORWARD = 0.7
    _ROAM_SPIN = 0.35
    _LIDAR_BLOCK_DIST = 35.0
    _LIDAR_SLOW_DIST = 52.0
    _LIDAR_PATH_CONE = math.pi / 9
    _LIDAR_SIDE_CONE = math.pi / 4
    _LIDAR_SIDE_GAP = 0.08
    _LIDAR_TARGET_MARGIN = 10.0
    _LIDAR_AVOID_ROTATION = 0.85
    _LIDAR_BLOCK_TRIGGER_STEPS = 6
    _LIDAR_ESCAPE_BACK_STEPS = 10
    _LIDAR_ESCAPE_TURN_STEPS = 16
    _LIDAR_ESCAPE_STRAFE_STEPS = 10
    _LIDAR_ESCAPE_BACK_SPEED = -0.55
    _LIDAR_ESCAPE_STRAFE_SPEED = 0.65
    _LIDAR_BOMB_ANGLE_TOL = math.pi / 18
    _LIDAR_BOMB_DIST_TOL = 18.0

    def __init__(self, identifier: Optional[int]=None, misc_data: Optional[MiscData]=None, **kwargs):
        super().__init__(identifier=identifier, misc_data=misc_data, display_lidar_graph=False, **kwargs)
        self._steps = 0
        self._explore_tick = 0
        self._lidar_blocked_steps = 0
        self._escape_back_steps_left = 0
        self._escape_turn_steps_left = 0
        self._escape_strafe_steps_left = 0
        self._escape_turn_sign = 1.0
        self._exploration_submitted = False
        self._peer_inventory: dict[int, int] = {}

    def define_message_for_all(self) -> _M00_TeamMessage:
        """广播本机 id、位姿与剩余可放置炸弹数。"""
        return (self.identifier, self.measured_gps_position(), self.measured_compass_angle(), self.carried_bombs_count())

    def control(self) -> CommandsDict:
        self._steps += 1
        cmd = self._control_roam()
        self._maybe_place_bomb(cmd)
        if self._is_leader():
            self._leader_try_submit_exploration_map()
        return cmd

    def _maybe_place_bomb(self, cmd: CommandsDict) -> None:
        if self._steps % self._PLACE_BOMB_EVERY_STEPS != 0:
            return
        if self.carried_bombs_count() > 0:
            cmd['place_bomb'] = 1

    def _is_leader(self) -> bool:
        return int(self.identifier or 0) == self._LEADER_DRONE_ID

    def _leader_try_submit_exploration_map(self) -> None:
        if self._exploration_submitted:
            return
        self._sync_peer_inventory_from_messages()
        if self._steps % self._INVENTORY_LOG_EVERY_STEPS == 0:
            self._log_team_inventory_status()
        if not self._all_teammates_report_zero_inventory():
            return
        if self.size_area is None:
            return
        self._log_team_inventory_status()
        print('[Drone0] 全队炸弹已放完，提交探索地图并结束回合')
        self._submit_random_exploration_map()
        self._exploration_submitted = True

    def _submit_random_exploration_map(self) -> None:
        (w, h) = self.size_area
        grid = np.random.randint(0, 2, size=(h, w), dtype=np.uint8)
        self.submit_exploration_map(grid)

    def _sync_peer_inventory_from_messages(self) -> None:
        self._peer_inventory[int(self.identifier)] = self.carried_bombs_count()
        for (_, msg) in self.communicator.received_messages:
            if not isinstance(msg, tuple) or len(msg) < 4:
                continue
            (drone_id, inv) = (msg[0], msg[3])
            if isinstance(drone_id, int) and isinstance(inv, (int, float)):
                self._peer_inventory[drone_id] = int(inv)

    def _team_size(self) -> Optional[int]:
        if self._misc_data is None:
            return None
        return self._misc_data.number_drones

    def _all_teammates_report_zero_inventory(self) -> bool:
        n = self._team_size()
        if n is None:
            return False
        for i in range(n):
            if i not in self._peer_inventory or self._peer_inventory[i] != 0:
                return False
        return True

    def _log_team_inventory_status(self) -> None:
        n = self._team_size()
        if n is None:
            print(f'[Drone0] 提交等待: number_drones 未知, 已记录={dict(sorted(self._peer_inventory.items()))}')
            return
        missing: list[int] = []
        not_zero: list[int] = []
        lines: list[str] = []
        for i in range(n):
            if i not in self._peer_inventory:
                lines.append(f'  #{i}: 未收到消息')
                missing.append(i)
            else:
                inv = self._peer_inventory[i]
                lines.append(f'  #{i}: {inv}')
                if inv != 0:
                    not_zero.append(i)
        ready = self._all_teammates_report_zero_inventory()
        in_range = len(self.communicator.comms_in_range)
        print(f'[Drone0] 库存汇总 step={self.elapsed_timestep} 已确认={len(self._peer_inventory)}/{n} 通信范围内={in_range} 可提交={ready}\n' + '\n'.join(lines))
        if missing:
            print(f'[Drone0] 仍缺消息: {missing}')
        if not_zero:
            print(f'[Drone0] 仍有库存: {not_zero}')

    def _control_roam(self) -> CommandsDict:
        cmd = self._empty_command()
        if self._run_lidar_escape_if_needed(cmd):
            return cmd
        cycle = self._EXPLORE_STRAIGHT_STEPS + self._EXPLORE_TURN_STEPS
        if self._explore_tick % cycle < self._EXPLORE_STRAIGHT_STEPS:
            cmd['forward'] = self._EXPLORE_FORWARD
            self._apply_lidar_avoidance(cmd)
        else:
            cmd['rotation'] = self._explore_turn_rate()
        self._explore_tick += 1
        return cmd

    def _roam_spin_sign(self) -> float:
        return self._ROAM_SPIN if int(self.identifier or 0) % 2 == 0 else -self._ROAM_SPIN

    def _explore_turn_rate(self) -> float:
        spin = self._roam_spin_sign()
        path_min = self._lidar_path_min()
        if path_min is not None and self._lidar_is_blocking(path_min, None):
            spin *= self._lidar_turn_sign()
        return spin

    def _lidar_path_min(self) -> Optional[float]:
        return self._lidar_sector_min(-self._LIDAR_PATH_CONE, self._LIDAR_PATH_CONE)

    def _lidar_sector_min(self, angle_min: float, angle_max: float) -> Optional[float]:
        values = self.lidar_values()
        if values is None:
            return None
        best = float('inf')
        found = False
        for (dist, ang) in zip(values, self.lidar_rays_angles()):
            if not math.isfinite(dist):
                continue
            a = float(ang)
            if not angle_min <= a <= angle_max:
                continue
            d = float(dist)
            if self._lidar_ray_hits_bomb(d, a):
                continue
            found = True
            best = min(best, d)
        return best if found else None

    def _lidar_ray_hits_bomb(self, dist: float, ang: float) -> bool:
        """语义确认为 BOMB 的 Lidar 读数忽略（放置阶段无碰撞）。"""
        semantic = self.semantic_values()
        if semantic is None:
            return False
        for obj in semantic:
            if obj.entity_type != DroneSemanticSensor.TypeEntity.BOMB:
                continue
            if bool(obj.grasped):
                continue
            if abs(float(obj.angle) - ang) > self._LIDAR_BOMB_ANGLE_TOL:
                continue
            if abs(float(obj.distance) - dist) <= self._LIDAR_BOMB_DIST_TOL:
                return True
        return False

    def _lidar_is_blocking(self, path_min: float, path_clearance: Optional[float]) -> bool:
        if path_min >= self._LIDAR_BLOCK_DIST:
            return False
        if path_clearance is None:
            return True
        return path_min < path_clearance - self._LIDAR_TARGET_MARGIN

    def _lidar_turn_sign(self) -> float:
        left = self._lidar_sector_min(self._LIDAR_SIDE_GAP, self._LIDAR_SIDE_CONE)
        right = self._lidar_sector_min(-self._LIDAR_SIDE_CONE, -self._LIDAR_SIDE_GAP)
        left_d = left if left is not None else float('inf')
        right_d = right if right is not None else float('inf')
        return 1.0 if left_d >= right_d else -1.0

    def _apply_lidar_avoidance(self, cmd: CommandsDict, *, path_clearance: Optional[float]=None) -> None:
        path_min = self._lidar_path_min()
        if path_min is None:
            self._lidar_blocked_steps = 0
            return
        fwd = float(cmd.get('forward', 0.0))
        blocked = self._lidar_is_blocking(path_min, path_clearance)
        if fwd <= 0.0 or not blocked:
            self._lidar_blocked_steps = 0
            return
        turn = self._lidar_turn_sign()
        self._lidar_blocked_steps += 1
        if self._lidar_blocked_steps >= self._LIDAR_BLOCK_TRIGGER_STEPS and (not self._lidar_escape_active()):
            self._start_lidar_escape(turn)
            cmd['forward'] = 0.0
            cmd['rotation'] = self._LIDAR_AVOID_ROTATION * turn
            return
        if path_min < self._LIDAR_BLOCK_DIST:
            cmd['forward'] = 0.0
            cmd['rotation'] = self._LIDAR_AVOID_ROTATION * turn
            return
        span = self._LIDAR_SLOW_DIST - self._LIDAR_BLOCK_DIST
        scale = max(0.0, min(1.0, (path_min - self._LIDAR_BLOCK_DIST) / span))
        cmd['forward'] = fwd * scale
        if scale < 0.55:
            cmd['rotation'] = 0.45 * turn

    def _lidar_escape_active(self) -> bool:
        return self._escape_back_steps_left > 0 or self._escape_turn_steps_left > 0 or self._escape_strafe_steps_left > 0

    def _start_lidar_escape(self, turn_sign: float) -> None:
        self._escape_turn_sign = 1.0 if turn_sign >= 0 else -1.0
        self._escape_back_steps_left = self._LIDAR_ESCAPE_BACK_STEPS
        self._escape_turn_steps_left = self._LIDAR_ESCAPE_TURN_STEPS
        self._escape_strafe_steps_left = self._LIDAR_ESCAPE_STRAFE_STEPS
        self._lidar_blocked_steps = 0

    def _run_lidar_escape_if_needed(self, cmd: CommandsDict) -> bool:
        if not self._lidar_escape_active():
            return False
        if self._escape_back_steps_left > 0:
            self._escape_back_steps_left -= 1
            cmd['forward'] = self._LIDAR_ESCAPE_BACK_SPEED
            cmd['lateral'] = 0.0
            cmd['rotation'] = 0.0
            return True
        if self._escape_turn_steps_left > 0:
            self._escape_turn_steps_left -= 1
            cmd['forward'] = 0.0
            cmd['lateral'] = 0.0
            cmd['rotation'] = self._LIDAR_AVOID_ROTATION * self._escape_turn_sign
            return True
        if self._escape_strafe_steps_left > 0:
            self._escape_strafe_steps_left -= 1
            cmd['forward'] = 0.25
            cmd['lateral'] = self._LIDAR_ESCAPE_STRAFE_SPEED * self._escape_turn_sign
            cmd['rotation'] = 0.0
            return True
        return False

    @staticmethod
    def _empty_command() -> CommandsDict:
        return {'forward': 0.0, 'lateral': 0.0, 'rotation': 0.0, 'grasper': 0, 'place_bomb': 0}

# ============================================================================
# Integrated source: red_v1_mapping
# ============================================================================

def _M01_wrap_angle(angle):
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi

class _M01_PoseFilter:
    """Predict with per-step odometry; gently correct using GPS and compass."""

    def __init__(self):
        self.position = None
        self.heading = None
        self.absolute_missing_steps = 0

    def update(self, gps, compass, odometry):
        valid_gps = gps is not None and np.shape(gps) == (2,) and np.isfinite(gps).all()
        valid_compass = compass is not None and math.isfinite(compass)
        self.absolute_missing_steps = 0 if valid_gps and valid_compass else self.absolute_missing_steps + 1
        if self.position is None:
            if not valid_gps or not valid_compass:
                return None
            self.position = np.array(gps, dtype=float)
            self.heading = float(compass)
        else:
            if odometry is not None and np.shape(odometry) == (3,) and np.isfinite(odometry).all():
                (distance, alpha, rotation) = odometry
                direction = self.heading + alpha
                self.position += distance * np.array([math.cos(direction), math.sin(direction)])
                self.heading = _M01_wrap_angle(self.heading + rotation)
            if valid_gps:
                self.position += 0.12 * (np.asarray(gps) - self.position)
            if valid_compass:
                self.heading = _M01_wrap_angle(self.heading + 0.15 * _M01_wrap_angle(compass - self.heading))
        return (self.position.copy(), self.heading)

class _M01_ObservationMap:
    """3-pixel raster. Repeated endpoints add evidence; ray interiors remove it."""
    CELL = 3

    def __init__(self, size):
        (self.width, self.height) = map(int, size)
        self.shape = ((self.height + self.CELL - 1) // self.CELL, (self.width + self.CELL - 1) // self.CELL)
        self.evidence = np.zeros(self.shape, dtype=np.float32)
        self.scans = 0

    def indices(self, points):
        points = np.asarray(points)
        cols = np.floor((points[:, 0] + self.width / 2) / self.CELL).astype(int)
        rows = np.floor((self.height / 2 - points[:, 1]) / self.CELL).astype(int)
        valid = (rows >= 0) & (rows < self.shape[0]) & (cols >= 0) & (cols < self.shape[1])
        return np.unique(rows[valid] * self.shape[1] + cols[valid])

    def observe(self, position, heading, distances, angles, excluded=None, max_range=_SF_LIDAR_RANGE):
        if distances is None:
            return
        (distances, angles) = (np.asarray(distances).ravel(), np.asarray(angles).ravel())
        valid = np.isfinite(distances) & np.isfinite(angles) & (distances > 4)
        if excluded is not None:
            valid &= ~np.asarray(excluded, dtype=bool)
        (distances, angles) = (distances[valid], angles[valid] + heading)
        if not len(distances):
            return
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        lengths = np.minimum(distances, max_range)
        samples = np.arange(12, max_range, 9)
        ray_lengths = np.broadcast_to(samples, (len(lengths), len(samples)))
        free = ray_lengths < lengths[:, None] - 15
        points = np.asarray(position) + directions[:, None, :] * ray_lengths[:, :, None]
        flat = self.evidence.ravel()
        free_indices = self.indices(points[free])
        flat[free_indices] = np.maximum(-3, flat[free_indices] - 0.3)
        hits = distances < max_range - _SF_HIT_MARGIN
        endpoints = np.asarray(position) + directions[hits] * distances[hits, None]
        hit_indices = self.indices(endpoints)
        flat[hit_indices] = np.minimum(20, flat[hit_indices] + 2)
        self.scans += 1

    def mask(self):
        return self.evidence >= 4

    def to_pixels(self, mask):
        return np.repeat(np.repeat(mask, self.CELL, axis=0), self.CELL, axis=1)[:self.height, :self.width].astype(np.uint8)

class _M01_SnapshotStore:
    """One immutable latest snapshot per origin. Relay preserves origin/version.

    Snapshots replace older masks, never add duplicate votes. All storage is
    per drone instance; there is no shared team memory or bypass of radio range.
    """

    def __init__(self, shape, number_drones):
        self.shape = tuple(shape)
        self.number_drones = number_drones
        self.snapshots = {}
        self.masks = {}
        self.accepted = 0
        self.duplicates = 0

    def publish(self, source, sequence, mask):
        payload = np.packbits(np.asarray(mask, dtype=np.uint8).ravel()).tobytes()
        self.accept((int(source), int(sequence), payload))

    def accept(self, snapshot):
        if not isinstance(snapshot, tuple) or len(snapshot) != 3:
            return False
        (source, sequence, payload) = snapshot
        if type(source) is not int or not 0 <= source < self.number_drones or type(sequence) is not int or (sequence < 0) or (not isinstance(payload, bytes)) or (len(payload) != (math.prod(self.shape) + 7) // 8):
            return False
        previous = self.snapshots.get(source)
        if previous is not None and sequence <= previous[1]:
            self.duplicates += 1
            return False
        self.snapshots[source] = snapshot
        self.masks[source] = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))[:math.prod(self.shape)].reshape(self.shape).astype(bool)
        self.accepted += 1
        return True

    def merged(self):
        result = np.zeros(self.shape, dtype=bool)
        for mask in self.masks.values():
            result |= mask
        return result

def _M01_submission_due(identifier, number_drones, step, max_steps, walltime, max_walltime):
    """Leader first, staggered backup slots; independent of missing inventory."""
    slot = 0.84 if identifier == 0 else min(0.96, 0.88 + 0.06 * identifier / max(1, number_drones - 1))
    return max_steps is not None and max_steps > 0 and (step >= slot * max_steps) or (max_walltime is not None and max_walltime > 0 and (walltime >= slot * max_walltime))

# ============================================================================
# Integrated source: my_drone_red_v1
# ============================================================================

class _M02_MyDroneRedV1(_M00_MyDronePlaceExample):
    """Reuse the example's local lidar escape; replace its mapping/end logic."""

    def __init__(self, identifier=None, misc_data=None, **kwargs):
        super().__init__(identifier=identifier, misc_data=misc_data, **kwargs)
        self._pose_filter = _M01_PoseFilter()
        self._map = _M01_ObservationMap(self.size_area) if self.size_area else None
        self._n = int(getattr(misc_data, 'number_drones', None) or 1)
        self._radio_maps = _M01_SnapshotStore(self._map.shape, self._n) if self._map else None
        self._status = {}
        self._home = None
        self._position = None
        self._heading = None
        self._nav_heading = None
        self._visits = {}
        self._initial_inventory = None
        self._last_inventory = None
        self._placements = []
        self._attempts = 0
        self._attempt_inventory = None
        self._attempt_step = None
        self._next_attempt = 0
        self._last_candidate = None
        self._stuck_position = None
        self._stuck_cycles = 0
        self._escape_cycles = 0

    def _event(self, kind, **data):
        print('V1_EVENT ' + json.dumps(dict(event=kind, drone=int(self.identifier or 0), step=int(self.elapsed_timestep), **data), sort_keys=True))

    def define_message_for_all(self):
        snapshot = None
        if self._radio_maps and self._radio_maps.snapshots:
            sources = sorted(self._radio_maps.snapshots)
            source = sources[self._steps // 8 % len(sources)]
            snapshot = self._radio_maps.snapshots[source]
        return ('red-v1', tuple(((source, *status) for (source, status) in sorted(self._status.items()))), snapshot)

    def _receive(self):
        for (_, message) in self.communicator.received_messages:
            if not isinstance(message, tuple) or len(message) != 3 or message[0] != 'red-v1':
                continue
            (statuses, snapshot) = message[1:]
            if isinstance(statuses, tuple) and len(statuses) <= self._n:
                for status in statuses:
                    if not isinstance(status, tuple) or len(status) != 3:
                        continue
                    (source, sequence, inventory) = status
                    if type(source) is int and 0 <= source < self._n and (type(sequence) is int) and (0 <= sequence <= self.elapsed_timestep) and (type(inventory) is int) and (inventory >= 0):
                        previous = self._status.get(source)
                        if previous is None or sequence > previous[0]:
                            self._status[source] = (sequence, inventory)
            if self._radio_maps and snapshot is not None:
                self._radio_maps.accept(snapshot)

    def _update_observation(self):
        pose = self._pose_filter.update(self.measured_gps_position(), self.measured_compass_angle(), self.odometer_values())
        if pose is None:
            return
        (self._position, self._heading) = pose
        if self._home is None:
            self._home = self._position.copy()
        if self._map and self._steps % 4 == 0:
            values = self.lidar_values()
            if values is not None:
                angles = np.asarray(self.lidar_rays_angles())
                distances = np.asarray(values).ravel()
                excluded = np.zeros(len(distances), dtype=bool)
                for obj in self.semantic_values() or []:
                    if not (math.isfinite(obj.distance) and math.isfinite(obj.angle)):
                        continue
                    difference = np.abs((angles - obj.angle + math.pi) % (2 * math.pi) - math.pi)
                    excluded |= (difference < 0.12) & (np.abs(distances - obj.distance) < 22)
                self._map.observe(self._position, self._heading, distances, angles, excluded)
        if self._map and (self._steps % 40 == 0 or self._steps == 1):
            self._radio_maps.publish(int(self.identifier or 0), self._steps, self._map.mask())

    def _progress(self):
        limits = self._misc_data
        fractions = []
        if limits and limits.max_timestep_limit:
            fractions.append(self.elapsed_timestep / limits.max_timestep_limit)
        if limits and limits.max_walltime_limit:
            fractions.append(self.elapsed_walltime / limits.max_walltime_limit)
        return max(fractions, default=self._steps / 1800)

    def _navigation(self):
        command = self._empty_command()
        if self._position is None:
            command['rotation'] = 0.35
            return command
        visit = tuple(np.floor(self._position / 45).astype(int))
        self._visits[visit] = self._visits.get(visit, 0) + 1
        if self._steps % 60 == 0:
            if self._stuck_position is not None and np.linalg.norm(self._position - self._stuck_position) < 12:
                self._stuck_cycles += 1
            else:
                self._stuck_cycles = 0
            self._stuck_position = self._position.copy()
            if self._stuck_cycles >= 2:
                self._start_lidar_escape((-1) ** (int(self.identifier or 0) + self._escape_cycles))
                self._escape_cycles += 1
                self._nav_heading = None
                self._stuck_cycles = 0
        if self._run_lidar_escape_if_needed(command):
            return command
        if self._nav_heading is None or self._steps % 45 == 0:
            values = self.lidar_values()
            if values is not None:
                distances = np.asarray(values).ravel()
                angles = np.asarray(self.lidar_rays_angles())
                candidates = []
                for angle in np.linspace(-math.pi, math.pi, 16, endpoint=False):
                    delta = np.abs((angles - angle + math.pi) % (2 * math.pi) - math.pi)
                    rays = distances[(delta < 0.19) & np.isfinite(distances)]
                    if not len(rays):
                        continue
                    clearance = min(200, float(np.min(rays)))
                    direction = self._heading + angle
                    target = self._position + max(30, clearance * 0.7) * np.array([math.cos(direction), math.sin(direction)])
                    cell = tuple(np.floor(target / 45).astype(int))
                    score = clearance - 15 * min(12, self._visits.get(cell, 0))
                    score += 12 * math.cos(direction - int(self.identifier or 0) * 2 * math.pi / self._n)
                    if self._progress() > 0.68:
                        home_angle = math.atan2(*(self._home - self._position)[::-1])
                        score += 65 * math.cos(direction - home_angle)
                    candidates.append((score, direction))
                if candidates:
                    self._nav_heading = max(candidates)[1]
        if self._nav_heading is None:
            return self._control_roam()
        difference = _M01_wrap_angle(self._nav_heading - self._heading)
        command['rotation'] = float(np.clip(difference * 1.5, -0.85, 0.85))
        command['forward'] = 0.7 * max(0, math.cos(difference))
        self._apply_lidar_avoidance(command)
        return command

    def _deployment(self, command):
        inventory = self.carried_bombs_count()
        if self._initial_inventory is None:
            self._initial_inventory = inventory
            self._last_inventory = inventory
        if inventory < self._last_inventory:
            self._placements.append(tuple(map(float, self._position)) if self._position is not None else None)
            self._event('placement_confirmed', inventory=inventory, attempted=self._attempt_step is not None)
            self._attempt_step = None
            self._attempt_inventory = None
            self._attempts = 0
            self._next_attempt = self._steps + 35
        self._last_inventory = inventory
        if inventory <= 0:
            return
        if self._attempt_step is not None:
            if self._steps - self._attempt_step < 4:
                command.update(forward=0, lateral=0, rotation=0, grasper=0, place_bomb=0)
                return
            self._event('placement_retry', inventory=inventory, attempts=self._attempts)
            self._attempt_step = None
            self._next_attempt = self._steps + (60 if self._attempts % 4 == 0 else 15)
        if self._attempts >= 20 or self._steps < self._next_attempt:
            return
        placed = self._initial_inventory - inventory
        due = 0.1 + 0.36 * (placed + 1) / (self._initial_inventory + 1)
        due += 0.008 * int(self.identifier or 0)
        if self._progress() < due:
            return
        path = self._lidar_path_min()
        if path is not None and path < 28 and (self._progress() < 0.72):
            return
        command.update(forward=0, lateral=0, rotation=0, grasper=0, place_bomb=1)
        self._attempt_step = self._steps
        self._attempt_inventory = inventory
        self._attempts += 1
        self._event('placement_requested', inventory=inventory, attempt=self._attempts)

    def _submit(self):
        if not self._map or self._exploration_submitted:
            return
        identifier = int(self.identifier or 0)
        self._radio_maps.publish(identifier, self._steps, self._map.mask())
        merged = self._radio_maps.merged()
        source_cells = {str(source): int(mask.sum()) for (source, mask) in self._radio_maps.masks.items()}
        self.submit_exploration_map(self._map.to_pixels(merged))
        self._exploration_submitted = True
        self._event('submitted', sources=source_cells, cells=int(merged.sum()), local_cells=int(self._map.mask().sum()), scans=self._map.scans, accepted=self._radio_maps.accepted, duplicates=self._radio_maps.duplicates, own_inventory=self.carried_bombs_count(), placements=len(self._placements), status_sources=sorted(self._status), escape_cycles=self._escape_cycles)

    def control(self):
        self._steps += 1
        self._receive()
        self._update_observation()
        identifier = int(self.identifier or 0)
        self._status[identifier] = (int(self.elapsed_timestep), int(self.carried_bombs_count()))
        command = self._navigation()
        self._deployment(command)
        limits = self._misc_data
        if _M01_submission_due(identifier, self._n, self.elapsed_timestep, getattr(limits, 'max_timestep_limit', None), self.elapsed_walltime, getattr(limits, 'max_walltime_limit', None)) or (not limits and self._steps >= 1500):
            self._submit()
        if self._steps % 240 == 0:
            self._event('status', inventory=self.carried_bombs_count(), sources=sorted(self._radio_maps.snapshots) if self._radio_maps else [], scans=self._map.scans if self._map else 0)
        return command

# ============================================================================
# Integrated source: red_v2_mapping
# ============================================================================

def _M03_fit_scan(distances, angles, excluded=None, max_range=_SF_LIDAR_RANGE):
    """Project only locally straight, contiguous hit groups onto fitted lines.

    No map geometry is assumed. Split at discontinuities and corners; never
    extrapolate past observed endpoints or bridge missing/excluded rays.
    """
    distances = np.asarray(distances, dtype=float).ravel().copy()
    angles = np.asarray(angles, dtype=float).ravel().copy()
    valid = np.isfinite(distances) & np.isfinite(angles) & (distances > 4) & (distances < max_range - _SF_HIT_MARGIN)
    if excluded is not None:
        valid &= ~np.asarray(excluded, dtype=bool)
    points = np.column_stack((distances * np.cos(angles), distances * np.sin(angles)))
    indices = np.flatnonzero(valid)
    if len(indices) < 6:
        return (distances, angles, 0)
    breaks = (np.diff(indices) != 1) | (np.linalg.norm(np.diff(points[indices], axis=0), axis=1) > 18)
    groups = np.split(indices, np.flatnonzero(breaks) + 1)
    fitted = 0

    def project(group, depth=0):
        nonlocal fitted
        if len(group) < 6:
            return
        data = points[group]
        center = data.mean(axis=0)
        centered = data - center
        (_, vectors) = np.linalg.eigh(centered.T @ centered)
        direction = vectors[:, -1]
        along = centered @ direction
        residual = centered @ vectors[:, 0]
        if np.max(np.abs(residual)) > 5 or np.sqrt(np.mean(residual ** 2)) > 2.8:
            if depth < 5 and len(group) >= 12:
                middle = len(group) // 2
                project(group[:middle + 1], depth + 1)
                project(group[middle:], depth + 1)
            return
        if np.ptp(along) < 18:
            return
        corrected = center + along[:, None] * direction
        distances[group] = np.linalg.norm(corrected, axis=1)
        angles[group] = np.arctan2(corrected[:, 1], corrected[:, 0])
        fitted += 1
    for group in groups:
        project(group)
    return (distances, angles, fitted)

class _M03_ObservationMapV2(_M01_ObservationMap):
    """Preserve the V1 evidence grid and radio format; refine only observations."""

    def __init__(self, size):
        super().__init__(size)
        self.static_nonwall = np.zeros(self.shape, dtype=np.uint8)
        self.fitted_segments = 0
        self.nonwall_observations = 0

    def observe(self, position, heading, distances, angles, excluded=None, max_range=_SF_LIDAR_RANGE):
        if distances is None:
            return
        (distances, angles, count) = _M03_fit_scan(distances, angles, excluded, max_range)
        self.fitted_segments += count
        super().observe(position, heading, distances, angles, excluded, max_range)

    def observe_nonwalls(self, position, heading, objects):
        points = []
        for obj in objects:
            if getattr(obj.entity_type, 'name', None) != 'DISPOSAL_CENTER':
                continue
            if not (math.isfinite(obj.distance) and math.isfinite(obj.angle)):
                continue
            direction = heading + obj.angle
            points.append(np.asarray(position) + obj.distance * np.array([math.cos(direction), math.sin(direction)]))
        if not points:
            return
        indices = self.indices(np.asarray(points))
        hits = np.zeros(self.shape, dtype=np.uint8)
        hits.ravel()[indices] = 1
        hits = cv2.dilate(hits, np.ones((3, 3), np.uint8))
        self.static_nonwall = np.minimum(8, self.static_nonwall.astype(np.uint16) + hits).astype(np.uint8)
        self.nonwall_observations += len(points)

    def mask(self):
        mask = super().mask()
        excluded = cv2.dilate((self.static_nonwall >= 2).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        return mask & ~excluded

    def to_pixels(self, mask):
        pixels = super().to_pixels(mask)
        pixels = cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        (count, labels, stats, _) = cv2.connectedComponentsWithStats(pixels, connectivity=8)
        keep = np.zeros(count, dtype=np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 27
        return keep[labels]

# ============================================================================
# Integrated source: my_drone_red_v2
# ============================================================================

class _M04_MyDroneRedV2(_M02_MyDroneRedV1):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.size_area:
            self._map = _M03_ObservationMapV2(self.size_area)
            self._radio_maps = _M01_SnapshotStore(self._map.shape, self._n)

    def _update_observation(self):
        super()._update_observation()
        if self._map and self._position is not None and (self._steps % 4 == 0):
            self._map.observe_nonwalls(self._position, self._heading, self.semantic_values() or [])

    def _event(self, kind, **data):
        if kind == 'submitted':
            data.update(fitted_segments=self._map.fitted_segments, nonwall_observations=self._map.nonwall_observations)
        print('V2_EVENT ' + json.dumps(dict(event=kind, drone=int(self.identifier or 0), step=int(self.elapsed_timestep), **data), sort_keys=True))

# ============================================================================
# Integrated source: red_v3_navigation
# ============================================================================

class _M05_ExplorationGrid:
    CELL = 20

    def __init__(self, size):
        (self.width, self.height) = size
        self.shape = (math.ceil(self.height / self.CELL), math.ceil(self.width / self.CELL))
        self.free = np.zeros(self.shape, bool)
        self.obstacle = np.zeros(self.shape, np.float32)
        self.visits = np.zeros(self.shape, np.uint16)
        self.shared_visited = np.zeros(self.shape, bool)
        self.goal_visits = np.zeros(self.shape, np.float32)

    def cell(self, position):
        return (int(np.clip((self.height / 2 - position[1]) / self.CELL, 0, self.shape[0] - 1)), int(np.clip((position[0] + self.width / 2) / self.CELL, 0, self.shape[1] - 1)))

    def point(self, cell):
        (r, c) = cell
        return np.array([(c + 0.5) * self.CELL - self.width / 2, self.height / 2 - (r + 0.5) * self.CELL])

    def indices(self, points):
        c = np.floor((points[:, 0] + self.width / 2) / self.CELL).astype(int)
        r = np.floor((self.height / 2 - points[:, 1]) / self.CELL).astype(int)
        good = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
        return np.unique(r[good] * self.shape[1] + c[good])

    def observe(self, position, heading, distances, angles, dynamic=None):
        d = np.asarray(distances).ravel()
        a = np.asarray(angles).ravel() + heading
        valid = np.isfinite(d) & np.isfinite(a) & (d > 4)
        if not valid.any():
            return
        dyn = np.zeros(len(d), bool) if dynamic is None else np.asarray(dynamic, bool)
        (d, a, dyn) = (d[valid], a[valid], dyn[valid])
        directions = np.column_stack((np.cos(a), np.sin(a)))
        steps = np.arange(0, _SF_LIDAR_RANGE, 12)
        pts = np.asarray(position) + directions[:, None, :] * steps[None, :, None]
        free = self.indices(pts[steps[None, :] < np.minimum(d[:, None], _SF_LIDAR_RANGE) - 26])
        self.free.ravel()[free] = True
        self.obstacle.ravel()[free] = np.maximum(0, self.obstacle.ravel()[free] - 0.5)
        hit = (d < _SF_HIT_LIMIT) & ~dyn
        indices = self.indices(np.asarray(position) + directions[hit] * d[hit, None])
        self.obstacle.ravel()[indices] = np.minimum(6, self.obstacle.ravel()[indices] + 1)
        cell = self.cell(position)
        self.free[cell] = True
        self.visits[cell] = min(65535, int(self.visits[cell]) + 1)

    def retire_goal(self, cell):
        """Penalize the whole inspected region, not just one interchangeable pixel."""
        (r, c) = cell
        (yy, xx) = np.ogrid[:self.shape[0], :self.shape[1]]
        self.goal_visits[(yy - r) ** 2 + (xx - c) ** 2 <= 16] += 1

    def line_clear(self, position, target):
        points = np.linspace(position, target, max(2, int(np.linalg.norm(target - position) / 8) + 1))
        walls = cv2.dilate((self.obstacle >= 2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        return all((self.free[self.cell(p)] and (not walls[self.cell(p)]) for p in points[1:]))

    def plan(self, position, home, identifier, number, claims, blocked_goals=(), rendezvous=False):
        """Search reachable observed free cells; never route through unknown."""
        walls = cv2.dilate((self.obstacle >= 2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))).astype(bool)
        allowed = self.free & ~walls
        start = self.cell(position)
        allowed[start] = True
        distance = {start: 0}
        parents = {}
        queue = deque([start])
        while queue:
            (r, c) = queue.popleft()
            for (dr, dc) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other = (r + dr, c + dc)
                if 0 <= other[0] < self.shape[0] and 0 <= other[1] < self.shape[1] and allowed[other] and (other not in distance):
                    distance[other] = distance[r, c] + 1
                    parents[other] = (r, c)
                    queue.append(other)
        if len(distance) < 3:
            return []
        unknown = cv2.filter2D((~self.free & ~walls).astype(np.float32), -1, np.ones((5, 5), np.float32))
        frontier = (unknown > 0) & allowed
        has_frontier = any((frontier[cell] and self.goal_visits[cell] < 2 for cell in distance))
        seen = self.shared_visited | (self.visits > 0)
        nearby = cv2.filter2D(seen.astype(np.float32), -1, np.ones((7, 7), np.float32))
        preferred = 2 * math.pi * identifier / max(1, number)
        best = None
        for (cell, steps) in distance.items():
            if steps < 3 or cell in blocked_goals:
                continue
            point = self.point(cell)
            delta = point - np.asarray(home)
            novelty = min(12, unknown[cell]) * 5 if has_frontier else 0
            score = novelty + (49 - nearby[cell]) * 2 - 0.65 * steps
            score -= min(100, int(self.visits[cell]) * 6) + self.goal_visits[cell] * 70
            score += 60 * math.cos(math.atan2(delta[1], delta[0]) - preferred)
            for goal in claims:
                score -= 160 * max(0, 1 - np.linalg.norm(point - goal) / 250)
            if rendezvous:
                score = -np.linalg.norm(point - np.asarray(home)) - 0.1 * steps
            candidate = (score, cell)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            return []
        path = [best[1]]
        while path[-1] != start:
            path.append(parents[path[-1]])
        return list(reversed(path))

def _M05_safe_velocity(distances, angles, target_angle, velocity, previous_angle=None):
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
    headings = np.linspace(-math.pi, math.pi, 48, endpoint=False)
    headings = np.append(headings, target_angle)
    (x, y) = (d * np.cos(a), d * np.sin(a))
    (cosine, sine) = (np.cos(headings)[:, None], np.sin(headings)[:, None])
    along = cosine * x + sine * y
    across = cosine * y - sine * x
    radius = 25.0
    intersects = (along > 0) & (np.abs(across) < radius) & (d[None, :] < _SF_HIT_LIMIT)
    entry = along - np.sqrt(np.maximum(0, radius ** 2 - across ** 2))
    clearance = np.maximum(0, np.min(np.where(intersects, entry, 220), axis=1))
    speed = np.minimum(6.5, np.sqrt(0.8 * np.maximum(0, clearance - 10)))
    scores = speed * np.cos(headings - target_angle) + 0.008 * np.minimum(clearance, 120)
    if previous_angle is not None:
        scores += 0.35 * np.cos(headings - previous_angle)
    best = int(np.argmax(scores))
    desired = speed[best] * np.array([math.cos(headings[best]), math.sin(headings[best])])
    command = 0.085 * desired + 0.65 * (desired - np.asarray(velocity))
    norm = np.linalg.norm(command)
    if norm > 1:
        command /= norm
    return (command, float(headings[best]))

# ============================================================================
# Integrated source: my_drone_red_v3
# ============================================================================

class _M06_MyDroneRedV3(_M04_MyDroneRedV2):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._explore = _M05_ExplorationGrid(self.size_area)
        self._peers = {}
        self._route = []
        self._goal_step = 0
        self._blocked_goals = {}
        self._last_route_step = -100
        self._last_motion_angle = None
        self._velocity = np.zeros(2)
        self._motion_anchor = None
        self._replans = 0
        self._yield_steps = 0
        self._stalled_steps = 0
        self._visited_snapshot = None
        self._gathering = False
        self._escape_until = 0
        self._escape_heading = None
        self._failed_place = None
        self._attempt_position = None
        self._total_attempts = 0
        self._waypoint_distance = float('inf')
        self._last_progress_step = 0
        self._yield_until = 0
        self._yield_heading = None

    def define_message_for_all(self):
        base = super().define_message_for_all()
        own = None
        if self._position is not None:
            goal = tuple(self._explore.point(self._route[-1])) if self._route else None
            own = (int(self.identifier or 0), self._steps, tuple(map(float, self._position)), goal)
        return ('red-v3', base, own, self._visited_snapshot)

    def _receive(self):
        for (_, message) in self.communicator.received_messages:
            if not isinstance(message, tuple) or len(message) != 4 or message[0] != 'red-v3':
                continue
            (base, peer, visited) = message[1:]
            if isinstance(base, tuple) and len(base) == 3 and (base[0] == 'red-v1') and isinstance(base[1], tuple) and (len(base[1]) <= self._n):
                for status in base[1]:
                    if not isinstance(status, tuple) or len(status) != 3:
                        continue
                    (source, seq, inventory) = status
                    if type(source) is int and 0 <= source < self._n and (type(seq) is int) and (0 <= seq <= self.elapsed_timestep) and (type(inventory) is int) and (inventory >= 0):
                        if seq > self._status.get(source, (-1, 0))[0]:
                            self._status[source] = (seq, inventory)
                if base[2] is not None:
                    self._radio_maps.accept(base[2])
            if self._valid_peer(peer):
                (source, seq, position, goal) = peer
                if seq > self._peers.get(source, (-1, None, None))[0]:
                    self._peers[source] = (seq, np.array(position), None if goal is None else np.array(goal))
            if isinstance(visited, bytes) and len(visited) == (math.prod(self._explore.shape) + 7) // 8:
                self._explore.shared_visited |= np.unpackbits(np.frombuffer(visited, np.uint8))[:math.prod(self._explore.shape)].reshape(self._explore.shape).astype(bool)

    def _valid_peer(self, peer):
        if not isinstance(peer, tuple) or len(peer) != 4:
            return False
        (source, seq, position, goal) = peer
        return type(source) is int and 0 <= source < self._n and (type(seq) is int) and (0 <= seq <= self.elapsed_timestep) and isinstance(position, tuple) and (len(position) == 2) and all((isinstance(v, (float, int)) and math.isfinite(v) for v in position)) and (goal is None or (isinstance(goal, tuple) and len(goal) == 2 and all((isinstance(v, (float, int)) and math.isfinite(v) for v in goal))))

    def _navigation(self):
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
        odom = self.odometer_values()
        if odom is not None and np.shape(odom) == (3,) and np.isfinite(odom).all():
            direction = odom[1] - odom[2]
            measured = odom[0] * np.array([math.cos(direction), math.sin(direction)])
            self._velocity = 0.4 * self._velocity + 0.6 * measured
        if self._steps % 6 == 0 or self._steps == 1:
            self._explore.observe(self._position, self._heading, d, angles, dynamic)
        if self._steps % 30 == 0:
            self._visited_snapshot = np.packbits(self._explore.shared_visited | (self._explore.visits > 0)).tobytes()
            self._radio_maps.publish(int(self.identifier or 0), self._steps, self._map.mask())
        fresh = {i: p for (i, p) in self._peers.items() if self._steps - p[0] < 90}
        claims = [p[2] for (i, p) in fresh.items() if p[2] is not None and i < int(self.identifier or 0)]
        if self._progress() > 0.72 and (not self._gathering):
            self._gathering = True
            self._route = []
        if self._steps % 60 == 0:
            if self._motion_anchor is not None and np.linalg.norm(self._position - self._motion_anchor) < 25:
                self._stalled_steps += 60
                self._start_recovery(d, angles, 'no_progress')
            self._motion_anchor = self._position.copy()
        self._blocked_goals = {g: t for (g, t) in self._blocked_goals.items() if t > self._steps}
        if self._route and np.linalg.norm(self._position - self._explore.point(self._route[-1])) < 40:
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
        while len(self._route) > 1 and np.linalg.norm(self._position - self._explore.point(self._route[0])) < 35:
            self._route.pop(0)
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
            target_angle = math.atan2(target[1], target[0]) - self._heading
        else:
            if self._escape_heading is None or self._steps >= self._escape_until:
                self._start_recovery(d, angles, 'disconnected')
            target_angle = self._escape_heading - self._heading
        if self._steps < self._escape_until and self._escape_heading is not None:
            target_angle = self._escape_heading - self._heading
        if self._steps >= self._yield_until:
            for (i, (seq, position, goal)) in fresh.items():
                delta = position - self._position
                if self._steps - seq < 8 and i < int(self.identifier or 0) and (np.linalg.norm(delta) < 75):
                    angle = math.atan2(delta[1], delta[0]) - self._heading
                    if math.cos(angle - target_angle) > 0.4:
                        self._yield_heading = self._heading + angle - math.pi / 2
                        self._yield_until = self._steps + 18
                        break
        if self._steps < self._yield_until:
            target_angle = self._yield_heading - self._heading
            self._yield_steps += 1
        steering_d = d.copy()
        steering_d[bomb_rays] = 80
        (move, self._last_motion_angle) = _M05_safe_velocity(steering_d, angles, _M01_wrap_angle(target_angle), self._velocity, self._last_motion_angle)
        command.update(forward=float(move[0]), lateral=float(move[1]), rotation=0.0)
        return command

    def _start_recovery(self, distances, angles, reason):
        if self._steps < self._escape_until:
            return
        if self._route:
            goal = self._route[-1]
            self._explore.retire_goal(goal)
            self._blocked_goals[goal] = self._steps + 600
        self._route = []
        choices = []
        for angle in np.linspace(-math.pi, math.pi, 32, endpoint=False):
            delta = np.abs((angles - angle + math.pi) % (2 * math.pi) - math.pi)
            rays = distances[(delta < 0.25) & np.isfinite(distances)]
            if not len(rays):
                continue
            clearance = min(180, float(np.min(rays)))
            direction = angle + self._heading
            target = self._position + min(100, clearance * 0.65) * np.array([math.cos(direction), math.sin(direction)])
            cell = self._explore.cell(target)
            score = clearance - 5 * min(20, int(self._explore.visits[cell]))
            score += 15 * math.cos(direction - 2 * math.pi * int(self.identifier or 0) / self._n)
            if self._escape_heading is not None:
                score += 12 * math.cos(direction - self._escape_heading)
            choices.append((score, direction))
        self._escape_heading = max(choices)[1] if choices else self._heading + math.pi
        self._escape_until = self._steps + 40
        self._escape_cycles += 1
        self._event('recovery', reason=reason, position=list(map(float, self._position)))

    def _deployment(self, command):
        inventory = self.carried_bombs_count()
        if self._initial_inventory is None:
            self._initial_inventory = self._last_inventory = inventory
        if inventory < self._last_inventory:
            self._placements.append(tuple(map(float, self._attempt_position)))
            self._event('placement_confirmed', inventory=inventory, attempted=self._attempt_step is not None)
            self._attempt_step = None
            self._attempts = 0
            self._failed_place = None
            self._next_attempt = self._steps + 35
        self._last_inventory = inventory
        if inventory <= 0 or self._position is None:
            return
        if self._attempt_step is not None:
            if self._steps - self._attempt_step < 4:
                command['place_bomb'] = 0
                return
            self._failed_place = self._attempt_position.copy()
            self._attempt_step = None
            self._next_attempt = self._steps + 30
            self._event('placement_retry', inventory=inventory, attempts=self._attempts)
            self._escape_until = 0
            self._start_recovery(np.asarray(self.lidar_values()).ravel(), np.asarray(self.lidar_rays_angles()), 'placement_failed')
        if self._steps < self._next_attempt or self._total_attempts >= 60:
            return
        if self._failed_place is not None and np.linalg.norm(self._position - self._failed_place) < 65:
            return
        placed = self._initial_inventory - inventory
        due = 0.1 + 0.36 * (placed + 1) / (self._initial_inventory + 1) + 0.008 * int(self.identifier or 0)
        if self._progress() < due:
            return
        if any((p is not None and np.linalg.norm(self._position - p) < 55 for p in self._placements)):
            return
        if any((getattr(o.entity_type, 'name', None) == 'BOMB' and o.distance < 45 for o in self.semantic_values() or [])):
            return
        command.update(grasper=0, place_bomb=1)
        self._attempt_step = self._steps
        self._attempt_position = self._position.copy()
        self._attempts += 1
        self._total_attempts += 1
        self._event('placement_requested', inventory=inventory, attempt=self._attempts)

    def _event(self, kind, **data):
        if kind == 'status' and self._position is not None:
            data.update(position=list(map(float, self._position)), speed=float(np.linalg.norm(self._velocity)), goal=list(map(float, self._explore.point(self._route[-1]))) if self._route else None)
        if kind == 'submitted':
            data.update(replans=self._replans, yield_steps=self._yield_steps, stalled_steps=self._stalled_steps, visited_cells=int((self._explore.visits > 0).sum()))
        print('V3_EVENT ' + json.dumps(dict(event=kind, drone=int(self.identifier or 0), step=int(self.elapsed_timestep), **data), sort_keys=True))

# ============================================================================
# Integrated source: red_v4_placement
# ============================================================================

def _M07_choose_bomb_route(grid, position, wall_mask, wall_cell, placed, claims, radius, max_steps=35):
    mask = np.asarray(wall_mask, np.float32).copy()
    (yy, xx) = np.indices(mask.shape)
    wx = (xx + 0.5) * wall_cell - grid.width / 2
    wy = grid.height / 2 - (yy + 0.5) * wall_cell
    edge = (np.abs(wx) > grid.width / 2 - 15) | (np.abs(wy) > grid.height / 2 - 15)
    mask[edge] *= 0.5
    for point in placed:
        mask[(wx - point[0]) ** 2 + (wy - point[1]) ** 2 <= radius ** 2] = 0
    for point in claims:
        mask[(wx - point[0]) ** 2 + (wy - point[1]) ** 2 <= radius ** 2] *= 0.15
    r = int(round(radius / wall_cell))
    (ky, kx) = np.ogrid[-r:r + 1, -r:r + 1]
    kernel = (kx * kx + ky * ky <= r * r).astype(np.float32)
    gain = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    walls = cv2.dilate((grid.obstacle >= 2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    allowed = grid.free & ~walls.astype(bool)
    start = grid.cell(position)
    allowed[start] = True
    distances = {start: 0}
    parents = {}
    queue = deque([start])
    best = None
    while queue:
        cell = queue.popleft()
        steps = distances[cell]
        point = grid.point(cell)
        if all((np.linalg.norm(point - p) > 55 for p in placed)):
            row = int(np.clip((grid.height / 2 - point[1]) / wall_cell, 0, mask.shape[0] - 1))
            col = int(np.clip((point[0] + grid.width / 2) / wall_cell, 0, mask.shape[1] - 1))
            value = float(gain[row, col])
            score = value / (1 + 0.035 * steps)
            if best is None or score > best[0]:
                best = (score, cell, value)
        if steps >= max_steps:
            continue
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = (cell[0] + dr, cell[1] + dc)
            if 0 <= other[0] < grid.shape[0] and 0 <= other[1] < grid.shape[1] and allowed[other] and (other not in distances):
                distances[other] = steps + 1
                parents[other] = cell
                queue.append(other)
    if best is None or best[2] <= 0:
        return ([], 0.0)
    path = [best[1]]
    while path[-1] != start:
        path.append(parents[path[-1]])
    return (list(reversed(path)), best[2])

# ============================================================================
# Integrated source: my_drone_red_v4
# ============================================================================

class _M08_MyDroneRedV4(_M06_MyDroneRedV3):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bomb_target = None
        self._bomb_target_step = 0
        self._last_bomb_plan = -100
        self._bomb_records = {}
        self._bomb_claims = {}
        self._bomb_gain = 0.0

    def define_message_for_all(self):
        source = int(self.identifier or 0)
        self._bomb_records[source] = (self._steps, tuple(self._placements))
        claim = None if self._bomb_target is None else tuple(map(float, self._bomb_target))
        records = tuple(((i, seq, points) for (i, (seq, points)) in sorted(self._bomb_records.items())))
        return ('red-v4', super().define_message_for_all(), (source, self._steps, claim), records)

    def _receive(self):
        stripped = []
        for (sender, message) in self.communicator.received_messages:
            if not isinstance(message, tuple) or len(message) != 4 or message[0] != 'red-v4':
                continue
            (base, claim, records) = message[1:]
            stripped.append((sender, base))
            if isinstance(claim, tuple) and len(claim) == 3:
                (source, seq, point) = claim
                if self._valid_peer((source, seq, (0.0, 0.0), point)) and seq > self._bomb_claims.get(source, (-1, None))[0]:
                    self._bomb_claims[source] = (seq, point)
            if not isinstance(records, tuple) or len(records) > self._n:
                continue
            for record in records:
                if not isinstance(record, tuple) or len(record) != 3:
                    continue
                (source, seq, points) = record
                if type(source) is int and 0 <= source < self._n and (type(seq) is int) and (0 <= seq <= self.elapsed_timestep) and isinstance(points, tuple) and (len(points) <= 100) and all((isinstance(p, tuple) and len(p) == 2 and all((isinstance(v, (int, float)) and math.isfinite(v) for v in p)) for p in points)) and (seq > self._bomb_records.get(source, (-1, ()))[0]):
                    self._bomb_records[source] = (seq, points)
        proxy = SimpleNamespace(_n=self._n, elapsed_timestep=self.elapsed_timestep, _status=self._status, _peers=self._peers, _radio_maps=self._radio_maps, _explore=self._explore, _valid_peer=self._valid_peer, communicator=SimpleNamespace(received_messages=stripped))
        _M06_MyDroneRedV3._receive(proxy)

    def _known_bombs(self):
        points = list(self._placements)
        for (source, (_, placed)) in self._bomb_records.items():
            if source != int(self.identifier or 0):
                points.extend(placed)
        return [np.array(p) for p in points if p is not None]

    def _navigation(self):
        if self._position is not None and self.carried_bombs_count() > 0 and (self._progress() < 0.68):
            initial = self._initial_inventory or self.carried_bombs_count()
            placed = initial - self.carried_bombs_count()
            due = 0.1 + 0.36 * (placed + 1) / (initial + 1) + 0.008 * int(self.identifier or 0)
            if self._bomb_target is not None and self._steps - self._bomb_target_step > 180:
                self._bomb_target = None
                self._route = []
            if self._progress() >= due and self._bomb_target is None and (self._steps - self._last_bomb_plan >= 45):
                claims = [np.array(p) for (i, (seq, p)) in self._bomb_claims.items() if i < int(self.identifier or 0) and self._steps - seq < 90 and (p is not None)]
                wall = self._radio_maps.merged() | self._map.mask()
                (route, gain) = _M07_choose_bomb_route(self._explore, self._position, wall, self._map.CELL, self._known_bombs() + ([] if self._failed_place is None else [self._failed_place]), claims, round(0.12 * min(self.size_area)))
                self._last_bomb_plan = self._steps
                if route:
                    self._bomb_target = self._explore.point(route[-1])
                    self._bomb_target_step = self._steps
                    self._route = route
                    self._goal_step = self._steps
                    self._bomb_gain = gain
                    self._event('bomb_target', target=list(map(float, self._bomb_target)), estimated_gain=gain)
        return super()._navigation()

    def _start_recovery(self, distances, angles, reason):
        if self._steps >= self._escape_until:
            self._bomb_target = None
            self._last_bomb_plan = self._steps
        super()._start_recovery(distances, angles, reason)

    def _deployment(self, command):
        confirming = self._attempt_step is not None or (self._last_inventory is not None and self.carried_bombs_count() < self._last_inventory)
        if not confirming and self._progress() < 0.68:
            if self._bomb_target is None or np.linalg.norm(self._position - self._bomb_target) > 40:
                return
        before = len(self._placements)
        super()._deployment(command)
        if len(self._placements) > before or self._failed_place is not None:
            self._bomb_target = None

# ============================================================================
# Integrated source: red_v4_navigation
# ============================================================================

class _M09_ExplorationGrid:
    CELL = 20

    def __init__(self, size):
        (self.width, self.height) = size
        self.shape = (math.ceil(self.height / self.CELL), math.ceil(self.width / self.CELL))
        self.free = np.zeros(self.shape, bool)
        self.obstacle = np.zeros(self.shape, np.float32)
        self.visits = np.zeros(self.shape, np.uint16)
        self.shared_visited = np.zeros(self.shape, bool)
        self.forward_direction = None
        self.goal_visits = np.zeros(self.shape, np.float32)

    def cell(self, position):
        return (int(np.clip((self.height / 2 - position[1]) / self.CELL, 0, self.shape[0] - 1)), int(np.clip((position[0] + self.width / 2) / self.CELL, 0, self.shape[1] - 1)))

    def point(self, cell):
        (r, c) = cell
        return np.array([(c + 0.5) * self.CELL - self.width / 2, self.height / 2 - (r + 0.5) * self.CELL])

    def indices(self, points):
        c = np.floor((points[:, 0] + self.width / 2) / self.CELL).astype(int)
        r = np.floor((self.height / 2 - points[:, 1]) / self.CELL).astype(int)
        good = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
        return np.unique(r[good] * self.shape[1] + c[good])

    def observe(self, position, heading, distances, angles, dynamic=None):
        d = np.asarray(distances).ravel()
        a = np.asarray(angles).ravel() + heading
        valid = np.isfinite(d) & np.isfinite(a) & (d > 4)
        if not valid.any():
            return
        dyn = np.zeros(len(d), bool) if dynamic is None else np.asarray(dynamic, bool)
        (d, a, dyn) = (d[valid], a[valid], dyn[valid])
        directions = np.column_stack((np.cos(a), np.sin(a)))
        steps = np.arange(0, _SF_LIDAR_RANGE, 12)
        pts = np.asarray(position) + directions[:, None, :] * steps[None, :, None]
        free = self.indices(pts[steps[None, :] < np.minimum(d[:, None], _SF_LIDAR_RANGE) - 26])
        self.free.ravel()[free] = True
        self.obstacle.ravel()[free] = np.maximum(0, self.obstacle.ravel()[free] - 0.5)
        hit = (d < _SF_HIT_LIMIT) & ~dyn
        indices = self.indices(np.asarray(position) + directions[hit] * d[hit, None])
        self.obstacle.ravel()[indices] = np.minimum(6, self.obstacle.ravel()[indices] + 1)
        cell = self.cell(position)
        self.free[cell] = True
        self.visits[cell] = min(65535, int(self.visits[cell]) + 1)

    def retire_goal(self, cell):
        """Penalize the whole inspected region, not just one interchangeable pixel."""
        (r, c) = cell
        (yy, xx) = np.ogrid[:self.shape[0], :self.shape[1]]
        self.goal_visits[(yy - r) ** 2 + (xx - c) ** 2 <= 16] += 1

    def line_clear(self, position, target):
        points = np.linspace(position, target, max(2, int(np.linalg.norm(target - position) / 8) + 1))
        walls = cv2.dilate((self.obstacle >= 2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        return all((self.free[self.cell(p)] and (not walls[self.cell(p)]) for p in points[1:]))

    def plan(self, position, home, identifier, number, claims, blocked_goals=(), rendezvous=False):
        """Search reachable observed free cells; never route through unknown."""
        walls = cv2.dilate((self.obstacle >= 2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))).astype(bool)
        allowed = self.free & ~walls
        start = self.cell(position)
        allowed[start] = True
        distance = {start: 0}
        parents = {}
        queue = deque([start])
        while queue:
            (r, c) = queue.popleft()
            for (dr, dc) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other = (r + dr, c + dc)
                if 0 <= other[0] < self.shape[0] and 0 <= other[1] < self.shape[1] and allowed[other] and (other not in distance):
                    distance[other] = distance[r, c] + 1
                    parents[other] = (r, c)
                    queue.append(other)
        if len(distance) < 3:
            return []
        unknown = cv2.filter2D((~self.free & ~walls).astype(np.float32), -1, np.ones((5, 5), np.float32))
        frontier = (unknown > 0) & allowed
        has_frontier = any((frontier[cell] and self.goal_visits[cell] < 2 for cell in distance))
        seen = self.shared_visited | (self.visits > 0)
        nearby = cv2.filter2D(seen.astype(np.float32), -1, np.ones((7, 7), np.float32))
        preferred = 2 * math.pi * identifier / max(1, number)
        best = None
        for (cell, steps) in distance.items():
            if steps < 3 or cell in blocked_goals:
                continue
            point = self.point(cell)
            delta = point - np.asarray(home)
            novelty = min(12, unknown[cell]) * 5 if has_frontier else 0
            score = novelty + (49 - nearby[cell]) * 3 - 0.4 * steps
            displacement = point - np.asarray(position)
            if self.forward_direction is not None and np.linalg.norm(displacement) > 1:
                score += 55 * float(np.dot(displacement / np.linalg.norm(displacement), self.forward_direction))
            score += min(180, np.linalg.norm(displacement)) * 0.18
            score -= min(100, int(self.visits[cell]) * 6) + self.goal_visits[cell] * 70
            score += 60 * math.cos(math.atan2(delta[1], delta[0]) - preferred)
            for goal in claims:
                score -= 160 * max(0, 1 - np.linalg.norm(point - goal) / 250)
            if rendezvous:
                score = -np.linalg.norm(point - np.asarray(home)) - 0.1 * steps
            candidate = (score, cell)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            return []
        path = [best[1]]
        while path[-1] != start:
            path.append(parents[path[-1]])
        return list(reversed(path))

def _M09_safe_velocity(distances, angles, target_angle, velocity, previous_angle=None):
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
    headings = np.linspace(-math.pi, math.pi, 48, endpoint=False)
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

# ============================================================================
# Integrated source: my_drone_red_v4_refined
# ============================================================================

class _M10_MyDroneRedV4Refined(_M08_MyDroneRedV4):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._explore = _M09_ExplorationGrid(self.size_area)
        self._direction_anchor = None
        self._recovery_origin = None
        self._slow_ticks = 0
        self._last_motion_recovery = -100
        self._launch_open = None
        self._launch_angle = 2 * math.pi * int(self.identifier or 0) / self._n

    def _navigation(self):
        if self._position is not None and self.carried_bombs_count() > 0 and (self._progress() < 0.68):
            initial = self._initial_inventory or self.carried_bombs_count()
            placed = initial - self.carried_bombs_count()
            due = 0.1 + 0.36 * (placed + 1) / (initial + 1) + 0.008 * int(self.identifier or 0)
            if self._bomb_target is not None and self._steps - self._bomb_target_step > 180:
                self._bomb_target = None
                self._route = []
            committed = self._route and self._steps - self._goal_step < 100 and (self._progress() < 0.55)
            if not committed and self._progress() >= due and (self._bomb_target is None) and (self._steps - self._last_bomb_plan >= 45):
                claims = [np.array(p) for (i, (seq, p)) in self._bomb_claims.items() if i < int(self.identifier or 0) and self._steps - seq < 90 and (p is not None)]
                wall = self._radio_maps.merged() | self._map.mask()
                (route, gain) = _M07_choose_bomb_route(self._explore, self._position, wall, self._map.CELL, self._known_bombs() + ([] if self._failed_place is None else [self._failed_place]), claims, round(0.12 * min(self.size_area)))
                self._last_bomb_plan = self._steps
                if route:
                    self._bomb_target = self._explore.point(route[-1])
                    self._bomb_target_step = self._steps
                    self._route = route
                    self._goal_step = self._steps
                    self._bomb_gain = gain
                    self._event('bomb_target', target=list(map(float, self._bomb_target)), estimated_gain=gain)
        return self._navigate_motion()

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
        self._slow_ticks = self._slow_ticks + 1 if np.linalg.norm(self._velocity) < 0.8 else max(0, self._slow_ticks - 2)
        if self._slow_ticks >= 24 and self._steps - self._last_motion_recovery >= 40:
            self._start_recovery(d, angles, 'no_progress')
            self._last_motion_recovery = self._steps
            self._slow_ticks = 0
        if self._steps % 6 == 0 or self._steps == 1:
            self._explore.observe(self._position, self._heading, d, angles, dynamic)
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
        if self._progress() > 0.72 and (not self._gathering):
            self._gathering = True
            self._route = []
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
            target_angle = math.atan2(target[1], target[0]) - self._heading
        else:
            if self._escape_heading is None or self._steps >= self._escape_until:
                self._start_recovery(d, angles, 'disconnected')
            target_angle = self._escape_heading - self._heading
        if self._steps < self._escape_until and self._escape_heading is not None:
            target_angle = self._escape_heading - self._heading
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
        steering_d = d.copy()
        steering_d[bomb_rays] = 80
        (move, self._last_motion_angle) = _M09_safe_velocity(steering_d, angles, _M01_wrap_angle(target_angle), self._velocity, self._last_motion_angle)
        command.update(forward=float(move[0]), lateral=float(move[1]), rotation=0.0)
        return command

    def _start_recovery(self, distances, angles, reason):
        if reason == 'no_progress':
            self._escape_until = 0
        if self._steps < self._escape_until:
            return
        previous = self._escape_heading
        failed = self._recovery_origin is not None and np.linalg.norm(self._position - self._recovery_origin) < 25
        super()._start_recovery(distances, angles, reason)
        choices = []
        d = np.asarray(distances).ravel()
        a = np.asarray(angles).ravel()
        valid = np.isfinite(d) & np.isfinite(a) & (d > 0)
        if valid.sum() < 12:
            return
        (d, a) = (d[valid], a[valid])
        (x, y) = (d * np.cos(a), d * np.sin(a))
        for angle in np.linspace(-math.pi, math.pi, 32, endpoint=False):
            along = x * math.cos(angle) + y * math.sin(angle)
            across = y * math.cos(angle) - x * math.sin(angle)
            hit = (along > 0) & (np.abs(across) < 25) & (d < _SF_HIT_LIMIT)
            clearance = max(0, float(np.min(np.where(hit, along - np.sqrt(np.maximum(0, 25 ** 2 - across ** 2)), 180))))
            direction = angle + self._heading
            target = self._position + min(100, clearance * 0.65) * np.array([math.cos(direction), math.sin(direction)])
            cell = self._explore.cell(target)
            score = clearance - 5 * min(20, int(self._explore.visits[cell]))
            score += 15 * math.cos(direction - self._launch_angle)
            if previous is not None:
                score += (-100 if failed else 12) * math.cos(direction - previous)
            choices.append((score, direction, clearance))
        feasible = [choice for choice in choices if choice[2] >= 45]
        self._escape_heading = max(feasible)[1] if feasible else max(choices, key=lambda c: c[2])[1]
        self._recovery_origin = self._position.copy()

# ============================================================================
# Integrated source: red_v4_loop_navigation
# ============================================================================

class _M11_LoopWatch:

    def __init__(self):
        self.samples = deque(maxlen=21)
        self.travel = 0.0
        self.cooldown = 0

    def update(self, step, position, measured_speed, known_cells):
        if not np.isfinite(measured_speed) or not np.isfinite(position).all():
            self.samples.clear()
            return False
        self.travel += max(0.0, float(measured_speed))
        if step % 6:
            return False
        self.samples.append((step, np.array(position).copy(), self.travel, known_cells))
        if len(self.samples) < 21 or step < self.cooldown:
            return False
        first = self.samples[0]
        xy = np.array([s[1] for s in self.samples])
        length = self.travel - first[2]
        net = np.linalg.norm(xy[-1] - xy[0])
        extent = np.linalg.norm(np.ptp(xy, axis=0))
        loop = length > 260 and net < 80 and (extent < 220) and (known_cells - first[3] < 8)
        if loop:
            self.cooldown = step + 180
            self.samples.clear()
        return bool(loop)

class _M11_ExplorationGrid(_M09_ExplorationGrid):

    def __init__(self, size):
        super().__init__(size)
        self.step = 0
        self.avoid_regions = []

    def avoid_region(self, position, until):
        self.avoid_regions.append((np.array(position).copy(), until))
        self.avoid_regions = self.avoid_regions[-8:]

    def plan(self, position, home, identifier, number, claims, blocked_goals=(), rendezvous=False):
        self.avoid_regions = [(p, t) for (p, t) in self.avoid_regions if t > self.step]
        excluded = set(blocked_goals)
        if not rendezvous and self.avoid_regions:
            (yy, xx) = np.indices(self.shape)
            wx = (xx + 0.5) * self.CELL - self.width / 2
            wy = self.height / 2 - (yy + 0.5) * self.CELL
            mask = np.zeros(self.shape, bool)
            for (point, until) in self.avoid_regions:
                mask |= (wx - point[0]) ** 2 + (wy - point[1]) ** 2 < 140 ** 2
            excluded.update(map(tuple, np.argwhere(mask)))
        route = super().plan(position, home, identifier, number, claims, excluded, rendezvous)
        if not route and excluded != set(blocked_goals):
            route = super().plan(position, home, identifier, number, claims, blocked_goals, rendezvous)
        return route

def _M11_safe_velocity(distances, angles, target_angle, velocity, previous_angle=None, target_distance=float('inf')):
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
    headings = np.linspace(-math.pi, math.pi, 48, endpoint=False)
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

# ============================================================================
# Integrated source: my_drone_red_v4_loops
# ============================================================================

class _M12_MyDroneRedV4Loops(_M10_MyDroneRedV4Refined):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._explore = _M11_ExplorationGrid(self.size_area)
        self._loop_watch = _M11_LoopWatch()
        self._holding = False

    def _try_hold(self, distances, dynamic, command):
        peers = [p[1] for (i, p) in self._peers.items() if i != int(self.identifier or 0) and self._steps - p[0] < 20]
        separation = min((np.linalg.norm(self._position - p) for p in peers), default=float('inf'))
        static = distances[np.isfinite(distances) & (distances > 0) & ~dynamic]
        clearance = float(np.min(static)) if len(static) >= 12 else 0.0
        home_distance = np.linalg.norm(self._position - self._home)
        synced = bool(self._radio_maps and len(self._radio_maps.snapshots) >= self._n)
        eligible = self._gathering and self.carried_bombs_count() == 0 and synced
        if self._holding and (not eligible or home_distance > 180 or separation < 40 or (clearance < 23)):
            self._holding = False
            self._event('hold_end')
        if not self._holding and eligible and (home_distance < 120) and (separation > 65) and (clearance > 38):
            self._holding = True
            self._event('hold_start', position=list(map(float, self._position)))
        if not self._holding:
            return False
        self._route = []
        self._bomb_target = None
        self._slow_ticks = 0
        self._motion_anchor = self._position.copy()
        if self._steps % 30 == 0:
            self._visited_snapshot = np.packbits(self._explore.shared_visited | (self._explore.visits > 0)).tobytes()
            self._radio_maps.publish(int(self.identifier or 0), self._steps, self._map.mask())
        move = np.clip(-0.7 * self._velocity, -1, 1)
        command.update(forward=float(move[0]), lateral=float(move[1]), rotation=0.0)
        return True

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
        steering_d = d.copy()
        steering_d[bomb_rays] = 80
        (move, self._last_motion_angle) = _M11_safe_velocity(steering_d, angles, _M01_wrap_angle(target_angle), self._velocity, self._last_motion_angle, target_distance)
        command.update(forward=float(move[0]), lateral=float(move[1]), rotation=0.0)
        return command

# ============================================================================
# Integrated source: red_v4_progress_navigation
# ============================================================================

def _M13_obstructed_route(grid, position, route):
    if not route:
        return False
    walls = cv2.dilate((grid.obstacle >= 2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    current = grid.cell(position)
    cells = route[:9] + [route[-1]]
    return any((walls[cell] and (cell != current or cell == route[-1]) for cell in cells))

class _M13_WaypointProgress:

    def __init__(self):
        self.target = None
        self.best = float('inf')
        self.last_advance = 0

    def reset(self):
        self.target = None

    def update(self, step, position, target):
        target = np.asarray(target)
        distance = float(np.linalg.norm(target - position))
        if self.target is None or np.linalg.norm(target - self.target) > 1:
            self.target = target.copy()
            self.best = distance
            self.last_advance = step
        elif distance < self.best - 6:
            self.best = distance
            self.last_advance = step
        return step - self.last_advance >= 36

# ============================================================================
# Integrated source: my_drone_red_v4_progress
# ============================================================================

class _M14_MyDroneRedV4Progress(_M12_MyDroneRedV4Loops):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._waypoint_progress = _M13_WaypointProgress()

    def _release_stale_route(self, reason):
        if self._route:
            self._event('route_released', reason=reason, waypoint=self._explore.point(self._route[0]).tolist(), destination=self._explore.point(self._route[-1]).tolist(), position=self._position.tolist())
            self._explore.retire_goal(self._route[-1])
            self._blocked_goals[self._route[-1]] = self._steps + 240
        self._route = []
        self._bomb_target = None
        self._last_bomb_plan = self._steps + 30
        self._last_route_step = self._steps - 12
        self._waypoint_progress.reset()

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
        (move, self._last_motion_angle) = _M11_safe_velocity(steering_d, angles, _M01_wrap_angle(target_angle), self._velocity, self._last_motion_angle, target_distance)
        command.update(forward=float(move[0]), lateral=float(move[1]), rotation=0.0)
        return command

# ============================================================================
# Integrated source: my_drone_red_v5
# ============================================================================

def _M15_ready_to_submit(identifier, number_drones, step, max_steps, statuses, snapshots):
    """Leader-only early finish; stale/missing peers retain V4 fallback deadline."""
    if identifier != 0 or not max_steps or max_steps <= 0 or (number_drones < 1) or (step < 0.72 * max_steps):
        return False
    oldest = max(0.7 * max_steps, step - 120)
    for source in range(number_drones):
        status = statuses.get(source)
        snapshot = snapshots.get(source)
        if status is None or snapshot is None:
            return False
        if status[1] != 0 or status[0] < oldest or snapshot[1] < oldest:
            return False
    return True

class _M15_MyDroneRedV5(_M14_MyDroneRedV4Progress):

    def control(self):
        command = super().control()
        if not self._exploration_submitted and self._radio_maps:
            if _M15_ready_to_submit(int(self.identifier or 0), self._n, self.elapsed_timestep, getattr(self._misc_data, 'max_timestep_limit', None), self._status, self._radio_maps.snapshots):
                self._event('early_finish_ready', sources=sorted(self._radio_maps.snapshots), snapshot_steps={str(k): v[1] for (k, v) in self._radio_maps.snapshots.items()}, inventory_status={str(k): list(v) for (k, v) in self._status.items()})
                self._submit()
        return command

# ============================================================================
# Integrated source: red_advanced_placement
# ============================================================================

def _M16_reachable(grid, position, limit):
    walls = cv2.dilate((grid.obstacle >= 2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    allowed = grid.free & ~walls.astype(bool)
    start = grid.cell(position)
    if grid.obstacle[start] >= 2:
        return ({}, {})
    allowed[start] = True
    distances = {start: 0}
    parents = {start: None}
    queue = deque([start])
    while queue:
        (r, c) = queue.popleft()
        if distances[r, c] >= limit:
            continue
        for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1] and allowed[cell] and (cell not in distances):
                distances[cell] = distances[r, c] + 1
                parents[cell] = (r, c)
                queue.append(cell)
    return (distances, parents)

def _M16_path_to(parents, cell):
    path = [cell]
    while parents[path[-1]] is not None:
        path.append(parents[path[-1]])
    return path[::-1]

class _M16_CoverageField:

    def __init__(self, wall_mask, cell_size, size, placed, claims, radius):
        self.cell = cell_size
        (self.width, self.height) = size
        self.radius = radius
        mask = np.asarray(wall_mask, dtype=np.uint8)
        skeleton = np.zeros_like(mask)
        rest = mask.copy()
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for _ in range(16):
            if not rest.any():
                break
            eroded = cv2.erode(rest, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            opened = cv2.dilate(eroded, cross)
            skeleton |= rest & ~opened
            rest = eroded
        skeleton |= rest
        (yy, xx) = np.indices(mask.shape)
        self.x = (xx + 0.5) * cell_size - self.width / 2
        self.y = self.height / 2 - (yy + 0.5) * cell_size
        weight = skeleton.astype(np.float32)
        edge = (np.abs(self.x) > self.width / 2 - 15) | (np.abs(self.y) > self.height / 2 - 15)
        weight[edge] *= 0.5
        self.placed = [np.asarray(p) for p in placed]
        self.claims = [np.asarray(p) for p in claims]
        for p in self.placed:
            weight[self.disk(p)] = 0
        self.unclaimed = weight.copy()
        for p in self.claims:
            weight[self.disk(p)] *= 0.05
        self.weight = weight
        r = int(np.ceil(radius / cell_size))
        (ky, kx) = np.ogrid[-r:r + 1, -r:r + 1]
        kernel = ((kx * cell_size) ** 2 + (ky * cell_size) ** 2 <= radius ** 2).astype(np.float32)
        self.gain = cv2.filter2D(weight, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    def disk(self, point):
        return (self.x - point[0]) ** 2 + (self.y - point[1]) ** 2 <= self.radius ** 2

    def value(self, point):
        return float(self.weight[self.disk(point)].sum())

    def sampled_value(self, point):
        row = int(np.clip((self.height / 2 - point[1]) / self.cell, 0, self.gain.shape[0] - 1))
        col = int(np.clip((point[0] + self.width / 2) / self.cell, 0, self.gain.shape[1] - 1))
        return max(0.0, float(self.gain[row, col]))

@dataclass
class _M16_PlacementPlan:
    route: list
    target: np.ndarray
    gain: float

def _M16_choose_placement(grid, position, field):
    (distances, parents) = _M16_reachable(grid, position, 50)
    candidates = []
    for (cell, steps) in distances.items():
        point = grid.point(cell)
        if any((np.linalg.norm(point - p) < 65 for p in field.placed)):
            continue
        value = field.sampled_value(point)
        if value > 0:
            candidates.append((value / (1 + 0.015 * steps), cell, steps))
    candidates.sort(reverse=True)
    best = None
    for (_, cell, steps) in candidates[:6]:
        center = grid.point(cell)
        for dx in (-8.0, 0.0, 8.0):
            for dy in (-8.0, 0.0, 8.0):
                point = center + np.array([dx, dy])
                if abs(point[0]) >= grid.width / 2 or abs(point[1]) >= grid.height / 2:
                    continue
                if not grid.line_clear(center, point):
                    continue
                if any((np.linalg.norm(point - p) < 65 for p in field.placed)):
                    continue
                value = field.value(point)
                score = value / (1 + 0.015 * (steps + np.linalg.norm(point - center) / grid.CELL))
                if value > 0 and (best is None or score > best[0]):
                    best = (score, cell, point, value)
    if best is None:
        return None
    return _M16_PlacementPlan(_M16_path_to(parents, best[1]), best[2], best[3])

# ============================================================================
# Integrated source: red_advanced2_placement
# ============================================================================

def _M17_transport_costs(grid, center):
    """Distances to an observed center approach; unknown is never a corridor."""
    if center is None:
        return {}
    walls = cv2.dilate((grid.obstacle >= 2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    allowed = grid.free & ~walls.astype(bool)
    cells = list(map(tuple, np.argwhere(allowed)))
    if not cells:
        return {}
    start = min(cells, key=lambda c: np.linalg.norm(grid.point(c) - center))
    approach = float(np.linalg.norm(grid.point(start) - center))
    if approach > 100:
        return {}
    distances = {start: approach}
    queue = deque([start])
    while queue:
        (r, c) = queue.popleft()
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1] and allowed[nxt] and (nxt not in distances):
                distances[nxt] = distances[r, c] + grid.CELL
                queue.append(nxt)
    return distances

def _M17_difficulty(point, route_distance, center, size):
    """Bounded proxy, not a prediction of the official difficulty score."""
    if center is None or route_distance is None:
        return 0.0
    direct = float(np.linalg.norm(point - center))
    scale = max(1.0, min(size))
    return float(0.7 * min(1.0, route_distance / scale) + 0.3 * min(1.0, max(0.0, route_distance - direct) / (0.5 * scale)))

def _M17_choose_placement(grid, position, field, center=None, event=None):
    baseline = _M16_choose_placement(grid, position, field)
    if baseline is None or center is None:
        return baseline
    (distances, parents) = _M16_reachable(grid, position, 50)
    transport = _M17_transport_costs(grid, center)
    if not transport:
        return baseline
    base_steps = max(0, len(baseline.route) - 1)
    base_value = baseline.gain / (1 + 0.015 * base_steps)
    base_diff = _M17_difficulty(baseline.target, transport.get(grid.cell(baseline.target)), center, (grid.width, grid.height))
    best_score = base_value * (1 + 0.35 * base_diff)
    best = baseline
    best_diff = base_diff
    coarse = []
    for (cell, steps) in distances.items():
        point = grid.point(cell)
        gain = field.sampled_value(point)
        if gain < 0.9 * baseline.gain or cell not in transport:
            continue
        diff = _M17_difficulty(point, transport[cell], center, (grid.width, grid.height))
        coarse.append((gain / (1 + 0.015 * steps) * (1 + 0.35 * diff), cell, steps))
    coarse.sort(reverse=True)
    selected = []
    for (_, cell, steps) in coarse:
        point = grid.point(cell)
        if any((np.linalg.norm(point - grid.point(c)) < 0.5 * field.radius for c in selected)):
            continue
        selected.append(cell)
        for dx in (-8.0, 0.0, 8.0):
            for dy in (-8.0, 0.0, 8.0):
                target = point + np.array([dx, dy])
                if abs(target[0]) >= grid.width / 2 or abs(target[1]) >= grid.height / 2:
                    continue
                if not grid.line_clear(point, target):
                    continue
                if any((np.linalg.norm(target - p) < 65 for p in field.placed)):
                    continue
                gain = field.value(target)
                value = gain / (1 + 0.015 * (steps + np.linalg.norm(target - point) / grid.CELL))
                if gain < 0.9 * baseline.gain or value < 0.9 * base_value:
                    continue
                diff = _M17_difficulty(target, transport[cell], center, (grid.width, grid.height))
                score = value * (1 + 0.35 * diff)
                if score > best_score:
                    (best_score, best_diff) = (score, diff)
                    best = _M16_PlacementPlan(_M16_path_to(parents, cell), target, gain)
        if len(selected) >= 12:
            break
    if event is not None:
        event('opponent_placement', changed=best is not baseline, baseline_gain=baseline.gain, selected_gain=best.gain, baseline_difficulty=base_diff, selected_difficulty=best_diff, observed_center=center.tolist())
    return best

# ============================================================================
# Integrated source: red_advanced_geometry
# ============================================================================

def _M18_complete_boundary(mask, free=None):
    result = np.asarray(mask, bool).copy()
    for (view, clear) in ((result, free), (result.T, None if free is None else free.T)):
        for reverse in (False, True):
            side = view[::-1] if reverse else view
            known_free = None if clear is None else clear[::-1] if reverse else clear
            support = side[:3].any(axis=0)
            if support.mean() < 0.65 or support.sum() < 40:
                continue
            if known_free is not None and known_free[:2].mean() > 0.02:
                continue
            side[:2] = True
    return result

def _M18_solid_hypotheses(mask, free=None, nonwall=None):
    """Compact enclosed rectangles may be solid: scoring estimate ONLY.

    Unknown room interiors cannot be proved solid from a contour; these masks
    must never be submitted as observations or used to block navigation.
    """
    closed = cv2.morphologyEx(np.asarray(mask, np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    (count, labels, stats, _) = cv2.connectedComponentsWithStats(1 - closed, connectivity=4)
    area = np.zeros_like(closed, dtype=bool)
    for i in range(1, count):
        (x, y, w, h, n) = stats[i]
        if x == 0 or y == 0 or x + w == closed.shape[1] or (y + h == closed.shape[0]):
            continue
        if min(w, h) < 5 or max(w, h) > 90 or max(w, h) / min(w, h) > 3 or (n < 0.88 * w * h):
            continue
        region = labels[y:y + h, x:x + w] == i
        if free is not None and np.any(free[y:y + h, x:x + w] & region):
            continue
        if nonwall is not None and np.any(nonwall[y:y + h, x:x + w] & region):
            continue
        area[y:y + h, x:x + w] |= region
    return area

class _M18_ObservationMapRefined(_M03_ObservationMapV2):

    def to_pixels(self, mask):
        return super().to_pixels(_M18_complete_boundary(mask, self.evidence < -1))

# ============================================================================
# Integrated source: red_advanced_survey
# ============================================================================

class _M19_SurveyGrid(_M11_ExplorationGrid):

    def __init__(self, size, event):
        super().__init__(size)
        self.event = event
        self.survey_enabled = False
        self.survey_count = 0
        self.survey_after = 0
        self.survey = None
        self.survey_history = []

    def unknown_near(self, cell):
        (r, c) = cell
        free = self.free[max(0, r - 3):r + 4, max(0, c - 3):c + 4]
        wall = self.obstacle[max(0, r - 3):r + 4, max(0, c - 3):c + 4] >= 2
        return int((~free & ~wall).sum())

    def plan(self, position, home, identifier, number, claims, blocked_goals=(), rendezvous=False):
        if not rendezvous and self.survey_enabled and (self.survey is None) and (self.survey_count < 2) and (self.step >= self.survey_after):
            wall = self.obstacle >= 2
            safety = cv2.dilate(wall.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            unknown = (~self.free & ~safety).astype(np.float32)
            frontier = cv2.filter2D(unknown, -1, np.ones((5, 5), np.float32))
            near_wall = cv2.dilate(wall.astype(np.uint8), np.ones((7, 7), np.uint8))
            (distances, parents) = _M16_reachable(self, position, 12)
            candidates = []
            for (cell, steps) in distances.items():
                if steps < 3 or cell in blocked_goals or (not near_wall[cell]) or (frontier[cell] < 3):
                    continue
                point = self.point(cell)
                if any((np.linalg.norm(point - p) < 160 for p in claims)):
                    continue
                if any((np.linalg.norm(point - p) < 140 for p in self.survey_history)):
                    continue
                if any((t > self.step and np.linalg.norm(point - p) < 140 for (p, t) in self.avoid_regions)):
                    continue
                if self.goal_visits[cell] >= 2:
                    continue
                candidates.append((float(frontier[cell]) - steps * 0.6 - self.visits[cell] * 0.5, cell))
            if candidates:
                (_, cell) = max(candidates)
                self.survey = dict(cell=cell, start=self.step, unknown=self.unknown_near(cell))
                self.survey_count += 1
                self.survey_after = self.step + 300
                self.survey_history.append(self.point(cell))
                self.event('survey_started', goal=self.point(cell).tolist(), budget=90)
                return _M16_path_to(parents, cell)
        return super().plan(position, home, identifier, number, claims, blocked_goals, rendezvous)

    def finish_survey(self, reason):
        task = self.survey
        if task is not None:
            self.event('survey_finished', reason=reason, duration=self.step - task['start'], newly_resolved=max(0, task['unknown'] - self.unknown_near(task['cell'])))
            self.retire_goal(task['cell'])
            self.survey = None

# ============================================================================
# Integrated source: my_drone_red_advanced2
# ============================================================================

class MyDroneRedAdvanced2(_M15_MyDroneRedV5):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._explore = _M19_SurveyGrid(self.size_area, self._event)
        self._advanced_field = None
        self._field_step = -100
        self._recheck_step = -100
        self._map = _M18_ObservationMapRefined(self.size_area)
        self._disposal_observed = None

    def _coverage_field(self, force=False):
        if force or self._advanced_field is None or self._steps - self._field_step >= 24:
            claims = [np.array(p) for (i, (seq, p)) in self._bomb_claims.items() if i < int(self.identifier or 0) and self._steps - seq < 90 and (p is not None)]
            walls = self._radio_maps.merged() | self._map.mask()
            self._advanced_field = _M16_CoverageField(walls, self._map.CELL, self.size_area, self._known_bombs(), claims, round(0.12 * min(self.size_area)))
            self._field_step = self._steps
        return self._advanced_field

    def _drop_placement(self, reason):
        self._event('placement_reconsidered', reason=reason)
        self._bomb_target = None
        self._route = []
        self._last_bomb_plan = self._steps

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
        command = super()._navigate_motion()
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

    def _deployment(self, command):
        confirming = self._attempt_step is not None or (self._last_inventory is not None and self.carried_bombs_count() < self._last_inventory)
        if not confirming and self._progress() < 0.68:
            if self._bomb_target is None or self._position is None:
                return
            if np.linalg.norm(self._position - self._bomb_target) > 28:
                return
            field = self._coverage_field(force=True)
            target_gain = field.value(self._bomb_target)
            actual_gain = field.value(self._position)
            if target_gain < max(1.0, 0.5 * self._bomb_gain):
                self._drop_placement('pre_release_conflict')
                return
            if actual_gain < 0.9 * target_gain or np.linalg.norm(self._velocity) > 1.2:
                return
            self._event('placement_checked', actual_gain=actual_gain, target_gain=target_gain, error=float(np.linalg.norm(self._position - self._bomb_target)))
        super()._deployment(command)

