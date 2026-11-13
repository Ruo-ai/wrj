"""
Swarm-Rescue 蓝队独立测试文件
============================
本文件包含完整蓝队控制器及导航算法，不依赖 red_release2.py、
my_drone_blue.py 或 blue_navigation.py；仍需要 Swarm-Rescue 仿真平台。

给测试者的使用说明：
1. 建议使用 Python 3.11，按原项目要求安装仿真依赖。
2. 备份测试项目原来的 src/swarm_rescue/solutions/my_drone_eval.py。
3. 将本文件复制到该目录，并重命名为 my_drone_eval.py。
4. 在测试项目根目录运行（将 RED_BOMBS.json 换成红队炸弹文件路径）：
   python -m swarm_rescue.launcher --competition \
       -c config/competition_rescue_eval_plan.yml --bombs-file RED_BOMBS.json
   如需隐藏窗口，在命令中加 --headless。
5. 此入口仅支持 rescue（蓝队）。请勿使用 place 或红蓝串联配置。

蓝队功能：多机协作搜索、雷达建图、通信共享、炸弹分配、路径规划、
局部避障、脱困、炸弹运输与处置、回合末返回出生位置。
所有状态存放于各无人机实例，通过平台通信交换，不读取真实地图坐标。
可设置环境变量 SWARM_BLUE_DEBUG=1 输出低频任务日志。

默认开发参考为 MyDroneBlueAdapted：量程边界适配与等价计算提速。
MyDroneBlue 保留未通过采用门槛的任务增强，须显式选择；本地结果不代表平台参考面板。
"""

import math
from collections import deque

import numpy as np
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR

LIDAR_RANGE = float(MAX_RANGE_LIDAR_SENSOR)
HIT_MARGIN = 4 * 2.5  # Current engine lidar sigma; maximum-range returns are not walls.


def steering_choice(distances, angles, desired, carrying=False):
    """Evaluate all local corridors together; only finite observed rays are used."""
    valid = np.isfinite(distances) & np.isfinite(angles) & (distances > 0)
    d, a = np.minimum(distances[valid], LIDAR_RANGE), angles[valid]
    if not len(d):
        return None
    candidates = np.linspace(-math.pi, math.pi, 49, endpoint=False)
    relative = (a[None, :] - candidates[:, None] + math.pi) % (2 * math.pi) - math.pi
    clearance = 43 if carrying else 31
    longitudinal = d[None, :] * np.cos(relative)
    corridor = (np.abs(relative) < .65) & (np.abs(d[None, :] * np.sin(relative)) < clearance)
    gaps = np.min(np.where(corridor, longitudinal, np.inf), axis=1)
    gaps[~np.isfinite(gaps)] = LIDAR_RANGE
    scores = -2.8 * np.abs((candidates - desired + math.pi) % (2 * math.pi) - math.pi)
    scores += np.minimum(gaps, 140) / 100 - 8 * (gaps < clearance)
    best = int(np.argmax(scores))
    return float(candidates[best]), float(gaps[best])


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def swept_steering(distances, angles, desired, carrying=False):
    """Local swept disk, with separate lateral footprint and stopping distance.

    The payload footprint is an operational margin, not a guarantee about a
    freely pivoting bomb. Narrow transit is slow; payload turns require more room.
    """
    valid = np.isfinite(distances) & np.isfinite(angles) & (distances > 0)
    d, a = np.minimum(distances[valid], LIDAR_RANGE), angles[valid]
    if not len(d):
        return None
    candidates = np.r_[np.linspace(-math.pi, math.pi, 25, endpoint=False), 0., wrap(desired)]
    relative = a[None, :] - candidates[:, None]
    longitudinal = d[None, :] * np.cos(relative)
    lateral = d[None, :] * np.sin(relative)
    radius = np.where(np.abs(candidates) < .22, 24., 32.) if carrying else np.full(len(candidates), 16.)
    corridor = (longitudinal >= 0) & (np.abs(lateral) < radius[:, None])
    contact = longitudinal - np.sqrt(np.maximum(0, radius[:, None]**2 - lateral**2))
    gaps = np.minimum(np.min(np.where(corridor, contact, np.inf), axis=1), LIDAR_RANGE-radius)
    scores = -2.8 * np.abs((candidates-desired+math.pi) % (2*math.pi)-math.pi)
    scores += np.minimum(np.maximum(gaps, 0), 140)/100 - 8*(gaps < 8)
    best = int(np.argmax(scores))
    near = (longitudinal[best] >= 0) & (longitudinal[best] < 65)
    tight = bool(np.any(near & (np.abs(lateral[best]) < (38 if carrying else 27)) & (d < 75)))
    return float(candidates[best]), float(gaps[best]), tight


def owner_for(point, agents):
    """Deterministic nearest free drone, with identifier as a tie breaker."""
    return min(agents, key=lambda i: (float(np.linalg.norm(np.asarray(agents[i]) - point)), i))


class RescueGrid:
    CELL = 16

    def __init__(self, size):
        self.width, self.height = size
        self.shape = (math.ceil(self.height / self.CELL), math.ceil(self.width / self.CELL))
        self.free = np.zeros(self.shape, bool)
        self.evidence = np.zeros(self.shape, np.float32)
        self.visits = np.zeros(self.shape, np.float32)
        self.remote = {}
        self._revision = 0
        self._layer_cache = None

    def cell(self, point):
        return (int(np.clip((self.height / 2 - point[1]) / self.CELL, 0, self.shape[0] - 1)),
                int(np.clip((point[0] + self.width / 2) / self.CELL, 0, self.shape[1] - 1)))

    def point(self, cell):
        r, c = cell
        return np.array([(c + .5) * self.CELL - self.width / 2,
                         self.height / 2 - (r + .5) * self.CELL])

    def observe(self, position, heading, distances, angles, dynamic):
        if distances is None or angles is None:
            return
        distances, angles = np.asarray(distances, float).ravel(), np.asarray(angles, float).ravel()
        if distances.shape != angles.shape or np.shape(dynamic) != distances.shape:
            return
        valid = np.isfinite(distances) & np.isfinite(angles) & (distances > 4)
        d, a = distances[valid], angles[valid] + heading
        excluded = np.asarray(dynamic, bool)[valid]
        directions = np.column_stack((np.cos(a), np.sin(a)))

        def indices(points):
            cols = np.floor((points[:, 0] + self.width / 2) / self.CELL).astype(int)
            rows = np.floor((self.height / 2 - points[:, 1]) / self.CELL).astype(int)
            good = (rows >= 0) & (rows < self.shape[0]) & (cols >= 0) & (cols < self.shape[1])
            return np.unique(rows[good] * self.shape[1] + cols[good])

        steps = np.arange(0, LIDAR_RANGE, 10)
        points = position + directions[:, None, :] * steps[None, :, None]
        cells = indices(points[steps[None, :] < np.minimum(d, LIDAR_RANGE)[:, None] - 20])
        self.free.ravel()[cells] = True
        self.evidence.ravel()[cells] = np.maximum(0, self.evidence.ravel()[cells] - .25)
        hits = (d < LIDAR_RANGE - HIT_MARGIN) & ~excluded
        cells = indices(position + directions[hits] * d[hits, None])
        self.evidence.ravel()[cells] = np.minimum(8, self.evidence.ravel()[cells] + 1)
        self.free[self.cell(position)] = True
        self.visits[self.cell(position)] += 1
        self._revision += 1

    def snapshot(self):
        return tuple(np.packbits(a.ravel()).tobytes() for a in
                     (self.free, self.evidence >= 2, self.visits > 0))

    def accept(self, source, step, payload):
        length = (math.prod(self.shape) + 7) // 8
        if not isinstance(payload, tuple) or len(payload) != 3:
            return
        if any(not isinstance(p, bytes) or len(p) != length for p in payload):
            return
        if source in self.remote and self.remote[source][0] >= step:
            return
        arrays = tuple(np.unpackbits(np.frombuffer(p, np.uint8))[:math.prod(self.shape)]
                       .reshape(self.shape).astype(bool) for p in payload)
        self.remote[source] = (step, arrays)
        self._revision += 1

    def layers(self):
        if self._layer_cache is not None and self._layer_cache[0] == self._revision:
            return self._layer_cache[1].copy(), self._layer_cache[2].copy()
        free, walls, seen = self.free.copy(), self.evidence >= 2, self.visits > 0
        for _, (other_free, other_walls, other_seen) in self.remote.values():
            free |= other_free
            # Recent local free-space evidence can clear stale remote obstacles.
            walls |= other_walls & ~(self.free & (self.evidence < .5))
            seen |= other_seen
        padded = np.pad(walls, 1)
        inflated = walls.copy()
        for dr, dc in ((0, 1), (2, 1), (1, 0), (1, 2)):
            inflated |= padded[dr:dr + self.shape[0], dc:dc + self.shape[1]]
        allowed = free & ~inflated
        self._layer_cache = (self._revision, allowed, seen)
        return allowed.copy(), seen.copy()

    def route(self, position, target=None, identifier=0, number=1, claims=()):
        allowed, seen = self.layers()
        start = self.cell(position)
        allowed[start] = True
        parents = {start: None}
        distance = {start: 0}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                other = r + dr, c + dc
                if (0 <= other[0] < self.shape[0] and 0 <= other[1] < self.shape[1]
                        and allowed[other] and other not in parents):
                    parents[other] = (r, c)
                    distance[other] = distance[r, c] + 1
                    queue.append(other)
        cells = list(parents)
        coords = np.asarray(cells)
        points = np.column_stack(((coords[:, 1] + .5) * self.CELL - self.width / 2,
                                  self.height / 2 - (coords[:, 0] + .5) * self.CELL))
        lengths = np.asarray([distance[c] for c in cells])
        if target is not None:
            # Use only reachable observed space, even when the goal is behind a wall.
            end = cells[int(np.argmin(np.linalg.norm(points - target, axis=1) + lengths * .15))]
        else:
            unknown = ~self.free & ~(self.evidence >= 2)
            for _, (remote_free, remote_walls, _) in self.remote.values():
                unknown &= ~remote_free & ~remote_walls
            padded = np.pad(unknown, 2)
            novelty = np.zeros(self.shape, int)
            for dr in range(5):
                for dc in range(5):
                    novelty += padded[dr:dr + self.shape[0], dc:dc + self.shape[1]]
            preferred = identifier * 2 * math.pi / max(1, number)

            indices = tuple(coords.T)
            delta = points - position
            values = 5 * novelty[indices] - .4 * lengths - 4 * self.visits[indices]
            values += 15 * (~seen[indices]) + 12 * np.cos(np.arctan2(delta[:, 1], delta[:, 0]) - preferred)
            for point in claims:
                values -= 65 * np.maximum(0, 1 - np.linalg.norm(points - point, axis=1) / 160)
            values[lengths < 3] = -1e6
            end = cells[int(np.argmax(values))]
        path = []
        while end is not None:
            path.append(self.point(end))
            end = parents[end]
        return list(reversed(path))[1:]

    def shortcut(self, position, route):
        """Look ahead along a route without cutting through walls or unknown."""
        if len(route) < 2:
            return route
        allowed, _ = self.layers()
        for index in range(min(12, len(route)) - 1, 0, -1):
            target = route[index]
            points = np.linspace(position, target, max(2, int(np.linalg.norm(target - position) / 6)))
            cols = np.clip(((points[:, 0] + self.width / 2) / self.CELL).astype(int), 0, self.shape[1] - 1)
            rows = np.clip(((self.height / 2 - points[:, 1]) / self.CELL).astype(int), 0, self.shape[0] - 1)
            if allowed[rows[1:], cols[1:]].all():
                return route[index:]
        return route

import math
import os

import numpy as np

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor


class MyDroneBlue(DroneAbstract):
    ENHANCED = True
    # Fixed-layout ablations: task changes are independent of swept steering.
    SWEPT_EMPTY = False
    SWEPT_PAYLOAD = False
    PEER_TTL = 90
    BOMB_TTL = 700

    def __init__(self, identifier=None, misc_data=None, **kwargs):
        super().__init__(identifier=identifier, misc_data=misc_data, **kwargs)
        self._grid = RescueGrid(self.size_area or (1250, 800))
        self._number = int(getattr(misc_data, 'number_drones', None) or 1)
        self._position = None
        self._heading = 0.
        self._home = None
        self._center = None
        self._center_step = -1
        self._center_hit = None
        self._peers = {}
        self._bombs = []  # [x, y, observation_step, available]
        self._target = None
        self._route = []
        self._route_step = -100
        self._mode = 'search'
        self._tick = 0
        self._snapshot = None
        self._anchor = None
        self._anchor_step = 0
        self._escape_until = 0
        self._escape_sign = 1
        self._was_carrying = False
        self._pickup = None
        self._delivery_ticks = 0
        self._task_started = 0
        self._deferred = []
        self._search_goal = None
        self._search_until = 0
        self._task_best_distance = float('inf')
        self._task_progress_step = 0
        self._debug = self.ENHANCED or os.environ.get('SWARM_BLUE_DEBUG') == '1'

    def define_message_for_all(self):
        if self._position is None:
            return None
        return ('blue-v1', int(self.identifier), self._tick, tuple(self._position),
                self._mode, tuple(self._target) if self._target is not None else None,
                tuple(tuple(b) for b in self._bombs),
                (self._center_step, tuple(self._center)) if self._center is not None else None,
                self._snapshot)

    def _record_bomb(self, point, step, available, create=True):
        for bomb in self._bombs:
            if np.linalg.norm(np.asarray(bomb[:2]) - point) < 32:
                if step >= bomb[2]:
                    if available:
                        bomb[:2] = [float(point[0]), float(point[1])]
                    bomb[2:] = [step, bool(available)]
                return
        if create:
            self._bombs.append([float(point[0]), float(point[1]), step, bool(available)])

    def _receive(self):
        for _, msg in self.communicator.received_messages:
            if not isinstance(msg, tuple) or len(msg) != 9 or msg[0] != 'blue-v1':
                continue
            _, source, step, position, mode, target, bombs, center, snapshot = msg
            if (type(source) is not int or not 0 <= source < self._number or source == self.identifier
                    or type(step) is not int or not 0 <= step <= self._tick):
                continue
            if self._tick - step > self.PEER_TTL:
                continue
            if np.shape(position) != (2,) or not np.isfinite(position).all():
                continue
            if (target is not None and (np.shape(target) != (2,) or not np.isfinite(target).all())):
                continue
            if mode not in ('search', 'collect', 'deliver', 'search_center', 'return'):
                continue
            if not isinstance(bombs, (list, tuple)) or len(bombs) > 64:
                continue
            previous = self._peers.get(source)
            if previous and previous[0] >= step:
                continue
            self._peers[source] = (step, np.asarray(position), mode, target)
            for bomb in bombs:
                if (len(bomb) == 4 and type(bomb[2]) is int and 0 <= bomb[2] <= self._tick
                        and np.isfinite(bomb[:2]).all()):
                    self._record_bomb(np.asarray(bomb[:2]), bomb[2], bomb[3])
            if center is not None and self._center is None:
                stamp, point = center
                if np.shape(point) == (2,) and np.isfinite(point).all() and stamp <= self._tick:
                    self._center_step, self._center = stamp, np.asarray(point)
            if snapshot is not None:
                stamp, payload = snapshot
                if type(stamp) is int and 0 <= stamp <= step:
                    self._grid.accept(source, stamp, payload)
        self._peers = {i: p for i, p in self._peers.items() if self._tick - p[0] <= self.PEER_TTL}
        self._bombs = [b for b in self._bombs if self._tick - b[2] <= self.BOMB_TTL]

    def _pose(self):
        gps, compass = self.measured_gps_position(), self.measured_compass_angle()
        good_gps = gps is not None and np.shape(gps) == (2,) and np.isfinite(gps).all()
        good_heading = compass is not None and np.isfinite(compass)
        if self._position is None:
            if not good_gps or not good_heading:
                return False
            self._position = np.asarray(gps, float).copy()
            self._heading = float(compass)
            self._home = self._position.copy()
        else:
            odo = self.odometer_values()
            if odo is not None and np.shape(odo) == (3,) and np.isfinite(odo).all():
                distance, alpha, rotation = odo
                angle = self._heading + alpha
                self._position += distance * np.array([math.cos(angle), math.sin(angle)])
                self._heading = wrap(self._heading + rotation)
            if good_gps:
                self._position += .18 * (np.asarray(gps) - self._position)
            if good_heading:
                self._heading = wrap(self._heading + .25 * wrap(compass - self._heading))
        return True

    def _sense(self, distances, angles):
        hits = []
        self._center_hit = None
        dynamic = np.zeros(len(distances), bool)
        entity = DroneSemanticSensor.TypeEntity
        for obj in self.semantic_values() or []:
            if not np.isfinite([obj.distance, obj.angle]).all() or obj.distance <= 0:
                continue
            bearing = self._heading + obj.angle
            point = self._position + obj.distance * np.array([math.cos(bearing), math.sin(bearing)])
            if obj.entity_type in (entity.BOMB, entity.DRONE, entity.DISPOSAL_CENTER):
                # Exclude identified objects from the persistent wall map only.
                dynamic |= ((np.abs((angles - obj.angle + math.pi) % (2 * math.pi) - math.pi)
                             < math.atan2(24, max(1, obj.distance)))
                            & (np.abs(distances - obj.distance) < 35))
            if obj.entity_type == entity.BOMB:
                # A bomb being transported is not a new search target at every
                # observed position. Retire nearby known reports without adding
                # a trail of tombstones that floods the team's radio traffic.
                self._record_bomb(point, self._tick, not obj.grasped, create=not obj.grasped)
                if not obj.grasped:
                    hits.append((obj, point))
            elif obj.entity_type == entity.DISPOSAL_CENTER:
                if self._center_hit is None or obj.distance < self._center_hit.distance:
                    self._center_hit = obj
                if self._center is None:
                    self._center = point
                else:
                    self._center += .15 * (point - self._center)
                self._center_step = self._tick
        if self._tick % 3 == 0:
            self._grid.observe(self._position, self._heading, distances, angles, dynamic)
        if self._tick % 18 == int(self.identifier) % 18 or self._snapshot is None:
            self._snapshot = (self._tick, self._grid.snapshot())
        return hits

    def _choose_bomb(self):
        agents = {int(self.identifier): self._position}
        agents.update({i: p[1] for i, p in self._peers.items() if p[2] in ('search', 'collect')})
        candidates = []
        for x, y, stamp, available in self._bombs:
            point = np.array([x, y])
            if not available or self._tick - stamp > self.BOMB_TTL:
                continue
            if any(until > self._tick and np.linalg.norm(point - p) < 40
                   for p, until in self._deferred):
                continue
            if self.ENHANCED:
                # A peer committed to a different bomb is not a free bidder.
                bidders = {int(self.identifier): self._position}
                committed = {}
                for i, p in self._peers.items():
                    if p[2] == 'search':
                        bidders[i] = p[1]
                    elif p[2] == 'collect' and p[3] is not None and np.linalg.norm(point - p[3]) < 40:
                        bidders[i] = p[1]
                        committed[i] = p[1]
                if self._mode == 'collect' and self._target is not None and np.linalg.norm(point - self._target) < 40:
                    committed[int(self.identifier)] = self._position
                if owner_for(point, committed or bidders) != int(self.identifier):
                    continue
            elif owner_for(point, agents) != int(self.identifier):
                continue
            cost = np.linalg.norm(point - self._position)
            if self._target is not None and self._mode == 'collect' and np.linalg.norm(point - self._target) < 35:
                cost -= 45  # Retain a task through small localization changes.
            candidates.append((cost, x, y))
        return np.array(min(candidates)[1:]) if candidates else None

    def _navigate(self, goal, distances, angles, carrying=False):
        cmd = dict(forward=0., lateral=0., rotation=0., grasper=int(carrying))
        claims = [np.asarray(p[3]) for p in self._peers.values() if p[3] is not None and p[2] == 'search']
        if self._tick - self._route_step >= 24:
            route_goal = goal
            if self.ENHANCED and goal is None:
                if (self._search_goal is not None and self._tick < self._search_until
                        and np.linalg.norm(self._search_goal - self._position) > 36):
                    route_goal = self._search_goal
                else:
                    self._search_goal = None
            self._route = self._grid.route(self._position, route_goal, int(self.identifier), self._number, claims)
            if self.ENHANCED and goal is None and self._search_goal is None and self._route:
                self._search_goal = self._route[-1].copy()
                self._search_until = self._tick + 180
            if self.ENHANCED and goal is None and not self._route:
                self._search_goal = None
            self._route_step = self._tick
        while self._route and np.linalg.norm(self._route[0] - self._position) < 20:
            self._route.pop(0)
        self._route = self._grid.shortcut(self._position, self._route)
        if goal is None:
            self._target = self._route[-1].copy() if self._route else None
        waypoint = self._route[0] if self._route else goal
        if waypoint is None:
            desired = wrap(int(self.identifier) * 2.399 + self._tick / 100 - self._heading)
        else:
            delta = waypoint - self._position
            desired = wrap(math.atan2(delta[1], delta[0]) - self._heading)
        valid = np.isfinite(distances) & np.isfinite(angles) & (distances > 0)
        d, a = distances[valid], angles[valid]
        if not len(d):
            return cmd  # No blind translation when the lidar is disabled.
        if self.ENHANCED and (self.SWEPT_PAYLOAD if carrying else self.SWEPT_EMPTY):
            angle, gap, tight = swept_steering(distances, angles, desired, carrying)
            clearance = 8  # Swept gap already subtracts the lateral footprint.
        else:
            clearance = 43 if carrying else 31
            angle, gap = steering_choice(distances, angles, desired, carrying)
            tight = False
        cmd['rotation'] = float(np.clip(angle * 1.5, -.85, .85))
        speed = (.62 if carrying else .72) * np.clip((gap - clearance) / 65, 0, 1)
        if tight:
            speed = min(speed, .25 if carrying else .35)
        if abs(angle) > 1.2:
            speed = 0.  # Turn first; do not translate along an unchecked side corridor.
        cmd['forward'] = float(speed * math.cos(angle))
        cmd['lateral'] = float(speed * math.sin(angle))
        if self._debug and self._tick % 240 == int(self.identifier):
            print('BLUE_NAV', self._tick, self.identifier, 'carrying', carrying,
                  'route_points', len(self._route), 'desired', round(desired, 3),
                  'angle', round(angle, 3), 'gap', round(gap, 1),
                  'speed', round(float(speed), 3), flush=True)
        # Detect persistent lack of translation; never release a carried bomb here.
        if self._anchor is None or np.linalg.norm(self._position - self._anchor) > 24:
            self._anchor, self._anchor_step = self._position.copy(), self._tick
        elif self._tick - self._anchor_step > 85:
            self._escape_until = self._tick + 24
            self._escape_sign = 1 if int(self.identifier) % 2 == 0 else -1
            self._anchor_step = self._tick
            self._route = []
            self._search_goal = None
            self._grid.visits[self._grid.cell(self._position)] += 30
        if self._tick < self._escape_until:
            rear = d[np.abs(a) > 2.5]
            cmd.update(forward=-.3 if len(rear) and rear.min() > 55 else 0.,
                       lateral=0., rotation=.65 * self._escape_sign)
        return cmd

    def control(self):
        self._tick = int(self.elapsed_timestep)
        stop = dict(forward=0., lateral=0., rotation=0., grasper=int(bool(self.grasped_bombs())))
        self._receive()
        if not self._pose():
            return stop
        raw = self.lidar_values()
        distances = np.asarray(raw if raw is not None else [], float).ravel()
        angles = np.asarray(self.lidar_rays_angles(), float).ravel()[:len(distances)]
        if len(angles) != len(distances):
            return stop
        hits = self._sense(distances, angles)
        carrying = bool(self.grasped_bombs())
        previous_mode = self._mode
        if carrying and not self._was_carrying:
            self._pickup = self._target.copy() if self._target is not None else self._position.copy()
            self._delivery_ticks = 0
        if self._pickup is not None and (carrying or self._was_carrying):
            self._record_bomb(self._pickup, self._tick, False)
        self._was_carrying = carrying
        if carrying:
            self._mode = 'deliver' if self._center is not None else 'search_center'
            goal = self._center
        else:
            self._pickup = None
            self._delivery_ticks = 0
            goal = self._choose_bomb()
            self._mode = 'collect' if goal is not None else 'search'
            limits = getattr(self._misc_data, 'max_timestep_limit', None)
            wall_limit = getattr(self._misc_data, 'max_walltime_limit', None)
            if ((self.ENHANCED and self.drone_health <= 8)
                    or (limits and self._tick > .92 * limits)
                    or (wall_limit and self.elapsed_walltime > .92 * wall_limit)):
                self._mode, goal = 'return', self._home
        changed = (previous_mode != self._mode or (goal is not None and
                   (self._target is None or np.linalg.norm(goal - self._target) > 40)))
        if changed:
            self._route = []
            self._route_step = -100
            self._task_started = self._tick
            self._task_best_distance = float('inf')
            self._task_progress_step = self._tick
            self._search_goal = None
        if goal is not None:
            self._target = np.asarray(goal).copy()
        if self._debug and self._tick % 240 == int(self.identifier):
            print('BLUE_STATUS', self._tick, self.identifier, self._mode,
                  'position', self._position.round(1).tolist(),
                  'target', self._target.round(1).tolist() if self._target is not None else None,
                  'carrying', carrying, 'known_bombs', len(self._bombs), flush=True)
        if self.ENHANCED and self._mode == 'collect':
            remaining = float(np.linalg.norm(goal - self._position))
            if remaining < self._task_best_distance - 12:
                self._task_best_distance, self._task_progress_step = remaining, self._tick
        expired = (self._tick - self._task_progress_step > 180 or self._tick - self._task_started > 650) if self.ENHANCED else self._tick - self._task_started > 260
        if self._mode == 'collect' and expired:
            # Leave a difficult corner to another approach rather than spend
            # the remaining round repeatedly pushing into the same wall.
            self._deferred = [(p, t) for p, t in self._deferred if t > self._tick]
            self._deferred.append((goal.copy(), self._tick + 180))
            self._mode, self._target, self._route = 'search', None, []
            goal = None
        # A visible assigned bomb gets precise bearing-based final approach.
        if self._mode == 'collect':
            visible = [(obj, point) for obj, point in hits if np.linalg.norm(point - goal) < 40]
            if visible:
                obj, _ = min(visible, key=lambda h: h[0].distance)
                if obj.distance < 85:
                    cmd = self._navigate(goal, distances, angles)
                    cmd['grasper'] = 1
                    if abs(obj.angle) > .18:
                        cmd.update(forward=0., lateral=0., rotation=float(np.clip(obj.angle * 2, -.7, .7)))
                    else:
                        cone = np.abs(angles - obj.angle) < .35
                        blockers = distances[cone & np.isfinite(distances) & (distances > 0)]
                        if not len(blockers) or blockers.min() >= obj.distance - 12:
                            cmd.update(forward=.3, lateral=0., rotation=float(np.clip(obj.angle, -.3, .3)))
                    return cmd
            elif np.linalg.norm(goal - self._position) < 45:
                # Reached a stale report: do not orbit an already removed bomb.
                self._record_bomb(goal, self._tick, False)
                self._target, self._route = None, []
                goal, self._mode = None, 'search'
        if self._mode == 'return' and np.linalg.norm(goal - self._position) < 25:
            return stop
        cmd = self._navigate(goal, distances, angles, carrying)
        if self._mode == 'collect':
            cmd['grasper'] = 1
        if carrying and self._center is not None:
            hit = self._center_hit
            distance = hit.distance if hit is not None else float('inf')
            if hit is not None and 18 <= distance < 105:
                cmd.update(forward=0., lateral=0.,
                           rotation=float(np.clip(hit.angle * 1.5, -.65, .65)))
                cone = np.abs((angles - hit.angle + math.pi) % (2 * math.pi) - math.pi) < .3
                blockers = distances[cone & np.isfinite(distances) & (distances > 0)]
                if len(blockers) and blockers.min() >= distance - 12:
                    cmd['forward'] = .3 * math.cos(hit.angle)
                    cmd['lateral'] = .3 * math.sin(hit.angle)
            if distance < 18:
                # The disposal center is physical. Turn the held bomb into it;
                # only release near the observed center, never during recovery.
                self._delivery_ticks += 1
                cmd.update(forward=0., lateral=0., rotation=.65 * self._escape_sign)
                if self._delivery_ticks >= 28:
                    cmd['grasper'] = 0
            else:
                self._delivery_ticks = 0
        return cmd


class MyDroneBlueAdapted(MyDroneBlue):
    """Ablation opponent: range correction and equivalent computation only."""
    ENHANCED = False

# Enhancement trials did not pass the fixed-layout adoption gate. Keep the
# adapted development reference; this does not certify all its runs as valid.
REFERENCE_BLUE_CLASS = MyDroneBlueAdapted
# Compatible with launchers expecting the older class-style entry point.
MyDroneEval = REFERENCE_BLUE_CLASS


def drone_class_for_mode(mode):
    """Current Swarm-Rescue launcher entry point; blue team only."""
    from swarm_rescue.simulation.reporting.team_mode import TeamMode
    if TeamMode.from_string(mode) != TeamMode.RESCUE:
        raise ValueError('This is a blue-only submission. Use team_mode: rescue.')
    return REFERENCE_BLUE_CLASS
