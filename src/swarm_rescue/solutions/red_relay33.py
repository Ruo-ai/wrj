"""Relay-only routing over legal known free space with fresh hull checks.

The old inflated coarse route is preferred. Its connectivity failure may be
diagnosed by a non-inflated known-free route, but that route is only a set of
waypoint suggestions: current lidar hull clearance and a stopping sweep gate
every movement. No navigation or submission evidence is mutated here.
"""
from collections import Counter, deque
import math

import cv2
import numpy as np

from swarm_rescue.solutions.my_drone_red_rn7 import rotate
from swarm_rescue.solutions.red_return30 import ReturnLocal30


class RelayNavigator33:
    REPLAN_STEPS = 18
    REPAIR_STEPS = 12
    LOOKAHEAD = 130.0
    ARRIVAL_RADIUS = 30.0
    MAX_SPEED = 3.2

    def __init__(self, identifier=0, local_factory=None):
        self.identifier = int(identifier)
        self.local_factory = ReturnLocal30 if local_factory is None else local_factory
        self.stats = Counter()
        self.reason = 'inactive'
        self.target = None
        self.waypoint = None
        self.route = []
        self.arrived = False
        self.blocked_ticks = 0
        self.planning_mode = None
        self.inflated_target_reachable = False
        self.target_reachable = False
        self.proxy_distance = None
        self.reachable_cells = 0
        self.tick = -1
        self._plan_tick = -10**9
        self._repair_tick = -10**9
        self._repair = []
        self._heading = None
        self._heading_tick = None
        self._velocity = np.zeros(2)
        self._last_local = None

    def reset(self):
        self.target = None
        self.route = []
        self._repair = []
        self.arrived = False
        self.blocked_ticks = 0
        self._plan_tick = -10**9
        self._repair_tick = -10**9
        self.reason = 'reset'

    def _brake(self, reason, blocked=False):
        self.reason = reason
        self.stats[reason] += 1
        if blocked:
            self.blocked_ticks += 1
        force = np.clip(-.65*self._velocity, -1., 1.)
        return dict(forward=float(force[0]), lateral=float(force[1]), rotation=0.)

    @staticmethod
    def _search(grid, position, target, inflated):
        wall = np.asarray(grid.obstacle) >= 2
        if inflated:
            wall = cv2.dilate(wall.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))).astype(bool)
        allowed = np.asarray(grid.free, bool) & ~wall
        start = tuple(grid.cell(position))
        destination = tuple(grid.cell(target))
        allowed[start] = True  # Own centre; execution still needs a fresh hull.
        queue = deque([start])
        distance = {start: 0}
        parents = {}
        h, w = allowed.shape
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other = r+dr, c+dc
                if (0 <= other[0] < h and 0 <= other[1] < w
                        and allowed[other] and other not in distance):
                    distance[other] = distance[(r, c)]+1
                    parents[other] = (r, c)
                    queue.append(other)
        center = np.asarray(grid.point(destination), float)
        # grid.cell clips coordinates at the canvas edge. A clipped index is
        # not evidence that a target outside the represented area is known.
        target_inside = bool(np.all(np.abs(center-target) <= float(grid.CELL)/2+1e-8))
        exact = target_inside and destination in distance and bool(grid.free[destination])
        if exact:
            goal = destination
        else:
            goal = min(distance, key=lambda cell: (
                float(np.linalg.norm(np.asarray(grid.point(cell))-target)),
                distance[cell], cell))
        path = [goal]
        while path[-1] != start:
            path.append(parents[path[-1]])
        path.reverse()
        proxy_distance = float(np.linalg.norm(np.asarray(grid.point(goal))-target))
        return dict(cells=path, exact=exact, count=len(distance), proxy_distance=proxy_distance)

    def _plan(self, grid, position, target):
        inflated = self._search(grid, position, target, True)
        selected = inflated
        self.inflated_target_reachable = inflated['exact']
        self.planning_mode = 'inflated_known_free'
        self.stats['inflated_plans'] += 1
        if not inflated['exact']:
            relaxed = self._search(grid, position, target, False)
            self.stats['unexpanded_connectivity_probes'] += 1
            if relaxed['exact'] or relaxed['proxy_distance']+1e-6 < inflated['proxy_distance']:
                selected = relaxed
                self.planning_mode = 'unexpanded_known_free_fresh_hull_required'
                self.stats['unexpanded_routes_selected'] += 1
        self.route = [np.asarray(grid.point(cell), float).copy() for cell in selected['cells'][1:]]
        if selected['exact']:
            # Actual point, not an implicit snap to a possibly blocked centre.
            self.route.append(np.asarray(target, float).copy())
        self.target_reachable = bool(selected['exact'])
        self.proxy_distance = selected['proxy_distance']
        self.reachable_cells = selected['count']
        self._plan_tick = self.tick

    def command(self, position, heading, velocity, lidar, angles, semantic, grid, target, tick):
        """Return body-frame force; velocity must be legal measured body velocity."""
        self.tick = int(tick)
        valid_velocity = velocity is not None and np.shape(velocity) == (2,) and np.isfinite(velocity).all()
        self._velocity = np.asarray(velocity, float).copy() if valid_velocity else np.zeros(2)
        if not valid_velocity:
            return self._brake('velocity_unavailable')
        if (position is None or np.shape(position) != (2,) or not np.isfinite(position).all()
                or heading is None or not math.isfinite(heading)):
            return self._brake('pose_unavailable')
        if target is None or np.shape(target) != (2,) or not np.isfinite(target).all():
            self.arrived = False
            return self._brake('target_unavailable')
        position = np.asarray(position, float).copy()
        target = np.asarray(target, float).copy()
        heading = float(heading)
        changed = self.target is None or np.linalg.norm(target-self.target) > 8.
        self.target = target.copy()
        if changed:
            self.route = []
            self._repair = []
            self._plan_tick = -10**9
            self.stats['target_changes'] += 1
        self.arrived = False
        old_heading, old_tick = self._heading, self._heading_tick
        self._heading, self._heading_tick = heading, self.tick
        if old_heading is not None and old_tick is not None and self.tick-old_tick <= 1:
            turn = abs((heading-old_heading+math.pi) % (2*math.pi)-math.pi)
            if turn > .12:
                return self._brake('heading_settle')
        if lidar is None or angles is None:
            return self._brake('scan_unavailable', True)
        d, a = np.asarray(lidar, float).ravel(), np.asarray(angles, float).ravel()
        if (len(d) < 24 or d.shape != a.shape or not np.isfinite(a).all()
                or np.count_nonzero(np.isfinite(d) & (d > 3)) < 12):
            return self._brake('scan_unavailable', True)
        fine = self.local_factory(d, a, semantic or ())
        self._last_local = fine
        self.stats['fresh_local_scans'] += 1
        if not fine.line_clear(np.zeros(2), np.zeros(2)):
            return self._brake('current_hull_not_observed_clear', True)
        direct = rotate(target-position, -heading)
        if np.linalg.norm(direct) <= self.ARRIVAL_RADIUS and fine.line_clear(np.zeros(2), direct):
            self.arrived = True
            self.blocked_ticks = 0
            self.route = []
            self._repair = []
            return self._brake('at_relay_target')
        if changed or self.tick-self._plan_tick >= self.REPLAN_STEPS:
            self._plan(grid, position, target)
        # A small corner is removed only if its successor is visible now.
        while len(self._repair) > 1 and np.linalg.norm(self._repair[0]-position) < 8.:
            successor = rotate(self._repair[1]-position, -heading)
            if not fine.line_clear(np.zeros(2), successor):
                self.stats['necessary_near_corner_retained'] += 1
                break
            self._repair.pop(0)
        if self._repair and np.linalg.norm(self._repair[0]-position) < .5:
            self._repair.pop(0)
        chosen = None
        repair_motion = False
        if self._repair:
            point = rotate(self._repair[0]-position, -heading)
            if fine.line_clear(np.zeros(2), point):
                chosen = point.copy()
                repair_motion = True
            else:
                self._repair = []
        if chosen is None:
            for point in self.route:
                body = rotate(point-position, -heading)
                length = np.linalg.norm(body)
                if length > self.LOOKAHEAD:
                    break
                if length > .5 and fine.line_clear(np.zeros(2), body):
                    chosen = body.copy()
            if (self.target_reachable and np.linalg.norm(direct) <= self.LOOKAHEAD
                    and fine.line_clear(np.zeros(2), direct)):
                chosen = direct.copy()
        if chosen is None and self.tick-self._repair_tick >= self.REPAIR_STEPS:
            self._repair_tick = self.tick
            nearby = [rotate(point-position, -heading) for point in self.route
                      if 12. < np.linalg.norm(point-position) <= self.LOOKAHEAD]
            # At most two fine searches; keep the single-CPU budget bounded.
            for body in nearby[:1]+nearby[-1:] if len(nearby) > 1 else nearby:
                path = fine.route(np.zeros(2), body.copy(), None, target=body.copy(), limit=165)
                self.stats['local_repair_searches'] += 1
                if path and np.linalg.norm(np.asarray(path[-1])-body) <= fine.CELL*math.sqrt(2)+1e-8:
                    self._repair = [position+rotate(np.asarray(p, float), heading) for p in path[1:]]
                    if self._repair:
                        candidate = rotate(self._repair[0]-position, -heading)
                        if fine.line_clear(np.zeros(2), candidate):
                            chosen = candidate.copy()
                            repair_motion = True
                            self.stats['local_repairs'] += 1
                            break
                elif path:
                    self.stats['repair_wrong_terminal_rejected'] += 1
        if chosen is None:
            return self._brake('no_fresh_hull_verified_segment', True)
        distance = float(np.linalg.norm(chosen))
        if distance <= .25:
            return self._brake('subpixel_waypoint')
        self.waypoint = (position+rotate(chosen, heading)).tolist()
        clearance = float(fine.sample(np.linspace(np.zeros(2), chosen, 8)).min())
        stop_margin = .25 if repair_motion or distance < 8. else 4.
        speed = min(self.MAX_SPEED, max(.7, .28*(clearance-fine.TRACK_RADIUS)),
                    math.sqrt(max(0., .8*(distance-stop_margin))))
        desired = chosen/distance*speed
        force = .085*desired+.65*(desired-self._velocity)
        force /= max(1., float(np.linalg.norm(force)))
        predicted = .95*self._velocity+.6*force
        stopping = predicted*3+predicted*np.linalg.norm(predicted)/1.2
        if not fine.line_clear(np.zeros(2), stopping):
            return self._brake('stopping_sweep_blocked', True)
        self.blocked_ticks = 0
        self.reason = 'following_fresh_local_repair' if repair_motion else 'following_verified_relay_route'
        self.stats['motion_ticks'] += 1
        if self.planning_mode == 'unexpanded_known_free_fresh_hull_required':
            self.stats['hull_verified_unexpanded_motion_ticks'] += 1
        return dict(forward=float(force[0]), lateral=float(force[1]), rotation=0.)

    step = command

    def status(self):
        return dict(reason=self.reason, target=None if self.target is None else self.target.tolist(),
                    waypoint=self.waypoint, arrived=self.arrived, route_cells=len(self.route),
                    repair_waypoints=len(self._repair), planning_mode=self.planning_mode,
                    inflated_target_reachable=self.inflated_target_reachable,
                    target_reachable=self.target_reachable, proxy_distance=self.proxy_distance,
                    reachable_cells=self.reachable_cells, blocked_ticks=self.blocked_ticks,
                    tick=self.tick, stats=dict(self.stats))
