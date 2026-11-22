"""Observed-path return and unfinished frontiers; no world/map/zone inputs."""
from collections import Counter, deque
import heapq
import math

import cv2
import numpy as np

from swarm_rescue.solutions.red_navigation32 import EvidenceNavigation32
from swarm_rescue.solutions.my_drone_red_stopturn import ray_clearance


class WalkGraph39:
    """Edges are actually walked, or checked against the current legal lidar.

    This is separate from occupancy: an old occupied grid cell must not erase
    the only record of a traversed doorway. Every executed edge still goes
    through live swept-disc avoidance. A graph edge is not collision immunity.
    """
    def __init__(self):
        self.points = []
        self.edges = []
        self.bins = {}
        self.last = None
        self.last_step = 0
        self.seen_step = 0
        self.blocked = {}
        self.stats = Counter()

    def near(self, position, radius):
        bx, by = np.floor(np.asarray(position)/50).astype(int)
        n = int(math.ceil(radius/50))
        return [i for x in range(bx-n, bx+n+1) for y in range(by-n, by+n+1)
                for i in self.bins.get((x, y), ())
                if np.linalg.norm(self.points[i]-position) <= radius]

    @staticmethod
    def visible(position, target, heading, distances, angles):
        delta = np.asarray(target)-position
        length = float(np.linalg.norm(delta))
        if length < 8:
            return True
        return ray_clearance(distances, angles,
            math.atan2(delta[1], delta[0])-heading, radius=20) >= length+3

    def observe(self, position, heading, distances, angles, step):
        if position is None or not np.isfinite(position).all():
            return
        p = np.asarray(position, float)
        continuous = step-self.seen_step < 4
        self.seen_step = int(step)
        if self.last is not None and np.linalg.norm(p-self.points[self.last]) < 22:
            return
        neighbors = self.near(p, 42)
        # Rejoin an old walked node only through a currently observed corridor.
        joined = [i for i in neighbors if np.linalg.norm(p-self.points[i]) < 24
                  and self.visible(p, self.points[i], heading, distances, angles)]
        if joined:
            node = min(joined, key=lambda i: np.linalg.norm(p-self.points[i]))
        else:
            node = len(self.points)
            self.points.append(p.copy()); self.edges.append({})
            self.bins.setdefault(tuple(np.floor(p/50).astype(int)), []).append(node)
            self.stats['walked_nodes'] += 1
        if self.last is not None and node != self.last:
            distance = float(np.linalg.norm(self.points[node]-self.points[self.last]))
            # Observed every control; reject a localization discontinuity.
            if distance < 70 and continuous:
                self.edges[node][self.last] = distance
                self.edges[self.last][node] = distance
                self.stats['walked_edges'] += 1
            else:
                self.stats['discontinuities'] += 1
        for other in neighbors:
            if other != node and other not in self.edges[node] and self.visible(
                    p, self.points[other], heading, distances, angles):
                distance = float(np.linalg.norm(self.points[node]-self.points[other]))
                # The new pose is within 24px of node; check both endpoints.
                if np.linalg.norm(p-self.points[node]) < 8:
                    self.edges[node][other] = distance
                    self.edges[other][node] = distance
                    self.stats['lidar_loop_closures'] += 1
        self.last = node; self.last_step = int(step)

    def path(self, position, heading, distances, angles, step, target=None):
        if not self.points:
            return [], float('inf')
        # The launch node is a legally observed fallback, never a map coordinate.
        root = 0 if target is None else min(range(len(self.points)),
            key=lambda i: np.linalg.norm(self.points[i]-target))
        costs = {root: 0.}; following = {}; queue = [(0., root)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != costs[node]:
                continue
            for other, length in self.edges[node].items():
                edge = tuple(sorted((node, other)))
                extra = 160 if self.blocked.get(edge, 0) > step else 0
                new = cost+length+extra
                if new < costs.get(other, float('inf')):
                    costs[other] = new; following[other] = node
                    heapq.heappush(queue, (new, other))
        nearby = self.near(position, 110)
        visible = [i for i in nearby if i in costs and self.visible(
            position, self.points[i], heading, distances, angles)]
        if not visible:
            visible = [self.last] if self.last in costs else []
        if not visible:
            return [], float('inf')
        node = min(visible, key=lambda i: costs[i]+np.linalg.norm(position-self.points[i]))
        cost = float(costs[node]+np.linalg.norm(position-self.points[node]))
        path = [node]
        while path[-1] != root:
            path.append(following[path[-1]])
        return path, cost


class FrontierQueue39(EvidenceNavigation32):
    """Retain unobserved branches; use remaining exploration time for frontiers.

    Original Clean evidence/clearance are inherited. Before extension_enabled,
    original exploration/placement decisions remain unchanged. Frontier tasks
    finish only after observations remove the unknown neighborhood, not simply
    because the aircraft passed near a goal once.
    """
    def __init__(self, size, event):
        super().__init__(size, event)
        self.extension_enabled = False
        self.pending39 = {}
        self.retry39 = {}
        self.audit39 = []
        self.stats39 = Counter()

    def retire_goal(self, cell):
        if self.extension_enabled and self.unknown_near(cell) >= 3:
            self.retry39[cell] = self.step+120
            self.stats39['unfinished_retained'] += 1
            return
        super().retire_goal(cell)

    def plan(self, position, home, identifier, number, claims, blocked_goals=(), rendezvous=False):
        if not self.extension_enabled or rendezvous:
            return super().plan(position, home, identifier, number, claims, blocked_goals, rendezvous)
        walls = cv2.dilate((self.obstacle >= 2).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))).astype(bool)
        allowed = self.free & ~walls; start = self.cell(position); allowed[start] = True
        distance = {start: 0}; parents = {}; queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                cell = (r+dr, c+dc)
                if (0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]
                        and allowed[cell] and cell not in distance):
                    distance[cell] = distance[r,c]+1; parents[cell] = (r,c); queue.append(cell)
        unknown = cv2.filter2D((~self.free & ~walls).astype(np.float32), -1, np.ones((5,5), np.float32))
        candidates = []
        for cell, steps in distance.items():
            if unknown[cell] < 3:
                self.pending39.pop(cell, None)
                continue
            self.pending39.setdefault(cell, self.step)
            if steps < 3 or cell in blocked_goals or self.retry39.get(cell,0) > self.step:
                continue
            point = self.point(cell)
            if any(t > self.step and np.linalg.norm(point-p) < 100 for p,t in self.avoid_regions):
                continue
            score = 7*min(15,float(unknown[cell]))-.55*steps
            score += min(35,(self.step-self.pending39[cell])*.035)
            score -= min(45,float(self.visits[cell])*3)
            for claim in claims:
                score -= 100*max(0,1-np.linalg.norm(point-claim)/160)
            candidates.append((float(score),cell,steps))
        if not candidates:
            self.stats39['no_reachable_frontier'] += 1
            return super().plan(position, home, identifier, number, claims, blocked_goals, False)
        candidates.sort(reverse=True); score,cell,steps = candidates[0]
        self.stats39['frontier_selected'] += 1
        if len(self.audit39) < 200:
            self.audit39.append(dict(step=self.step,goal=list(cell),score=score,
                unknown=float(unknown[cell]),distance=steps,pending=len(self.pending39),
                top=[(s,list(c),n) for s,c,n in candidates[:4]]))
        self.event('d39_frontier',goal=self.point(cell).tolist(),unknown=float(unknown[cell]),pending=len(self.pending39))
        path = [cell]
        while path[-1] != start:
            path.append(parents[path[-1]])
        return list(reversed(path))
