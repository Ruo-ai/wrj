"""Fresh-scan narrow passages for a fixed physical hull orientation.

Only observed free space is usable. Collision geometry is the existing RN7
convex hull, expanded by 2 px plus raster uncertainty. Turning is preempted;
this module never substitutes the drone's abstract radius for its actual hull.
"""
import math
from collections import deque

import cv2
import numpy as np

from swarm_rescue.solutions.my_drone_red_rn7 import (
    HULL, LocalPassage, PassageMotion, CoarseView,
)


class HullPassage28(LocalPassage):
    SAFETY_MARGIN = 2.0
    _FOOTPRINT = None

    @classmethod
    def footprint(cls):
        if cls._FOOTPRINT is None:
            extent = int(math.ceil((np.abs(HULL).max() + 8) / cls.CELL))
            kernel = np.zeros((2*extent+1, 2*extent+1), np.uint8)
            vertices = np.rint(HULL * np.array([1., -1.]) / cls.CELL).astype(np.int32) + extent
            cv2.fillConvexPoly(kernel, vertices, 1)
            # The rounded polygon may under-cover a point by half a cell
            # diagonal. Add that uncertainty in addition to physical clearance.
            radius = int(math.ceil((cls.SAFETY_MARGIN + cls.CELL/math.sqrt(2)) / cls.CELL))
            yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
            expansion = (xx*xx + yy*yy <= radius*radius).astype(np.uint8)
            cls._FOOTPRINT = cv2.dilate(kernel, expansion)
        return cls._FOOTPRINT

    @property
    def clearance(self):
        if self._clearance is None:
            known = (self.free & ~self.static).astype(np.uint8)
            centers = cv2.erode(known, self.footprint(), borderType=cv2.BORDER_CONSTANT,
                                borderValue=0).astype(bool)
            extra = cv2.distanceTransform(np.pad(centers.astype(np.uint8), 1),
                                         cv2.DIST_L2, 5)[1:-1, 1:-1] * self.CELL
            # Preserve the original route/motion API. This value describes
            # hull-feasible centre-space, not the radius of a circular drone.
            self._clearance = np.where(centers,
                self.PLAN_RADIUS + np.maximum(0., extra.astype(float)-self.CELL), 0.)
        return self._clearance

    @property
    def squeeze(self):
        if self._squeeze is None:
            yy, xx = np.mgrid[-29:30, -29:30]
            xx, yy = xx*self.CELL, yy*self.CELL
            walls = self.static.astype(np.uint8)
            both = np.zeros(self.shape, bool)
            for angle in np.arange(8)*math.pi/8:
                across = xx*math.cos(angle) + yy*math.sin(angle)
                along = -xx*math.sin(angle) + yy*math.cos(angle)
                side = ((across >= 12) & (across <= 56) & (np.abs(along) <= 8)).astype(np.uint8)
                both |= (cv2.dilate(walls, side) > 0) & (cv2.dilate(walls, side[::-1, ::-1]) > 0)
            self._squeeze = both & (self.clearance >= self.PLAN_RADIUS) & (self.clearance < self.PLAN_RADIUS+20)
        return self._squeeze


class PassageMotion28(PassageMotion):
    """Bounded passage attempts; each motion and stopping sweep is reobserved."""
    def __init__(self, identifier=0):
        super().__init__(identifier)
        self.angular_increment = 0.
        self._angular_samples = deque(maxlen=3)

    def update(self, odometry):
        super().update(odometry)
        value = (float(odometry[2]) if odometry is not None
                 and np.shape(odometry) == (3,) and np.isfinite(odometry).all() else float('inf'))
        if math.isfinite(value):
            self._angular_samples.append(value)
            self.angular_increment = value if abs(value) > .08 else float(np.mean(self._angular_samples))
        else:
            self._angular_samples.clear()
            self.angular_increment = float('inf')

    def command(self, d, a, coarse, position, heading, desired, objects=(), allow=True):
        if not allow or not self.valid_odom or d is None or a is None:
            if self.path:
                self.release('mission_preempted')
            return None
        if abs(self.angular_increment) > .02:
            if self.path:
                self.stats['angular_brake'] += 1
                return self.brake()
            return None
        if not self.path:
            if self.tick < self.cooldown or self.episodes >= 24:
                return None
            self.cooldown = self.tick + 30 + self.identifier*3
            fine = HullPassage28(d, a, objects)
            self.stats['probes'] += 1
            preferred = np.array([math.cos(desired), math.sin(desired)])
            path = fine.route(np.zeros(2), preferred, CoarseView(coarse, position, heading), limit=165)
            if not path:
                self.stats['reject_'+fine.last_reason] += 1
                return None
            self.path = path[1:]
            self.since = self.tick
            self.last_progress = self.tick
            self.best_remaining = float('inf')
            self.episodes += 1
            self.blocked_ticks = 0
            self.snapshot = fine
            self.last_refresh = self.tick
            self.event('entered')
        if self.tick-self.since > 220:
            self.release('task_timeout')
            return self.brake()
        while self.path and np.linalg.norm(self.path[0]) < 6:
            self.path.pop(0)
        if not self.path:
            self.release('completed')
            return self.brake()
        remaining = float(sum(np.linalg.norm(v) for v in np.diff([np.zeros(2), *self.path], axis=0)))
        if remaining < self.best_remaining-5:
            self.best_remaining = remaining
            self.last_progress = self.tick
        if self.tick-self.last_progress > 40:
            self.release('no_progress')
            return self.brake()
        peers = [o for o in objects or () if getattr(o.entity_type, 'name', '') == 'DRONE'
                 and o.distance < 60
                 and abs((o.angle-math.atan2(self.path[0][1], self.path[0][0])+math.pi) % (2*math.pi)-math.pi) < .8]
        if peers:
            self.wait += 1
            self.stats['peer_brake_ticks'] += 1
            if self.wait > 7+3*self.identifier:
                self.release('peer_wait_budget')
            return self.brake()
        self.wait = 0
        fine = HullPassage28(d, a, objects)
        target = self.path[0].copy()
        if not fine.line_clear(np.zeros(2), target):
            self.blocked_ticks += 1
            if self.tick-self.repair_tick >= 6:
                self.repair_tick = self.tick
                repaired = []
                for endpoint in reversed(self.path):
                    if np.linalg.norm(endpoint) < 25:
                        continue
                    normal = np.array([-endpoint[1], endpoint[0]])/np.linalg.norm(endpoint)
                    offsets = np.arange(-12., 13., 2.)
                    choices = endpoint+offsets[:, None]*normal
                    clearance = fine.sample(choices)
                    chosen = int(np.argmax(clearance-.04*np.abs(offsets)))
                    if clearance[chosen] < fine.PLAN_RADIUS:
                        continue
                    endpoint = choices[chosen]
                    repaired = fine.route(np.zeros(2), endpoint, CoarseView(coarse, position, heading),
                                          target=endpoint, limit=165)
                    break
                if repaired:
                    self.path = repaired[1:]
                    self.blocked_ticks = 0
                    self.event('local_replan')
            if self.blocked_ticks >= 12:
                self.release('fresh_scan_obstructed')
            return self.brake()
        self.blocked_ticks = 0
        for point in self.path[1:4]:
            if np.linalg.norm(point) > 32 or not fine.line_clear(np.zeros(2), point):
                break
            target = point.copy()
        distance = float(np.linalg.norm(target))
        direction = target/max(distance, 1.)
        margin = max(0., float(fine.sample(np.linspace(np.zeros(2), target, 8)).min())-fine.PLAN_RADIUS)
        speed = min(3.2, 1.1+.28*margin, math.sqrt(max(0., .8*(distance-3))))
        goal = direction*speed
        force = .085*goal+.65*(goal-self.velocity)
        force /= max(1., float(np.linalg.norm(force)))
        predicted = .95*self.velocity+.6*force
        stopping = predicted*3 + predicted*np.linalg.norm(predicted)/1.2
        if not fine.line_clear(np.zeros(2), stopping):
            self.stats['inertial_brake'] += 1
            return self.brake()
        self.stats['active_ticks'] += 1
        return dict(forward=float(force[0]), lateral=float(force[1]), rotation=0.)
