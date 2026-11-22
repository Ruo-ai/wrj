"""Passage 30: preserve route targets and required turns in Passage 28 geometry.

The inherited planner normalizes its direction in place. A target passed as the
same numpy array would otherwise shrink to a unit vector during local repair.
Required nearby turns are kept until the next segment is currently visible.
No clearance, hull, observation, exploration budget, or timing parameter changes.
"""
import math

import numpy as np

from swarm_rescue.solutions.red_passage28 import HullPassage28, PassageMotion28
from swarm_rescue.solutions.my_drone_red_rn7 import CoarseView


class HullPassage30(HullPassage28):
    """The exact 28 local geometry, with independent planner-owned arrays."""

    def route(self, position, preferred, coarse, avoid=(), target=None, limit=220):
        # Copy independently even when preferred and target share a base/view.
        direction = np.array(preferred, dtype=float, copy=True)
        destination = None if target is None else np.array(target, dtype=float, copy=True)
        return super().route(position, direction, coarse, avoid=avoid,
                             target=destination, limit=limit)


class PassageMotion30(PassageMotion28):
    """Passage 28 control with alias-safe routes and necessary nearby turns.

    Copying command keeps the frozen 28 implementation untouched. Its original
    stopping sweep remains authoritative for every proposed motion command.
    """
    LOCAL_FACTORY = HullPassage30

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
            fine = self.LOCAL_FACTORY(d, a, objects)
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
        fine = self.LOCAL_FACTORY(d, a, objects)
        while self.path and np.linalg.norm(self.path[0]) < 6:
            # Near a turn is not the same as having cleared the corner. Skip
            # only if the current complete connecting segment is admissible.
            if len(self.path) > 1 and not fine.line_clear(np.zeros(2), self.path[1]):
                break
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
        required_near_corner = len(self.path) > 1 and np.linalg.norm(self.path[0]) < 6
        stopping_reserve = 0. if required_near_corner else 3.
        speed = min(3.2, 1.1+.28*margin,
                    math.sqrt(max(0., .8*(distance-stopping_reserve))))
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
