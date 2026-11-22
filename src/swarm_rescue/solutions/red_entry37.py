"""Bounded outage-entry retrace using only this vehicle's legal observations.

No map identity, true pose, sensor mutation, mapper reset or world-axis prior.
This controller returns to recently traversed breadcrumbs, waits for stable
absolute sensing, aligns to the measured inbound course and re-enters once.
The existing exploration/stop-turn controller resumes after a short entry.
"""
from collections import Counter, deque
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_stopturn import available_pose, body_motion, ray_clearance, turn_clear
from swarm_rescue.solutions.semifinal_localization import wrap


class Entry37:
    MAX_EPISODE = 300
    MAX_TOTAL = 420
    MAX_ATTEMPTS = 2
    START_BEFORE = .50
    FINISH_BEFORE = .60
    RETURN_SPEED = 2.4
    ENTRY_SPEED = 3.2
    CALIBRATION_FRAMES = 24

    def __init__(self):
        self.state = 'IDLE'
        self.tick = self.since = self.started = self.total = self.good = self.missing = 0
        self.history = deque(maxlen=80)
        self.samples = deque(maxlen=self.CALIBRATION_FRAMES)
        self.events = []
        self.stats = Counter()
        self.attempted = []
        self.velocity = np.zeros(2)
        self.angular = deque(maxlen=5)
        self.still = 0
        self.blocked = 0
        self.return_path = []
        self.reentry_path = []
        self.calibrated = False
        self.recovered_at = None
        self.target_heading = 0.

    @property
    def active(self):
        return self.state != 'IDLE'

    def event(self, kind, **details):
        self.events.append(dict(step=self.tick, event=kind, **details))

    def transition(self, state, reason):
        self.event('transition', old=self.state, new=state, reason=reason)
        self.state, self.since, self.still, self.blocked = state, self.tick, 0, 0

    def release(self, reason):
        self.stats['finish_' + reason] += 1
        self.event('finish', reason=reason, duration=self.tick-self.started,
                   calibrated=self.calibrated)
        self.transition('IDLE', reason)

    def observe(self, tick, position, heading, gps, compass, odometry):
        self.tick = int(tick)
        self.position = None if position is None else np.asarray(position, float).copy()
        self.heading = float(heading) if heading is not None else 0.
        self.absolute = all(available_pose(gps, compass))
        self.good = self.good+1 if self.absolute else 0
        self.missing = 0 if self.absolute else self.missing+1
        self.measured = body_motion(odometry)
        if self.measured is not None:
            angle = float(odometry[2])
            c, s = math.cos(angle), math.sin(angle)
            self.velocity = .35*np.array([[c,s],[-s,c]])@self.velocity + .65*self.measured
            self.angular.append(angle)
        self.gps = np.asarray(gps, float).copy() if self.absolute else None
        self.compass = float(compass) if self.absolute else None
        if self.absolute and self.position is not None and not self.active:
            if self.good >= 8 and (not self.history or
                    np.linalg.norm(self.position-self.history[-1]['position']) >= 5):
                self.history.append(dict(step=self.tick, position=self.position.copy(), heading=self.heading))

    def brake(self):
        v = np.clip(-.60*self.velocity, -1., 1.)
        return dict(forward=float(v[0]), lateral=float(v[1]), rotation=0.)

    def stopped(self):
        return bool(np.linalg.norm(self.velocity) < .65 and self.angular and
                    abs(float(np.mean(self.angular))) < .045)

    def begin(self):
        recent = [p for p in self.history if self.tick-p['step'] <= 90]
        if len(recent) < 3:
            return False
        anchor = recent[-1]['position']
        if any(np.linalg.norm(anchor-p) < 120 for p in self.attempted):
            self.stats['same_entry_suppressed'] += 1
            return False
        previous = next((p['position'] for p in reversed(recent[:-1])
                         if np.linalg.norm(p['position']-anchor) >= 25), None)
        if previous is None:
            return False
        direction = anchor-previous
        direction /= np.linalg.norm(direction)
        # Actual observed route, decimated spatially; do not infer a straight
        # backward ray through corners from only the final GPS coordinate.
        path = []
        for p in reversed(recent):
            point = p['position']
            if not path or np.linalg.norm(point-path[-1]) >= 12:
                path.append(point.copy())
            if np.linalg.norm(point-anchor) >= 65:
                break
        if len(path) < 3:
            return False
        self.anchor = anchor.copy()
        self.loss_position = self.position.copy()
        self.inbound = direction
        self.target_heading = math.atan2(direction[1], direction[0])
        self.return_path = path
        self.reentry_path = []
        self.attempted.append(anchor.copy())
        self.started = self.tick
        self.calibrated = False
        self.recovered_at = None
        self.samples.clear()
        self.stats['attempts'] += 1
        self.event('entry_detected', position=self.position.tolist(), anchor=anchor.tolist(),
                   heading=self.heading, speed=float(np.linalg.norm(self.velocity)),
                   inbound=direction.tolist(), breadcrumbs=[p.tolist() for p in path])
        self.transition('BRAKE', 'joint_absolute_loss')
        return True

    def drive(self, target, speed, distances, angles):
        delta = target-self.position
        length = float(np.linalg.norm(delta))
        direction = wrap(math.atan2(delta[1], delta[0])-self.heading)
        clearance = ray_clearance(distances, angles, direction, radius=18.5)
        actual_speed = float(np.linalg.norm(self.velocity))
        stop = (actual_speed+.4)**2/(2*.35)+2*actual_speed+4
        if clearance <= stop+4:
            self.blocked += 1
            self.stats['blocked_ticks'] += 1
            return self.brake()
        self.blocked = max(0, self.blocked-1)
        desired = min(speed, math.sqrt(.7*max(0., length-3)),
                      math.sqrt(.7*max(0., clearance-stop-4)))
        goal = desired*np.array([math.cos(direction), math.sin(direction)])
        force = .085*goal+.65*(goal-self.velocity)
        force /= max(1., float(np.linalg.norm(force)))
        return dict(forward=float(force[0]), lateral=float(force[1]), rotation=0.)

    def step(self, distances, angles, eligible, progress):
        if self.position is None or self.measured is None:
            if self.active:
                self.release('invalid_observation')
            return None
        if not self.active:
            # The first transition only: never start a stale return in a deep
            # outage or repeatedly rediscover one border after release.
            if not (eligible and progress < self.START_BEFORE and 3 <= self.missing <= 5
                    and len(self.attempted) < self.MAX_ATTEMPTS and self.total < self.MAX_TOTAL):
                return None
            if not self.begin():
                return None
        if not eligible or progress >= self.FINISH_BEFORE:
            self.release('task_preempted')
            return None
        self.total += 1
        self.stats['ticks_'+self.state] += 1
        if self.tick-self.started >= self.MAX_EPISODE or self.total >= self.MAX_TOTAL:
            self.release('budget')
            return None
        if self.blocked >= 24:
            self.release('blocked')
            return None
        age = self.tick-self.since
        if age > dict(BRAKE=35, RETREAT=120, CALIBRATE=75, ALIGN=65, REENTER=120)[self.state]:
            self.release('timeout_'+self.state)
            return None
        if self.state == 'BRAKE':
            self.still = self.still+1 if self.stopped() else 0
            if self.still >= 3:
                self.transition('RETREAT', 'stopped_before_retrace')
            return self.brake()
        if self.state == 'RETREAT':
            if self.absolute and self.recovered_at is None:
                self.recovered_at = self.position.copy()
                self.event('sensors_recovered', position=self.position.tolist())
            if not self.absolute:
                self.recovered_at = None
            if self.good >= 8 and self.recovered_at is not None and (
                    np.linalg.norm(self.position-self.recovered_at) >= 12 or not self.return_path):
                self.transition('CALIBRATE', 'recovered_with_margin')
                return self.brake()
            while self.return_path and np.linalg.norm(self.return_path[0]-self.position) < 8:
                self.return_path.pop(0)
            if not self.return_path:
                if self.good >= 8:
                    self.transition('CALIBRATE', 'observed_return_end')
                elif self.absolute:
                    return self.brake()
                else:
                    self.release('return_route_exhausted')
                    return None
                return self.brake()
            return self.drive(self.return_path[0], self.RETURN_SPEED, distances, angles)
        if self.state == 'CALIBRATE':
            if not self.absolute:
                self.samples.clear()
                self.transition('RETREAT', 'recovery_flicker')
                return self.brake()
            if self.stopped():
                self.samples.append((self.gps.copy(), self.compass))
            else:
                self.samples.clear()
            if len(self.samples) == self.CALIBRATION_FRAMES:
                positions = np.array([s[0] for s in self.samples])
                headings = np.array([s[1] for s in self.samples])
                mean = math.atan2(float(np.sin(headings).mean()),float(np.cos(headings).mean()))
                spread = float(np.max(np.abs((headings-mean+math.pi)%(2*math.pi)-math.pi)))
                gps_spread = float(np.max(np.linalg.norm(positions-np.median(positions,axis=0),axis=1)))
                if spread <= .12 and gps_spread <= 14:
                    self.calibrated = True
                    self.stats['calibrations'] += 1
                    self.event('calibrated', position=self.position.tolist(), heading=self.heading,
                        gps_center=np.median(positions,axis=0).tolist(), compass_mean=mean,
                        gps_spread=gps_spread, compass_spread=spread,
                        target_heading=self.target_heading, frames=len(self.samples))
                    # Reverse the actually traversed prefix of the known route.
                    prefix = [p['position'].copy() for p in self.history
                              if self.started-p['step'] <= 90]
                    start = int(np.argmin([np.linalg.norm(p-self.position) for p in prefix]))
                    self.reentry_path = [p for p in prefix[start+1:]]
                    self.reentry_path.append(self.loss_position.copy())
                    self.reentry_path.append(self.loss_position+35*self.inbound)
                    self.transition('ALIGN', 'stable_absolute_samples')
            return self.brake()
        if self.state == 'ALIGN':
            if not self.absolute:
                self.release('lost_during_alignment')
                return None
            error = wrap(self.target_heading-self.heading)
            if abs(error) > .08 and not turn_clear(distances, angles):
                self.blocked += 1
                return self.brake()
            self.still = self.still+1 if self.stopped() and abs(error) < .07 else 0
            if self.still >= 4:
                self.event('aligned', position=self.position.tolist(), heading=self.heading,
                           target_heading=self.target_heading)
                self.transition('REENTER', 'heading_confirmed')
                return self.brake()
            command = self.brake()
            if np.linalg.norm(self.velocity) < .65:
                command['rotation'] = float(np.clip(.75*error-.4*np.mean(self.angular),-.25,.25))
            return command
        if self.state == 'REENTER':
            if self.missing == 1:
                self.stats['reentries'] += 1
                self.event('reentered', position=self.position.tolist(), heading=self.heading,
                           speed=float(np.linalg.norm(self.velocity)))
            while self.reentry_path and np.linalg.norm(self.reentry_path[0]-self.position) < 9:
                self.reentry_path.pop(0)
            if not self.reentry_path:
                self.release('reentered' if self.missing else 'route_complete_outside')
                return self.brake()
            command = self.drive(self.reentry_path[0],self.ENTRY_SPEED,distances,angles)
            error = wrap(self.target_heading-self.heading)
            if abs(error) > .20:
                # Stop translation before substantial heading adjustment.
                command = self.brake()
                if np.linalg.norm(self.velocity) < .65 and turn_clear(distances,angles):
                    command['rotation'] = float(np.clip(.75*error,-.25,.25))
            elif abs(error) > .025:
                command['rotation'] = float(np.clip(.35*error,-.06,.06))
            return command
        raise AssertionError(self.state)
