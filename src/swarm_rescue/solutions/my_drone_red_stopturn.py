"""S6: optional measured-motion stop/turn execution on retained A3.

No world geometry, simulator pose, map identity or command integration is used.
The private motion estimate never replaces navigation or submission coordinates.
"""
from collections import Counter, deque
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_mapping import MyDroneRedMapping
from swarm_rescue.solutions.semifinal_localization import LocalMatcher, scan_points, wrap


def available_pose(gps, compass):
    return (bool(gps is not None and np.shape(gps)==(2,) and np.isfinite(gps).all()),
            bool(compass is not None and np.ndim(compass)==0 and np.isfinite(compass)))


def body_motion(odometry):
    if odometry is None or np.shape(odometry)!=(3,) or not np.isfinite(odometry).all():
        return None
    distance, alpha, theta = map(float, odometry)
    return distance*np.array([math.cos(alpha-theta), math.sin(alpha-theta)])


def wall_rank(points):
    """Reject translation correction in a single wall/parallel corridor.

    Fit small contiguous runs, not separate world x/y probes. Noise does not
    magically make the long direction of an oblique corridor observable.
    """
    normals = []
    for i in range(0, len(points)-4, 3):
        group = points[i:i+5]
        if np.max(np.linalg.norm(np.diff(group, axis=0), axis=1)) > 35: continue
        centered = group-group.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T@centered)
        if values[1] < 40 or values[0] > .08*values[1]: continue
        normals.append(vectors[:, 0])
    if len(normals) < 3: return 0
    values = np.linalg.eigvalsh(np.asarray(normals).T@np.asarray(normals))
    return 2 if values[0] > .18*values.sum() else 1


class MotionEstimate:
    """Every-step legal odometry, with explicitly heuristic uncertainty."""
    def __init__(self, size, radar=False):
        self.position = None; self.heading = 0.; self.velocity = np.zeros(2)
        self.angular_increment = 0.; self.odometry_valid = False
        self.step = 0; self.good = 0; self.missing = 0; self.absolute = (False, False)
        self.position_variance = 25.; self.heading_variance = math.radians(4)**2
        self.reliable = None; self.radar = radar
        self.matcher = LocalMatcher(size) if radar else None
        self.stats = Counter(); self.rejected_streak = 0; self.last_rank = 0

    def update(self, gps, compass, odometry, distances=None, angles=None, objects=()):
        self.step += 1; self.absolute = available_pose(gps, compass)
        joint = all(self.absolute)
        self.good = self.good+1 if joint else 0
        self.missing = 0 if joint else self.missing+1
        measured = body_motion(odometry); self.odometry_valid = measured is not None
        if self.position is None:
            if not joint: return
            self.position = np.asarray(gps, float).copy(); self.heading = float(compass)
            # No preceding anchored pose: do not double-integrate the first tick.
        elif self.odometry_valid:
            d, alpha, rotation = map(float, odometry)
            self.position += d*np.array([math.cos(self.heading+alpha), math.sin(self.heading+alpha)])
            self.heading = wrap(self.heading+rotation)
            self.position_variance += .2**2+d*d*(math.radians(8)**2+self.heading_variance)
            self.heading_variance += math.radians(1)**2
            self.stats['integrated_steps'] += 1
        if self.odometry_valid:
            self.velocity = measured; self.angular_increment = float(odometry[2])
        else: self.stats['invalid_odometry'] += 1
        # Recovery requires BOTH sensors; partial recovery never declares lock.
        # This private estimate blends bounded corrections, not a map jump.
        if joint and self.good >= 8:
            delta = np.asarray(gps)-self.position
            self.position += delta*min(.12, 2./max(1e-9, float(np.linalg.norm(delta))))
            self.heading = wrap(self.heading+np.clip(.15*wrap(compass-self.heading), -.04, .04))
            self.position_variance = max(25., .8*self.position_variance)
            self.heading_variance = max(math.radians(4)**2, .8*self.heading_variance)
            self.reliable = dict(step=self.step, position=self.position.copy(), heading=self.heading,
                position_sigma=math.sqrt(self.position_variance), heading_sigma=math.sqrt(self.heading_variance))
        if self.matcher is not None and self.step % 8 == 0:
            points = scan_points(distances, angles, objects)
            self.last_rank = wall_rank(points)
            if not joint and self.last_rank == 2:
                offset, info = self.matcher.match(points, self.position, self.heading)
                self.stats['match_attempts'] += 1
                if (offset is not None and all(info['observable']) and
                        np.linalg.norm(offset[:2]) <= 4 and abs(offset[2]) <= .035):
                    self.position += offset[:2]; self.heading = wrap(self.heading+offset[2])
                    self.stats['matches'] += 1; self.rejected_streak = 0
                else:
                    self.stats['match_rejections'] += 1; self.rejected_streak += 1
            elif not joint: self.stats['degenerate_or_sparse'] += 1
            if self.good == 8 or self.rejected_streak >= 4:
                self.matcher.frames.clear(); self.matcher.distance = None
                self.rejected_streak = 0; self.stats['reference_resets'] += 1
            # Always match before adding; adding grants no absolute confidence.
            self.matcher.add(points, self.position, self.heading)


def ray_clearance(distances, angles, direction, radius=22.):
    if distances is None or angles is None: return 0.
    d = np.asarray(distances, float).ravel(); a = np.asarray(angles, float).ravel()
    if d.shape != a.shape or len(d) < 12: return 0.
    good = np.isfinite(d) & np.isfinite(a) & (d > 0)
    delta = (a-direction+math.pi)%(2*math.pi)-math.pi
    front = np.abs(delta) < .35
    if front.sum()<3 or good[front].mean()<.9: return 0.
    along = np.minimum(d[good],230)*np.cos(delta[good]); across = d[good]*np.sin(delta[good])
    hit = (along > 0) & (np.abs(across) < radius)
    entry = along-np.sqrt(np.maximum(0, radius*radius-across*across))
    return max(0., min(208., float(np.min(entry[hit])))) if hit.any() else 0.


def turn_clear(distances, angles):
    if distances is None or angles is None: return False
    d = np.asarray(distances).ravel(); a = np.asarray(angles).ravel()
    if d.shape != a.shape or len(d)<24 or not np.isfinite(d).all() or not np.isfinite(a).all(): return False
    # Circumscribed native hull radius 16.4012px + lidar/noise margin.
    bins = np.floor(((a+math.pi)%(2*math.pi))/(math.pi/4)).astype(int)
    return len(np.unique(bins))==8 and bool(np.min(d) > 22.)


class SegmentMotion:
    CRUISE_SPEED = 4.
    SEGMENT_LENGTH = 80.
    BRAKE_DECEL = .35  # conservative against native brake probe, not thrust=velocity
    PAUSED = frozenset(('BRAKE','SETTLE','OBSERVE','TURN','WAIT','RECOVER'))

    def __init__(self, size, identifier=0, radar=False):
        self.pose = MotionEstimate(size, radar); self.identifier = identifier
        self.state = 'NORMAL'; self.since = 0; self.cooldown = 0; self.episode = 0
        self.target_heading = 0.; self.origin = None; self.still = 0; self.turn_good = 0
        self.direction_changes = 0; self.stats = Counter(); self.events = deque(maxlen=128)
        self.record_events = False; self.entry_anchor = None; self.reason = 'initial'
        self.last_clearance = 0.; self.last_brake_distance = 0.

    def transition(self, state, reason):
        if state != self.state:
            self.stats['enter_'+state] += 1
            if self.record_events:
                self.events.append(dict(step=self.pose.step,old=self.state,state=state,reason=reason))
        self.state = state; self.since = self.pose.step; self.reason = reason
        self.still = 0; self.turn_good = 0

    def release(self, reason, cooldown=60):
        self.transition('COOLDOWN', reason); self.cooldown = self.pose.step+cooldown+3*self.identifier
        self.stats['fallback_'+reason] += 1

    def stopping_distance(self):
        speed = float(np.linalg.norm(self.pose.velocity))
        return (speed+.6)**2/(2*self.BRAKE_DECEL)+2*speed+3

    def brake(self):
        v = np.clip(-.55*self.pose.velocity,-1,1)
        return dict(forward=float(v[0]),lateral=float(v[1]),rotation=0.)

    def step(self, distances, angles, desired, target_distance=160., objects=(), allow=True):
        p = self.pose; tick = p.step
        self.stats['ticks_'+self.state] += 1
        if not allow:
            if self.state not in ('NORMAL','COOLDOWN'): self.release('preempted')
            return None
        if self.state == 'COOLDOWN':
            if tick < self.cooldown: return None
            self.transition('NORMAL','cooldown_complete')
        if self.state == 'NORMAL':
            if p.missing < 3 or p.reliable is None: return None
            self.entry_anchor = dict(p.reliable); self.episode = tick
            self.transition('BRAKE','confirmed_absolute_loss')
        if not p.odometry_valid:
            self.release('invalid_odometry'); return None
        if all(p.absolute):
            if p.good >= 8:
                self.transition('NORMAL','absolute_recovered'); return None
            if self.state != 'RECOVER': self.transition('RECOVER','confirm_recovery')
            return self.brake()
        if self.state == 'RECOVER': self.transition('BRAKE','recovery_interrupted')
        if tick-self.episode > 320:
            self.release('episode_budget',90); return None
        if desired is None or not np.isfinite(desired):
            self.release('no_navigation_direction'); return None
        age = tick-self.since
        if age > {'BRAKE':28,'SETTLE':12,'OBSERVE':15,'TURN':55,'WAIT':24,'ACCEL':30,'CRUISE':80}.get(self.state,55):
            self.release('state_timeout'); return None
        speed = float(np.linalg.norm(p.velocity))
        clearance = ray_clearance(distances,angles,desired)
        self.last_clearance = clearance; self.last_brake_distance = self.stopping_distance()
        if distances is None or angles is None:
            if self.state != 'BRAKE': self.transition('BRAKE','lidar_missing')
            return self.brake()
        peers = [o for o in objects or () if getattr(o.entity_type,'name','')=='DRONE'
                 and np.isfinite([o.distance,o.angle]).all() and o.distance<60
                 and abs(wrap(o.angle-desired))<.7]
        if peers:
            if self.state != 'WAIT': self.transition('WAIT','dynamic_peer')
            if tick-self.since >= 8+3*self.identifier:
                self.release('peer_wait_budget'); return None
            return self.brake()
        if self.state == 'WAIT': self.transition('OBSERVE','peer_cleared')
        if self.state in ('BRAKE','SETTLE'):
            stopped = speed<.45 and abs(p.angular_increment)<.045
            self.still = self.still+1 if stopped else 0
            if self.state == 'BRAKE' and stopped:
                self.transition('SETTLE','low_measured_speed'); self.still = 1
            elif self.state == 'SETTLE' and self.still >= 3:
                self.transition('OBSERVE','stopped_confirmed')
            return self.brake()
        if self.state == 'OBSERVE':
            if abs(desired)>.12 and not turn_clear(distances,angles):
                self.release('rotation_clearance'); return None
            if clearance < 30: return self.brake()
            if tick-self.since < 2: return self.brake()
            self.target_heading = wrap(p.heading+desired)
            if abs(desired)>.12:
                if not turn_clear(distances,angles):
                    self.release('rotation_clearance'); return None
                self.transition('TURN','direction_selected')
            else:
                self.origin = p.position.copy(); self.transition('ACCEL','straight_selected')
            return self.brake()
        error = wrap(self.target_heading-p.heading)
        if self.state == 'TURN':
            if not turn_clear(distances,angles):
                self.release('rotation_obstructed'); return self.brake()
            if speed > .8:
                self.transition('BRAKE','translation_during_turn'); return self.brake()
            self.turn_good = self.turn_good+1 if abs(error)<.06 and abs(p.angular_increment)<.045 else 0
            if self.turn_good >= 3:
                self.transition('OBSERVE','turn_complete'); return self.brake()
            cmd = self.brake(); cmd['rotation'] = float(np.clip(error/.3*.65,-.35,.35))
            self.stats['turn_ticks'] += 1
            return cmd
        if self.state in ('ACCEL','CRUISE'):
            displacement = p.position-self.origin
            along = float(displacement@np.array([math.cos(self.target_heading),math.sin(self.target_heading)]))
            forward_clear = ray_clearance(distances,angles,error)
            remaining = min(self.SEGMENT_LENGTH-along, max(0.,target_distance-8.))
            self.direction_changes = self.direction_changes+1 if abs(wrap(desired-error))>.4 else 0
            if (forward_clear <= self.last_brake_distance or remaining <= self.last_brake_distance
                    or (self.direction_changes>=3 and along>15) or abs(error)>.3):
                self.transition('BRAKE','segment_end_or_risk'); return self.brake()
            if self.state == 'ACCEL' and (speed>=3.2 or age>=12): self.transition('CRUISE','accelerated')
            cross = float(displacement@np.array([-math.sin(self.target_heading),math.cos(self.target_heading)]))
            goal = self.CRUISE_SPEED*np.array([math.cos(error),math.sin(error)])
            goal[1] += float(np.clip(-.05*cross,-.4,.4))
            cmd = .085*goal+.65*(goal-p.velocity)
            cmd /= max(1., float(np.linalg.norm(cmd)))
            return dict(forward=float(cmd[0]),lateral=float(cmd[1]),rotation=float(np.clip(error/.3*.4,-.08,.08)))
        return self.brake()


class MyDroneRedStopTurn(MyDroneRedMapping):
    SEGMENTED_MOTION = True
    MOTION_RADAR = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._segment_motion = SegmentMotion(self.size_area,int(self.identifier or 0),self.MOTION_RADAR)

    def _update_observation(self):
        if self.SEGMENTED_MOTION:
            self._segment_motion.pose.update(self.measured_gps_position(),self.measured_compass_angle(),
                self.odometer_values(),self.lidar_values(),self.lidar_rays_angles(),self.semantic_values())
        super()._update_observation()

    def _navigation(self):
        if not self.SEGMENTED_MOTION: return super()._navigation()
        motion = self._segment_motion
        if motion.state in motion.PAUSED and motion.pose.step-motion.since<=55:
            # A planned pause must not itself retire the mission as a stall.
            # New-wall checks, mission age and all original safety logic remain.
            self._slow_ticks = 0
            self._waypoint_progress.last_advance = self._steps
            if self._steps % 60 == 0: self._motion_anchor = None
        original = super()._navigation()
        target_distance = 160.
        if self._route and self._position is not None:
            target_distance = float(np.linalg.norm(self._explore.point(self._route[0])-self._position))
        near_bomb = (self._bomb_target is not None and self._position is not None and
                     np.linalg.norm(self._bomb_target-self._position)<40)
        allow = (self.drone_health>8 and self._progress()<.60 and not self._exploration_submitted
                 and self._attempt_step is None and not near_bomb and not self._gathering
                 and self._steps>=self._escape_until and self._steps>=self._yield_until
                 and self._position is not None and bool(self._route))
        command = motion.step(self.lidar_values(),self.lidar_rays_angles(),self._last_motion_angle,
                              target_distance,self.semantic_values(),allow)
        if command is not None: original.update(command)
        return original


class MyDroneRedStopTurnRadar(MyDroneRedStopTurn):
    MOTION_RADAR = True


class MyDroneRedStopTurnOff(MyDroneRedStopTurn):
    SEGMENTED_MOTION = False
