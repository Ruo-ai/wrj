"""Instance-local observed-route return with an explicit delivery-time reserve.

Inputs are estimated poses, odometry, current lidar/semantics, and radio peer
reports supplied by the controller. No map files, truth, simulator, or team
globals. A breadcrumb is evidence of this drone's past travel, not permission to
cross an obstacle now: every commanded segment is rechecked in the current scan.
"""
from collections import Counter, deque
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_rn7 import rotate
from swarm_rescue.solutions.my_drone_red_stopturn import body_motion


class ReturnPlanner28:
    SPACING = 12.
    LOOP_RADIUS = 24.
    REACHED = 10.
    HOME_RADIUS = 70.
    LOOKAHEAD = 140.
    MAX_NODES = 4096
    DELIVERY_RESERVE = 540
    ROUTE_FACTOR = 1.55
    MAX_SPEED = 3.2

    def __init__(self, identifier=0, local_factory=None):
        self.identifier = int(identifier)
        if local_factory is None:
            from swarm_rescue.solutions.red_passage28 import HullPassage28
            local_factory = HullPassage28
        self.local_factory = local_factory
        self.trail = []
        self.trail_steps = []
        self.home = None
        self.position = None
        self.heading = 0.
        self.step = 0
        self.returning = False
        self.arrived = False
        self.cursor = None
        self.stats = Counter()
        self.events = deque(maxlen=96)
        self.last_target = None
        self._target_index = None
        self.last_reason = 'exploring'
        self.last_budget = {}
        self.velocity = np.zeros(2)
        self.valid_odometry = False
        self.angular_increment = 0.
        self._angular_samples = deque(maxlen=3)
        self._angular_hold_until = 0
        self._speed = deque(maxlen=120)
        self._last_seconds = None
        self.seconds_per_step = .04
        self._last_position = None
        self._last_step = None
        self._gap = False
        self._local = None
        self._local_step = -1
        self._repair = []
        self._repair_step = -100
        self._best_remaining = float('inf')
        self._last_advance = 0
        self._blocked_ticks = 0
        self._full_scan = False
        self._recovery_since = None
        self._recovery_cooldown = 0

    def set_home(self, point):
        value = np.asarray(point, float)
        if value.shape != (2,) or not np.isfinite(value).all():
            return False
        self.home = value.copy()
        return True

    def _event(self, reason, **fields):
        self.last_reason = reason
        self.stats[reason] += 1
        self.events.append(dict(step=self.step, reason=reason, **fields))

    def _scan(self, lidar, angles, semantic):
        if self._local_step == self.step:
            return self._local
        self._local_step = self.step
        self._local = None
        self._full_scan = False
        if lidar is None or angles is None:
            return None
        d, a = np.asarray(lidar, float).ravel(), np.asarray(angles, float).ravel()
        if d.shape != a.shape or len(d) < 24 or not np.isfinite(a).all() or np.count_nonzero(np.isfinite(d) & (d > 3)) < 12:
            return None
        self._full_scan = bool(np.isfinite(d).all() and (d > 3).all())
        self._local = self.local_factory(d, a, semantic or ())
        return self._local

    def observe(self, step, position, heading, odometry, lidar, angles, semantic=(), elapsed_seconds=None):
        """Record own travel; retain no drone objects or received global state."""
        if type(step) is not int or step <= self.step:
            self.stats['nonincreasing_step_ignored'] += 1
            return
        self.step = step
        point = np.asarray(position, float) if position is not None else None
        if point is None or point.shape != (2,) or not np.isfinite(point).all() or heading is None or not math.isfinite(heading):
            self.valid_odometry = False
            self.stats['invalid_pose'] += 1
            return
        self.position, self.heading = point.copy(), float(heading)
        measured = body_motion(odometry)
        self.valid_odometry = measured is not None
        if measured is not None:
            self.angular_increment = float(odometry[2])
            self._angular_samples.append(self.angular_increment)
            same_direction=(len(self._angular_samples)==3 and
                (all(v>0 for v in self._angular_samples) or all(v<0 for v in self._angular_samples)))
            if abs(self.angular_increment)>.08 or (same_direction and abs(float(np.mean(self._angular_samples)))>.02):
                self._angular_hold_until=step+3
            self.velocity = .4 * rotate(self.velocity, -self.angular_increment) + .6 * measured
            speed = float(np.linalg.norm(measured))
            if .4 < speed < 12:
                self._speed.append(speed)
        if elapsed_seconds is not None and math.isfinite(elapsed_seconds):
            if self._last_seconds is not None and self._last_step is not None:
                dt = (float(elapsed_seconds)-self._last_seconds)/max(1, step-self._last_step)
                if .001 < dt < 1.:
                    self.seconds_per_step = .95*self.seconds_per_step + .05*dt
            self._last_seconds = float(elapsed_seconds)
        if self._last_position is not None:
            move = float(np.linalg.norm(point-self._last_position))
            travel = float(abs(odometry[0])) if measured is not None else 0.
            if move > max(32., 4*travel+12) and step-self._last_step <= 2:
                self._gap = True
                self._event('pose_discontinuity', distance=move)
        self._last_position, self._last_step = point.copy(), step
        if self.returning:
            return
        if not self.trail:
            self.trail = [point.copy()]
            self.trail_steps = [step]
            return
        # Erase a travelled loop only through a connector verified in this scan.
        if len(self.trail) > 5 and (step % 6 == 0 or self._gap):
            old = np.asarray(self.trail[:-4])
            candidates = np.flatnonzero(np.linalg.norm(old-point, axis=1) <= self.LOOP_RADIUS)
            if len(candidates):
                fine = self._scan(lidar, angles, semantic)
                for index in candidates:
                    target = rotate(old[index]-point, -self.heading)
                    if fine is not None and fine.line_clear(np.zeros(2), target):
                        erased = len(self.trail)-int(index)-1
                        self.trail = self.trail[:int(index)+1]
                        self.trail_steps = self.trail_steps[:int(index)+1]
                        self.stats['loop_nodes_erased'] += erased
                        self._gap = False
                        break
        if self._gap:
            self.stats['unlinked_pose_samples'] += 1
            return
        if np.linalg.norm(point-self.trail[-1]) >= self.SPACING:
            if len(self.trail) >= self.MAX_NODES:
                self._event('trail_capacity')
                return
            self.trail.append(point.copy())
            self.trail_steps.append(step)

    def remaining_path_length(self):
        if not self.trail or self.position is None:
            return 0.
        last = len(self.trail)-1 if self.cursor is None else self.cursor
        points = np.asarray(self.trail[:last+1])
        length = float(np.linalg.norm(self.position-points[-1]))
        if len(points)>1:
            length += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if self.home is not None:
            length += max(0., float(np.linalg.norm(points[0]-self.home))-self.HOME_RADIUS)
        return length

    def should_return(self, remaining_steps, remaining_seconds, inventory):
        """Take the tighter native budget; return is a strategy, not a guarantee."""
        if self.returning:
            return True
        length = self.remaining_path_length()
        speed = min(2.5, max(.8, float(np.percentile(self._speed, 25)))) if self._speed else 1.8
        route_steps = length/speed*self.ROUTE_FACTOR
        reserve = self.DELIVERY_RESERVE + 90*max(0,int(inventory))
        effective = float('inf')
        if remaining_steps is not None:
            effective = min(effective, max(0.,float(remaining_steps)))
        if remaining_seconds is not None:
            effective = min(effective, max(0.,float(remaining_seconds))/max(.01,self.seconds_per_step))
        required = route_steps+reserve
        self.last_budget = dict(path_length=length, conservative_speed=speed, estimated_route_steps=route_steps,
                                delivery_and_inventory_reserve=reserve, required_steps=required,
                                effective_remaining_steps=effective, seconds_per_step=self.seconds_per_step)
        due = bool(self.trail and (effective <= required or self._gap or len(self.trail)>=self.MAX_NODES))
        if due:
            self.start_return('budget_due' if not self._gap else 'pose_gap_return')
        return due

    def start_return(self, reason='controller_requested'):
        if not self.returning:
            self.returning = True
            self.cursor = max(0,len(self.trail)-1)
            self._last_advance = self.step
            self._event(reason, nodes=len(self.trail), length=self.remaining_path_length())

    def _brake(self, reason):
        self.last_reason = reason
        self.stats[reason] += 1
        force = np.clip(-.65*self.velocity,-1,1)
        return dict(forward=float(force[0]), lateral=float(force[1]), rotation=0.)

    def _home_relay(self, peers):
        if self.home is None or self.position is None:
            return False
        for source, value in (peers or {}).items():
            if int(source)==self.identifier or not isinstance(value,(tuple,list)) or len(value)<2:
                continue
            seq, point = value[:2]
            point=np.asarray(point,float)
            if (type(seq) is int and 0<=self.step-seq<=8 and point.shape==(2,) and np.isfinite(point).all()
                    and np.linalg.norm(point-self.home)<=70 and np.linalg.norm(point-self.position)<=130):
                self.stats['fresh_near_home_relay'] += 1
                return True
        return False

    def _clearance_recovery(self, fine):
        """Retreat from an initially over-conservative/overlapping footprint.

        Keep the SAME hull, physical margin and raster allowance. Every swept
        footprint may intersect only a shrinking subset of the initially unsafe
        pixels. No newly occupied unsafe pixel is permitted. This is a bounded
        low-speed escape, not an assertion that the initial pose was safe.
        """
        if not self._full_scan or not all(hasattr(fine,k) for k in ('footprint','free','static','cell')):
            return None
        if self.step < self._recovery_cooldown:
            return self._brake('clearance_recovery_cooldown')
        if np.linalg.norm(self.velocity)>1.:
            return self._brake('clearance_recovery_settle')
        if self._recovery_since is None:self._recovery_since=self.step
        if self.step-self._recovery_since>=60:
            self._recovery_since=None;self._recovery_cooldown=self.step+12
            return self._brake('clearance_recovery_budget')
        known=fine.free & ~fine.static
        kernel=fine.footprint().astype(bool)
        offsets=np.argwhere(kernel)-np.array(kernel.shape)//2
        origin=np.asarray(fine.cell(np.zeros(2)))
        initial=origin+offsets
        if np.any(initial<0) or np.any(initial>=np.array(known.shape)):
            return None
        initial_bad=~known[initial[:,0],initial[:,1]]
        count=int(initial_bad.sum())
        if not count:return None
        unsafe=np.zeros_like(known)
        unsafe[initial[initial_bad,0],initial[initial_bad,1]]=True

        def admissible(vector, strict=False):
            previous=count
            for p in np.linspace(np.zeros(2),vector,max(2,int(np.linalg.norm(vector)/1.)+1))[1:]:
                cells=np.asarray(fine.cell(p))+offsets
                if np.any(cells<0) or np.any(cells>=np.array(known.shape)):return False,previous
                y,x=cells[:,0],cells[:,1];bad=~known[y,x]
                current=int(bad.sum())
                if np.any(bad & ~unsafe[y,x]) or current>previous:return False,current
                previous=current
            return (previous<count if strict else True),previous

        choices=[]
        for angle in np.linspace(-math.pi,math.pi,24,endpoint=False):
            direction=np.array([math.cos(angle),math.sin(angle)])
            allowed,remaining=admissible(direction*10,True)
            if allowed:choices.append((count-remaining,angle,direction))
        for _,_,direction in sorted(choices,key=lambda value:(-value[0],value[1])):
            desired=direction*.8
            force=.085*desired+.65*(desired-self.velocity)
            force/=max(1.,float(np.linalg.norm(force)))
            predicted=.95*self.velocity+.6*force
            stopping=predicted*3+predicted*np.linalg.norm(predicted)/1.2
            safe,_=admissible(stopping)
            if safe:
                self.stats['monotone_clearance_recovery']+=1
                self.last_reason='monotone_clearance_recovery'
                return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)
        return None

    def command(self, position, heading, velocity, lidar, angles, semantic=(), peers=None):
        if not self.returning:
            return None
        if position is None or heading is None or not np.isfinite(position).all() or not math.isfinite(heading):
            return self._brake('return_pose_unavailable')
        position=np.asarray(position,float)
        if position.shape!=(2,) or not self.trail:
            return self._brake('return_no_recorded_route')
        self.position=position.copy();self.heading=float(heading)
        if not self.valid_odometry:
            candidate=np.asarray(velocity,float) if velocity is not None else None
            if candidate is not None and candidate.shape==(2,) and np.isfinite(candidate).all():self.velocity=candidate.copy()
            return self._brake('return_odometry_unavailable')
        if self.step < self._angular_hold_until:
            return self._brake('return_wait_fixed_heading')
        if self.home is not None and np.linalg.norm(position-self.home)<=self.HOME_RADIUS:
            self.arrived=True
            return self._brake('at_rendezvous')
        if self._home_relay(peers):
            return self._brake('at_near_home_relay')
        self.arrived=False
        fine=self._scan(lidar,angles,semantic)
        if fine is None:
            return self._brake('return_scan_unavailable')
        if any(getattr(getattr(obj,'entity_type',None),'name','')=='DRONE' and obj.distance<42 for obj in semantic or ()):
            return self._brake('return_peer_clearance')
        if not fine.line_clear(np.zeros(2),np.zeros(2)):
            recovered=self._clearance_recovery(fine)
            return recovered if recovered is not None else self._brake('origin_clearance_no_safe_retreat')
        self._recovery_since=None
        if (self._target_index is not None and self._target_index<=self.cursor and
                np.linalg.norm(position-self.trail[self._target_index])<self.REACHED):
            self.cursor=max(0,self._target_index-1)
            self._target_index=None
            self._last_advance=self.step
            self._repair=[]
        while self.cursor>0 and np.linalg.norm(position-self.trail[self.cursor])<self.REACHED:
            self.cursor-=1
            self._last_advance=self.step
            self._repair=[]
        target=None;chosen=None
        # Shortcuts are current observed-free segments, never global straight lines.
        for index in range(max(0,self.cursor-12),self.cursor+1):
            body=rotate(self.trail[index]-position,-heading)
            if np.linalg.norm(body)<=self.LOOKAHEAD and fine.line_clear(np.zeros(2),body):
                target=body;chosen=index;break
        if self.cursor==0 and self.home is not None:
            body=rotate(self.home-position,-heading)
            if np.linalg.norm(body)<=self.LOOKAHEAD and fine.line_clear(np.zeros(2),body):
                target=body;chosen=0
        if target is None:
            self._blocked_ticks+=1
            if self.step-self._repair_step>=12:
                self._repair_step=self.step
                for index in range(max(0,self.cursor-6),self.cursor+1):
                    body=rotate(self.trail[index]-position,-heading)
                    if not 15<np.linalg.norm(body)<=160:continue
                    path=fine.route(np.zeros(2),body,None,target=body,limit=165)
                    if path:
                        self._repair=[position+rotate(p,heading) for p in path[1:]]
                        self.stats['observed_local_repair']+=1
                        break
            while self._repair and np.linalg.norm(self._repair[0]-position)<8:self._repair.pop(0)
            if self._repair:
                body=rotate(self._repair[0]-position,-heading)
                if fine.line_clear(np.zeros(2),body):target=body
            if target is None:
                return self._brake('recorded_route_currently_blocked')
        else:
            self._blocked_ticks=0
            self._repair=[]
            if chosen is not None and np.linalg.norm(target)<self.REACHED:
                self.cursor=max(0,chosen-1)
                self._last_advance=self.step
                return self._brake('return_waypoint_reached')
        self.last_target=(position+rotate(target,heading)).tolist()
        self._target_index=chosen
        distance=float(np.linalg.norm(target))
        if distance<1:return self._brake('return_zero_target')
        clearance=float(fine.sample(np.linspace(np.zeros(2),target,8)).min())
        speed=min(self.MAX_SPEED,max(.7,.28*(clearance-fine.TRACK_RADIUS)),math.sqrt(max(0.,.8*(distance-4))))
        desired=target/distance*speed
        force=.085*desired+.65*(desired-self.velocity)
        force/=max(1.,float(np.linalg.norm(force)))
        predicted=.95*self.velocity+.6*force
        stopping=predicted*3+predicted*np.linalg.norm(predicted)/1.2
        if not fine.line_clear(np.zeros(2),stopping):
            return self._brake('return_inertial_clearance')
        self.last_reason='following_observed_return_route'
        self.stats['return_motion_ticks']+=1
        return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)

    def status(self):
        return dict(returning=self.returning,arrived=self.arrived,step=self.step,trail_nodes=len(self.trail),
            cursor=self.cursor,home=None if self.home is None else self.home.tolist(),
            remaining_path_length=self.remaining_path_length(),last_target=self.last_target,
            last_reason=self.last_reason,pose_gap=self._gap,budget=dict(self.last_budget),stats=dict(self.stats))
