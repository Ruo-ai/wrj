"""Observed return with deployment priority and bounded bomb-occlusion memory.

No engine objects, truth, map files or team-global state. The only remembered
free space is one recent unobscured legal scan. It can fill explicitly identified
BOMB-occluded bearings, never new wall/peer observations or arbitrary missing rays.
"""
from collections import deque
import math

import cv2
import numpy as np

from swarm_rescue.solutions.red_return28 import ReturnPlanner28
from swarm_rescue.solutions.my_drone_red_rn7 import rotate


class ReturnPlanner29(ReturnPlanner28):
    CACHE_STEPS = 80
    CACHE_TRAVEL = 55.
    CACHE_ANGLE = .30

    def __init__(self, identifier=0, local_factory=None):
        super().__init__(identifier, local_factory)
        self.return_requested = False
        self._inventory29 = None
        self._clear_frame29 = None
        self._cache_shift29 = np.zeros(2)
        self._cache_angle29 = 0.
        self._cache_used29 = False
        self._turn_window29 = deque(maxlen=6)
        self._clock_window29 = deque(maxlen=241)

    def request_return(self, reason='requested'):
        if not self.return_requested:
            self.return_requested = True
            self._event('return_requested', request_reason=str(reason), inventory=self._inventory29)

    def start_return(self, reason='controller_requested'):
        self.request_return(reason)
        if self._inventory29 is not None and self._inventory29 > 0:
            self.stats['return_deferred_for_inventory'] += 1
            return
        super().start_return(reason)

    def should_return(self, remaining_steps, remaining_seconds, inventory):
        self._inventory29 = max(0, int(inventory))
        super().should_return(remaining_steps, remaining_seconds, inventory)
        return self.returning

    @staticmethod
    def _bomb_rays(d, a, semantic):
        matched = np.zeros(len(d), bool)
        groups={'BOMB':[], 'DRONE':[]}
        for obj in semantic or ():
            kind=getattr(getattr(obj,'entity_type',None),'name','')
            if kind in groups and np.isfinite([obj.distance,obj.angle]).all() and (kind!='BOMB' or obj.distance<18):
                groups[kind].append((obj.distance,obj.angle))
        for kind in ('BOMB','DRONE'):
            if not groups[kind]:continue
            values=np.asarray(groups[kind],float)
            delta=np.abs((a[:,None]-values[None,:,1]+math.pi)%(2*math.pi)-math.pi)
            limit=12 if kind=='BOMB' else 25
            hits=np.any((delta<.19)&(np.abs(d[:,None]-values[None,:,0])<=limit),axis=1)&np.isfinite(d)
            if kind=='BOMB':matched|=hits
            else:matched&=~hits
        return matched

    @staticmethod
    def _cache_affine(fine, shift, angle):
        """Exact rigid pixel-centre transform; never fit rounded cell indices."""
        c,s=math.cos(angle),math.sin(angle)
        matrix=np.array([[c,-s],[s,c]])
        center=np.array([fine.width/(2*fine.CELL)-.5,fine.height/(2*fine.CELL)-.5])
        translation=rotate(-np.asarray(shift,float),-angle)*np.array([1.,-1.])/fine.CELL
        return np.column_stack((matrix,center-matrix@center+translation))

    def observe(self, step, position, heading, odometry, lidar, angles, semantic=(), elapsed_seconds=None):
        old_step = self.step
        if self._clear_frame29 is not None and step > old_step:
            if odometry is None or np.shape(odometry) != (3,) or not np.isfinite(odometry).all() or step-old_step != 1:
                self._clear_frame29 = None
            else:
                distance, alpha, theta = map(float, odometry)
                shift = distance*np.array([math.cos(alpha), math.sin(alpha)])
                self._cache_shift29 += rotate(shift, self._cache_angle29)
                self._cache_angle29 += theta
        super().observe(step, position, heading, odometry, lidar, angles, semantic, elapsed_seconds)
        if step <= old_step:
            return
        # A robust window of the actual native wall-clock cost avoids a single
        # expensive planner call permanently latching an early return request.
        if elapsed_seconds is not None and math.isfinite(elapsed_seconds):
            self._clock_window29.append((step, float(elapsed_seconds)))
            first = self._clock_window29[0]
            if step-first[0] >= 60:
                self.seconds_per_step = max(.01, (float(elapsed_seconds)-first[1])/(step-first[0]))
        # Both estimated heading change and sustained odometry must support a
        # turn. The heading filter also uses odometry, so these are not assumed
        # statistically independent. One-frame odometer noise is insufficient.
        self._angular_hold_until = 0
        if self.valid_odometry and heading is not None and math.isfinite(heading):
            self._turn_window29.append((float(odometry[2]), float(heading)))
            if len(self._turn_window29) == 6:
                rotations = np.array([row[0] for row in self._turn_window29])
                heading_change = (heading-self._turn_window29[0][1]+math.pi) % (2*math.pi)-math.pi
                if abs(rotations.sum()) > .18 and abs(heading_change) > .08 and rotations.sum()*heading_change > 0:
                    self._angular_hold_until = step+1
        else:
            self._turn_window29.clear()
        if lidar is None or angles is None:
            return
        d, a = np.asarray(lidar, float).ravel(), np.asarray(angles, float).ravel()
        if d.shape != a.shape or len(d) < 24 or not np.isfinite(a).all():
            return
        # Copying an unobscured raw frame is cheap; build its raster lazily only
        # if a later placed bomb actually blinds a sensor sector.
        clean = np.isfinite(d).all() and (d > 3).all() and not self._bomb_rays(d, a, semantic).any()
        if clean:
            self._clear_frame29 = dict(step=step, ranges=d.copy(), angles=a.copy(), semantic=tuple(semantic or ()), fine=None)
            self._cache_shift29 = np.zeros(2)
            self._cache_angle29 = 0.

    def _scan(self, lidar, angles, semantic):
        if self._local_step == self.step:
            return self._local
        self._local_step = self.step
        self._local = None
        self._cache_used29 = False
        self._full_scan = False
        if lidar is None or angles is None:
            return None
        d, a = np.asarray(lidar, float).ravel(), np.asarray(angles, float).ravel()
        if d.shape != a.shape or len(d) < 24 or not np.isfinite(a).all():
            return None
        bomb = self._bomb_rays(d, a, semantic)
        good = np.isfinite(d) & (d > 3)
        memory = self._clear_frame29
        usable = (memory is not None and bomb.any() and self.step-memory['step'] <= self.CACHE_STEPS
                  and np.linalg.norm(self._cache_shift29) <= self.CACHE_TRAVEL
                  and abs(self._cache_angle29) <= self.CACHE_ANGLE)
        if good.sum() < 12 and not usable:
            return None
        fine = self.local_factory(d, a, semantic or ())
        self._full_scan = bool(good.all())
        if usable and all(hasattr(fine, key) for key in ('free', 'static', 'indices', 'CELL')):
            if memory['fine'] is None:
                memory['fine'] = self.local_factory(memory['ranges'], memory['angles'], memory['semantic'])
            old = memory['fine']
            # Transform original evidence once; never accumulate a drifting OR.
            transform = self._cache_affine(fine,self._cache_shift29,self._cache_angle29)
            known = (old.free & ~old.static).astype(np.uint8)
            age = self.step-memory['step']
            uncertainty = .5+.04*age+.08*np.linalg.norm(self._cache_shift29)+20*abs(self._cache_angle29)
            radius = int(math.ceil(uncertainty/fine.CELL))
            if radius:
                yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
                known = cv2.erode(known, ((xx*xx+yy*yy) <= radius*radius).astype(np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
            warped = cv2.warpAffine(known, transform, (fine.shape[1], fine.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            yy, xx = np.indices(fine.shape)
            x = (xx+.5)*fine.CELL-fine.width/2
            y = fine.height/2-(yy+.5)*fine.CELL
            bearing = np.arctan2(y, x)
            # Rays are checked in angular order; unrecognized/missing rays
            # remain closed. Only explicit BOMB sectors can use old evidence.
            order = np.argsort(a); sorted_angles = a[order]
            index = np.searchsorted(sorted_angles, bearing).clip(1, len(a)-1)
            left, right = order[index-1], order[index]
            limits = np.where(bomb, np.inf, np.where(good, np.maximum(0.,d-3), 0.))
            limit = np.minimum(limits[left], limits[right])
            sector = bomb[left] & bomb[right]
            addition = warped & sector & (np.hypot(x,y) <= limit) & ~fine.static
            fine.free |= addition
            fine._clearance = None
            self._cache_used29 = bool(addition.any())
            if self._cache_used29:
                self.stats['bomb_occlusion_memory_frames'] += 1
        self._local = fine
        return fine

    def _clearance_recovery(self, fine):
        if self._full_scan:
            return super()._clearance_recovery(fine)
        # A validated recent scan may support retreat from its conservative
        # margin even though the current bomb-obscured scan is incomplete.
        # Never relabel the current sensor frame as complete.
        if not self._cache_used29 or not all(hasattr(fine,k) for k in ('footprint','free','static','cell')):
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
        origin=np.asarray(fine.cell(np.zeros(2)));initial=origin+offsets
        if np.any(initial<0) or np.any(initial>=np.array(known.shape)):return None
        initial_bad=~known[initial[:,0],initial[:,1]];count=int(initial_bad.sum())
        if not count:return None
        unsafe=np.zeros_like(known);unsafe[initial[initial_bad,0],initial[initial_bad,1]]=True
        def admissible(vector,strict=False):
            previous=count
            for point in np.linspace(np.zeros(2),vector,max(2,int(np.linalg.norm(vector))+1))[1:]:
                cells=np.asarray(fine.cell(point))+offsets
                if np.any(cells<0) or np.any(cells>=np.array(known.shape)):return False,previous
                yy,xx=cells[:,0],cells[:,1];bad=~known[yy,xx];current=int(bad.sum())
                if np.any(bad & ~unsafe[yy,xx]) or current>previous:return False,current
                previous=current
            return (previous<count if strict else True),previous
        choices=[]
        for angle in np.linspace(-math.pi,math.pi,24,endpoint=False):
            direction=np.array([math.cos(angle),math.sin(angle)])
            allowed,remaining=admissible(direction*10,True)
            if allowed:choices.append((count-remaining,angle,direction))
        for _,_,direction in sorted(choices,key=lambda value:(-value[0],value[1])):
            desired=direction*.8;force=.085*desired+.65*(desired-self.velocity)
            force/=max(1.,float(np.linalg.norm(force)))
            predicted=.95*self.velocity+.6*force
            stopping=predicted*3+predicted*np.linalg.norm(predicted)/1.2
            if admissible(stopping)[0]:
                self.stats['bomb_memory_monotone_recovery']+=1
                self.last_reason='bomb_memory_monotone_recovery'
                return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)
        return None

    def _yield_move(self, fine, semantic, peers):
        close = [o for o in semantic or () if getattr(getattr(o, 'entity_type', None), 'name', '') == 'DRONE' and o.distance < 60]
        if not close:
            return None
        nearest = min(close, key=lambda obj: obj.distance)
        direction = np.array([math.cos(nearest.angle), math.sin(nearest.angle)])
        identified = []
        for key, value in (peers or {}).items():
            if int(key) == self.identifier or not isinstance(value, (tuple,list)) or len(value) < 2:
                continue
            seq, point = value[:2]
            if type(seq) is not int or not 0 <= self.step-seq <= 8:
                continue
            delta = rotate(np.asarray(point,float)-self.position, -self.heading)
            if np.shape(delta) == (2,) and np.isfinite(delta).all() and np.linalg.norm(delta) < 90:
                if delta @ direction > 0 and abs(direction[0]*delta[1]-direction[1]*delta[0]) < 30:
                    identified.append(int(key))
        # The lower ID can continue only through fresh hull-feasible space.
        # The higher ID backs/steps aside. This removes the symmetric
        # unconditional stop; blocked geometry may still prevent progress.
        yield_now = bool(identified and self.identifier > min(identified))
        if not yield_now and nearest.distance >= 24:
            return None
        away = -direction
        side = rotate(away, (1 if self.identifier % 2 else -1)*math.pi/2)
        options = [away, .7*away+.7*side, side, .7*away-.7*side]
        for vector in options:
            vector = vector/max(1.,np.linalg.norm(vector))
            target = vector*18
            if target @ direction > .01 or not fine.line_clear(np.zeros(2), target):
                continue
            desired = vector*1.2
            force = .085*desired+.65*(desired-self.velocity)
            force /= max(1.,float(np.linalg.norm(force)))
            predicted = .95*self.velocity+.6*force
            stopping = predicted*3+predicted*np.linalg.norm(predicted)/1.2
            if stopping @ direction > .01 or not fine.line_clear(np.zeros(2), stopping):
                continue
            self.stats['priority_peer_yield_motion'] += 1
            self.last_reason = 'priority_peer_yield_motion'
            return dict(forward=float(force[0]), lateral=float(force[1]), rotation=0.)
        return None

    def command(self, position, heading, velocity, lidar, angles, semantic=(), peers=None):
        if not self.returning:
            return None
        if position is not None and heading is not None and np.shape(position) == (2,) and np.isfinite(position).all() and math.isfinite(heading):
            self.position, self.heading = np.asarray(position,float).copy(), float(heading)
        fine = self._scan(lidar, angles, semantic)
        if fine is not None and self.valid_odometry and self.step >= self._angular_hold_until:
            yielded = self._yield_move(fine, semantic, peers)
            if yielded is not None:
                return yielded
            # Prefer an earlier visited node if the entire connecting segment
            # is visible now; the cached/occluded case stays short and slow.
            if self.cursor is not None and self.step % 6 == 0:
                old = np.asarray(self.trail[:self.cursor+1])
                candidates = np.flatnonzero(np.linalg.norm(old-self.position,axis=1) <= (40 if self._cache_used29 else self.LOOKAHEAD))
                for index in candidates:
                    if index >= max(0,self.cursor-12):break
                    body = rotate(old[index]-self.position, -self.heading)
                    if fine.line_clear(np.zeros(2),body):
                        self.cursor = int(index)+1
                        self.stats['visible_return_shortcut'] += 1
                        break
        # _scan above includes every live DRONE. Remove only the old blanket
        # 42-px stop predicate; geometric paths still include those obstacles.
        without_peer_predicate = [o for o in semantic or () if getattr(getattr(o,'entity_type',None),'name','') != 'DRONE']
        old_speed = self.MAX_SPEED
        if self._cache_used29:self.MAX_SPEED = 1.2
        try:
            return super().command(position,heading,velocity,lidar,angles,without_peer_predicate,peers)
        finally:
            self.MAX_SPEED = old_speed

    def status(self):
        value = super().status()
        value.update(return_requested=self.return_requested, inventory_pending=self._inventory29,
            bomb_occlusion_memory_used=self._cache_used29,
            memory_age=None if self._clear_frame29 is None else self.step-self._clear_frame29['step'],
            memory_displacement=float(np.linalg.norm(self._cache_shift29)))
        return value
