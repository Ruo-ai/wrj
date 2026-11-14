"""N8 body-frame configuration-space local planning, no hidden geometry.

Raw observed free space is eroded ONCE by the native collision hull plus a
2--4px margin. No old 18.5/22/25px circle is applied to this result. The body
frame makes current physical orientation explicit even with no compass.
"""
from functools import lru_cache
import heapq
import math
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_red_advanced5_narrow import NarrowLocalMap
from swarm_rescue.solutions.my_drone_red_advanced2 import _M03_fit_scan, _SF_LIDAR_RANGE, _SF_HIT_LIMIT
from swarm_rescue.solutions.my_drone_red_rn7 import HULL,PassageMotion,CoarseView,rotate


@lru_cache(maxsize=6)
def hull_kernel(margin=3.,cell=2.):
    if not 2<=margin<=4:raise ValueError('Additional physical margin must be 2..4px')
    radius=int(math.ceil((np.linalg.norm(HULL,axis=1).max()+margin+cell/math.sqrt(2))/cell))
    kernel=np.zeros((2*radius+1,2*radius+1),np.uint8)
    polygon=HULL.astype(np.float32)
    for r in range(-radius,radius+1):
        for c in range(-radius,radius+1):
            # Half-cell diagonal is a raster bound, not a second body radius.
            if cv2.pointPolygonTest(polygon,(c*cell,-r*cell),True)>=-(margin+cell/math.sqrt(2)):
                kernel[r+radius,c+radius]=1
    return kernel


class FootprintMap(NarrowLocalMap):
    CELL=2.
    PLAN_RADIUS=.5  # path helper threshold for residual C-space clearance
    TRACK_RADIUS=0. # hull is already included, never subtract it a second time
    def __init__(self,distances,angles,objects=(),margin=3.):
        super().__init__((440,440));self.margin=margin
        d=np.asarray(distances,float).ravel();a=np.asarray(angles,float).ravel()
        dynamic=np.zeros(len(d),bool)
        for obj in objects or ():
            if getattr(obj.entity_type,'name','') in ('DRONE','BOMB') and np.isfinite([obj.angle,obj.distance]).all():
                dynamic|=(np.abs((a-obj.angle+math.pi)%(2*math.pi)-math.pi)<.19)&(np.abs(d-obj.distance)<25)
        # Existing traditional scan fitting only refines contiguous observed
        # straight segments; no extrapolation across doors/excluded objects.
        raw_d=d.copy();raw_a=a.copy()
        fitted_d,fitted_a,_=_M03_fit_scan(d,a,dynamic)
        fitted=fitted_d[:,None]*np.column_stack((np.cos(fitted_a),np.sin(fitted_a)))
        # Fitted point bearings are NOT physical beam bearings. Joining their
        # perturbed angles can leave fictitious unknown wedges through our own
        # body. Intersect a supported fitted segment with the ORIGINAL beam.
        if len(d)>=3:
            chord=fitted[2:]-fitted[:-2];span=np.linalg.norm(chord,axis=1)
            normal=np.column_stack((-chord[:,1],chord[:,0]))/np.maximum(span,1.)[:,None]
            middle=fitted[1:-1];residual=np.abs(np.sum((middle-fitted[:-2])*normal,axis=1))
            beam=np.column_stack((np.cos(raw_a[1:-1]),np.sin(raw_a[1:-1])))
            denom=np.sum(normal*beam,axis=1)
            with np.errstate(divide='ignore',invalid='ignore'):
                corrected=np.sum(normal*middle,axis=1)/denom
            good=(span>4)&(span<30)&(residual<.35)&(np.abs(denom)>.1)&np.isfinite(corrected)
            good&=(corrected>4)&(np.abs(corrected-raw_d[1:-1])<7.5)&~dynamic[1:-1]
            d[1:-1]=np.where(good,corrected,raw_d[1:-1])
        a=raw_a
        valid=np.isfinite(d)&np.isfinite(a)&(d>3)
        raw=np.clip(np.where(valid,d,0.),0,_SF_LIDAR_RANGE)
        safe=np.minimum(np.minimum(raw,np.roll(raw,1)),np.roll(raw,-1))
        points=safe[:,None]*np.column_stack((np.cos(a),np.sin(a)))
        xy=self.indices(points);origin=self.indices(np.zeros((1,2)))[0]
        seen=np.zeros(self.shape,np.uint8)
        # No additional 3px inward endpoint shift: margin is in the hull kernel.
        if valid.all() and np.max(np.abs(np.diff(a)))<.12 and abs(a[-1]-a[0])>6:
            cv2.fillPoly(seen,[xy],1)
        else:
            for i in range(len(d)-1):
                if valid[i] and valid[i+1] and 0<abs(a[i+1]-a[i])<.12:
                    cv2.fillConvexPoly(seen,np.array([origin,xy[i],xy[i+1]]),1)
        static=valid&~dynamic&(d<_SF_HIT_LIMIT)
        end=d[static,None]*np.column_stack((np.cos(a[static]),np.sin(a[static])))
        ix=self.indices(end)
        if len(ix):
            x,y=ix[:,0],ix[:,1];ok=(x>=0)&(y>=0)&(x<self.shape[1])&(y<self.shape[0])
            self.hits[y[ok],x[ok]]=1;seen[y[ok],x[ok]]=0
        self.free=seen.astype(bool)
        self.configuration=cv2.erode(seen,hull_kernel(margin,self.CELL),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        self.wall_clearance=cv2.distanceTransform(np.pad(seen,1),cv2.DIST_L2,5)[1:-1,1:-1]*self.CELL
        self._clearance=cv2.distanceTransform(np.pad(self.configuration.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]*self.CELL

    @property
    def squeeze(self):
        if self._squeeze is None:
            yy,xx=np.mgrid[-28:29,-28:29];xx=xx*self.CELL;yy=yy*self.CELL
            both=np.zeros(self.shape,bool);walls=(self.hits>0).astype(np.uint8)
            for angle in np.arange(8)*math.pi/8:
                across=xx*math.cos(angle)+yy*math.sin(angle);along=-xx*math.sin(angle)+yy*math.cos(angle)
                kernel=((across>=12)&(across<=54)&(np.abs(along)<=8)).astype(np.uint8)
                both|=(cv2.dilate(walls,kernel)>0)&(cv2.dilate(walls,kernel[::-1,::-1])>0)
            self._squeeze=both&self.configuration&(self.wall_clearance<45)
        return self._squeeze

    def route(self,position,preferred,coarse,avoid=(),target=None,limit=165):
        start=self.cell(position);clear=self.clearance
        if clear[start]<self.PLAN_RADIUS:self.last_reason='start_footprint';return []
        direction=np.asarray(preferred,float);direction/=max(1.,float(np.linalg.norm(direction)))
        squeeze=self.squeeze
        if target is None and not squeeze.any():self.last_reason='no_two_sided_passage';return []
        target_cell=None if target is None else self.cell(target)
        parents={start:None};cost={start:0.};passed={start:False};queue=[(0.,start)]
        best=None;end=None;visited=0
        while queue and visited<6000:
            distance,cell=heapq.heappop(queue)
            if distance!=cost[cell]:continue
            visited+=1;point=self.point(cell);net=float(np.linalg.norm(point-position))
            if target_cell is not None and cell==target_cell:end=cell;break
            if target is None and passed[cell] and 65<net<=limit and distance<limit*1.5:
                count=int(coarse.visits[coarse.cell(point)])
                if count<2 and not any(np.linalg.norm(point-p)<45 for p in avoid):
                    score=float((point-position)@direction)*.35+min(12.,clear[cell])*.7+net*.08-distance*.07-count*5
                    if best is None or score>best[0]:best=(score,cell)
            if distance>=limit*1.5:continue
            r,c=cell
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                nxt=(r+dr,c+dc)
                if not(0<=nxt[0]<self.shape[0] and 0<=nxt[1]<self.shape[1]):continue
                if clear[nxt]<self.PLAN_RADIUS:continue
                if dr and dc and min(clear[r+dr,c],clear[r,c+dc])<self.PLAN_RADIUS:continue
                value=distance+self.CELL*(math.sqrt(2) if dr and dc else 1.)*(1+.12*max(0.,4.-clear[nxt]))
                if value<cost.get(nxt,float('inf')):
                    parents[nxt]=cell;cost[nxt]=value;passed[nxt]=passed[cell] or (distance>20 and squeeze[nxt])
                    heapq.heappush(queue,(value,nxt))
        if end is None and best is not None:end=best[1]
        if end is None:self.last_reason='no_observed_route';return []
        cells=[end]
        while parents[cells[-1]] is not None:cells.append(parents[cells[-1]])
        cells.reverse();path=[np.asarray(position).copy()]+[self.point(c) for c in cells[1:]]
        result=[path[0]];i=0
        while i<len(path)-1:
            j=min(len(path)-1,i+15)
            while j>i+1 and not self.line_clear(path[i],path[j]):j-=1
            if not self.line_clear(path[i],path[j]):self.last_reason='swept_footprint';return []
            result.append(path[j]);i=j
        self.last_reason='passable';return result


class FootprintMotion(PassageMotion):
    SAFETY_MARGIN=3.
    REFRESH_BUDGET=True
    MAX_TASK_TICKS=240
    def __init__(self,identifier=0):
        super().__init__(identifier);self.entries=[];self.active_budget=0
        self.last_theta=0.;self.factory=FootprintMap
        self.axis=np.array([1.,0.]);self.continuations=0

    def update(self,odometry):
        super().update(odometry)
        self.last_theta=float(odometry[2]) if self.valid_odom else 0.
        if self.valid_odom:self.axis=rotate(self.axis,-self.last_theta)

    def budget_available(self):
        self.entries=[t for t in self.entries if self.tick-t<600]
        if not self.REFRESH_BUDGET:return self.episodes<8
        return len(self.entries)<8 and self.episodes<32 and self.active_budget<1400

    def command(self,d,a,coarse,position,heading,desired,objects=(),allow=True):
        if not allow or not self.valid_odom or d is None or a is None:
            if self.path:self.release('mission_preempted')
            return None
        if not self.path:
            if self.tick<self.cooldown:return None
            if not self.budget_available():
                self.stats['budget_paused_ticks']+=1;return None
            self.cooldown=self.tick+40+self.identifier*3
            fine=self.factory(d,a,objects,self.SAFETY_MARGIN);self.stats['probes']+=1
            path=fine.route(np.zeros(2),[math.cos(desired),math.sin(desired)],CoarseView(coarse,position,heading))
            if not path:self.stats['reject_'+fine.last_reason]+=1;return None
            self.path=path[1:];self.since=self.tick;self.last_progress=self.tick
            self.best_remaining=float('inf');self.episodes+=1;self.entries.append(self.tick)
            self.axis=path[-1]/max(1.,float(np.linalg.norm(path[-1])))
            self.continuations=0
            self.blocked_ticks=0;self.event('entered')
        self.active_budget+=1
        if self.tick-self.since>self.MAX_TASK_TICKS or self.active_budget>1400:
            self.release('task_or_active_budget');return self.brake()
        while self.path and np.linalg.norm(self.path[0])<7:self.path.pop(0)
        if not self.path:
            # Reaching a short local waypoint is not proof we have exited the
            # passage. Continue the same bounded task while two walls remain;
            # handing a still-enclosed drone back to the 22/25px primitive can
            # otherwise trap it for the whole cooldown.
            fine=self.factory(d,a,objects,self.SAFETY_MARGIN)
            near=[fine.cell(p) for p in (np.zeros(2),self.axis*16)]
            path=[]
            if self.continuations<2 and self.tick-self.since<self.MAX_TASK_TICKS-40 and any(fine.squeeze[c] for c in near):
                path=fine.route(np.zeros(2),self.axis,CoarseView(coarse,position,heading))
            if path:
                self.path=path[1:];self.continuations+=1;self.best_remaining=float('inf')
                self.last_progress=self.tick;self.event('continue_inside_passage')
            else:self.release('completed');return self.brake()
        remaining=float(sum(np.linalg.norm(v) for v in np.diff([np.zeros(2),*self.path],axis=0)))
        if remaining<self.best_remaining-6:self.best_remaining=remaining;self.last_progress=self.tick
        if self.tick-self.last_progress>35:self.release('no_progress');return self.brake()
        target=self.path[0].copy();target_angle=math.atan2(target[1],target[0])
        peers=[o for o in objects or () if getattr(o.entity_type,'name','')=='DRONE' and o.distance<60
               and abs((o.angle-target_angle+math.pi)%(2*math.pi)-math.pi)<.8]
        if peers:
            self.wait+=1;self.stats['peer_brake_ticks']+=1
            if self.wait>7+3*self.identifier:self.release('peer_wait_budget')
            return self.brake()
        self.wait=0;fine=self.factory(d,a,objects,self.SAFETY_MARGIN)
        # Rotation is not commanded in tight space. If an observed rotation is
        # substantial, first brake instead of assuming the hull stayed fixed.
        if abs(self.last_theta)>.07:self.stats['angular_brake_ticks']+=1;return self.brake()
        if not fine.line_clear(np.zeros(2),target):
            self.blocked_ticks+=1
            if self.tick-self.repair_tick>=6:
                self.repair_tick=self.tick
                for endpoint in reversed(self.path):
                    length=float(np.linalg.norm(endpoint))
                    if length<25:continue
                    normal=np.array([-endpoint[1],endpoint[0]])/length
                    offsets=np.arange(-12.,13.,2.);choices=endpoint+offsets[:,None]*normal
                    clear=fine.sample(choices);ix=int(np.argmax(clear-.04*np.abs(offsets)))
                    if clear[ix]<fine.PLAN_RADIUS:continue
                    repair=fine.route(np.zeros(2),endpoint,CoarseView(coarse,position,heading),target=choices[ix])
                    if repair:self.path=repair[1:];self.blocked_ticks=0;self.event('local_replan')
                    break
            if self.blocked_ticks>=12:self.release('fresh_scan_obstructed')
            return self.brake()
        self.blocked_ticks=0
        for point in self.path[1:4]:
            if np.linalg.norm(point)>32 or not fine.line_clear(np.zeros(2),point):break
            target=point.copy()
        distance=float(np.linalg.norm(target));direction=target/max(distance,1.)
        slack=float(fine.sample(np.linspace(np.zeros(2),target,8)).min())
        speed=min(3.4,1.1+.3*slack,math.sqrt(max(0.,.8*(distance-3))))
        goal=direction*speed;force=.085*goal+.65*(goal-self.velocity)
        force/=max(1.,float(np.linalg.norm(force)))
        predicted=.95*self.velocity+.6*force
        # Measured speed determines the next-step + stopping sweep. This is a
        # measured-motion safety check, never position integration of commands.
        stopping=np.linalg.norm(predicted)/.6+1.
        end=predicted*max(2.,stopping/2)
        if not fine.line_clear(np.zeros(2),end):self.stats['inertial_brake_ticks']+=1;return self.brake()
        self.stats['active_ticks']+=1
        return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)
