"""B5-A: bounded, empty-only lidar corridor control. No map truth online.

The hull below is the offline-audited stock DroneBase collision polygon, not an
observed map. Coarse planning/inflation and payload policy stay with B3 compute.
This is an experimental selector, never the official entry or B0 default.
"""
from collections import Counter, deque
import math
import numpy as np
from swarm_rescue.solutions.my_drone_blue_transport import MyDroneBlueTransportCompute
from swarm_rescue.solutions.my_drone_eval_blue import wrap


HULL = np.array([[-13.,-10.],[-11.,-12.],[5.,-12.],[12.,-11.],
                 [12.,11.],[10.,13.],[-10.,13.],[-13.,10.]])


def lateral_support(angle):
    y = HULL[:,0]*math.sin(angle)+HULL[:,1]*math.cos(angle)
    return float(y.min()), float(y.max())


def _line(points):
    if len(points)<4 or np.ptp(points[:,0])<22:return None
    # Door end caps must not be fitted as the side wall: bounded RANSAC, with
    # range noise projected onto the wall normal (grazing rays are more precise).
    sample=np.linspace(0,len(points)-1,min(12,len(points))).astype(int)
    pairs=[(i,j) for n,i in enumerate(sample) for j in sample[n+1:] if abs(points[j,0]-points[i,0])>22]
    if not pairs:return None
    ij=np.asarray(pairs);delta=points[ij[:,1]]-points[ij[:,0]]
    slopes=delta[:,1]/delta[:,0];intercepts=points[ij[:,0],1]-slopes*points[ij[:,0],0]
    tolerance=1.2+4*np.abs(points[:,1])/np.maximum(10,np.linalg.norm(points,axis=1))
    residual=np.abs(points[None,:,1]-slopes[:,None]*points[None,:,0]-intercepts[:,None])
    masks=(residual<tolerance)&(np.abs(slopes[:,None])<.6)
    scores=masks.sum(axis=1)-.01*np.minimum(residual,10).sum(axis=1)
    keep=masks[int(np.argmax(scores))]
    for _ in range(3):
        x,y=points[keep].T
        if len(x)<4 or np.ptp(x)<22:return None
        matrix=np.column_stack((x,np.ones(len(x))))
        m,b=np.linalg.lstsq(matrix,y,rcond=None)[0]
        residual=points[:,1]-m*points[:,0]-b
        sigma=max(.3,1.4826*float(np.median(np.abs(residual[keep]-np.median(residual[keep])))))
        keep=np.abs(residual)<tolerance
    if abs(m)>.6 or sigma>3.6:return None
    # Uncertainty of wall intercept, not raw per-ray range sigma.
    uncertainty=max(.2,sigma/math.sqrt(keep.sum())*1.8)
    return float(m),float(b),uncertainty,points[keep]


def corridor_geometry(distances, angles, desired=0.):
    """Fit two locally observed parallel wall segments; no unknown-space route."""
    d,a=np.asarray(distances),np.asarray(angles)
    valid=np.isfinite(d)&np.isfinite(a)&(d>5)&(d<220)
    if valid.sum()<14:return None
    points=np.column_stack((d[valid]*np.cos(a[valid]),d[valid]*np.sin(a[valid])))
    best=None
    # Bounded search around the intended local direction. No global SLAM.
    for axis in (float(np.clip(desired,-.65,.65)),0.):
        c,s=math.cos(axis),math.sin(axis)
        q=points@np.array([[c,-s],[s,c]])
        slab=(q[:,0]>-18)&(q[:,0]<115)&(np.abs(q[:,1])<65)
        left=_line(q[slab&(q[:,1]>7)])
        right=_line(q[slab&(q[:,1]<-7)])
        if left is None or right is None:continue
        ml,bl,ul,pl=left;mr,br,ur,pr=right
        if abs(ml-mr)>.10:continue
        # Refit a SHARED wall direction. Separate slope extrapolation from far
        # door lips can otherwise inflate a 27px opening into a spurious 35px.
        centered=[p-p.mean(axis=0) for p in (pl,pr)]
        m=sum(float(np.dot(p[:,0],p[:,1])) for p in centered)/sum(float(np.dot(p[:,0],p[:,0])) for p in centered)
        scale=math.sqrt(1+m*m)
        bl=float(np.median(pl[:,1]-m*pl[:,0]));br=float(np.median(pr[:,1]-m*pr[:,0]))
        width=(bl-br)/scale
        angle=wrap(axis+math.atan(m))
        if not 24.<width<78. or abs(wrap(angle-desired))>.8:continue
        # Both boundaries must be seen ahead, rather than inferred behind a wall.
        overlap=min(pl[:,0].max(),pr[:,0].max())-max(pl[:,0].min(),pr[:,0].min())
        if overlap<24:continue
        error=(bl+br)/(2*scale)
        geo=dict(width=float(width),angle=angle,center=float(error),uncertainty=ul+ur,
                 start=float(max(pl[:,0].min(),pr[:,0].min())),
                 end=float(min(pl[:,0].max(),pr[:,0].max())),points=int(len(pl)+len(pr)))
        if best is None or geo['uncertainty']<best['uncertainty']:best=geo
    return best


class NarrowPassage:
    """Observe -> align -> transit -> exit; finite abort/cooldown, empty only.

    Odometer supplies velocity and short progress, lidar supplies all geometry.
    A width bound permits examination of +2px passages but does not guarantee
    collision-free execution under noisy sensing. Payload is deliberately gated.
    """
    MAX_ATTEMPTS=12

    def __init__(self):
        self.state='idle';self.until=0;self.started=0;self.age=0
        self.history=deque(maxlen=10);self.stats=Counter();self.reason=None
        self.progress=0.;self.anchor_progress=0.;self.anchor_tick=0
        self.filtered=None;self.lost=0;self.exit_distance=0.;self.last_angle=0.
        self.last_geometry=None;self.pending=False
        self.exit_remaining=None

    def _stop(self,velocity,carrying=False):
        return dict(forward=float(np.clip(-velocity[0],-.5,.5)),
                    lateral=float(np.clip(-velocity[1],-.35,.35)),rotation=0.,grasper=int(carrying))

    def abort(self,tick,reason):
        self.stats['abort_'+reason]+=1;self.reason=reason
        self.state='cooldown';self.until=tick+100;self.history.clear();self.filtered=None

    def step(self,distances,angles,odo,tick,desired=0.,carrying=False,dynamic=False,health=25):
        velocity=np.zeros(2)
        if odo is not None and np.shape(odo)==(3,) and np.isfinite(odo).all():
            velocity=np.array([odo[0]*math.cos(odo[1]-odo[2]),odo[0]*math.sin(odo[1]-odo[2])])
        if carrying:
            if self.state not in ('idle','cooldown'):self.abort(tick,'payload_acquired')
            return None
        if self.state=='cooldown':
            if tick<self.until:return None
            self.state='idle'
        if self.state=='idle' and tick%6:return None
        if self.state=='idle' and abs(desired)>.8:return None
        if self.state=='idle' and self.stats['attempts']>=self.MAX_ATTEMPTS:
            self.stats['attempt_budget_exhausted']=1
            return None
        d,a=np.asarray(distances),np.asarray(angles)
        valid=np.isfinite(d)&np.isfinite(a)&(d>0)
        if valid.sum()<30:
            if self.state!='idle':self.abort(tick,'scan_missing');return self._stop(velocity)
            return None
        geo=corridor_geometry(d,a,desired if self.state=='idle' else self.last_angle)
        self.last_geometry=geo
        if self.state=='idle':
            if geo is None or dynamic or health<8:return None
            # Observation takes over only near a measured opening, not at a remote
            # unknown goal. Keep the normal coarse map and route untouched.
            if geo['start']>72 or abs(geo['angle'])>.55:return None
            self.state='observe';self.started=tick;self.age=0;self.history.clear()
            self.progress=0.;self.anchor_progress=0.;self.anchor_tick=tick
            self.filtered=None;self.lost=0;self.pending=True;self.reason=None
            self.exit_remaining=None
            self.stats['attempts']+=1
        self.age+=1;self.progress+=float(velocity[0])
        if dynamic:
            self.abort(tick,'dynamic_obstacle');return self._stop(velocity)
        if tick-self.started>480:
            self.abort(tick,'time_budget');return self._stop(velocity)
        if health<8:
            self.abort(tick,'health');return self._stop(velocity)
        if abs(self.progress-self.anchor_progress)>6:
            self.anchor_progress=self.progress;self.anchor_tick=tick
        elif tick-self.anchor_tick>100 and self.state not in ('observe','align'):
            self.abort(tick,'no_progress');return self._stop(velocity)
        if self.exit_remaining is not None:self.exit_remaining-=max(0.,float(velocity[0]))
        if geo is None:
            self.lost+=1
            # Loss alone is not proof of exit. Require a forward/side fan clear
            # for the entire body, then continue straight until the tail clears.
            fan=valid&(np.abs(a)<.5)
            open_exit=fan.sum()>10 and np.min(d[fan])>45
            if self.state in ('transit','exit') and open_exit and self.progress>25 and self.exit_remaining is not None:
                if self.state!='exit':self.stats['exit_observed']+=1
                self.state='exit'
                if self.exit_remaining<0:
                    self.stats['passed']+=1;self.state='cooldown';self.until=tick+70;self.pending=False
                    return self._stop(velocity)
                return dict(forward=float(np.clip((.45-velocity[0])+.038,-.5,.35)),
                            lateral=float(np.clip(-velocity[1],-.3,.3)),rotation=0.,grasper=0)
            if self.lost>5:self.abort(tick,'geometry_lost')
            return self._stop(velocity)
        self.lost=0;self.last_angle=geo['angle']
        if self.state=='transit' and -5<geo['end']<32:
            # Last observed wall tip plus rear hull and 10px exit margin. Only a
            # clear current forward fan admits this bounded straight extension.
            self.exit_remaining=geo['end']+23
        if self.state=='observe':
            self.history.append(geo)
            if len(self.history)<10 or np.linalg.norm(velocity)>.18:return self._stop(velocity)
            width=float(np.median([g['width'] for g in self.history]))
            uncertainty=float(np.median([g['uncertainty'] for g in self.history]))/math.sqrt(5)
            if width-uncertainty<26.:
                self.abort(tick,'width_uncertain');return self._stop(velocity)
            self.stats['confirmed']+=1;self.state='align';self.age=0
        angle=geo['angle'];lo,hi=lateral_support(-angle)
        # The real polygon is asymmetric: at 0deg y is [-12,13], so its center
        # must sit half a pixel below the geometric corridor center.
        center=geo['center']-(lo+hi)/2
        if self.filtered is None:self.filtered=np.array([center,angle,geo['width']])
        else:
            # Transform last wall offsets by legal odometry before filtering.
            self.filtered[0]-=velocity[1]
            if odo is not None and np.isfinite(odo).all():self.filtered[1]-=odo[2]
            self.filtered=.78*self.filtered+.22*np.array([center,angle,geo['width']])
        center,angle,width=self.filtered
        frontal=valid&(np.abs(a)<.22)
        front=float(np.min(d[frontal])) if frontal.any() else 0.
        if front<19:
            self.abort(tick,'front_blocked');return self._stop(velocity)
        if self.state=='align':
            # Large rotations only before entry where the measured radial hull
            # fits. Inside the throat, permit only tiny alignment corrections.
            if abs(angle)>.07 and geo['start']<17 and np.min(d[valid])<19:
                self.abort(tick,'no_turn_space');return self._stop(velocity)
            if abs(angle)<.035 and abs(center)<max(.65,(width-25)/5):
                self.state='transit';self.stats['entered']+=1;self.age=0
            elif self.age>100:
                self.abort(tick,'alignment_timeout');return self._stop(velocity)
        # Velocity feedback brakes inertia. Avoid sidestepping/turning sharply
        # in a narrow slot; center correction is bounded and decreases at +2.
        margin=max(0.,width-(hi-lo))
        target_v=0. if self.state=='align' else min(.95,.32+.08*margin)
        target_v*=max(0.,1-abs(angle)/.18)
        if abs(center)>max(.7,margin/3):target_v=min(target_v,.12)
        target_v=min(target_v,max(0.,(front-20)/30))
        vy=float(np.clip(center*.22,-.25,.25))
        return dict(forward=float(np.clip(target_v-velocity[0]+.085*target_v,-.5,.4)),
                    lateral=float(np.clip(vy-velocity[1],-.22,.22)),
                    rotation=float(np.clip(angle*.9,-.12,.12)),grasper=0)


class MyDroneBlueNarrow(MyDroneBlueTransportCompute):
    EMPTY_NARROW=True
    PAYLOAD_NARROW=False
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self._narrow=NarrowPassage();self._narrow_action=None

    def control(self):
        self._narrow_action=None
        command=super().control()
        if self._narrow_action is not None and not self.grasped_bombs():
            # The inherited visible-bomb approach must not replace the already
            # checked narrow motion with a large turn or an unbraked .3 thrust.
            command=dict(self._narrow_action,grasper=command.get('grasper',0))
            self._command_source='narrow_'+self._narrow.state
        return command

    def _navigate(self,goal,distances,angles,carrying=False):
        normal=super()._navigate(goal,distances,angles,carrying)
        self._command_source='navigation'
        if not self.EMPTY_NARROW:return normal
        if carrying:
            if self._narrow.state not in ('idle','cooldown'):self._narrow.abort(self._tick,'payload_acquired')
            return normal
        waypoint=self._route[0] if self._route else goal
        desired=0.
        if waypoint is not None:
            delta=waypoint-self._position
            desired=wrap(math.atan2(delta[1],delta[0])-self._heading)
        dynamic=any(getattr(s.entity_type,'name','')=='DRONE' and s.distance<85
                    and abs(s.angle)<1.2 for s in self.semantic_values() or ())
        action=self._narrow.step(distances,angles,self.odometer_values(),self._tick,
                                desired,carrying,dynamic,self.drone_health)
        if action is not None:
            self._narrow_action=dict(action)
            self._command_source='narrow_'+self._narrow.state
            # Suppress the inherited oscillating stuck recovery only while the
            # bounded local maneuver owns navigation. Preserve the task goal.
            self._escape_until=0;self._anchor_step=self._tick
            return action
        return normal
