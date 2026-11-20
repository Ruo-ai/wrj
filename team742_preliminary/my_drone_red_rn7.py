"""RN7: observed-space narrow detours for retained A3 / S6-B / S6-C.

Only the local motion executor changes. No true pose, map identity, hidden
geometry, command integration, shared localization or submitted-map edits.
The N0 free-fan/path primitive is reused with a CURRENT BODY-FRAME snapshot,
corrected hull clearance, bounded tasks and no accumulation of drifting scans.
"""
from collections import Counter, deque
import math
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_red_advanced5_narrow import NarrowLocalMap
from swarm_rescue.solutions.my_drone_red_mapping import MyDroneRedMapping
from swarm_rescue.solutions.my_drone_red_stopturn import (
    MyDroneRedStopTurn, MyDroneRedStopTurnRadar, body_motion)


HULL = np.array([[-13,-10],[-11,-12],[5,-12],[12,-11],
                 [12,11],[10,13],[-10,13],[-13,10]],float)


def rotate(points, angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.asarray(points) @ np.array([[c,s],[-s,c]])


class LocalPassage(NarrowLocalMap):
    CELL=2.
    BODY_RADIUS=float(np.linalg.norm(HULL,axis=1).max())
    TRACK_RADIUS=18.5
    PLAN_RADIUS=TRACK_RADIUS+CELL/math.sqrt(2)

    def __init__(self,distances,angles,objects=()):
        super().__init__((440,440))
        d=np.asarray(distances,float).ravel(); a=np.asarray(angles,float).ravel()
        dynamic=np.zeros(len(d),bool)
        for obj in objects or ():
            if getattr(obj.entity_type,'name','') in ('DRONE','BOMB'):
                dynamic |= (np.abs((a-obj.angle+math.pi)%(2*math.pi)-math.pi)<.19)&(np.abs(d-obj.distance)<25)
        # A new snapshot on every refresh: no long-term drifting free union.
        self.observe(np.zeros(2),0.,d,a,dynamic)
        self.static=(self.hits>0)

    @property
    def squeeze(self):
        if self._squeeze is None:
            # Require actual static endpoints on both sides, not unknown edges.
            yy,xx=np.mgrid[-29:30,-29:30];xx=xx*self.CELL;yy=yy*self.CELL
            walls=self.static.astype(np.uint8); both=np.zeros(self.shape,bool)
            for angle in np.arange(8)*math.pi/8:
                across=xx*math.cos(angle)+yy*math.sin(angle)
                along=-xx*math.sin(angle)+yy*math.cos(angle)
                kernel=((across>=18)&(across<=56)&(np.abs(along)<=8)).astype(np.uint8)
                both |= (cv2.dilate(walls,kernel)>0)&(cv2.dilate(walls,kernel[::-1,::-1])>0)
            self._squeeze=both&(self.clearance>=self.PLAN_RADIUS)&(self.clearance<43)
        return self._squeeze


class CoarseView:
    """Only visits are consulted; global free/obstacle evidence is not rewritten."""
    def __init__(self,coarse,position,heading):
        self.coarse=coarse;self.position=np.asarray(position);self.heading=heading
        self.visits=coarse.visits
    def cell(self,point):
        return self.coarse.cell(self.position+rotate(point,self.heading))


class PassageMotion:
    def __init__(self,identifier=0):
        self.identifier=identifier;self.path=[];self.velocity=np.zeros(2)
        self.tick=0;self.since=0;self.cooldown=75+identifier*11
        self.stats=Counter();self.events=deque(maxlen=160)
        self.snapshot=None;self.last_refresh=-99;self.last_progress=0
        self.best_remaining=float('inf');self.wait=0;self.episodes=0
        self.valid_odom=False
        self.repair_tick=-99;self.blocked_ticks=0

    def event(self,reason):
        self.stats[reason]+=1
        self.events.append(dict(tick=self.tick,reason=reason))

    def update(self,odometry):
        self.tick+=1;measured=body_motion(odometry)
        self.valid_odom=measured is not None
        if measured is None:
            if self.path:self.release('invalid_odometry')
            return
        distance,alpha,theta=map(float,odometry)
        self.velocity=.4*rotate(self.velocity,-theta)+.6*measured
        if self.path:
            shift=distance*np.array([math.cos(alpha),math.sin(alpha)])
            self.path=[rotate(p-shift,-theta) for p in self.path]

    def release(self,reason):
        self.event(reason);self.path=[];self.snapshot=None
        self.cooldown=self.tick+90+self.identifier*7;self.wait=0

    def brake(self):
        force=np.clip(-.65*self.velocity,-1,1)
        return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)

    def command(self,d,a,coarse,position,heading,desired,objects=(),allow=True):
        if not allow or not self.valid_odom or d is None or a is None:
            if self.path:self.release('mission_preempted')
            return None
        if not self.path:
            if self.tick<self.cooldown or self.episodes>=8:return None
            self.cooldown=self.tick+40+self.identifier*3
            fine=LocalPassage(d,a,objects);self.stats['probes']+=1
            preferred=np.array([math.cos(desired),math.sin(desired)])
            path=fine.route(np.zeros(2),preferred,CoarseView(coarse,position,heading),limit=165)
            if not path:
                self.stats['reject_'+fine.last_reason]+=1;return None
            # The inherited bounded search returns only swept, observed-free paths.
            self.path=path[1:];self.since=self.tick;self.last_progress=self.tick
            self.best_remaining=float('inf');self.episodes+=1
            self.blocked_ticks=0
            self.snapshot=fine;self.last_refresh=self.tick;self.event('entered')
        if self.tick-self.since>170:
            self.release('task_timeout');return self.brake()
        while self.path and np.linalg.norm(self.path[0])<9:self.path.pop(0)
        if not self.path:
            self.release('completed');return self.brake()
        remaining=float(sum(np.linalg.norm(v) for v in np.diff([np.zeros(2),*self.path],axis=0)))
        if remaining<self.best_remaining-6:
            self.best_remaining=remaining;self.last_progress=self.tick
        if self.tick-self.last_progress>35:
            self.release('no_progress');return self.brake()
        peers=[o for o in objects or () if getattr(o.entity_type,'name','')=='DRONE'
               and o.distance<60 and abs((o.angle-math.atan2(self.path[0][1],self.path[0][0])+math.pi)%(2*math.pi)-math.pi)<.8]
        if peers:
            self.wait+=1;self.stats['peer_brake_ticks']+=1
            if self.wait>7+3*self.identifier:self.release('peer_wait_budget')
            return self.brake()
        self.wait=0
        # Rebuild from fresh legal rays. A changed wall/peer cannot be made free
        # by an old snapshot; invalidate rather than push through unknown space.
        fine=LocalPassage(d,a,objects)
        target=self.path[0].copy()
        if not fine.line_clear(np.zeros(2),target):
            self.blocked_ticks+=1
            # A noisy scan or odometric lateral slip should first replan the
            # local path, not discard the parent exploration/bomb mission.
            if self.tick-self.repair_tick>=6:
                self.repair_tick=self.tick
                repaired=[]
                for endpoint in reversed(self.path):
                    if np.linalg.norm(endpoint)<25:continue
                    # Recenter only this local waypoint within observed space.
                    # This repairs lateral odometry drift, not global pose/map.
                    normal=np.array([-endpoint[1],endpoint[0]])/np.linalg.norm(endpoint)
                    choices=endpoint+np.arange(-12.,13.,2.)[:,None]*normal
                    clear=fine.sample(choices)
                    ix=int(np.argmax(clear-.04*np.abs(np.arange(-12.,13.,2.))))
                    if clear[ix]<fine.PLAN_RADIUS:continue
                    endpoint=choices[ix]
                    repaired=fine.route(np.zeros(2),endpoint,
                        CoarseView(coarse,position,heading),target=endpoint,limit=165)
                    break
                if repaired:
                    self.path=repaired[1:];self.blocked_ticks=0
                    self.event('local_replan')
            if self.blocked_ticks>=12:self.release('fresh_scan_obstructed')
            return self.brake()
        self.blocked_ticks=0
        # Short string pulling preserves corners; all segments swept checked.
        for point in self.path[1:4]:
            if np.linalg.norm(point)>36 or not fine.line_clear(np.zeros(2),point):break
            target=point.copy()
        distance=float(np.linalg.norm(target));direction=target/max(distance,1.)
        clearance=float(fine.sample(np.linspace(np.zeros(2),target,8)).min())
        speed=min(3.4,max(1.1,.32*(clearance-self.TRACK_RADIUS)),math.sqrt(max(0.,.8*(distance-4))))
        goal=direction*speed
        force=.085*goal+.65*(goal-self.velocity)
        force/=max(1.,float(np.linalg.norm(force)))
        # Fresh scan also checks measured inertial drift under this next command.
        predicted=.95*self.velocity+.6*force
        if not fine.line_clear(np.zeros(2),predicted*3):
            self.release('inertial_risk');return self.brake()
        self.stats['active_ticks']+=1
        return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)

    TRACK_RADIUS=LocalPassage.TRACK_RADIUS


class NarrowMixin:
    NARROW_ENABLED=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._rn7=PassageMotion(int(self.identifier or 0))

    def _update_observation(self):
        self._rn7.update(self.odometer_values())
        super()._update_observation()

    def _navigation(self):
        narrow=self._rn7
        if narrow.path:
            self._slow_ticks=0;self._waypoint_progress.last_advance=self._steps
            if self._steps%60==0:self._motion_anchor=None
        original=super()._navigation()
        if not self.NARROW_ENABLED:return original
        near_bomb=(self._bomb_target is not None and self._position is not None
                   and np.linalg.norm(self._bomb_target-self._position)<55)
        allow=(self._position is not None and self.drone_health>10
               and self._progress()<.68 and not self._gathering
               and not self._exploration_submitted and self._attempt_step is None
               and not near_bomb and self._steps>=self._yield_until)
        desired=self._last_motion_angle if self._last_motion_angle is not None else 0.
        command=narrow.command(self.lidar_values(),self.lidar_rays_angles(),self._explore,
                               self._position,self._heading,desired,self.semantic_values(),allow)
        if command is not None:
            motion=getattr(self,'_segment_motion',None)
            if motion is not None and motion.state not in ('NORMAL','COOLDOWN'):
                motion.release('rn7_local_passage')
            original.update(command)
        return original


class MyDroneRedRN7A(NarrowMixin,MyDroneRedMapping):pass
class MyDroneRedRN7B(NarrowMixin,MyDroneRedStopTurn):pass
class MyDroneRedRN7C(NarrowMixin,MyDroneRedStopTurnRadar):pass
class MyDroneRedRN7Off(MyDroneRedRN7A):NARROW_ENABLED=False
