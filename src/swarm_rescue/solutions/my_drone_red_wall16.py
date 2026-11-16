"""RL16-A: finite single-wall, measured-angle entry alignment on RL11.

No map identity/truth, no world-axis assumptions, no navigation pose correction.
Only a bounded entry maneuver changes; original S6/RN7/mapper/bombs survive.
"""
from collections import Counter,deque
import math
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_red_lines11 import MyDroneRedLines11,fit_lines,line_angle
from swarm_rescue.solutions.my_drone_red_stopturn import SegmentMotion,ray_clearance
from swarm_rescue.solutions.my_drone_red_rn7 import HULL,rotate
from swarm_rescue.solutions.my_drone_red_advanced5_narrow import NarrowLocalMap
from swarm_rescue.solutions.semifinal_localization import wrap


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def supported_sweep(distances,angles,delta,yaw):
    """Conservative convex enclosure of full native hull translation/rotation.

    Fresh 2px observed fan (including its 3px endpoint retreat), 3-degree/1px
    samples plus raster/angular padding. Convex enclosure can reject safe turns;
    it cannot justify routing across an unobserved corner. Noise is not bounded.
    """
    if distances is None or angles is None:return False
    d=np.asarray(distances,float).ravel();a=np.asarray(angles,float).ravel()
    if d.shape!=a.shape or len(d)<24 or not np.isfinite(d).all():return False
    f=NarrowLocalMap((440,440));f.CELL=2.;f.__init__((440,440))
    f.observe(np.zeros(2),0.,d,a,np.zeros(len(d),bool))
    delta=np.asarray(delta,float);yaw=wrap(float(yaw))
    count=max(1,int(math.ceil(np.linalg.norm(delta))),int(math.ceil(abs(yaw)/math.radians(3))))
    points=np.vstack([rotate(HULL,yaw*t)+delta*t for t in np.linspace(0,1,count+1)])
    if np.any(np.abs(points)>=210):return False
    polygon=cv2.convexHull(f.indices(points));mask=np.zeros(f.shape,np.uint8)
    cv2.fillConvexPoly(mask,polygon,1)
    # 6px conservative bound: 2px extra + 1.414 raster + <=.43 angular + .5 path.
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    return bool(f.free[mask.astype(bool)].all())


def match_wall(reference,lines):
    """Finite overlap and normal consistency; reject ambiguous associations."""
    if reference is None:return None
    v=reference[1]-reference[0];t=v/max(np.linalg.norm(v),1.)
    n=np.array([-t[1],t[0]]);center=reference.mean(0);angle=math.atan2(t[1],t[0]);choices=[]
    for wall in lines:
        vector=wall[1]-wall[0];length=float(np.linalg.norm(vector))
        if length<55:continue
        da=float(line_angle(math.atan2(vector[1],vector[0])-angle))
        normal=abs(float((wall.mean(0)-center)@n))
        overlap=min(float(np.max(wall@t)),float(np.max(reference@t)))-max(float(np.min(wall@t)),float(np.min(reference@t)))
        if abs(da)<.085 and normal<7 and overlap>30:
            choices.append((normal+40*abs(da),wall))
    choices.sort(key=lambda x:x[0])
    if not choices or len(choices)>1 and choices[1][0]-choices[0][0]<1.5:return None
    return choices[0][1].copy()


def choose_wall(lines,desired,distances,angles):
    candidates=[]
    for wall in lines:
        v=wall[1]-wall[0];length=float(np.linalg.norm(v))
        if length<65:continue
        axis=math.atan2(v[1],v[0])
        for k in range(4):
            direction=wrap(axis+k*math.pi/2)
            if abs(wrap(direction-desired))>.9:continue
            end=42*np.array([math.cos(direction),math.sin(direction)])
            if ray_clearance(distances,angles,direction)<50:continue
            if not supported_sweep(distances,angles,np.zeros(2),direction):continue
            # Body rotates first, then translates; check the complete convex
            # enclosure too, deliberately conservative near corners.
            if not supported_sweep(distances,angles,end,direction):continue
            score=2*math.cos(direction-desired)+.003*min(length,140)-.002*np.linalg.norm(wall.mean(0))
            candidates.append((score,dict(wall=wall.copy(),direction=direction,
                relation=float(line_angle(axis-direction)),length=length)))
    if not candidates:return None
    candidates.sort(key=lambda x:x[0],reverse=True)
    if len(candidates)>1 and candidates[0][0]-candidates[1][0]<.05:
        # Equivalent directions on a continuous wall are harmless; competing
        # directions at a corner are not silently declared unique.
        if abs(wrap(candidates[0][1]['direction']-candidates[1][1]['direction']))>.15:return None
    return candidates[0][1]


class SingleWallMotion(SegmentMotion):
    ANGLE_TOLERANCE=math.radians(3)
    MAX_ENTRY_TICKS=125
    def __init__(self,size,identifier=0):
        super().__init__(size,identifier,False)
        self.align_state='IDLE';self.align_since=0;self.entry_tick=0;self.next_entry=0
        self.wall=None;self.remaining=0.;self.relation=0.;self.stable=0;self.retests=0
        self.last_anchor=None;self.wall_anchor=None;self.wall_stats=Counter();self.wall_events=deque(maxlen=1800)
        self.last_step=0;self.good_wall_step=-99;self.task_angle=None;self.external_busy=False
        self.travel=0.;self.last_relative_error=None;self.outage_serial=0;self.attempted=False

    def note(self,event,**kw):
        self.wall_stats[event]+=1
        self.wall_events.append(dict(step=self.pose.step,event=event,phase=self.align_state,
            speed=float(np.linalg.norm(self.pose.velocity)),increment=float(self.pose.angular_increment),
            relative_error=self.last_relative_error,**plain(kw)))

    def phase(self,state,reason):
        self.align_state=state;self.align_since=self.pose.step;self.stable=0;self.note(reason)

    def fallback(self,reason):
        self.note('fallback_'+reason);self.align_state='IDLE';self.wall=None
        self.next_entry=self.pose.step+180
        # The original RL11 S6 controller resumes; no navigation/map pose edits.
        self.state='NORMAL'

    def release(self,reason,cooldown=60):
        if self.align_state!='IDLE':self.fallback(reason)
        super().release(reason,cooldown)

    def step(self,d,a,desired,target_distance=160.,objects=(),allow=True):
        p=self.pose;tick=p.step
        if self.last_step!=tick:
            if self.wall is not None and p.odometry_valid:
                self.wall=rotate(self.wall,-p.angular_increment)-p.velocity
                self.remaining=wrap(self.remaining-p.angular_increment)
            self.last_step=tick
        if all(p.absolute) and p.good>=8:
            self.attempted=False
            if tick%4==0 and p.reliable is not None:
                self.last_anchor=dict(p.reliable,lines=[v.tolist() for v in fit_lines(d,a,objects)])
            if self.align_state!='IDLE':self.fallback('absolute_recovered')
        desired=self.task_angle if self.task_angle is not None else desired
        if self.align_state=='IDLE':
            if not (allow and not self.external_busy and not self.attempted and p.missing>=3
                    and p.reliable is not None and tick>=self.next_entry):
                return super().step(d,a,desired,target_distance,objects,allow)
            self.attempted=True;self.outage_serial+=1;self.entry_tick=tick
            self.wall_anchor=self.last_anchor;self.retests=0
            self.phase('BRAKE','triggered');self.note('entry_anchor',anchor=self.wall_anchor)
        if not allow or self.external_busy:
            self.fallback('task_preempted');return super().step(d,a,desired,target_distance,objects,allow)
        if not p.odometry_valid or d is None or desired is None or not np.isfinite(desired):
            self.fallback('missing_observation');return self.brake()
        if tick-self.entry_tick>self.MAX_ENTRY_TICKS:
            self.fallback('entry_timeout');return super().step(d,a,desired,target_distance,objects,allow)
        if all(p.absolute):return self.brake()  # confirm partial/brief recovery
        if any(getattr(o.entity_type,'name','')=='DRONE' and o.distance<55 for o in objects or ()):
            self.fallback('dynamic_peer');return self.brake()
        speed=float(np.linalg.norm(p.velocity));lines=fit_lines(d,a,objects)
        self.state='OBSERVE'  # original pause bookkeeping preserves the mission
        self.since=tick
        if self.align_state=='BRAKE':
            self.stable=self.stable+1 if speed<.45 and abs(p.angular_increment)<.045 else 0
            if self.stable>=3:self.phase('SELECT','stopped')
            elif tick-self.align_since>28:self.fallback('cannot_stop')
            return self.brake()
        if self.align_state=='SELECT':
            selected=choose_wall(lines,desired,d,a)
            if selected is None:
                if tick-self.align_since>10:self.fallback('short_ambiguous_or_no_space')
                return self.brake()
            matched=match_wall(self.wall,[selected['wall']])
            self.stable=self.stable+1 if matched is not None else 1
            self.wall=selected['wall'];self.remaining=selected['direction'];self.relation=selected['relation']
            if self.stable>=3:
                self.note('wall_selected',wall=self.wall.tolist(),target_angle=self.remaining,relation=self.relation,
                    task_angle=float(desired),wall_length=selected['length'])
                self.good_wall_step=tick;self.phase('TURN','turn_started')
            return self.brake()
        matched=match_wall(self.wall,lines)
        if matched is not None:
            self.wall=matched;self.good_wall_step=tick
            v=matched[1]-matched[0]
            self.last_relative_error=float(line_angle(math.atan2(v[1],v[0])-self.relation))
        if tick-self.good_wall_step>6:
            self.fallback('wall_lost_or_ambiguous');return self.brake()
        if self.align_state=='TURN':
            error=self.remaining
            if matched is not None and abs(error)<.25 and abs(self.last_relative_error-error)<.12:
                error=self.last_relative_error
            if speed>.8 or not supported_sweep(d,a,p.velocity*3,error):
                self.fallback('rotation_or_slip_space');return self.brake()
            if abs(error)<self.ANGLE_TOLERANCE and abs(p.angular_increment)<.045:
                self.phase('VERIFY','turn_stopped');return self.brake()
            cmd=self.brake();cmd['rotation']=float(np.clip(error/.18*.55,-.45,.45))
            self.note('turn_control',error=float(error),command=cmd.copy());return cmd
        if self.align_state=='VERIFY':
            ok=matched is not None and abs(self.last_relative_error)<self.ANGLE_TOLERANCE and speed<.55
            self.stable=self.stable+1 if ok else 0
            self.note('relative_retest',ok=bool(ok))
            if self.stable>=3:
                self.phase('MOVE','alignment_verified');self.travel=0.;self.remaining=0.
            elif tick-self.align_since>=7:
                if self.retests>=2:self.fallback('retest_failed')
                else:
                    self.retests+=1;self.remaining=self.last_relative_error or 0.;self.phase('TURN','small_correction')
            return self.brake()
        if self.align_state=='MOVE':
            self.travel+=float(np.linalg.norm(p.velocity))
            if self.travel>=36 or tick-self.align_since>28:
                self.note('movement_complete',distance=self.travel);self.fallback('entry_maneuver_complete')
                return super().step(d,a,desired,target_distance,objects,allow)
            error=self.last_relative_error if matched is not None else self.remaining
            if abs(error)>.12 or abs(wrap(desired-error))>.65:
                self.fallback('task_changed_or_angle_drift');return self.brake()
            goal=3.*np.array([math.cos(error),math.sin(error)])
            force=.085*goal+.65*(goal-p.velocity);force/=max(1.,float(np.linalg.norm(force)))
            stopping=self.stopping_distance();end=(p.velocity/max(speed,1.))*stopping if speed>1 else goal*4
            if ray_clearance(d,a,error)<stopping+10 or not supported_sweep(d,a,end,0):
                self.fallback('translation_brake_space');return self.brake()
            cmd=dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)
            self.note('aligned_motion',distance=self.travel,command=cmd.copy());return cmd
        return self.brake()


class MyDroneRedWall16(MyDroneRedLines11):
    SINGLE_WALL_ALIGN=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.SINGLE_WALL_ALIGN:self._segment_motion=SingleWallMotion(self.size_area,int(self.identifier or 0))

    def _navigation(self):
        if not self.SINGLE_WALL_ALIGN:return super()._navigation()
        m=self._segment_motion;m.external_busy=bool(self._rn7.path);m.task_angle=None
        if self._route and self._position is not None:
            delta=self._explore.point(self._route[0])-self._position
            if np.linalg.norm(delta)>22:m.task_angle=wrap(math.atan2(delta[1],delta[0])-self._heading)
        # Do not let RN7 silently overwrite a stopped alignment command. Its
        # existing active task preempts alignment; new RN7 tasks wait <=125 ticks.
        saved=self.NARROW_ENABLED
        if not m.external_busy and (m.align_state!='IDLE' or (m.pose.missing>=3 and not m.attempted)):
            self.NARROW_ENABLED=False
        try:return super()._navigation()
        finally:self.NARROW_ENABLED=saved


class MyDroneRedWall16Off(MyDroneRedWall16):SINGLE_WALL_ALIGN=False
