"""RL13-E / RL13-S: independent motion experiments on immutable RL11.

E uses continuous measured opposite wall segments to select a body-relative
entry direction. S changes segment length/turn anticipation using measured
odometry. Neither writes a radar relative angle into global navigation pose.
No forced wall symmetry; no map identity, region coordinates or truth input.
"""
from collections import deque
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_lines11 import MyDroneRedLines11, fit_lines, line_angle
from swarm_rescue.solutions.my_drone_red_stopturn import SegmentMotion, ray_clearance, turn_clear
from swarm_rescue.solutions.semifinal_localization import wrap


def corridor_direction(lines,desired):
    """Two finite overlapping opposite walls, close to the requested route.

    Returns a relative angle and center offset, not an absolute compass.
    Single walls, openings, nonparallel walls and unsupported forward extent
    produce no guide. The current vehicle need not be centered.
    """
    choices=[]
    for i,left in enumerate(lines):
        v=left[1]-left[0];length=float(np.linalg.norm(v))
        if length<45:continue
        angle=math.atan2(v[1],v[0]);angle=wrap(desired+float(line_angle(angle-desired)))
        if abs(wrap(angle-desired))>.35:continue
        t=np.array([math.cos(angle),math.sin(angle)]);n=np.array([-t[1],t[0]])
        a=float(left.mean(axis=0)@n)
        for right in lines[i+1:]:
            rv=right[1]-right[0];rl=float(np.linalg.norm(rv))
            if rl<45 or abs(float(line_angle(math.atan2(rv[1],rv[0])-angle)))>.055:continue
            b=float(right.mean(axis=0)@n)
            if a*b>=0 or min(abs(a),abs(b))<24:continue
            width=abs(a-b)
            if not 52<=width<=180:continue
            overlap=min(float(np.max(left@t)),float(np.max(right@t)))-max(0.,float(np.min(left@t)),float(np.min(right@t)))
            if overlap<35:continue
            other=wrap(desired+float(line_angle(math.atan2(rv[1],rv[0])-desired)))
            mean=wrap(angle+.5*wrap(other-angle))
            choices.append((overlap,dict(angle=mean,center=(a+b)/2,width=width,overlap=overlap)))
    return max(choices,key=lambda x:x[0])[1] if choices else None


class EntryMotion(SegmentMotion):
    ENTRY_ALIGN=True
    CENTERING=False  # separately switchable; not enabled without physical evidence.

    def __init__(self,size,identifier=0,radar=False):
        super().__init__(size,identifier,radar)
        self.guide_history=deque(maxlen=3);self.guide=None;self.guide_step=-100
        self.last_wall_anchor=None;self.entry_wall_anchor=None

    def step(self,distances,angles,desired,target_distance=160.,objects=(),allow=True):
        if not self.ENTRY_ALIGN:
            return super().step(distances,angles,desired,target_distance,objects,allow)
        p=self.pose
        if all(p.absolute) and p.good>=8 and p.step%4==0 and p.position is not None:
            self.last_wall_anchor=dict(step=p.step,position=p.position.copy(),heading=p.heading,
                position_sigma=math.sqrt(p.position_variance),heading_sigma=math.sqrt(p.heading_variance),
                lines=fit_lines(distances,angles,objects))
        if p.missing==1:self.entry_wall_anchor=self.last_wall_anchor
        if p.missing==0:self.guide_history.clear();self.guide=None
        candidate=None
        if (allow and desired is not None and np.isfinite(desired) and
                3<=p.missing<=96 and p.step%4==0):
            candidate=corridor_direction(fit_lines(distances,angles,objects),desired)
            if candidate is None:
                self.guide_history.clear();self.guide=None;self.stats['entry_no_pair']+=1
            else:
                candidate['relative_anchor']=wrap(p.heading+candidate['angle'])
                self.guide_history.append(candidate)
                headings=[r['relative_anchor'] for r in self.guide_history]
                if (len(headings)==3 and max(abs(wrap(h-headings[-1])) for h in headings)<.065
                        and np.ptp([r['width'] for r in self.guide_history])<12):
                    self.guide=candidate;self.guide_step=p.step;self.stats['entry_confirmed']+=1
        selected=desired
        if self.guide is not None and p.step-self.guide_step<=8 and 3<=p.missing<=96:
            relative=wrap(self.guide['relative_anchor']-p.heading)
            if (desired is not None and abs(wrap(relative-desired))<=.35
                    and ray_clearance(distances,angles,relative)>self.stopping_distance()+20
                    and turn_clear(distances,angles)):
                selected=relative;self.stats['entry_guided_ticks']+=1
        command=super().step(distances,angles,selected,target_distance,objects,allow)
        if (self.CENTERING and command is not None and self.guide is not None
                and p.step-self.guide_step<=4 and self.state in ('ACCEL','CRUISE')
                and abs(wrap(self.guide['relative_anchor']-p.heading))<.06
                and 3<abs(self.guide['center'])<10 and np.linalg.norm(p.velocity)<1.5):
            side=math.copysign(math.pi/2,self.guide['center'])
            if ray_clearance(distances,angles,side)>self.stopping_distance()+10:
                command['lateral']=float(np.clip(command['lateral']+.025*self.guide['center'],-1,1))
                self.stats['centering_ticks']+=1
        return command


class MeasuredSegmentMotion(SegmentMotion):
    ADAPTIVE_SEGMENTS=True

    def step(self,distances,angles,desired,target_distance=160.,objects=(),allow=True):
        if not self.ADAPTIVE_SEGMENTS:
            return super().step(distances,angles,desired,target_distance,objects,allow)
        if (allow and self.pose.missing>=3 and self.pose.odometry_valid
                and self.state in ('NORMAL','OBSERVE')
                and target_distance<self.stopping_distance()+20):
            # A nearby coarse waypoint cannot support an accelerate/brake
            # segment. Hand it back, instead of repeatedly starting and braking.
            self.release('insufficient_segment_budget',30)
            return None
        previous=self.state
        command=super().step(distances,angles,desired,target_distance,objects,allow)
        if self.state=='ACCEL' and previous!=self.state:
            clearance=ray_clearance(distances,angles,wrap(self.target_heading-self.pose.heading))
            self.SEGMENT_LENGTH=float(np.clip(min(clearance-12,target_distance-8),80,140))
            self.stats['measured_segment_selected']+=1
        if command is not None and self.state=='TURN' and abs(command['rotation'])>0:
            # One-step angular anticipation from the actual noisy increment;
            # integration and residual translation remain in MotionEstimate.
            error=wrap(self.target_heading-self.pose.heading)
            anticipated=error-float(np.clip(self.pose.angular_increment,-.08,.08))*.6
            command['rotation']=float(np.clip(anticipated/.3*.65,-.35,.35))
            self.stats['measured_turn_ticks']+=1
        return command


class MyDroneRedEntry13(MyDroneRedLines11):
    ENTRY_ALIGN=True

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.ENTRY_ALIGN:
            self._segment_motion=EntryMotion(self.size_area,int(self.identifier or 0))


class MyDroneRedSegments13(MyDroneRedLines11):
    ADAPTIVE_SEGMENTS=True

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.ADAPTIVE_SEGMENTS:
            self._segment_motion=MeasuredSegmentMotion(self.size_area,int(self.identifier or 0))


class MyDroneRedEntry13Off(MyDroneRedEntry13):ENTRY_ALIGN=False
class MyDroneRedSegments13Off(MyDroneRedSegments13):ADAPTIVE_SEGMENTS=False
