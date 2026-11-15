"""RL11: measured wall-line registration and independent cautious motion on RN7-B.

No Manhattan/world-axis assumption, true pose, map files or region identities.
Only submission pose changes in the line experiment. Navigation remains RN7-B.
"""
from collections import Counter
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_rn7 import MyDroneRedRN7B
from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper, MappingSnapshots
from swarm_rescue.solutions.my_drone_red_stopturn import available_pose, body_motion
from swarm_rescue.solutions.semifinal_localization import LocalMatcher, transform


def line_angle(value):
    return (np.asarray(value)+math.pi/2)%math.pi-math.pi/2


def fit_lines(distances, angles, objects=()):
    """Split contiguous legal hits, fitting arbitrary oriented finite segments."""
    if distances is None or angles is None:return []
    d=np.asarray(distances,float).ravel();a=np.asarray(angles,float).ravel()
    if d.shape!=a.shape or len(d)<8:return []
    valid=np.isfinite(d)&np.isfinite(a)&(d>8)&(d<215)
    for o in objects or ():
        if np.isfinite([o.distance,o.angle]).all():
            valid &= ~((np.abs((a-o.angle+math.pi)%(2*math.pi)-math.pi)<.18)&(np.abs(d-o.distance)<30))
    points=np.column_stack((d*np.cos(a),d*np.sin(a)));lines=[]
    def split(p,depth=0):
        if len(p)<7:return
        center=p.mean(axis=0);q=p-center
        values,vectors=np.linalg.eigh(q.T@q/len(p));direction=vectors[:,1];normal=vectors[:,0]
        residual=np.abs(q@normal);along=q@direction;span=float(np.ptp(along))
        if residual.max()>5.5 or values[0]>7.:
            if depth>=6 or len(p)<14:return
            cut=int(np.argmax(residual))
            if cut<6 or cut>len(p)-7:cut=len(p)//2
            split(p[:cut+1],depth+1);split(p[cut:],depth+1);return
        if span<45:return
        ends=center+np.array([along.min(),along.max()])[:,None]*direction
        lines.append(ends)
    start=None
    for i in range(len(d)+1):
        ok=i<len(d) and valid[i]
        disconnected=ok and start is not None and np.linalg.norm(points[i]-points[i-1])>28
        if start is not None and (not ok or disconnected):
            split(points[start:i]);start=None
        if ok and start is None:start=i
    return sorted(lines,key=lambda p:np.linalg.norm(p[1]-p[0]),reverse=True)[:16]


def line_registration(lines, references, position, heading):
    """Orientation plus observable wall-normal translation; no corridor-axis fix.

    Matches existing finite segments before any current observation is inserted.
    The eigenspace gate retains odometry in unobservable translation directions.
    """
    if len(lines)<2 or len(references)<2:return None,dict(reason='few_lines')
    refs=np.asarray(references);rc=refs.mean(axis=1);rv=refs[:,1]-refs[:,0]
    lengths=np.linalg.norm(rv,axis=1);t=rv/lengths[:,None];n=np.column_stack((-t[:,1],t[:,0]))
    ra=np.arctan2(t[:,1],t[:,0]);matches={}
    for line in lines:
        world=transform(line,position,heading);center=world.mean(axis=0)
        vector=world[1]-world[0];length=float(np.linalg.norm(vector))
        angle=float(math.atan2(vector[1],vector[0]))
        da=line_angle(ra-angle);normal_distance=(n*(center-rc)).sum(axis=1)
        along=np.abs((t*(center-rc)).sum(axis=1))
        valid=(np.abs(da)<.16)&(np.abs(normal_distance)<26)&(along<(lengths+length)/2+25)
        score=np.where(valid,np.abs(normal_distance)+40*np.abs(da),np.inf)
        index=int(np.argmin(score))
        if not np.isfinite(score[index]):continue
        record=(score[index],line,float(da[index]),min(length,120.))
        if index not in matches or score[index]<matches[index][0]:matches[index]=record
    if len(matches)<2:return None,dict(reason='few_matches',matches=len(matches))
    indexes=list(matches);angles=np.array([matches[i][2] for i in indexes])
    median=float(np.median(angles));keep=np.abs(line_angle(angles-median))<.035
    indexes=[i for i,k in zip(indexes,keep) if k]
    if len(indexes)<2:return None,dict(reason='angular_disagreement')
    weights=np.array([matches[i][3] for i in indexes]);weights/=weights.sum()
    rotation=float(np.sum(weights*np.array([matches[i][2] for i in indexes])))
    if abs(rotation)>.10:return None,dict(reason='angle_bound')
    centers=np.array([transform(matches[i][1],position,heading+rotation).mean(axis=0) for i in indexes])
    normals=n[indexes];residual=np.sum(normals*(centers-rc[indexes]),axis=1)
    gram=(normals.T*weights)@normals;values,vectors=np.linalg.eigh(gram)
    observable=values>max(.04,.12*values.max())
    inverse=(vectors*np.where(observable,1/np.maximum(values,1e-12),0.))@vectors.T
    shift=-inverse@(normals.T@(weights*residual))
    error=float(np.sqrt(np.sum(weights*(residual+normals@shift)**2)))
    if np.linalg.norm(shift)>12 or error>4.5:return None,dict(reason='translation_or_residual_bound',residual=error)
    info=dict(reason='line_registered',matches=len(indexes),rank=int(observable.sum()),
              residual=error,angle_spread=float(np.sqrt(np.sum(weights*line_angle(angles[keep]-rotation)**2))))
    return np.r_[shift,rotation],info


class WallLineMatcher(LocalMatcher):
    MAX_REFERENCES=64
    def __init__(self,size):
        super().__init__(size);self.lines=[];self.references=[];self.absolute=False
        self.stats=Counter();self.last_info={}

    def match(self,points,position,heading):
        offset,info=line_registration(self.lines,self.references,position,heading)
        self.last_info=info;self.stats['attempts']+=1
        if offset is not None:
            self.stats['accepted']+=1;self.stats['rank_'+str(info['rank'])]+=1
            return offset,info
        self.stats['fallback_'+info['reason']]+=1
        return super().match(points,position,heading)

    def add(self,points,position,heading):
        if not self.frames:self.references.clear()
        super().add(points,position,heading)
        for line in self.lines:
            world=transform(line,position,heading);center=world.mean(axis=0)
            vector=world[1]-world[0];length=np.linalg.norm(vector);angle=math.atan2(vector[1],vector[0])
            match=None
            if self.references:
                refs=np.asarray(self.references);rc=refs.mean(axis=1);rv=refs[:,1]-refs[:,0]
                rl=np.linalg.norm(rv,axis=1);t=rv/rl[:,None];n=np.column_stack((-t[:,1],t[:,0]))
                valid=(np.abs(line_angle(np.arctan2(t[:,1],t[:,0])-angle))<.06)
                valid&=np.abs(((center-rc)*n).sum(axis=1))<6
                valid&=np.abs(((center-rc)*t).sum(axis=1))<(rl+length)/2+20
                indexes=np.flatnonzero(valid)
                if len(indexes):match=int(indexes[0])
            if match is None:
                self.references.append(world)
            else:
                # Keep established wall direction/normal during an outage.
                # Extend along the measured segment only, no filled wall pixels.
                ref=self.references.pop(match);rc=ref.mean(axis=0);t=(ref[1]-ref[0]);t/=np.linalg.norm(t)
                if self.absolute:
                    tv=vector/length
                    if tv@t<0:tv=-tv
                    t=.9*t+.1*tv;t/=np.linalg.norm(t);rc=.9*rc+.1*center
                projected=(np.vstack((ref,world))-rc)@t
                self.references.append(rc+np.array([projected.min(),projected.max()])[:,None]*t)
            self.references=self.references[-self.MAX_REFERENCES:]
        self.stats['max_references']=max(self.stats['max_references'],len(self.references))


class WallLineMapper(SubmissionMapper):
    LINE_REGISTRATION=True
    def __init__(self,size):
        super().__init__(size)
        if self.LINE_REGISTRATION:self.matcher=WallLineMatcher(size)

    def update(self,gps,compass,odometry,distances,angles,objects=()):
        if self.LINE_REGISTRATION and (self.step+1)%8==0:
            self.matcher.lines=fit_lines(distances,angles,objects)
            self.matcher.absolute=all(available_pose(gps,compass))
        super().update(gps,compass,odometry,distances,angles,objects)
        if self.LINE_REGISTRATION and self.step%8==0:
            self.stats.update({'line_'+k:int(v) for k,v in self.matcher.stats.items()})


class MyDroneRedLines11(MyDroneRedRN7B):
    LINE_MAPPING=True
    SLOW_OUTAGE=False
    OUTAGE_SPEED=3.2
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._l11_stats=Counter();self._l11_missing=0;self._l11_good=0;self._l11_active=False
        if self.LINE_MAPPING:
            self._submission_mapper=WallLineMapper(self.size_area)
            self._radio_maps=MappingSnapshots(self._map.shape,self._n,self._submission_mapper)

    def _update_observation(self):
        joint=all(available_pose(self.measured_gps_position(),self.measured_compass_angle()))
        self._l11_missing=0 if joint else self._l11_missing+1
        self._l11_good=self._l11_good+1 if joint else 0
        if self._l11_missing>=3:self._l11_active=True
        if self._l11_good>=8:self._l11_active=False
        super()._update_observation()

    def _navigation(self):
        command=super()._navigation()
        if not self.SLOW_OUTAGE or not self._l11_active:return command
        if self._progress()>=.68 or self._gathering or self._attempt_step is not None or self.drone_health<=10:return command
        # The RN7 passage executor already has a lower measured speed target.
        if self._rn7.path:return command
        velocity=body_motion(self.odometer_values())
        if velocity is None:return command
        speed=float(np.linalg.norm(velocity));force=np.array([command['forward'],command['lateral']])
        if abs(command['rotation'])>.02 and speed>.8:
            force=np.clip(-.65*velocity,-1,1);command['rotation']=0.
            self._l11_stats['turn_translation_brake']+=1
        elif speed>self.OUTAGE_SPEED:
            # Subtract only along measured momentum. Never convert a brake to thrust.
            force-=min(1.2,.65*(speed-self.OUTAGE_SPEED))*velocity/speed
            self._l11_stats['speed_brake']+=1
        command['rotation']=float(np.clip(command['rotation'],-.25,.25))
        force/=max(1.,float(np.linalg.norm(force)))
        command.update(forward=float(force[0]),lateral=float(force[1]))
        self._l11_stats['guard_ticks']+=1
        return command


class MyDroneRedLines11Slow(MyDroneRedLines11):
    LINE_MAPPING=False
    SLOW_OUTAGE=True


class MyDroneRedLines11Combined(MyDroneRedLines11):SLOW_OUTAGE=True
class MyDroneRedLines11Off(MyDroneRedLines11):LINE_MAPPING=False
