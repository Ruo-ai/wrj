"""Bounded local correlative matching, inspired by Olson 2009; original code.
Only own past lidar is used. No map files, entities, or shared global state.
"""
import math
from collections import deque
import cv2
import numpy as np
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR


def wrap(x):
    return (float(x)+math.pi)%(2*math.pi)-math.pi


def scan_points(distances, angles, objects=()):
    if distances is None or angles is None:
        return np.empty((0,2))
    d=np.asarray(distances,float).ravel();a=np.asarray(angles,float).ravel()
    if d.shape!=a.shape:
        return np.empty((0,2))
    valid=np.isfinite(d)&np.isfinite(a)&(d>12)&(d<float(MAX_RANGE_LIDAR_SENSOR)-10)
    for o in objects or ():
        if getattr(o.entity_type,'name','') in ('BOMB','DRONE','DISPOSAL_CENTER'):
            if np.isfinite([o.distance,o.angle]).all():
                valid &= ~((np.abs((a-o.angle+math.pi)%(2*math.pi)-math.pi)<.18)&(np.abs(d-o.distance)<35))
    indices=np.flatnonzero(valid)
    if len(indices)>48:
        indices=indices[np.linspace(0,len(indices)-1,48).astype(int)]
    return np.column_stack((d[indices]*np.cos(a[indices]),d[indices]*np.sin(a[indices])))


def transform(points, position, heading):
    c,s=math.cos(heading),math.sin(heading)
    return points @ np.array([[c,s],[-s,c]])+position


class LocalMatcher:
    CELL=4
    def __init__(self,size):
        self.width,self.height=size
        self.shape=(math.ceil(self.height/4),math.ceil(self.width/4))
        self.frames=deque(maxlen=8)
        self.distance=None
        self.last_candidates=0

    def add(self, points, position, heading):
        if len(points)<12:
            return
        self.frames.append(transform(points,position,heading))
        mask=np.ones(self.shape,np.uint8)
        for frame in self.frames:
            pix=np.column_stack(((frame[:,0]+self.width/2)/4,(self.height/2-frame[:,1])/4)).astype(int)
            valid=(pix[:,0]>=0)&(pix[:,0]<self.shape[1])&(pix[:,1]>=0)&(pix[:,1]<self.shape[0])
            for i,(p,ok) in enumerate(zip(pix,valid)):
                if ok:
                    mask[p[1],p[0]]=0
                if i and ok and valid[i-1] and np.linalg.norm(frame[i]-frame[i-1])<28:
                    cv2.line(mask,tuple(pix[i-1]),tuple(p),0,1)
        self.distance=cv2.distanceTransform(mask,cv2.DIST_L2,3)*4

    def costs(self,points,position,heading,offsets):
        angles=heading+offsets[:,2]
        c,s=np.cos(angles)[:,None],np.sin(angles)[:,None]
        x=points[None,:,0]*c-points[None,:,1]*s+position[0]+offsets[:,0,None]
        y=points[None,:,0]*s+points[None,:,1]*c+position[1]+offsets[:,1,None]
        cols=np.floor((x+self.width/2)/4).astype(int);rows=np.floor((self.height/2-y)/4).astype(int)
        good=(cols>=0)&(cols<self.shape[1])&(rows>=0)&(rows<self.shape[0])
        d=self.distance[np.clip(rows,0,self.shape[0]-1),np.clip(cols,0,self.shape[1]-1)]
        d=np.where(good,d,24.)
        return np.minimum(d,24).mean(axis=1),(d<9).mean(axis=1)

    def match(self,points,position,heading):
        self.last_candidates=0
        if self.distance is None or len(points)<12:
            return None,dict(reason='insufficient_support',candidates=0)
        coarse=np.array([(x,y,t) for x in (-12,-6,0,6,12) for y in (-12,-6,0,6,12)
                         for t in (-.08,-.04,0,.04,.08)],float)
        raw,support=self.costs(points,position,heading,coarse)
        coarse_cost=raw.copy()
        penalty=.004*(coarse[:,:2]**2).sum(axis=1)+8*coarse[:,2]**2
        best=coarse[np.argmin(raw+penalty)]
        fine=best+np.array([(x,y,t) for x in (-2,0,2) for y in (-2,0,2) for t in (-.015,0,.015)])
        raw,support=self.costs(points,position,heading,fine)
        rank=raw+.004*(fine[:,:2]**2).sum(axis=1)+8*fine[:,2]**2
        i=int(np.argmin(rank));best=fine[i].copy()
        initial=self.costs(points,position,heading,np.zeros((1,3)))[0][0]
        # Axis-wise flat directions retain odometry, rather than inventing certainty.
        probes=best+np.array([(8,0,0),(-8,0,0),(0,8,0),(0,-8,0)])
        probe_cost=self.costs(points,position,heading,probes)[0]
        # Diagnostics are emitted as JSON by both controllers; NumPy bools are not serializable.
        observable=[bool(min(probe_cost[:2])-raw[i]>.3),bool(min(probe_cost[2:])-raw[i]>.3)]
        best[:2]*=observable
        cost,inliers=self.costs(points,position,heading,best[None,:])
        self.last_candidates=125+27+1+4+1
        # A distinct coarse minimum must also be considered; the fine stencil
        # alone cannot expose repetitive-wall ambiguity outside its 4 px span.
        alternatives=(np.linalg.norm((coarse[:,:2]-best[:2])*observable,axis=1)>10)
        ambiguous=bool(alternatives.any() and coarse_cost[alternatives].min()<cost[0]+.2)
        accepted=(cost[0]<=6.5 and inliers[0]>=.65 and cost[0]<=initial+.1
                  and np.max(np.abs(best[:2]))<14 and abs(best[2])<.095 and not ambiguous)
        return (best if accepted else None),dict(reason='matched' if accepted else 'rejected',
            residual=float(cost[0]),prior_residual=float(initial),support=float(inliers[0]),
            observable=observable,candidates=self.last_candidates)


class PoseGuard:
    def __init__(self,size,gps_gain=.18,heading_gain=.25):
        self.position=None;self.heading=0.;self.absolute_missing_steps=0
        self.matcher=LocalMatcher(size);self.scan=np.empty((0,2));self.step=0
        self.gps_gain=gps_gain;self.heading_gain=heading_gain
        self.trusted=False;self.last_match=-100;self.relocked=False;self.good_steps=0
        self.had_absolute=False
        self.diagnostic=dict(reason='initial');self.enabled=True

    def update(self,gps,compass,odometry):
        self.step+=1;self.relocked=False
        good=gps is not None and np.shape(gps)==(2,) and np.isfinite(gps).all()
        angle=compass is not None and np.ndim(compass)==0 and np.isfinite(compass)
        was_missing=self.absolute_missing_steps
        self.absolute_missing_steps=0 if good and angle else was_missing+1
        self.good_steps=self.good_steps+1 if good and angle else 0
        if self.position is None:
            if not good or not angle:return None
            self.position=np.array(gps,float);self.heading=float(compass)
        else:
            if odometry is not None and np.shape(odometry)==(3,) and np.isfinite(odometry).all():
                distance,alpha,rotation=odometry
                self.position+=distance*np.array([math.cos(self.heading+alpha),math.sin(self.heading+alpha)])
                self.heading=wrap(self.heading+rotation)
            if good and angle and was_missing>0:
                residual=float(np.linalg.norm(np.asarray(gps)-self.position))
                if residual>30 or abs(wrap(compass-self.heading))>.3:
                    # Old trusted coordinates remain in the absolute frame; discard
                    # transient goals, not globally translate every old landmark.
                    self.position=np.array(gps,float);self.heading=float(compass)
                    self.matcher.frames.clear();self.matcher.distance=None
                    self.last_match=-100
                    self.relocked=True;self.diagnostic=dict(reason='relock_reset',residual=residual)
            if good:self.position+=self.gps_gain*(np.asarray(gps)-self.position)
            if angle:self.heading=wrap(self.heading+self.heading_gain*wrap(compass-self.heading))
        if not(good and angle) and self.enabled and self.step%8==0:
            offset,info=self.matcher.match(self.scan,self.position,self.heading)
            self.diagnostic=info
            if offset is not None:
                self.position+=offset[:2];self.heading=wrap(self.heading+offset[2])
                if all(info['observable']):self.last_match=self.step
        absolute=bool(good and angle)
        if absolute and self.good_steps>=4:self.had_absolute=True
        self.trusted=bool(self.good_steps>=4 if absolute else
                          self.had_absolute and (self.absolute_missing_steps<=8 or
                          (self.absolute_missing_steps<=240 and self.step-self.last_match<=16)))
        # Match against history BEFORE inserting the current scan. Low-confidence
        # scans cannot self-validate or contaminate the reference.
        if self.trusted and self.step%8==0:self.matcher.add(self.scan,self.position,self.heading)
        return self.position.copy(),self.heading
