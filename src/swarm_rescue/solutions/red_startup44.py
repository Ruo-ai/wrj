"""Bounded local launch planning from legal range scans and received bids."""
import math
import warnings
import numpy as np

ANGLES = np.arange(24) * (2 * math.pi / 24)


class InitialMap44:
    """Finite stationary scan map, used ONLY by startup; never a submission map.

    Range registration is bounded to +/-3deg. Moving frames are rejected instead
    of integrating uncertain odometry into this short stationary map. Unknown
    (including robot-occluded) bearings stay unknown, never free space.
    """
    def __init__(self):
        self.frames=[]; self.headings=[]; self.positions=[]; self.shifts=[]
        self.angles=None; self.ranges=None; self.support=None; self.grid=None
        self.origin=None; self.heading=None; self.rejected=0

    def add(self, distances, angles, position, heading, speed, semantics=()):
        if distances is None or angles is None or position is None or not np.isfinite(heading):return
        d=np.asarray(distances,float).ravel().copy(); a=np.asarray(angles,float).ravel()
        p=np.asarray(position,float)
        if d.shape!=a.shape or len(d)<24 or not np.isfinite(d).all() or not np.isfinite(p).all():return
        if speed>.55 or (self.positions and np.linalg.norm(p-np.median(self.positions,axis=0))>14):
            self.rejected+=1;return
        for obj in semantics or ():
            if getattr(obj.entity_type,'name',None)=='DRONE':
                d[(np.abs((a-obj.angle+np.pi)%(2*np.pi)-np.pi)<.19)&(np.abs(d-obj.distance)<25)]=np.nan
        shift=0.
        if self.frames:
            ref=self.frames[0]; best=0.; cost0=None; best_cost=float('inf')
            for deg in (0.,-.5,.5,-1.,1.,-1.5,1.5,-2.,2.,-2.5,2.5,-3.,3.):
                trial=np.interp(a-math.radians(deg),a,d,left=np.nan,right=np.nan)
                valid=np.isfinite(ref)&np.isfinite(trial)&(ref<220)&(trial<220)
                if valid.sum()<24:continue
                cost=float(np.mean(np.minimum(np.abs(ref[valid]-trial[valid]),15)))
                if deg==0:cost0=cost
                if cost<best_cost:best_cost=cost;best=math.radians(deg)
            if cost0 is not None and cost0-best_cost>1.5:shift=best
        aligned=np.interp(a-shift,a,d,left=np.nan,right=np.nan)
        self.frames.append(aligned); self.headings.append(float(heading-shift))
        self.positions.append(p.copy());self.shifts.append(shift);self.angles=a.copy()

    def finish(self):
        if len(self.frames)<6:return None
        frames=np.asarray(self.frames);self.support=np.isfinite(frames).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            med=np.nanmedian(frames,axis=0)
        med[self.support<max(4,math.ceil(len(frames)*.6))]=np.nan
        self.ranges=med
        self.heading=float(math.atan2(np.mean(np.sin(self.headings)),np.mean(np.cos(self.headings))))
        self.origin=np.median(self.positions,axis=0)
        # An independent 4px map: -1 unknown, 0 supported free, 1 wall return.
        y,x=np.mgrid[-240:241:4,-240:241:4];radius=np.hypot(x,y)
        bearing=np.arctan2(y,x)
        index=np.argmin(np.abs((bearing[...,None]-self.angles+np.pi)%(2*np.pi)-np.pi),axis=-1)
        limits=med[index];self.grid=np.full(radius.shape,-1,np.int8)
        self.grid[np.isfinite(limits)&(radius<limits-5)]=0
        self.grid[np.isfinite(limits)&(limits<225)&(np.abs(radius-limits)<3)]=1
        return scan_lengths(np.nan_to_num(med,nan=0.),self.angles,self.heading)

    def audit(self):
        return dict(frames=len(self.frames),rejected=self.rejected,shifts=self.shifts,
            heading=self.heading,origin=None if self.origin is None else self.origin.tolist(),
            known_bearings=0 if self.ranges is None else int(np.isfinite(self.ranges).sum()),
            independent=True,used_for_submission=False)


def scan_lengths(distances, angles, heading, semantics=()):
    """Swept 24px radius clearance, conservative three-ray angular support."""
    if distances is None or angles is None or not np.isfinite(heading):
        return None
    d = np.asarray(distances, float).ravel()
    a = np.asarray(angles, float).ravel()
    if d.shape != a.shape or len(d) < 24 or not np.isfinite(d).all():
        return None
    dynamic = np.zeros(len(d), bool)
    for obj in semantics or ():
        if getattr(obj.entity_type, 'name', None) == 'DRONE':
            dynamic |= (np.abs((a-obj.angle+np.pi)%(2*np.pi)-np.pi)<.19) & (np.abs(d-obj.distance)<25)
    out = []
    for theta in ANGLES:
        delta = (a + heading - theta + np.pi) % (2*np.pi) - np.pi
        cone = (np.abs(delta)<.13) & ~dynamic
        if cone.sum()<3:
            out.append(0.); continue
        along, across = d*np.cos(delta), np.abs(d*np.sin(delta))
        hit = (d<225) & (along>0) & (across<24) & ~dynamic
        cap = float(np.min(along[hit]-24)) if hit.any() else 160.
        out.append(float(np.clip(min(cap, np.min(d[cone])-24), 0, 160)))
    return np.asarray(out)


def valid_bid(bid, number, clock):
    return (isinstance(bid,tuple) and len(bid)==4 and type(bid[0]) is int and 0<=bid[0]<number
        and type(bid[1]) is int and 1<=bid[1]<=min(clock,34)
        and isinstance(bid[2],tuple) and len(bid[2])==2
        and all(isinstance(x,(int,float)) and math.isfinite(x) for x in bid[2])
        and isinstance(bid[3],tuple) and len(bid[3])==24
        and all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=160 for x in bid[3]))


def assignment(bids):
    """Stable ID order, broad exits and 10px utility deadband, received bids only.

    Do not search 120 permutations: tiny sensor changes must not arbitrarily
    swap all five roles. A narrow observed exit may be shared if necessary.
    """
    ids=sorted(bids)
    if not ids:return ()
    origins={i:np.asarray(bids[i][2]) for i in ids}
    lengths={i:np.asarray(bids[i][3]) for i in ids}
    dirs=np.column_stack((np.cos(ANGLES),np.sin(ANGLES)))
    chosen={};points=[];indices=[];starts=[]
    for i in ids:
        endpoints=origins[i]+dirs*np.minimum(lengths[i],140)[:,None]
        broad=np.minimum(np.roll(lengths[i],1),np.roll(lengths[i],-1))
        quality=.8*(10*np.round(lengths[i]/10))+.2*(10*np.round(broad/10))
        for p,k,start in zip(points,indices,starts):
            quality-=85*np.exp(-np.linalg.norm(endpoints-p,axis=1)/80)
            quality-=35*np.exp(-((1-np.cos(ANGLES-ANGLES[k]))/.35))
            # Penalize crossing launch segments, without inventing peer poses.
            def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
            u=endpoints-origins[i];v=p-start
            crossed=(cross(u,start-origins[i])*cross(u,p-origins[i])<0)&(cross(v,origins[i]-start)*cross(v,endpoints-start)<0)
            quality-=20*crossed
        feasible=lengths[i]>=60
        if not feasible.any():continue
        quality[~feasible]=-1e6
        near=np.flatnonzero(feasible&(quality>=quality.max()-10))
        # Wide cone wins near-ties; final tie uses world sector, not random yaw.
        width=10*np.round(broad[near]/10);k=int(near[np.argmax(width)])
        chosen[i]=k;points.append(endpoints[k]);indices.append(k);starts.append(origins[i])
    return tuple((i,int(k),float(min(lengths[i][k],140))) for i,k in sorted(chosen.items()))


def valid_plan(plan, number, clock):
    return (isinstance(plan,tuple) and len(plan)==2 and type(plan[0]) is int and 24<=plan[0]<=min(clock,48)
        and isinstance(plan[1],tuple) and len(plan[1])<=number
        and all(isinstance(r,tuple) and len(r)==3 and type(r[0]) is int and 0<=r[0]<number
                and type(r[1]) is int and 0<=r[1]<24 and isinstance(r[2],(int,float))
                and math.isfinite(r[2]) and 60<=r[2]<=140 for r in plan[1])
        and len({r[0] for r in plan[1]})==len(plan[1]))
