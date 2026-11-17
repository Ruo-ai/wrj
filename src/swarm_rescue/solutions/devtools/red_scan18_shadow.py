"""RL18-B independent short-window lidar/odometry experiment, OFFLINE ONLY.

Original NumPy implementation. Uses finite scan segments, not copied RF2O/SRF
code or a claim to reproduce those algorithms. No truth, map name, peers, file
I/O, baseline pose or navigation is accepted by ShortWindow18.
"""
from collections import Counter
from dataclasses import dataclass
import math
import time

import numpy as np

from swarm_rescue.solutions.my_drone_red_lines11 import fit_lines
from swarm_rescue.solutions.semifinal_localization import transform, wrap


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def compose(a, b):
    return np.r_[a[:2]+rotation(a[2])@b[:2], wrap(a[2]+b[2])]


def relative(a, b):
    return np.r_[rotation(-a[2])@(b[:2]-a[:2]), wrap(b[2]-a[2])]


def integrate(pose, odometry):
    if odometry is None or np.shape(odometry)!=(3,) or not np.isfinite(odometry).all():
        return pose.copy()
    d, alpha, theta = odometry
    return compose(pose, np.array([d*math.cos(alpha), d*math.sin(alpha), theta]))


@dataclass
class Scan:
    points: np.ndarray
    segments: np.ndarray


def make_scan(distances, angles, objects=()):
    if distances is None or angles is None:
        return Scan(np.empty((0, 2)), np.empty((0, 2, 2)))
    d, a = np.asarray(distances, float).ravel(), np.asarray(angles, float).ravel()
    if d.shape != a.shape or len(d)>1000:
        return Scan(np.empty((0, 2)), np.empty((0, 2, 2)))
    good = np.isfinite(d) & np.isfinite(a) & (d>12) & (d<218)
    for obj in objects or ():
        if getattr(obj.entity_type, 'name', '') in ('BOMB','DRONE','DISPOSAL_CENTER') and np.isfinite([obj.distance,obj.angle]).all():
            good &= ~((np.abs((a-obj.angle+np.pi)%(2*np.pi)-np.pi)<.18) & (np.abs(d-obj.distance)<35))
    ids=np.flatnonzero(good)
    if len(ids)>90: ids=ids[np.linspace(0,len(ids)-1,90).astype(int)]
    points=np.column_stack((d[ids]*np.cos(a[ids]), d[ids]*np.sin(a[ids])))
    # Reuse only the baseline's legal finite-segment extractor, never its map.
    segments=np.asarray(fit_lines(distances,angles,objects),float).reshape(-1,2,2)
    return Scan(points,segments)


def geometry(segments):
    middle=segments.mean(axis=1); edge=segments[:,1]-segments[:,0]
    length=np.linalg.norm(edge,axis=1); tangent=edge/length[:,None]
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))
    same=(np.abs(normal@normal.T)>.998) & (np.abs(np.einsum('ijk,ik->ij',middle[None,:,:]-middle[:,None,:],normal))<5)
    return middle, length, tangent, normal, same


def associate(points, pose, geo):
    middle,length,tangent,normal,same=geo
    q=transform(points,pose[:2],pose[2]); difference=q[:,None,:]-middle[None,:,:]
    residual=np.einsum('ijk,jk->ij',difference,normal)
    along=np.einsum('ijk,jk->ij',difference,tangent)
    costs=np.where((np.abs(along)<=length/2+3)&(np.abs(residual)<16),np.abs(residual),np.inf)
    best=costs.argmin(axis=1); near=costs[np.arange(len(points)),best]
    alternate=np.where(same[best],np.inf,costs).min(axis=1)
    valid=np.isfinite(near) & ((alternate>near+3)|~np.isfinite(alternate))
    ids=np.flatnonzero(valid); n=normal[best[ids]]; r=residual[ids,best[ids]]
    return q[ids],n,r,valid


def solve_seed(points, geo, prior, yaw_seed):
    pose=prior.copy(); pose[2]+=yaw_seed; basis=None; info={}
    for iteration in range(5):
        q,n,r,valid=associate(points,pose,geo)
        if len(r)<14 or valid.mean()<.55:
            return None,dict(reason='support_or_ambiguity',support=float(valid.mean()))
        weight=np.minimum(1.,3./np.maximum(np.abs(r),1e-9)); weight/=weight.sum()
        values,vectors=np.linalg.eigh((n.T*weight)@n)
        keep=values>max(.04,.10*values.max()); current=vectors[:,keep]
        if basis is None: basis=current
        elif current.shape!=basis.shape or np.linalg.norm(current@current.T-basis@basis.T)>.15:
            return None,dict(reason='changing_observability')
        local=q-pose[:2]; angular=np.sum(n*np.column_stack((-local[:,1],local[:,0])),axis=1)/100
        j=np.column_stack((n@basis,angular)); h=(j.T*weight)@j
        spectrum=np.linalg.eigvalsh(h)
        if spectrum[0]<max(.008,.015*spectrum[-1]):
            return None,dict(reason='coupled_degeneracy')
        info=dict(reason='matched',rank=int(keep.sum()),support=float(valid.mean()),
                  rms=float(np.sqrt(np.sum(weight*r*r))),cost=float(np.minimum(np.abs(r),9).mean()),
                  eigenvalues=values.tolist(),observable_projector=(basis@basis.T).tolist(),
                  hessian_eigenvalues=spectrum.tolist())
        if iteration==4:
            if info['rms']>4.: return None,dict(info,reason='residual')
            return pose,info
        delta=-np.linalg.solve(h,j.T@(weight*r))
        pose[:2]+=basis@delta[:-1];pose[2]=wrap(pose[2]+delta[-1]/100)
        if np.linalg.norm(pose[:2]-prior[:2])>15 or abs(wrap(pose[2]-prior[2]))>.15:
            return None,dict(info,reason='prior_bound')


def match_pair(current, reference, prior):
    """Return current-to-reference SE(2). Rank-null displacement stays odometry.

    Three yaw initializations expose nearby competing minima, not global
    uniqueness. Reverse residual checks reject one-sided overlap/occlusion.
    """
    if len(current.points)<14 or not len(reference.segments):
        return None,dict(reason='few_features')
    geo=geometry(reference.segments); candidates=[]; reasons=Counter()
    for seed in (0.,-.06,.06):
        pose,info=solve_seed(current.points,geo,prior,seed)
        if pose is not None: candidates.append((pose,info))
        else: reasons[info['reason']]+=1
    if not candidates: return None,dict(reason='all_seeds_rejected',details=dict(reasons))
    candidates.sort(key=lambda x:x[1]['cost']); pose,info=candidates[0]
    for other,other_info in candidates[1:]:
        delta=relative(pose,other)
        if other_info['cost']<info['cost']+.4 and (np.linalg.norm(delta[:2])>4 or abs(delta[2])>.035):
            return None,dict(info,reason='multiple_solutions')
    if not len(current.segments) or len(reference.points)<14:
        return None,dict(info,reason='reverse_features')
    inverse=relative(pose,np.zeros(3))
    _,_,r,valid=associate(reference.points,inverse,geometry(current.segments))
    if len(r)<14 or valid.mean()<.5 or np.sqrt(np.mean(np.minimum(r*r,81)))>5:
        return None,dict(info,reason='reverse_support')
    return pose,dict(info,reverse_support=float(valid.mean()))


class ShortWindow18:
    """Independent 4-step scan cadence, previous scan + <=24-step key scan.

    Missing/degenerate/disagreeing constraints retain odometry. Reference
    eviction never grants absolute trust. No pose is returned to a controller.
    """
    STRIDE=4
    WINDOW=24

    def __init__(self):
        self.pose=None; self.step=0; self.missing=0
        self.previous=None; self.key=None; self.stats=Counter();self.last={}

    def update(self,gps,compass,odometry,distances,angles,objects=()):
        started=time.perf_counter();self.step+=1
        good=gps is not None and np.shape(gps)==(2,) and np.isfinite(gps).all()
        good=bool(good and compass is not None and np.ndim(compass)==0 and np.isfinite(compass))
        self.last=dict(reason='odometry',accepted=False)
        if self.pose is None:
            if not good: return None
            self.pose=np.r_[gps,compass].astype(float)
        else:
            self.pose=integrate(self.pose,odometry)
            if good:
                if self.missing:
                    self.pose=np.r_[gps,compass].astype(float);self.previous=None;self.key=None
                else:
                    self.pose[:2]+=.12*(np.asarray(gps)-self.pose[:2])
                    self.pose[2]=wrap(self.pose[2]+.15*wrap(compass-self.pose[2]))
        self.missing=0 if good else self.missing+1
        if self.step%self.STRIDE==0:
            scan=make_scan(distances,angles,objects)
            if self.missing and self.previous is not None:
                old_tick,old_pose,old_scan=self.previous
                answer,info=match_pair(scan,old_scan,relative(old_pose,self.pose))
                self.stats['attempts']+=1
                if answer is not None:
                    candidate=compose(old_pose,answer)
                    if self.key is not None and self.key[0]!=old_tick:
                        kt,kp,ks=self.key
                        key_answer,key_info=match_pair(scan,ks,relative(kp,self.pose))
                        if key_answer is not None:
                            key_candidate=compose(kp,key_answer)
                            disagreement=relative(candidate,key_candidate)
                            projector=np.array(info['observable_projector'])
                            # Compare in the previous-scan axes; preserve its null direction.
                            observed=projector@(rotation(-old_pose[2])@(candidate[:2]-key_candidate[:2]))
                            if np.linalg.norm(observed)>5 or abs(disagreement[2])>.04:
                                answer=None;info=dict(info,reason='key_disagreement')
                            else:
                                candidate=key_candidate;info=key_info
                        else:
                            info=dict(info,key_fallback=key_info['reason'])
                    if answer is not None:
                        self.pose=candidate;info=dict(info,accepted=True)
                        self.stats['accepted']+=1;self.stats['rank_'+str(info['rank'])]+=1
                self.last=dict(info,accepted=answer is not None)
                self.stats[info['reason']]+=1
            frame=(self.step,self.pose.copy(),scan)
            self.previous=frame
            if good or self.key is None or self.step-self.key[0]>=self.WINDOW:
                self.key=frame
        self.last['seconds']=time.perf_counter()-started
        return self.pose.copy()
