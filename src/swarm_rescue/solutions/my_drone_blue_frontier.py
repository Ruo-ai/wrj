"""B5-B: observed-space frontier tasks with occlusion-bounded semantic gain.

Independent of narrow control, terminal delivery and localization experiments.
Visited body cells, lidar-free cells and semantic opportunities remain distinct.
"""
import math
import time
from collections import Counter
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_blue_search import SemanticSearchGrid,MyDroneBlueSemanticSearch


class FrontierTaskGrid(SemanticSearchGrid):
    def __init__(self,size):
        super().__init__(size)
        self.tick=0;self.task=None;self.task_until=0;self.task_state='none'
        self.progress_step=0;self.best_distance=float('inf');self.cooldowns=[]
        self.stats=Counter();self.last_gain=0.;self.priority_claims=[]
        a=np.linspace(-math.pi,math.pi,35,endpoint=False)
        self._ray_offsets=np.rint(np.stack((np.sin(a),np.cos(a)),axis=1)[:,None,:]*
                                  np.arange(1,8)[None,:,None]).astype(int)

    def visibility_gain(self,cell,free,walls,seen):
        """35 semantic rays <=112px: stop at known walls AND first unknown.

        Only the first unknown boundary is an uncertain exploration opportunity;
        invisible free/unknown regions behind it contribute nothing.
        """
        p=np.asarray(cell)+self._ray_offsets
        valid=(p[:,:,0]>=0)&(p[:,:,0]<self.shape[0])&(p[:,:,1]>=0)&(p[:,:,1]<self.shape[1])
        r=np.clip(p[:,:,0],0,self.shape[0]-1);c=np.clip(p[:,:,1],0,self.shape[1]-1)
        stop=(~valid)|walls[r,c]|~free[r,c]
        before=np.concatenate((np.ones((35,1),bool),~np.maximum.accumulate(stop,axis=1)[:,:-1]),axis=1)
        visible=valid&before&~walls[r,c]
        ids=np.unique(r[visible]*self.shape[1]+c[visible])
        new_known=free.ravel()[ids]&~seen.ravel()[ids]
        uncertain=~free.ravel()[ids]
        return float(new_known.sum()+.35*uncertain.sum())

    def route(self,position,target=None,identifier=0,number=1,claims=()):
        if target is not None:return super().route(position,target,identifier,number,claims)
        started=time.perf_counter();allowed,_=self.layers()
        start,lengths,parents=self._shortest(allowed,position)
        free=self.free.copy();walls=self.evidence>=2;seen=self.search_coverage()
        for _,(rf,rw,_) in self.remote.values():
            walls |= rw&~(self.free&(self.evidence<.5));free |= rf
        reachable=np.isfinite(lengths).reshape(self.shape)&allowed
        unknown=~free&~walls
        # Inflation remains intact. An observation point must be reachable;
        # dilation finds safe viewpoints set back from a boundary, not shortcuts.
        near_unknown=cv2.dilate(unknown.astype(np.uint8),np.ones((5,5),np.uint8))>0
        front=reachable&(near_unknown|(~seen&free))
        self.cooldowns=[(q,t) for q,t in self.cooldowns if t>self.tick]
        end=None
        if self.task is not None:
            cell=self.cell(self.task);flat=np.ravel_multi_index(cell,self.shape)
            distance=float(np.linalg.norm(position-self.task))
            if distance+8<self.best_distance:self.best_distance=distance;self.progress_step=self.tick
            conflict=any(np.linalg.norm(self.task-q)<80 for q in self.priority_claims)
            gain=self.visibility_gain(cell,free,walls,seen)
            reason=('reached' if distance<26 else 'claim_conflict' if conflict else
                    'unreachable' if not reachable[cell] else 'searched' if gain<1 else
                    'blocked' if self.tick-self.progress_step>120 else
                    'expired' if self.tick>=self.task_until else None)
            if reason:
                self.stats[reason]+=1;self.task_state=reason
                if reason in ('blocked','unreachable','claim_conflict'):self.cooldowns.append((self.task.copy(),self.tick+100))
                self.task=None
            else:end=int(flat);self.last_gain=gain;self.task_state='active'
        if end is None:
            count,labels=cv2.connectedComponents(front.astype(np.uint8),connectivity=4)
            candidates=[]
            # Keep a bounded number of reachable regions and representatives;
            # full-map visibility raycasts for every cell would exceed budget.
            groups=[]
            for label in range(1,count):
                indices=np.flatnonzero(labels.ravel()==label)
                indices=indices[lengths[indices]>=3]
                if len(indices):groups.append((float(lengths[indices].min()),indices))
            for _,indices in sorted(groups,key=lambda g:g[0])[:12]:
                ordered=indices[np.argsort(lengths[indices],kind='stable')]
                picks=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
                for index in dict.fromkeys(picks):
                    point=self._points[index]
                    if any(np.linalg.norm(point-q)<48 for q,_ in self.cooldowns):continue
                    cell=np.unravel_index(index,self.shape);gain=self.visibility_gain(cell,free,walls,seen)
                    if gain<1:continue
                    delta=point-position
                    score=3*gain-.5*lengths[index]-2*self.visits.ravel()[index]
                    score+=7*math.cos(math.atan2(delta[1],delta[0])-identifier*2*math.pi/max(1,number))
                    for q in claims:score-=50*max(0,1-float(np.linalg.norm(point-q))/128)
                    candidates.append((score,int(index),gain))
            if candidates:
                _,end,self.last_gain=max(candidates)
                self.task=self._points[end].copy();self.task_until=self.tick+240
                self.best_distance=float(np.linalg.norm(position-self.task));self.progress_step=self.tick
                self.task_state='active';self.stats['assigned']+=1
            else:
                self.task_state='no_reachable_viewpoint';self.stats['no_viewpoint']+=1
                # Preserve established search fallback, without crossing unknown.
                return super().route(position,None,identifier,number,claims)
        self.last_endpoint=self._points[end].copy();self.last_complete=False
        self.last_route_length=float(lengths[end]*self.CELL)
        path=[];node=end
        while node!=start and node>=0:
            path.append(self._points[node].copy());node=int(parents[node])
        self.route_calls+=1;self.route_seconds+=time.perf_counter()-started
        return list(reversed(path))


class MyDroneBlueFrontier(MyDroneBlueSemanticSearch):
    FRONTIER_TASKS=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.FRONTIER_TASKS:self._grid=FrontierTaskGrid(self.size_area)

    def _sense(self,distances,angles):
        if self.FRONTIER_TASKS:
            self._grid.tick=self._tick
            self._grid.priority_claims=[np.asarray(p[3]) for ident,p in self._peers.items()
                if ident<int(self.identifier) and p[2]=='search' and p[3] is not None]
        return super()._sense(distances,angles)
