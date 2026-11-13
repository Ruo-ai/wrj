"""Independent B4: search coverage uses semantic range, not lidar free space.

Coverage is a geometric sensing opportunity, not proof that every bomb was seen.
Only own legal scans and received bounded snapshots contribute. No truth online.
"""
import time
import math
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_blue_transport import MyDroneBlueTransportCompute,TransportGrid


class SemanticSearchGrid(TransportGrid):
    def __init__(self,size):
        super().__init__(size)
        self.searched=np.zeros(self.shape,bool);self.semantic_enabled=True

    def observe(self,position,heading,distances,angles,dynamic):
        super().observe(position,heading,distances,angles,dynamic)
        if not self.semantic_enabled:return
        valid=np.isfinite(distances)&np.isfinite(angles)&(distances>4)
        d=np.minimum(distances[valid]-20,118.)
        a=angles[valid]+heading
        samples=np.arange(0,119,12)
        directions=np.column_stack((np.cos(a),np.sin(a)))
        points=position+directions[:,None,:]*samples[None,:,None]
        points=points[samples[None,:]<d[:,None]]
        if not len(points):return
        c=np.floor((points[:,0]+self.width/2)/self.CELL).astype(int)
        r=np.floor((self.height/2-points[:,1])/self.CELL).astype(int)
        good=(r>=0)&(r<self.shape[0])&(c>=0)&(c<self.shape[1])
        self.searched[r[good],c[good]]=True

    def snapshot(self):
        return tuple(np.packbits(a.ravel()).tobytes() for a in (self.free,self.evidence>=2,self.searched))

    def search_coverage(self):
        seen=self.searched.copy()
        for _,(_,_,remote_seen) in self.remote.values():seen|=remote_seen
        return seen

    def route(self,position,target=None,identifier=0,number=1,claims=()):
        if target is not None:return super().route(position,target,identifier,number,claims)
        started=time.perf_counter();allowed,_=self.layers()
        start,lengths,parents=self._shortest(allowed,position)
        cells=self._bfs_order;points=self._points[cells];seen=self.search_coverage()
        walls=self.evidence>=2;free=self.free.copy()
        for _,(rf,rw,_) in self.remote.values():
            walls |= rw&~(self.free&(self.evidence<.5));free |= rf
        unsearched=free&~seen&~walls
        gain=cv2.boxFilter(unsearched.astype(np.float32),-1,(11,11),normalize=False,
                          borderType=cv2.BORDER_CONSTANT).ravel()
        delta=points-position
        frontier=cv2.boxFilter((~free&~walls).astype(np.float32),-1,(5,5),normalize=False,
                              borderType=cv2.BORDER_CONSTANT).ravel()
        values=2.0*gain[cells]+5*frontier[cells]-.4*lengths[cells]-4*self.visits.ravel()[cells]
        values+=15*(~seen.ravel()[cells])+12*np.cos(np.arctan2(delta[:,1],delta[:,0])-identifier*2*math.pi/max(1,number))
        for claim in claims:values-=65*np.maximum(0,1-np.linalg.norm(points-claim,axis=1)/160)
        values[lengths[cells]<3]=-1e6
        end=int(cells[np.argmax(values)])
        self.last_endpoint=self._points[end].copy();self.last_complete=False
        self.last_route_length=float(lengths[end]*self.CELL)
        path=[];node=end
        while node!=start and node>=0:
            path.append(self._points[node].copy());node=int(parents[node])
        self.route_calls+=1;self.route_seconds+=time.perf_counter()-started
        return list(reversed(path))


class MyDroneBlueSemanticSearch(MyDroneBlueTransportCompute):
    SEMANTIC_SEARCH=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.SEMANTIC_SEARCH:self._grid=SemanticSearchGrid(self.size_area)

    def _sense(self,distances,angles):
        if self.SEMANTIC_SEARCH:self._grid.semantic_enabled=not self.semantic_is_disabled()
        return super()._sense(distances,angles)
