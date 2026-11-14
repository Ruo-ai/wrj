"""P9: exact incremental cache of P8 unclosed-outage evidence, no new scoring.

A3 pending-frame poses are immutable until the outage is reanchored. Therefore
replaying the same buffered frames on every radio publish is unnecessary.
"""
import cv2
import numpy as np
from swarm_rescue.solutions.red_pixel_faces import PixelFaceMapper,enclosed_fill,PixelSnapshots
from swarm_rescue.solutions.semifinal_localization import transform
from swarm_rescue.solutions.my_drone_red_rn7 import MyDroneRedRN7B


class IncrementalPixelMapper(PixelFaceMapper):
    def __init__(self,size):
        super().__init__(size)
        self._provisional=np.zeros(self.shape,np.float32)
        self._provisional_support=np.zeros(self.shape,np.float32)
        self._cached_pending_steps=[]
        self.stats.update(provisional_frame_integrations=0,provisional_rebuilds=0)

    def update(self,*args,**kwargs):
        super().update(*args,**kwargs)
        steps=[frame[0] for frame in self.pending]
        previous=self._cached_pending_steps
        prefix=len(previous)<=len(steps) and steps[:len(previous)]==previous
        if not prefix:
            self._provisional.fill(0);self._provisional_support.fill(0)
            previous=[];self.stats['provisional_rebuilds']+=1
        for _,points,position,heading,weight,free in self.pending[len(previous):]:
            self._erase(self._provisional,transform(free,position,heading),weight)
            self._insert(self._provisional,transform(points,position,heading),weight)
            self.stats['provisional_frame_integrations']+=1
        self._cached_pending_steps=steps

    def evidence(self):
        if self._cached_step==self.step and self._cached is not None:return self._cached
        surface=(((self.hits>0)&(self.support>=2.4))|
                 ((self._provisional>0)&(self._provisional_support>=2.4))).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(surface,connectivity=8)
        keep=np.zeros(count,np.uint8);keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=5
        surface=keep[labels].astype(bool)
        bits=np.zeros(self.shape,np.uint8)
        for k in range(8):bits|=((self.direction_counts[k]>=3).astype(np.uint8)<<k)
        fill=enclosed_fill(surface,self.free_seen,bits,self.PAIR_FILL)
        self.stats.update(fill_pixels=int(fill.sum()),surface_pixels=int(surface.sum()),permanent_free_pixels=int(self.free_seen.sum()))
        self._cached_step=self.step;self._cached=(surface,fill,self.free_seen)
        return self._cached


class MyDroneRedPixel9(MyDroneRedRN7B):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._submission_mapper=IncrementalPixelMapper(self.size_area)
        self._radio_maps=PixelSnapshots(self._map.shape,self._n,self._submission_mapper)
