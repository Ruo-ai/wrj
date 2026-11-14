"""P8: independent 1 px submission evidence, retaining A3 pose estimation.

No unconditional closing/dilation of submission walls. Filling requires a
closed observed surface, repeated outward-facing evidence and no observed free
pixel inside it. Navigation remains the inherited coarse map and radio data.
"""
import math
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper
from swarm_rescue.solutions.my_drone_red_advanced2 import _M01_SnapshotStore


def densify_neighbours(points, maximum_gap):
    points=np.asarray(points,dtype=np.float32)
    if len(points)<2:return points
    delta=np.diff(points,axis=0);length=np.linalg.norm(delta,axis=1)
    pairs=np.flatnonzero((length>1.)&(length<=maximum_gap))
    if not len(pairs):return points
    counts=np.ceil(length[pairs]).astype(int)
    index=np.repeat(np.arange(len(pairs)),counts-1)
    offset=np.arange((counts-1).sum())-np.repeat(np.cumsum(counts-1)-(counts-1),counts-1)+1
    extra=points[pairs[index]]+delta[pairs[index]]*(offset/counts[index])[:,None]
    return np.vstack((points,extra)).astype(np.float32)


def enclosed_fill(surface,free_seen,directions,enabled=True):
    """Only a CLOSED observed surface with outward observations can bound solid.

    A room's walls face its interior, so observing a room perimeter must not
    turn its free interior into solid. Open ends, one face, opposing corridor
    faces and any trusted free evidence veto the entire inferred component.
    directions is a uint8 bitmask of repeatedly observed viewing sectors.
    """
    fill=np.zeros(surface.shape,np.uint8)
    if not enabled:return fill.astype(bool)
    contours,hierarchy=cv2.findContours(surface.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return fill.astype(bool)
    for i,contour in enumerate(contours):
        if hierarchy[0,i,3]<0:continue
        area=cv2.contourArea(contour)
        if not 8<=area<=12000:continue
        x,y,w,h=cv2.boundingRect(contour)
        local=np.zeros((h,w),np.uint8)
        cv2.drawContours(local,[contour-np.array([[[x,y]]])],-1,1,-1)
        hole=local.astype(bool)&~surface[y:y+h,x:x+w]
        if not hole.any() or np.any(hole&free_seen[y:y+h,x:x+w]):continue
        boundary=cv2.dilate(hole.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&surface[y:y+h,x:x+w]
        yy,xx=np.nonzero(boundary)
        if len(xx)<12:continue
        bits=directions[y+yy,x+xx]
        cx,cy=np.mean(np.nonzero(hole)[1]),np.mean(np.nonzero(hole)[0])
        radial=np.column_stack((xx-cx,cy-yy));radial/=np.maximum(1,np.linalg.norm(radial,axis=1))[:,None]
        sector=np.arange(8)*math.pi/4
        dot=radial@np.column_stack((np.cos(sector),np.sin(sector))).T
        supported=(bits[:,None]&(1<<np.arange(8)))!=0
        outward=(supported&(dot>.5)).any(axis=1)
        inward=(supported&(dot<-.5)).any(axis=1)
        if outward.sum()<max(12,.75*len(xx)) or inward.sum()>.10*len(xx):continue
        used=np.flatnonzero((supported&(dot>.5)).sum(axis=0)>=3)
        if not any((int(a)+4)%8 in used for a in used):continue
        fill[y:y+h,x:x+w][hole]=1
    return fill.astype(bool)


class PixelFaceMapper(SubmissionMapper):
    CELL=1
    PAIR_FILL=True
    def __init__(self,size):
        super().__init__(size)
        self.shape=(self.height,self.width)
        self.hits=np.zeros(self.shape,np.float32)
        self.support=np.zeros(self.shape,np.float32)
        self._provisional=None;self._provisional_support=None
        self.free_seen=np.zeros(self.shape,bool)
        self.direction_counts=np.zeros((8,*self.shape),np.uint8)
        self._trusted=False;self._cached_step=-1;self._cached=None
        self.stats.update(pixel_resolution=1,fill_pixels=0,surface_pixels=0,
                          permanent_free_pixels=0,maximum_snapshot_bytes=0)

    def update(self,gps,compass,odometry,distances,angles,objects=()):
        # Leave A3's pose, matching, outage buffering and reanchoring unchanged.
        # Direction votes from reanchored batches are withheld: the inherited
        # insertion hook does not expose each historical frame's corrected origin.
        self._trusted=bool(self.missing==0 and gps is not None and np.shape(gps)==(2,)
            and np.isfinite(gps).all() and compass is not None and np.ndim(compass)==0 and np.isfinite(compass))
        super().update(gps,compass,odometry,distances,angles,objects)

    def _point_indices(self,points):
        x=np.floor(points[:,0]+self.width/2).astype(int)
        y=np.floor(self.height/2-points[:,1]).astype(int)
        valid=(x>=0)&(x<self.width)&(y>=0)&(y<self.height)
        return y[valid]*self.width+x[valid],valid

    def _insert(self,grid,points,weight):
        # Existing scan interpolation has <=2px segments. Only densify those;
        # do not connect separate rays/surfaces across arbitrary unknown gaps.
        points=densify_neighbours(points,2.5)
        index,valid=self._point_indices(points);unique=np.unique(index)
        grid.ravel()[unique]=np.minimum(30,grid.ravel()[unique]+weight)
        support=self.support if grid is self.hits else self._provisional_support if grid is self._provisional else None
        if support is not None and len(unique):
            # Fuse independent scans within 2px, while preserving exact 1px hit
            # locations. This uncertainty neighbourhood is NOT painted as wall.
            yy,xx=unique//self.width,unique%self.width
            left=max(0,int(xx.min())-2);right=min(self.width,int(xx.max())+3)
            top=max(0,int(yy.min())-2);bottom=min(self.height,int(yy.max())+3)
            scan=np.zeros((bottom-top,right-left),np.uint8);scan[yy-top,xx-left]=1
            vote=cv2.dilate(scan,np.ones((5,5),np.uint8)).astype(bool)
            patch=support[top:bottom,left:right]
            patch[vote]=np.minimum(30,patch[vote]+weight)
        if grid is self.hits and self._trusted and len(index):
            vector=self.position-points[valid]
            bins=np.rint(np.arctan2(vector[:,1],vector[:,0])/(math.pi/4)).astype(int)%8
            keys=np.unique(bins*self.hits.size+index)
            self.direction_counts.ravel()[keys]=np.minimum(30,self.direction_counts.ravel()[keys]+1)

    def _erase(self,grid,points,weight):
        # Densify only consecutive samples of the same measured beam interior;
        # A3's 18px endpoint guard remains, without filling angular sectors.
        points=densify_neighbours(points,9.5)
        index,_=self._point_indices(points);index=np.unique(index)
        grid.ravel()[index]=np.maximum(0,grid.ravel()[index]-.35*weight)
        support=self.support if grid is self.hits else self._provisional_support if grid is self._provisional else None
        if support is not None:support.ravel()[index]=np.maximum(0,support.ravel()[index]-.35*weight)
        if grid is self.hits and self._trusted:self.free_seen.ravel()[index]=True

    def evidence(self):
        if self._cached_step==self.step and self._cached is not None:return self._cached
        # Per-scan unique insertion prevents dense rays from becoming multiple
        # independent observations. Unclosed outages stay provisional, as A3.
        from swarm_rescue.solutions.semifinal_localization import transform
        self._provisional=np.zeros(self.shape,np.float32)
        self._provisional_support=np.zeros(self.shape,np.float32)
        for _,points,position,heading,weight,free in self.pending:
            self._erase(self._provisional,transform(free,position,heading),weight)
            self._insert(self._provisional,transform(points,position,heading),weight)
        surface=(((self.hits>0)&(self.support>=2.4))|
                 ((self._provisional>0)&(self._provisional_support>=2.4))).astype(np.uint8)
        self._provisional=None;self._provisional_support=None
        count,labels,stats,_=cv2.connectedComponentsWithStats(surface,connectivity=8)
        keep=np.zeros(count,np.uint8);keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=5
        surface=keep[labels].astype(bool)
        direction_bits=np.zeros(self.shape,np.uint8)
        for k in range(8):direction_bits|=((self.direction_counts[k]>=3).astype(np.uint8)<<k)
        fill=enclosed_fill(surface,self.free_seen,direction_bits,self.PAIR_FILL)
        self.stats.update(fill_pixels=int(fill.sum()),surface_pixels=int(surface.sum()),
                          permanent_free_pixels=int(self.free_seen.sum()))
        self._cached_step=self.step;self._cached=(surface,fill,self.free_seen)
        return self._cached

    def mask(self):
        surface,fill,free=self.evidence()
        return surface|(fill&~free)

    def to_pixels(self,mask):
        assert mask.shape==self.shape
        return np.asarray(mask,np.uint8)  # no 3px expansion, closing or dilation


class PixelSnapshots(_M01_SnapshotStore):
    """Red-owned payload extension; navigation prefix and origin versions intact."""
    def __init__(self,shape,number_drones,mapper):
        super().__init__(shape,number_drones)
        self.mapper=mapper;self.mapping_masks={};self.fill_masks={};self.free_masks={}
        self.nav_bytes=(math.prod(shape)+7)//8
        self.pixel_bytes=(math.prod(mapper.shape)+7)//8

    def publish(self,source,sequence,mask):
        surface,fill,free=self.mapper.evidence()
        payload=b''.join(np.packbits(np.asarray(m,np.uint8).ravel()).tobytes() for m in (mask,surface,fill,free))
        self.mapper.stats['maximum_snapshot_bytes']=max(self.mapper.stats['maximum_snapshot_bytes'],len(payload))
        self.accept((int(source),int(sequence),payload))

    def accept(self,snapshot):
        if not isinstance(snapshot,tuple) or len(snapshot)!=3:return False
        source,sequence,payload=snapshot
        if not isinstance(payload,bytes) or len(payload)!=self.nav_bytes+3*self.pixel_bytes:return False
        if not super().accept((source,sequence,payload[:self.nav_bytes])):return False
        self.snapshots[source]=snapshot
        for i,store in enumerate((self.mapping_masks,self.fill_masks,self.free_masks)):
            start=self.nav_bytes+i*self.pixel_bytes
            store[source]=np.unpackbits(np.frombuffer(payload[start:start+self.pixel_bytes],np.uint8))[:math.prod(self.mapper.shape)].reshape(self.mapper.shape).astype(bool)
        return True

    def submission(self):
        surface=np.zeros(self.mapper.shape,bool);fill=surface.copy();free=surface.copy()
        for source in self.mapping_masks:
            surface|=self.mapping_masks[source];fill|=self.fill_masks[source];free|=self.free_masks[source]
        return surface|(fill&~free)
