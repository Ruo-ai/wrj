"""RL13-L: rigid short submaps at RL11 absolute-pose recovery.

Only provisional legal observations are reprojected. Navigation, motion,
wall interpolation, free evidence and radio rules remain RL11. A short block
uses one SE(2) correction instead of deforming every scan differently.
Entry pose and recovery pose are noisy anchors, never ground truth.
"""
from collections import Counter, deque
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_lines11 import MyDroneRedLines11, WallLineMapper
from swarm_rescue.solutions.my_drone_red_mapping import MappingSnapshots, anchor_frames
from swarm_rescue.solutions.my_drone_red_stopturn import available_pose
from swarm_rescue.solutions.semifinal_localization import transform, wrap


def rigid_blocks(frames, start, end_position, end_heading, measured_position,
                 measured_heading, end_step, span=48, turn=.18):
    """Bounded blocks split on elapsed time or accumulated estimated turning.

    The anchor correction at each block midpoint is shared by that block.
    Thus measured within-block distances/angles are preserved exactly; gaps
    between blocks, entry bias and nonrigid drift remain explicit limitations.
    No line or unobserved wall pixels are invented.
    """
    if not frames:
        return [], 0
    blocks=[]; block=[]
    for frame in frames:
        if block and (frame[0]-block[0][0] >= span or
                      abs(wrap(frame[3]-block[0][3])) > turn):
            blocks.append(block);block=[]
        block.append(frame)
    if block:blocks.append(block)
    result=[]
    for block in blocks:
        representative=block[len(block)//2]
        corrected=anchor_frames([representative],start,end_position,end_heading,
                                measured_position,measured_heading,end_step)[0]
        angle=wrap(corrected[3]-representative[3])
        origin=representative[2];target=corrected[2]
        for f in block:
            pos=transform(np.asarray([f[2]-origin]),target,angle)[0]
            result.append((f[0],f[1],pos,wrap(f[3]+angle),f[4],*f[5:]))
    return result,len(blocks)


class LocalBlockMapper(WallLineMapper):
    LOCAL_BLOCKS=True
    BLOCK_SPAN=48  # 12 scans at the unchanged four-tick scan cadence.
    ENTRY_HOLD=24  # Buffer entry scans; bounded fallback, never discard them.

    def __init__(self,size):
        super().__init__(size)
        self.local_stats=Counter();self.local_events=deque(maxlen=128)
        self.entry=None;self.entry_confirmations=0;self.entry_last_match=-1

    def update(self,gps,compass,odometry,distances,angles,objects=()):
        if not self.LOCAL_BLOCKS:
            return super().update(gps,compass,odometry,distances,angles,objects)
        good=all(available_pose(gps,compass))
        if not good and self.missing==0 and self.position is not None:
            self.entry=dict(step=self.step,position=self.position.copy(),heading=self.heading,
                            position_sigma=5.,heading_sigma=math.radians(4),
                            references=[x.copy() for x in self.matcher.references[-16:]])
            self.entry_confirmations=0;self.entry_last_match=-1
            self.local_stats['entries']+=1
        if good and self.missing and self.position is not None and self.start is not None:
            end=self.position.copy();heading=self.heading
            if odometry is not None and np.shape(odometry)==(3,) and np.isfinite(odometry).all():
                distance,direction,rotation=map(float,odometry)
                end+=distance*np.array([math.cos(heading+direction),math.sin(heading+direction)])
                heading=wrap(heading+rotation)
            frames=self.pending
            corrected,blocks=rigid_blocks(frames,self.start,end,heading,np.asarray(gps),
                                          float(compass),self.step+1,self.BLOCK_SPAN)
            # Commit only this closed outage. Its free rays use RL11's .8
            # recovery weight; no global pose, reference or old map is shifted.
            for _,points,pos,angle,_,free in corrected:
                self._erase(self.hits,transform(free,pos,angle),.8)
                self._insert(self.hits,transform(points,pos,angle),.8)
            self.stats['anchored']+=len(corrected)
            self.local_stats['reprojected_scans']+=len(corrected)
            self.local_stats['closed_blocks']+=blocks
            self.local_events.append(dict(step=self.step+1,event='reproject_closed_outage',
                scans=len(corrected),blocks=blocks,missing=self.missing,
                endpoint_distance=float(np.linalg.norm(np.asarray(gps)-end)),
                endpoint_angle=float(wrap(compass-heading))))
            self.pending=[]  # parent performs recovery once, without duplicate insertion.
        super().update(gps,compass,odometry,distances,angles,objects)
        if self.missing and self.step%8==0:
            info=self.matcher.last_info
            if self.last_match==self.step and info.get('reason')=='line_registered':
                self.entry_confirmations+=1;self.entry_last_match=self.step
            else:self.entry_confirmations=0
        self.local_stats['maximum_pending']=max(self.local_stats['maximum_pending'],len(self.pending))

    def mask(self):
        if (self.LOCAL_BLOCKS and 0<self.missing<self.ENTRY_HOLD and self.entry is not None
                and self.entry_confirmations<2):
            # Preserve the whole entry tape, including free rays, until bounded
            # geometric confirmation. No irreversible dropping of entrance walls.
            self.local_stats['entry_publications_deferred']+=1
            return self.hits>=2.4
        return super().mask()


class MyDroneRedLocal13(MyDroneRedLines11):
    LOCAL_BLOCKS=True

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.LOCAL_BLOCKS:
            mapper=LocalBlockMapper(self.size_area)
            self._submission_mapper=mapper
            self._radio_maps=MappingSnapshots(self._map.shape,self._n,mapper)


class MyDroneRedLocal13Off(MyDroneRedLocal13):
    LOCAL_BLOCKS=False
