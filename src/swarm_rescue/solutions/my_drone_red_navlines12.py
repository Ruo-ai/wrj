"""RL12-N: independently apply legal wall geometry to RN7 navigation pose.

RL11 submission mapping and RN7 movement/placement remain inherited. All old
files remain unchanged. Current observations match past own references before
insertion; parallel corridors only constrain observable normal translation.
"""
from collections import Counter
import numpy as np
from swarm_rescue.solutions.my_drone_red_advanced2 import _M01_PoseFilter
from swarm_rescue.solutions.my_drone_red_lines11 import (
    MyDroneRedLines11, WallLineMatcher, fit_lines, line_registration)
from swarm_rescue.solutions.my_drone_red_stopturn import available_pose
from swarm_rescue.solutions.semifinal_localization import scan_points, wrap


class NavigationLinePose(_M01_PoseFilter):
    ENABLED = True

    def __init__(self, size):
        super().__init__()
        self.matcher=WallLineMatcher(size);self.observation=(None,None,())
        self.tick=0;self.good=0;self.rejections=0;self.stats=Counter()

    def prepare(self, distances, angles, objects=()):
        self.observation=(distances,angles,objects)

    def update(self, gps, compass, odometry):
        self.tick+=1
        result=super().update(gps,compass,odometry)
        if not self.ENABLED or result is None:return result
        absolute=all(available_pose(gps,compass))
        self.good=self.good+1 if absolute else 0
        if self.good==8:
            # Drop outage reference when absolute sensing has stably recovered.
            self.matcher.frames.clear();self.matcher.distance=None
            self.matcher.references.clear();self.rejections=0
            self.stats['recovery_reference_reset']+=1
        if self.tick%8:return result
        distances,angles,objects=self.observation
        self.matcher.lines=fit_lines(distances,angles,objects)
        self.matcher.absolute=absolute
        points=scan_points(distances,angles,objects)
        accepted=False
        if not absolute:
            offset,info=line_registration(self.matcher.lines,self.matcher.references,self.position,self.heading)
            self.stats['attempts']+=1
            if offset is not None:
                self.position+=offset[:2];self.heading=wrap(self.heading+offset[2]);accepted=True
                self.stats['accepted']+=1;self.stats['rank_'+str(info['rank'])]+=1
            else:self.stats['reject_'+info['reason']]+=1
        self.rejections=0 if absolute or accepted else self.rejections+1
        if self.rejections>=4:
            self.matcher.frames.clear();self.matcher.distance=None
            self.matcher.references.clear();self.rejections=0
            self.stats['local_reference_reset']+=1
        if absolute or accepted or self.matcher.distance is None:
            self.matcher.add(points,self.position,self.heading)
        return self.position.copy(),self.heading


class MyDroneRedNavLines12(MyDroneRedLines11):
    NAV_LINE_POSE=True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.NAV_LINE_POSE:self._pose_filter=NavigationLinePose(self.size_area)

    def _update_observation(self):
        if self.NAV_LINE_POSE:
            self._pose_filter.prepare(self.lidar_values(),self.lidar_rays_angles(),self.semantic_values())
        super()._update_observation()


class MyDroneRedNavLines12Off(MyDroneRedNavLines12):
    NAV_LINE_POSE=False
