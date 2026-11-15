"""RL12: bounded lag smoothing of provisional RL11 outage scans.

All integration, matching and recovery remain RL11. A successful wall match
also corrects the short uncommitted interval since the last match, instead of
leaving its earlier scan in the pre-correction frame. No truth, region identity,
new wall filling, communication or navigation changes.
"""
import numpy as np
from swarm_rescue.solutions.my_drone_red_lines11 import MyDroneRedLines11, WallLineMapper, WallLineMatcher
from swarm_rescue.solutions.my_drone_red_mapping import MappingSnapshots, anchor_frames
from swarm_rescue.solutions.my_drone_red_mapping_anchor import StableAnchorMapper


class RecordingLineMatcher(WallLineMatcher):
    def __init__(self, size):
        super().__init__(size)
        self.accepted_pose = None

    def match(self, points, position, heading):
        offset, info = super().match(points, position, heading)
        # Only the geometrically checked line path authorizes retrospective work.
        self.accepted_pose = ((position.copy(), float(heading), offset.copy())
                              if offset is not None and info.get('reason') == 'line_registered' else None)
        return offset, info


class ConsistentMapper(WallLineMapper):
    SMOOTH_PENDING = True
    MAX_LAG = 32

    def __init__(self, size):
        super().__init__(size)
        self.matcher = RecordingLineMatcher(size)
        self.lag_anchor = None
        self.stats.update(lag_corrections=0, lag_scans=0, lag_max_scans=0)

    def update(self, gps, compass, odometry, distances, angles, objects=()):
        if not self.SMOOTH_PENDING:
            return super().update(gps, compass, odometry, distances, angles, objects)
        self.matcher.accepted_pose = None
        previous = None if self.position is None else (self.step, self.position.copy())
        was_missing = self.missing
        super().update(gps, compass, odometry, distances, angles, objects)
        if not self.missing:
            self.lag_anchor = None
            return
        if not was_missing:
            self.lag_anchor = previous
        match = self.matcher.accepted_pose
        if match is None:
            return
        before_position, before_heading, offset = match
        anchor = self.lag_anchor
        if anchor is None or self.step-anchor[0] > self.MAX_LAG:
            # Bounded uncertainty: do not bend a long unobserved interval.
            choices = [f for f in self.pending if 0 < self.step-f[0] <= self.MAX_LAG]
            anchor = (choices[0][0], choices[0][2].copy()) if choices else None
        if anchor is not None:
            start = next((i for i, f in enumerate(self.pending) if f[0] > anchor[0]), len(self.pending))
            # The current frame is already written at the corrected pose by A3.
            stop = next((i for i in range(start, len(self.pending)) if self.pending[i][0] >= self.step), len(self.pending))
            frames = self.pending[start:stop]
            if frames:
                self.pending[start:stop] = anchor_frames(frames, anchor, before_position, before_heading,
                                                        self.position, self.heading, self.step)
                self.stats['lag_corrections'] += 1
                self.stats['lag_scans'] += len(frames)
                self.stats['lag_max_scans'] = max(self.stats['lag_max_scans'], len(frames))
        self.lag_anchor = (self.step, self.position.copy())


class MyDroneRedConsistent12(MyDroneRedLines11):
    SMOOTH_PENDING = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mapper = ConsistentMapper(self.size_area)
        mapper.SMOOTH_PENDING = self.SMOOTH_PENDING
        self._submission_mapper = mapper
        self._radio_maps = MappingSnapshots(self._map.shape, self._n, mapper)


class MyDroneRedConsistent12Off(MyDroneRedConsistent12):
    SMOOTH_PENDING = False


class ContinuousHeadingMatcher(WallLineMatcher):
    """Use historical wall directions also while compass is available.

    Only a bounded heading correction is applied, before this scan becomes a
    reference. This prevents the reference being continually reoriented solely
    by correlated compass noise. It cannot remove an unknown global yaw bias.
    """
    def __init__(self, size, owner):
        super().__init__(size)
        self.owner = owner

    def add(self, points, position, heading):
        if self.absolute and self.frames:
            from swarm_rescue.solutions.my_drone_red_lines11 import line_registration
            from swarm_rescue.solutions.semifinal_localization import wrap
            offset, info = line_registration(self.lines, self.references, position, heading)
            self.stats['absolute_heading_attempts'] += 1
            if offset is not None:
                correction = float(np.clip(.5*offset[2], -.035, .035))
                self.owner.heading = wrap(heading+correction)
                heading = self.owner.heading
                self.stats['absolute_heading_accepted'] += 1
        return super().add(points, position, heading)


class ContinuousHeadingMapper(WallLineMapper):
    CONTINUOUS_HEADING = True

    def __init__(self, size):
        super().__init__(size)
        if self.CONTINUOUS_HEADING:
            self.matcher = ContinuousHeadingMatcher(size, self)


class MyDroneRedHeading12(MyDroneRedLines11):
    """Independent absolute-heading reference experiment; lag smoothing OFF."""
    CONTINUOUS_HEADING = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.CONTINUOUS_HEADING:
            mapper = ContinuousHeadingMapper(self.size_area)
            self._submission_mapper = mapper
            self._radio_maps = MappingSnapshots(self._map.shape, self._n, mapper)


class StableLineMapper(StableAnchorMapper, WallLineMapper):
    """Reuse retained A5 recovery, independently of the other RL12 experiments.

    Cooperative inheritance calls RL11's wall fitter on every normal/recovery
    update. All recovery thresholds remain exactly A5, with no parameter search.
    """
    pass


class MyDroneRedStable12(MyDroneRedLines11):
    STABLE_ANCHOR = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mapper = StableLineMapper(self.size_area)
        mapper.STABLE_ANCHOR = self.STABLE_ANCHOR
        self._submission_mapper = mapper
        self._radio_maps = MappingSnapshots(self._map.shape, self._n, mapper)
