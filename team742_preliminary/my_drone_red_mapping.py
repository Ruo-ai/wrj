"""A3: independent submission mapping; A0 motion and placement stay inherited.

Legal sensor observations only. Local scan matching and endpoint anchoring are
bounded traditional estimation, not a global SLAM system. Confidence weights
are engineering heuristics, not calibrated probabilities. No map-file access.
"""
import math
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_red_advanced2 import (
    MyDroneRedAdvanced2, _M01_SnapshotStore, _M03_fit_scan, _SF_LIDAR_RANGE,
    _SF_HIT_MARGIN,
)
from swarm_rescue.solutions.semifinal_localization import LocalMatcher, scan_points, transform, wrap


def wall_points(distances, angles, objects=()):
    """Interpolate only adjacent observed hits, never max-range/excluded gaps."""
    if distances is None or angles is None:
        return np.empty((0, 2), np.float32)
    d = np.asarray(distances, float).ravel()
    a = np.asarray(angles, float).ravel()
    if d.shape != a.shape:
        return np.empty((0, 2), np.float32)
    excluded = np.zeros(len(d), bool)
    for obj in objects or ():
        if np.isfinite([obj.distance, obj.angle]).all():
            delta = np.abs((a-obj.angle+math.pi) % (2*math.pi)-math.pi)
            excluded |= (delta < .12) & (np.abs(d-obj.distance) < 22)
    d, a, _ = _M03_fit_scan(d, a, excluded)
    valid = np.isfinite(d) & np.isfinite(a) & (d > 4) & (d < _SF_LIDAR_RANGE-_SF_HIT_MARGIN) & ~excluded
    p = np.column_stack((d*np.cos(a), d*np.sin(a)))
    pairs = np.flatnonzero(valid[:-1] & valid[1:] & (np.linalg.norm(np.diff(p, axis=0), axis=1) < 18))
    if not len(pairs):
        return p[valid].astype(np.float32)
    delta = p[pairs+1]-p[pairs]
    counts = np.maximum(1, np.ceil(np.linalg.norm(delta, axis=1)/2).astype(int))
    index = np.repeat(np.arange(len(pairs)), counts)
    offset = np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts, counts)
    interpolated = p[pairs[index]]+delta[index]*(offset/counts[index])[:, None]
    return np.vstack((p[valid], interpolated)).astype(np.float32)


def anchor_frames(frames, start, end_position, end_heading, measured_position, measured_heading, end_step):
    """Correct only one buffered outage, fixing its start and measured end.

    Position/angle corrections vary continuously along the segment. Historical
    confirmed map evidence is never globally shifted. No hidden truth is used.
    """
    start_step, origin = start
    angle = wrap(measured_heading-end_heading)
    c, s = math.cos(angle), math.sin(angle)
    rotated_end = np.array([[c, -s], [s, c]]) @ (end_position-origin)
    translation = measured_position-origin-rotated_end
    result = []
    for frame in frames:
        step, points, position, heading, weight = frame[:5]
        u = float(np.clip((step-start_step)/max(1, end_step-start_step), 0, 1))
        c, s = math.cos(u*angle), math.sin(u*angle)
        corrected = origin+np.array([[c, -s], [s, c]]) @ (position-origin)+u*translation
        result.append((step, points, corrected, wrap(heading+u*angle), weight, *frame[5:]))
    return result


def free_points(distances, angles):
    """Observed beam interiors only; stop 18 px before an obstacle endpoint."""
    if distances is None or angles is None:
        return np.empty((0, 2), np.float32)
    d = np.asarray(distances, float).ravel()[::3]
    a = np.asarray(angles, float).ravel()[::3]
    if d.shape != a.shape:
        return np.empty((0, 2), np.float32)
    good = np.isfinite(d) & np.isfinite(a) & (d > 30)
    d, a = np.minimum(d[good], _SF_LIDAR_RANGE), a[good]
    samples = np.arange(12, _SF_LIDAR_RANGE, 9)
    lengths = np.broadcast_to(samples, (len(d), len(samples)))
    points = np.stack((np.cos(a), np.sin(a)), axis=1)[:, None, :]*lengths[:, :, None]
    return points[lengths < d[:, None]-18].astype(np.float32)


class SubmissionMapper:
    CELL = 3
    MAX_PENDING = 1400  # > one scan per four ticks for a full official round

    def __init__(self, size):
        self.width, self.height = map(int, size)
        self.shape = ((self.height+2)//3, (self.width+2)//3)
        self.hits = np.zeros(self.shape, np.float32)
        self.position = None
        self.heading = 0.
        self.step = 0
        self.missing = 0
        self.matcher = LocalMatcher(size)
        self.pending = []
        self.start = None
        self.last_match = -100
        self.rejected_streak = 0
        self.stats = dict(scans=0, buffered=0, anchored=0, relocks=0,
                          matched=0, rejected=0, reference_resets=0, dropped=0)

    def update(self, gps, compass, odometry, distances, angles, objects=()):
        self.step += 1
        good = gps is not None and np.shape(gps) == (2,) and np.isfinite(gps).all()
        good = bool(good and compass is not None and np.ndim(compass) == 0 and np.isfinite(compass))
        if self.position is None:
            if not good:
                return
            self.position = np.array(gps, float)
            self.heading = float(compass)
        else:
            previous_position = self.position.copy()
            if odometry is not None and np.shape(odometry) == (3,) and np.isfinite(odometry).all():
                distance, direction, rotation = odometry
                self.position += distance*np.array([math.cos(self.heading+direction), math.sin(self.heading+direction)])
                self.heading = wrap(self.heading+rotation)
            if not good and self.missing == 0:
                self.start = (self.step-1, previous_position)
            if good and self.missing:
                corrected = anchor_frames(self.pending, self.start, self.position, self.heading,
                                          np.asarray(gps), float(compass), self.step)
                for _, points, pos, heading, _, free in corrected:
                    self._erase(self.hits, transform(free, pos, heading), .8)
                    self._insert(self.hits, transform(points, pos, heading), .8)
                self.stats['anchored'] += len(corrected)
                self.stats['relocks'] += 1
                self.pending.clear()
                self.position = np.array(gps, float)
                self.heading = float(compass)
                self.matcher.frames.clear()
                self.matcher.distance = None
                self.rejected_streak = 0
            elif good:
                self.position += .12*(np.asarray(gps)-self.position)
                self.heading = wrap(self.heading+.15*wrap(compass-self.heading))
        self.missing = 0 if good else self.missing+1
        if self.step % 8 == 0:
            scan = scan_points(distances, angles, objects)
            accepted = False
            if not good:
                offset, info = self.matcher.match(scan, self.position, self.heading)
                if offset is not None:
                    self.position += offset[:2]
                    self.heading = wrap(self.heading+offset[2])
                    self.last_match = self.step
                    self.stats['matched'] += 1
                    accepted = True
                else:
                    self.stats['rejected'] += 1
            self.rejected_streak = 0 if good or accepted else self.rejected_streak+1
            # A rejected old reference must not indefinitely stop local sensing.
            # Starting a new local reference grants NO absolute-position trust.
            if self.rejected_streak >= 4:
                self.matcher.frames.clear()
                self.matcher.distance = None
                self.stats['reference_resets'] += 1
                self.rejected_streak = 0
            if good or accepted or self.matcher.distance is None:
                self.matcher.add(scan, self.position, self.heading)
        if self.step % 4:
            return
        points = wall_points(distances, angles, objects)
        free = free_points(distances, angles)
        self.stats['scans'] += 1
        if good:
            self._erase(self.hits, transform(free, self.position, self.heading), 1.)
            self._insert(self.hits, transform(points, self.position, self.heading), 1.)
        else:
            weight = .45 if self.step-self.last_match <= 24 else .2
            self.pending.append((self.step, points, self.position.copy(), self.heading, weight, free))
            self.stats['buffered'] += 1
            if len(self.pending) > self.MAX_PENDING:
                self.pending.pop(0)
                self.stats['dropped'] += 1

    def _indices(self, points):
        x = np.floor((points[:, 0]+self.width/2)/self.CELL).astype(int)
        y = np.floor((self.height/2-points[:, 1])/self.CELL).astype(int)
        valid = (x >= 0) & (x < self.shape[1]) & (y >= 0) & (y < self.shape[0])
        return np.unique(y[valid]*self.shape[1]+x[valid])

    def _insert(self, grid, points, weight):
        index = self._indices(points)
        grid.ravel()[index] = np.minimum(30, grid.ravel()[index]+weight)

    def _erase(self, grid, points, weight):
        index = self._indices(points)
        grid.ravel()[index] = np.maximum(0, grid.ravel()[index]-.35*weight)

    def mask(self):
        provisional = np.zeros(self.shape, np.float32)
        # Unclosed segments remain provisional and need repeated support.
        # Their free beams may clean their OWN local evidence, never erase the
        # independently anchored map before an absolute recovery.
        for _, points, position, heading, weight, free in self.pending:
            self._erase(provisional, transform(free, position, heading), weight)
            self._insert(provisional, transform(points, position, heading), weight)
        return (self.hits >= 2.4) | (provisional >= 2.4)

    def to_pixels(self, mask):
        pixels = np.repeat(np.repeat(mask, 3, 0), 3, 1)[:self.height, :self.width].astype(np.uint8)
        pixels = cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(pixels, connectivity=8)
        keep = np.zeros(count, np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 27
        return keep[labels]


class MappingSnapshots(_M01_SnapshotStore):
    """Preserve A0's navigation masks; append independently versioned map data."""
    def __init__(self, shape, number_drones, mapper):
        super().__init__(shape, number_drones)
        self.mapper = mapper
        self.mapping_masks = {}
        self.bytes_per_mask = (math.prod(shape)+7)//8

    def publish(self, source, sequence, mask):
        nav = np.packbits(np.asarray(mask, np.uint8).ravel()).tobytes()
        mapped = np.packbits(self.mapper.mask().ravel()).tobytes()
        self.accept((int(source), int(sequence), nav+mapped))

    def accept(self, snapshot):
        if not isinstance(snapshot, tuple) or len(snapshot) != 3:
            return False
        source, sequence, payload = snapshot
        if not isinstance(payload, bytes) or len(payload) != 2*self.bytes_per_mask:
            return False
        if not super().accept((source, sequence, payload[:self.bytes_per_mask])):
            return False
        self.snapshots[source] = snapshot  # relay both masks without creating extra votes
        self.mapping_masks[source] = np.unpackbits(np.frombuffer(payload[self.bytes_per_mask:], np.uint8))[:math.prod(self.shape)].reshape(self.shape).astype(bool)
        return True

    def submission(self):
        result = np.zeros(self.shape, bool)
        for mask in self.mapping_masks.values():
            result |= mask
        return result


class MyDroneRedMapping(MyDroneRedAdvanced2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submission_mapper = SubmissionMapper(self.size_area)
        self._radio_maps = MappingSnapshots(self._map.shape, self._n, self._submission_mapper)

    def _update_observation(self):
        self._submission_mapper.update(self.measured_gps_position(), self.measured_compass_angle(),
            self.odometer_values(), self.lidar_values(), self.lidar_rays_angles(), self.semantic_values())
        super()._update_observation()

    def _submit(self):
        if self._exploration_submitted:
            return
        self._radio_maps.publish(int(self.identifier or 0), self._steps, self._map.mask())
        merged = self._radio_maps.submission()
        self.submit_exploration_map(self._submission_mapper.to_pixels(merged))
        self._exploration_submitted = True
        self._event('submitted', sources={str(k): int(v.sum()) for k, v in self._radio_maps.mapping_masks.items()},
                    cells=int(merged.sum()), mapping_stats=self._submission_mapper.stats,
                    pending_scans=len(self._submission_mapper.pending), placements=len(self._placements))
