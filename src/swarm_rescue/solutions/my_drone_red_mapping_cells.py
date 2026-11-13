"""A4: one-factor A3 submission-map experiment, denser free-beam evidence.

Only radial free-space sampling changes (9 px -> 3 px). Beam selection, the
18 px endpoint guard, evidence weights, pose estimation and radio stay A3.
No truth, map names, navigation changes, or third-party code are introduced.
"""
import math
import numpy as np

from swarm_rescue.solutions.my_drone_red_mapping import (
    MyDroneRedMapping, SubmissionMapper, MappingSnapshots, free_points,
    wall_points, anchor_frames, scan_points, transform, wrap, _SF_LIDAR_RANGE,
)


def cell_free_points(distances, angles):
    """Sample measured beam interiors at map-cell spacing; never fill a wedge.

    This improves radial coverage, not an assertion that every intersected
    diagonal cell is known free. Each scan still contributes at most one
    negative vote per cell, through SubmissionMapper._erase's unique indices.
    """
    if distances is None or angles is None:
        return np.empty((0, 2), np.float32)
    distances = np.asarray(distances, float).ravel()
    angles = np.asarray(angles, float).ravel()
    if distances.shape != angles.shape:
        return np.empty((0, 2), np.float32)
    d, a = distances[::3], angles[::3]
    good = np.isfinite(d) & np.isfinite(a) & (d > 30)
    d, a = np.minimum(d[good], _SF_LIDAR_RANGE), a[good]
    samples = np.arange(12, _SF_LIDAR_RANGE, SubmissionMapper.CELL)
    lengths = np.broadcast_to(samples, (len(d), len(samples)))
    points = np.stack((np.cos(a), np.sin(a)), axis=1)[:, None, :] * lengths[:, :, None]
    return points[lengths < d[:, None] - 18].astype(np.float32)


class CellSubmissionMapper(SubmissionMapper):
    DENSE_FREE_RAYS = True

    def update(self, gps, compass, odometry, distances, angles, objects=()):
        if not self.DENSE_FREE_RAYS:
            return super().update(gps, compass, odometry, distances, angles, objects)
        # A3 update is deliberately kept local because A3 has no sampling hook.
        # Except for cell_free_points below, this is the frozen A3-v2 update.
        # Do not change localization here: that is a separate experiment.
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
        free = cell_free_points(distances, angles)
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


class MyDroneRedMappingCells(MyDroneRedMapping):
    DENSE_FREE_RAYS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mapper = CellSubmissionMapper(self.size_area)
        mapper.DENSE_FREE_RAYS = self.DENSE_FREE_RAYS
        self._submission_mapper = mapper
        self._radio_maps = MappingSnapshots(self._map.shape, self._n, mapper)


class MyDroneRedMappingCellsOff(MyDroneRedMappingCells):
    """Explicit rollback/ablation, without altering the retained A3 file."""
    DENSE_FREE_RAYS = False
