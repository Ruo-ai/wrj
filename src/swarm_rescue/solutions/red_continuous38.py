"""Continuous legal heading constraints and shared navigation/mapping pose.

The orientation family is learned with absolute sensing, never supplied from
a map. During an outage its reference is immutable. Translation is corrected
only in the observable wall-normal subspace. All failed matches retain the
pre-outage reference; newly seen geometry remains a separate local reference.
"""
from collections import Counter
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_lines11 import fit_lines, line_registration
from swarm_rescue.solutions.my_drone_red_mapping import wall_points, free_points
from swarm_rescue.solutions.red_structural28 import StructuralMapper28, coherent_axis, quarter_angle
from swarm_rescue.solutions.semifinal_localization import scan_points, transform, wrap


def fixed_heading_translation(lines, references, position, heading):
    """Fit translation with observed heading held fixed; rank-one is legal."""
    if len(lines) < 2 or len(references) < 2:
        return None, {'reason': 'few_lines'}
    refs = np.asarray(references)
    centers = refs.mean(axis=1)
    vectors = refs[:, 1]-refs[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    tangents = vectors/np.maximum(lengths[:, None], 1e-9)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    directions = np.arctan2(tangents[:, 1], tangents[:, 0])
    selected = {}
    for line in lines:
        world = transform(line, position, heading)
        center = world.mean(axis=0)
        vector = world[1]-world[0]
        length = float(np.linalg.norm(vector))
        angle = math.atan2(vector[1], vector[0])
        difference = (directions-angle+math.pi/2) % math.pi-math.pi/2
        residual = np.sum(normals*(center-centers), axis=1)
        along = np.abs(np.sum(tangents*(center-centers), axis=1))
        valid = ((np.abs(difference) < .12) & (np.abs(residual) < 22)
                 & (along < (lengths+length)/2+20))
        score = np.where(valid, np.abs(residual)+40*np.abs(difference), np.inf)
        index = int(np.argmin(score))
        if np.isfinite(score[index]) and (index not in selected or score[index] < selected[index][0]):
            selected[index] = (score[index], residual[index], min(length, 120.))
    if len(selected) < 2:
        return None, {'reason': 'few_matches'}
    keys = list(selected)
    n = normals[keys]
    residual = np.array([selected[k][1] for k in keys])
    weights = np.array([selected[k][2] for k in keys]); weights /= weights.sum()
    values, vectors = np.linalg.eigh((n.T*weights)@n)
    observable = values > max(.04, .12*values.max())
    inverse = (vectors*np.where(observable, 1/np.maximum(values, 1e-12), 0.))@vectors.T
    shift = -inverse@(n.T@(weights*residual))
    error = float(np.sqrt(np.sum(weights*(residual+n@shift)**2)))
    info = dict(reason='fixed_heading', rank=int(observable.sum()), residual=error,
                matches=len(keys))
    if np.linalg.norm(shift) > 10 or error > 4.5:
        return None, dict(info, reason='translation_bound')
    return np.r_[shift, 0.], info


def segmented_recovery(frames, metadata, endpoint, measured, heading_delta):
    """Piecewise correction weighted by observed uncertainty, not elapsed time.

    Knots are short blocks of observations. Heading already constrained by
    geometry is not rotated again by an unrelated compass endpoint residual.
    This is a bounded chain correction, not a claim of global SLAM.
    """
    if not frames:
        return []
    blocks = []
    for start in range(0, len(frames), 16):
        stop = min(start+16, len(frames))
        uncertainty = sum(metadata[j]['uncertainty'] for j in range(start, stop))
        blocks.append((start, stop, max(.1, uncertainty)))
    total = sum(b[2] for b in blocks)
    delta = np.asarray(measured)-endpoint
    result = []; accumulated = 0.
    weak_total = sum(not m['heading_supported'] for m in metadata)
    weak_seen = 0
    for start, stop, uncertainty in blocks:
        for j in range(start, stop):
            frame = frames[j]
            fraction = (accumulated+uncertainty*(j-start+1)/(stop-start))/total
            step, points, position, heading, weight, free = frame
            supported = metadata[j]['heading_supported']
            weak_seen += int(not supported)
            rotation = 0. if supported else heading_delta*weak_seen/max(1, weak_total)
            result.append((step, points, position+fraction*delta, wrap(heading+rotation), weight, free))
        accumulated += uncertainty
    return result


class ContinuousMapper38(StructuralMapper28):
    """Original raster/renderer interface; continuous pose also serves flight."""
    AXIS_GAIN = .75
    AXIS_STEP_BOUND = math.radians(6)
    AXIS_CAPTURE_BOUND = math.radians(20)

    def __init__(self, size):
        super().__init__(size)
        self.trusted_references38 = []
        self.outage_axis38 = None
        self.axis_last38 = -1000
        self.pending_meta38 = []
        self.events38 = []
        self.stats38 = Counter()
        self.last_info38 = {}
        self.last_translation_rank38 = 0
        self.recovery38 = None
        self.pose_step38 = 0
        self.last_position38 = None

    def _event38(self, kind, **values):
        if len(self.events38) < 2500:
            self.events38.append(dict(step=self.pose_step38, event=kind, **values))

    def _recover38(self, gps, compass):
        correction = np.asarray(gps)-self.position
        heading_delta = wrap(compass-self.heading)
        corrected = segmented_recovery(self.pending, self.pending_meta38,
                                       self.position, np.asarray(gps), heading_delta)
        for _, points, pos, heading, weight, free in corrected:
            # A recovery endpoint does not promote weak scans to .8 confidence.
            self._erase(self.hits, transform(free, pos, heading), weight)
            self._insert(self.hits, transform(points, pos, heading), weight)
        self.recovery38 = {int(f[0]): (f[2].copy(), float(f[3])) for f in corrected}
        self.stats['anchored'] += len(corrected)
        self.stats['relocks'] += 1
        self.stats38['recovered_segments'] += math.ceil(len(corrected)/16)
        self._event38('recovery', frames=len(corrected), segments=math.ceil(len(corrected)/16),
                      position_correction=float(np.linalg.norm(correction)),
                      heading_delta=float(heading_delta))
        self.pending.clear(); self.pending_meta38.clear()
        self.matcher.frames.clear(); self.matcher.references.clear(); self.matcher.distance = None
        self.position = np.asarray(gps, float).copy(); self.heading = float(compass)

    def update(self, gps, compass, odometry, distances, angles, objects=()):
        self.pose_step38 += 1
        tick = self.pose_step38
        sealed = self.final_mask is not None
        if not sealed:
            self.step = tick
        self.recovery38 = None
        good = bool(gps is not None and np.shape(gps) == (2,) and np.isfinite(gps).all()
                    and compass is not None and np.ndim(compass) == 0 and np.isfinite(compass))
        odom_good = odometry is not None and np.shape(odometry) == (3,) and np.isfinite(odometry).all()
        self.last_position38 = None if self.position is None else self.position.copy()
        if self.position is None:
            if not good:
                return
            self.position = np.asarray(gps, float).copy(); self.heading = float(compass)
        else:
            if not good and not self.missing:
                self.start = (tick-1, self.position.copy())
                self.outage_axis38 = self.structure28.axis
                self.trusted_references38 = [x.copy() for x in self.matcher.references]
                self.stats38['outages'] += 1
                self._event38('outage', axis=self.outage_axis38,
                              trusted_lines=len(self.trusted_references38))
            if odom_good:
                d, alpha, rotation = odometry
                self.position += d*np.array([math.cos(self.heading+alpha), math.sin(self.heading+alpha)])
                self.heading = wrap(self.heading+rotation)
            if good and self.missing:
                if not sealed:
                    self._recover38(gps, compass)
                else:
                    self.position = np.asarray(gps, float).copy(); self.heading = float(compass)
                    self.matcher.frames.clear(); self.matcher.references.clear(); self.matcher.distance = None
            elif good:
                self.position += .12*(np.asarray(gps)-self.position)
                self.heading = wrap(self.heading+.15*wrap(compass-self.heading))
        self.missing = 0 if good else self.missing+1
        unstable = self.structure28.motion(odometry)
        lines = fit_lines(distances, angles, objects) if tick % 4 == 0 else []
        if good and not unstable and tick % 4 == 0:
            corrected = self.structure28.observe(lines, compass)
            if self.structure28.last_correction_applied:
                self.heading = wrap(corrected)
        elif not good and tick % 4 == 0 and self.outage_axis38 is not None:
            geometry = coherent_axis(lines, self.heading)
            if geometry is not None:
                correction = quarter_angle(self.outage_axis38-geometry[0])
                if abs(correction) <= self.AXIS_CAPTURE_BOUND:
                    applied = float(np.clip(self.AXIS_GAIN*correction,
                                            -self.AXIS_STEP_BOUND, self.AXIS_STEP_BOUND))
                    self.heading = wrap(self.heading+applied)
                    self.axis_last38 = tick
                    self.stats38['axis_corrections'] += 1
                    self.last_info38 = dict(reason='observed_axis', correction=applied,
                                            concentration=geometry[1])
                else:
                    self.stats38['axis_ambiguity_rejected'] += 1
        axis_fresh = not good and tick-self.axis_last38 <= 8
        accepted = False; info = {}
        if tick % 8 == 0:
            scan = scan_points(None if good and unstable else distances, angles, objects)
            self.matcher.lines = [] if good and unstable else lines
            self.matcher.absolute = good
            if not good:
                if axis_fresh:
                    offset, info = fixed_heading_translation(lines, self.trusted_references38,
                                                            self.position, self.heading)
                    if offset is None:
                        offset, info = fixed_heading_translation(lines, self.matcher.references,
                                                                self.position, self.heading)
                else:
                    offset, info = self.matcher.match(scan, self.position, self.heading)
                    if offset is None:
                        offset, info = line_registration(lines, self.trusted_references38,
                                                         self.position, self.heading)
                self.last_info38 = dict(info)
                if offset is not None:
                    self.last_translation_rank38 = info.get('rank', 0)
                    self.position += offset[:2]; self.heading = wrap(self.heading+offset[2])
                    self.last_match = tick; accepted = True
                    self.stats['matched'] += 1
                else:
                    self.stats['rejected'] += 1
                    self.stats38['reference_preserved_on_reject'] += 1
            # Local extension can use observed heading, but never changes the
            # frozen absolute reference or grants absolute translation trust.
            if good or accepted or axis_fresh:
                self.matcher.add(scan, self.position, self.heading)
                if not good and not accepted:
                    self.stats38['provisional_local_extensions'] += 1
            if not good:
                self._event38('match', accepted=accepted, axis_supported=axis_fresh,
                              info=info, protected_lines=len(self.trusted_references38))
        if sealed:
            self.stats['after_cutoff_observations_not_mapped'] = self.stats.get('after_cutoff_observations_not_mapped', 0)+1
            return
        if tick % 4:
            return
        values = None if good and unstable else distances
        points = wall_points(values, angles, objects); free = free_points(values, angles)
        self.stats['scans'] += 1
        if good:
            self._erase(self.hits, transform(free, self.position, self.heading), 1.)
            self._insert(self.hits, transform(points, self.position, self.heading), 1.)
        else:
            weight = .45 if tick-self.last_match <= 24 else .2
            self.pending.append((tick, points, self.position.copy(), self.heading, weight, free))
            # Error allocation respects the measured support in each block.
            uncertainty = 1. if axis_fresh and tick-self.last_match <= 8 and self.last_translation_rank38 == 2 else 6.
            self.pending_meta38.append(dict(heading_supported=axis_fresh, uncertainty=uncertainty))
            self.stats['buffered'] += 1
            if len(self.pending) > self.MAX_PENDING:
                self.pending.pop(0); self.pending_meta38.pop(0); self.stats['dropped'] += 1


class SharedPose38:
    """Read the mapper's already updated pose; never integrate a second time."""
    def __init__(self, mapper):
        self.mapper = mapper

    @property
    def absolute_missing_steps(self):
        return self.mapper.missing

    @property
    def position(self):
        return self.mapper.position

    @property
    def heading(self):
        return self.mapper.heading

    def update(self, gps, compass, odometry):
        if self.mapper.position is None:
            return None
        return self.mapper.position.copy(), self.mapper.heading
