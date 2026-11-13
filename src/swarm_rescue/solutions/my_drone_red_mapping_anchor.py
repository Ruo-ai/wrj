"""A5: bounded multi-frame absolute re-anchoring on top of retained A3.

Only the recovery of the submission mapper changes. A3's scan matcher, free
evidence, radio, motion and placement are reused unchanged. No truth or map
identity is read. A stable measurement bias is NOT observable by this method.
"""
from collections import deque
import math
import numpy as np

from swarm_rescue.solutions.my_drone_red_mapping import (
    SubmissionMapper, MyDroneRedMapping, MappingSnapshots, anchor_frames,
    transform, wrap,
)


def absolute_pose_available(gps, compass):
    return bool(gps is not None and np.shape(gps) == (2,) and np.isfinite(gps).all()
                and compass is not None and np.ndim(compass) == 0 and np.isfinite(compass))


def fit_recovery(samples, origin):
    """Fit one SE(2) correction to moving poses and legal absolute readings.

    Each row is (estimated_position, estimated_heading, GPS, compass).
    Rotation uses a circular median of compass-minus-estimate differences;
    translation is a coordinate median AFTER motion/rotation compensation.
    Scatter measures internal agreement, not global accuracy or independence.
    """
    if len(samples) < 3:
        return None, dict(reason='too_few_samples', samples=len(samples))
    positions = np.array([s[0] for s in samples], float)
    gps = np.array([s[2] for s in samples], float)
    deltas = np.array([wrap(s[3]-s[1]) for s in samples])
    pivot = float(deltas[0])
    unwrapped = pivot + (deltas-pivot+math.pi) % (2*math.pi)-math.pi
    angle = wrap(float(np.median(unwrapped)))
    rotated = transform(positions-origin, np.zeros(2), angle)
    translations = gps-origin-rotated
    translation = np.median(translations, axis=0)
    position_error = np.linalg.norm(translations-translation, axis=1)
    angle_error = np.abs((deltas-angle+math.pi) % (2*math.pi)-math.pi)
    # Engineering agreement gates, not calibrated statistical confidence.
    inliers = (position_error <= 10.) & (angle_error <= .12)
    required = max(3, math.ceil(.75*len(samples)))
    diagnostics = dict(samples=len(samples), inliers=int(inliers.sum()),
                       position_scatter_p90=float(np.quantile(position_error, .9)),
                       heading_scatter_p90=float(np.quantile(angle_error, .9)),
                       reason='accepted' if inliers.sum() >= required else 'inconsistent')
    if inliers.sum() < required:
        return None, diagnostics
    # Refine only from mutually consistent measurements, still in one frame.
    angle = wrap(float(np.median(unwrapped[inliers])))
    rotated = transform(positions[inliers]-origin, np.zeros(2), angle)
    translation = np.median(gps[inliers]-origin-rotated, axis=0)
    return (translation, angle), diagnostics


class StableAnchorMapper(SubmissionMapper):
    STABLE_ANCHOR = True
    RECOVERY_SAMPLES = 12
    MAX_RECOVERY_TICKS = 48
    RECORD_ANCHORS = False

    def __init__(self, size):
        super().__init__(size)
        self.recovery_samples = deque(maxlen=self.RECOVERY_SAMPLES)
        self.recovery_started = None
        self.anchor_events = deque(maxlen=32)
        self.stats.update(anchor_windows=0, anchor_commits=0, anchor_fallbacks=0,
                          anchor_rejections=0, anchor_interruptions=0,
                          max_pending_scans=0, max_recovery_samples=0)

    def _record(self, event, **values):
        if self.RECORD_ANCHORS:
            self.anchor_events.append(dict(event=event, step=int(self.step), **values))

    def _clear_recovery(self):
        self.recovery_samples.clear()
        self.recovery_started = None

    def update(self, gps, compass, odometry, distances, angles, objects=()):
        if not self.STABLE_ANCHOR:
            return super().update(gps, compass, odometry, distances, angles, objects)
        good = absolute_pose_available(gps, compass)
        recovering = self.position is not None and self.missing > 0 and good
        if not recovering:
            if not good and self.recovery_samples:
                self.recovery_samples.clear()
                self.stats['anchor_interruptions'] += 1
                self._record('absolute_dropout')
            result = super().update(gps, compass, odometry, distances, angles, objects)
            self.stats['max_pending_scans'] = max(self.stats['max_pending_scans'], len(self.pending))
            return result
        if self.recovery_started is None:
            self.recovery_started = self.step+1
            self.stats['anchor_windows'] += 1
        if self.step+1-self.recovery_started >= self.MAX_RECOVERY_TICKS:
            # Bound the additional delay. Fall back to A3's original recovery
            # only on an actually available absolute observation; never invent it.
            self.stats['anchor_fallbacks'] += 1
            self._record('fallback_to_a3', waited=self.step+1-self.recovery_started)
            self._clear_recovery()
            return super().update(gps, compass, odometry, distances, angles, objects)
        # Keep scans provisional while collecting recovery evidence. Local
        # matching remains match-before-add inside A3; no global map freeze.
        super().update(None, None, odometry, distances, angles, objects)
        self.recovery_samples.append((self.position.copy(), float(self.heading),
                                      np.asarray(gps, float).copy(), float(compass)))
        self.stats['max_pending_scans'] = max(self.stats['max_pending_scans'], len(self.pending))
        self.stats['max_recovery_samples'] = max(self.stats['max_recovery_samples'], len(self.recovery_samples))
        if len(self.recovery_samples) < self.RECOVERY_SAMPLES:
            return
        origin = self.start[1]
        correction, diagnostics = fit_recovery(self.recovery_samples, origin)
        if correction is None:
            self.stats['anchor_rejections'] += 1
            self._record('inconsistent_recovery', **diagnostics)
            return
        translation, angle = correction
        endpoint = origin + transform((self.position-origin)[None, :], np.zeros(2), angle)[0] + translation
        end_heading = wrap(self.heading+angle)
        corrected = anchor_frames(self.pending, self.start, self.position, self.heading,
                                  endpoint, end_heading, self.step)
        for _, points, pos, heading, _, free in corrected:
            self._erase(self.hits, transform(free, pos, heading), .8)
            self._insert(self.hits, transform(points, pos, heading), .8)
        self.stats['anchored'] += len(corrected)
        self.stats['relocks'] += 1
        self.stats['anchor_commits'] += 1
        self._record('stable_anchor', buffered=len(corrected),
                     correction_px=float(np.linalg.norm(endpoint-self.position)),
                     correction_rad=float(angle), **diagnostics)
        self.pending.clear()
        self.position = endpoint
        self.heading = end_heading
        self.missing = 0
        self.matcher.frames.clear()
        self.matcher.distance = None
        self.rejected_streak = 0
        self._clear_recovery()


class MyDroneRedMappingAnchor(MyDroneRedMapping):
    STABLE_ANCHOR = True
    RECORD_ANCHORS = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mapper = StableAnchorMapper(self.size_area)
        mapper.STABLE_ANCHOR = self.STABLE_ANCHOR
        mapper.RECORD_ANCHORS = self.RECORD_ANCHORS
        self._submission_mapper = mapper
        self._radio_maps = MappingSnapshots(self._map.shape, self._n, mapper)


class MyDroneRedMappingAnchorOff(MyDroneRedMappingAnchor):
    STABLE_ANCHOR = False
