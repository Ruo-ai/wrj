"""FS1 isolated traditional candidates derived from frozen Advanced2.

Weighted marginal coverage inspired by Nemhauser/Wolsey/Fisher (1978),
Mathematical Programming 14:265-294. Original implementation; no external code.
Support weights are engineering proxies, NOT calibrated wall probabilities.
No unknown interiors are filled and neither submission nor navigation maps change.
"""
import cv2
import numpy as np

from swarm_rescue.solutions.my_drone_red_advanced2 import (
    MyDroneRedAdvanced2, _M16_CoverageField,
)


def supported_area(observation, remote_mask):
    """Count observed cells only, with local free/non-wall evidence taking priority."""
    evidence = observation.evidence
    own = observation.mask()
    # Binary peer maps do not contain confidence or independent view counts.
    support = np.where(remote_mask, .35, 0.).astype(np.float32)
    support[own] = np.clip((evidence[own] - 2.) / 6., 0., 1.)
    nonwall = cv2.dilate((observation.static_nonwall >= 2).astype(np.uint8),
                        np.ones((3, 3), np.uint8)).astype(bool)
    support[(evidence < -1.) | nonwall] = 0.
    # Partial cells at the image border do not own a full cell's area.
    cell = observation.CELL
    heights = np.minimum(cell, observation.height - np.arange(support.shape[0]) * cell)
    widths = np.minimum(cell, observation.width - np.arange(support.shape[1]) * cell)
    return support * heights[:, None] * widths[None, :]


class AreaCoverageField(_M16_CoverageField):
    """Exact residual weighted area on the observed grid, not circle separation.

    Claims keep the baseline soft discount for this ablation. Confirmed disks
    remove a union once; duplicates cannot repeatedly subtract confirmed area.
    """
    def __init__(self, area, cell_size, size, placed, claims, radius):
        self.cell = cell_size
        self.width, self.height = size
        self.radius = radius
        yy, xx = np.indices(area.shape)
        self.x = (xx + .5) * cell_size - self.width / 2
        self.y = self.height / 2 - (yy + .5) * cell_size
        weight = np.maximum(0., np.asarray(area, np.float32)).copy()
        # Deliberately retain A0 boundary proxy to isolate the area mechanism.
        edge = (np.abs(self.x) > self.width / 2 - 15) | (np.abs(self.y) > self.height / 2 - 15)
        weight[edge] *= .5
        self.standalone = weight.copy()
        self.placed = [np.asarray(p, dtype=float) for p in placed]
        self.claims = [np.asarray(p, dtype=float) for p in claims]
        for p in self.placed:
            weight[self.disk(p)] = 0.
        self.unclaimed = weight.copy()  # factual residual, before promises
        for p in self.claims:
            weight[self.disk(p)] *= .05
        self.weight = weight
        r = int(np.ceil(radius / cell_size))
        ky, kx = np.ogrid[-r:r+1, -r:r+1]
        kernel = ((kx*cell_size)**2 + (ky*cell_size)**2 <= radius**2).astype(np.float32)
        self.gain = cv2.filter2D(weight, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    def marginal(self, point):
        """Confirmed-union residual; uncompleted claims are never facts."""
        return float(self.unclaimed[self.disk(point)].sum())

    def risk_samples(self, point, displacement):
        point = np.asarray(point)
        return [self.marginal(point + offset) for offset in (
            (0., 0.), (displacement, 0.), (-displacement, 0.),
            (0., displacement), (0., -displacement))]


def release_quality(field, point, missing_steps, progress):
    """Five deterministic stress samples, NOT a statistical confidence bound."""
    displacement = 6. if missing_steps == 0 else 15.
    values = field.risk_samples(point, displacement)
    actual, mean = values[0], float(np.mean(values))
    enough = actual >= field.cell**2 and mean >= .5 * actual
    deadline = progress >= .76
    return dict(allow=bool(enough or deadline), reason='supported_gain' if enough else
                'deadline_fallback' if deadline else 'low_marginal_gain',
                marginal_area=actual, risk_mean=mean, risk_min=min(values),
                displacement=displacement, missing_steps=int(missing_steps))


class MyDroneRedFS1Area(MyDroneRedAdvanced2):
    """A1a: area-only ablation; all task/navigation/deployment rules inherited."""
    CHECK_RELEASE = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fs1_signature = None
        self._fs1_pending = None
        self._fs1_last_reject = -100

    def _coverage_field(self, force=False):
        placed = self._known_bombs()
        claims = [np.array(p) for i, (seq, p) in self._bomb_claims.items()
                  if i < int(self.identifier or 0) and 0 <= self._steps-seq < 90 and p is not None]
        signature = (tuple(tuple(p) for p in placed), tuple(tuple(p) for p in claims))
        if force or self._advanced_field is None or self._steps-self._field_step >= 24 or signature != self._fs1_signature:
            # Exclude our last radio snapshot: it is not independent peer evidence.
            remote = np.zeros(self._map.shape, bool)
            for source, mask in self._radio_maps.masks.items():
                if source != int(self.identifier or 0):
                    remote |= mask
            area = supported_area(self._map, remote)
            self._advanced_field = AreaCoverageField(area, self._map.CELL, self.size_area,
                placed, claims, round(.12 * min(self.size_area)))
            self._field_step = self._steps
            self._fs1_signature = signature
        return self._advanced_field

    def _deployment(self, command):
        pending = self._attempt_step is not None or (
            self._last_inventory is not None and self.carried_bombs_count() < self._last_inventory)
        near = self._bomb_target is not None and self._position is not None and np.linalg.norm(self._position-self._bomb_target) <= 28
        eligible = (not pending and self.carried_bombs_count() > 0 and self._position is not None
                    and self._steps >= self._next_attempt and (near or self._progress() >= .68))
        if self.CHECK_RELEASE and eligible:
            quality = release_quality(self._coverage_field(force=True), self._position,
                self._pose_filter.absolute_missing_steps, self._progress())
            if not quality['allow']:
                if self._steps-self._fs1_last_reject >= 30:
                    self._event('fs1_release_deferred', **quality)
                    self._fs1_last_reject = self._steps
                return
        super()._deployment(command)

    def _event(self, kind, **data):
        # Diagnostic estimates only; never consult physics entities or offline truth.
        if kind == 'placement_requested':
            field = self._coverage_field(force=True)
            quality = release_quality(field, self._position,
                self._pose_filter.absolute_missing_steps, self._progress())
            target = self._bomb_target
            self._fs1_pending = dict(quality, source=int(self.identifier or 0),
                request_step=self._steps, estimated_position=self._position.tolist(),
                target=None if target is None else target.tolist(),
                target_error=None if target is None else float(np.linalg.norm(self._position-target)),
                claim_discounted_area=field.value(self._position),
                confidence='uncalibrated_pose_and_wall_support', check_enabled=self.CHECK_RELEASE)
            data['fs1'] = self._fs1_pending
        elif kind == 'placement_confirmed':
            data['fs1'] = dict(request=self._fs1_pending, confirmation_step=self._steps,
                               record_count=len(self._placements))
            self._fs1_pending = None
            self._fs1_signature = None
        elif kind == 'placement_retry':
            self._fs1_pending = None
        super()._event(kind, **data)


class MyDroneRedFS1Gain(MyDroneRedFS1Area):
    """A1b: area plus bounded pre-release marginal checks and deadline fallback."""
    CHECK_RELEASE = True
