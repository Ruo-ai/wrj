"""RL17: bounded point-to-finite-line submission-pose experiment on frozen RL11.

Original NumPy implementation; no external source copied. Odometry, navigation,
wall evidence, radio versions, recovery projection and submission deadlines are
inherited unchanged. References are OWN past observed segments, not truth or a
peer's private map. Confidence is geometric, not an absolute-position guarantee.
"""
from collections import Counter
import time

import numpy as np

from swarm_rescue.solutions.my_drone_red_lines11 import (
    MyDroneRedLines11, WallLineMapper, WallLineMatcher,
)
from swarm_rescue.solutions.my_drone_red_mapping import MappingSnapshots
from swarm_rescue.solutions.semifinal_localization import transform


def point_line_registration(points, references, position, heading, deadline=None):
    """Three local IRLS steps, retaining odometry in unobservable directions.

    Units: pixels, radians; heading Jacobian normalized by 100 pixels. Finite
    segment support prevents an opening from becoming an infinite reference.
    Ambiguous near-equidistant *different* surfaces are not counted as inliers.
    No output state is mutated; any failed gate discards the whole correction.
    """
    p = np.asarray(points, float)
    refs = np.asarray(references, float)
    info = dict(reason='insufficient_support',points=len(p),iterations=0)
    if p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 12 or refs.size == 0:
        return None, info
    if refs.ndim != 3 or refs.shape[1:] != (2, 2):
        return None, info
    if not np.isfinite(p).all() or not np.isfinite(refs).all():
        return None, dict(info, reason='nonfinite')
    # Hard work bounds independent of input or map dimensions.
    p = p[np.linspace(0, len(p)-1, min(len(p), 48)).astype(int)]
    refs = refs[-64:]
    edge = refs[:, 1]-refs[:, 0]
    length = np.linalg.norm(edge, axis=1)
    keep = length >= 45
    refs, edge, length = refs[keep], edge[keep], length[keep]
    if not len(refs):
        return None, info
    tangent = edge/length[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    centers = refs.mean(axis=1)
    # Several cached collinear pieces of one wall are not independent surfaces.
    parallel = np.abs(normal@normal.T) > .999
    separation = np.abs(np.einsum('ijk,ik->ij', centers[None, :, :]-centers[:, None, :], normal))
    same_surface = parallel & (separation < 4.)
    offset = np.zeros(3)
    initial_cost = None
    basis = None
    for iteration in range(4):  # 3 solves + final confidence measurement
        if deadline is not None and time.perf_counter() >= deadline:
            return None, dict(info, reason='budget')
        world = transform(p, position+offset[:2], heading+offset[2])
        delta = world[:, None, :]-centers[None, :, :]
        residuals = np.einsum('ijk,jk->ij', delta, normal)
        along = np.einsum('ijk,jk->ij', delta, tangent)
        finite = (np.abs(along) <= length[None, :]/2+2.) & (np.abs(residuals) < 18.)
        cost = np.where(finite, np.abs(residuals), np.inf)
        index = np.argmin(cost, axis=1)
        nearest = cost[np.arange(len(p)), index]
        other = np.where(~same_surface[index], cost, np.inf).min(axis=1)
        unique = (other > nearest+3.) | ~np.isfinite(other)
        valid = np.isfinite(nearest) & unique
        ids = np.flatnonzero(valid)
        info.update(points=len(p),inliers=len(ids),ambiguous=int((np.isfinite(nearest)&~unique).sum()),
                    support=float(valid.mean()),iterations=min(iteration,3))
        if len(ids) < 12 or valid.mean() < .55:
            return None, dict(info, reason='support_or_ambiguity')
        ni = normal[index[ids]]
        r = residuals[ids, index[ids]]
        weights = np.minimum(1., 3./np.maximum(np.abs(r), 1e-9))
        # Cap contribution from any one reference, so a long nearby wall does
        # not overwhelm a shorter perpendicular wall solely by ray density.
        counts = np.bincount(index[ids], minlength=len(refs))
        weights *= np.minimum(1., 16./counts[index[ids]])
        weights /= weights.sum()
        gram = (ni.T*weights)@ni
        values, vectors = np.linalg.eigh(gram)
        observable = values > max(.04, .12*values.max())
        rank = int(observable.sum())
        current_basis = vectors[:, observable]
        if basis is None:
            basis = current_basis
        elif rank != basis.shape[1] or np.linalg.norm(basis@basis.T-current_basis@current_basis.T) > .12:
            return None, dict(info, reason='observability_changed')
        q = world[ids]-position-offset[:2]
        angular = np.sum(ni*np.column_stack((-q[:, 1], q[:, 0])), axis=1)/100.
        jacobian = np.column_stack((ni@basis, angular))
        hessian = (jacobian.T*weights)@jacobian
        spectrum = np.linalg.eigvalsh(hessian)
        info.update(rank=rank,normal_eigenvalues=values.tolist(),hessian_eigenvalues=spectrum.tolist(),
                    residual=float(np.sqrt(np.sum(weights*r*r))),references=int(len(np.unique(index[ids]))))
        if spectrum[0] < max(.008, .015*spectrum[-1]):
            return None, dict(info, reason='coupled_degeneracy')
        # Require observed extent to constrain yaw even for a single wall.
        span = float(np.linalg.norm(np.ptp(world[ids], axis=0)))
        if span < 65.:
            return None, dict(info, reason='small_extent')
        robust_cost = float(np.minimum(np.abs(r), 9.).mean())
        if initial_cost is None:
            initial_cost = robust_cost
        info.update(prior_residual=initial_cost,robust_residual=robust_cost,extent=span)
        if iteration == 3:
            if info['residual'] > 4.5 or robust_cost > initial_cost+.1:
                return None, dict(info, reason='residual')
            return offset, dict(info, reason='p2l_registered',offset=offset.tolist())
        step = -np.linalg.solve(hessian, jacobian.T@(weights*r))
        offset[:2] += basis@step[:-1]
        offset[2] += step[-1]/100.
        if np.linalg.norm(offset[:2]) > 12. or abs(offset[2]) > .10:
            return None, dict(info, reason='prior_bound')
    raise AssertionError('bounded iteration invariant')


class PointLineMatcher17(WallLineMatcher):
    MAX_CALL_SECONDS = .004
    MAX_TOTAL_SECONDS = .65  # per aircraft, incremental matcher work only

    def __init__(self, size):
        super().__init__(size)
        self.p2l_stats = Counter()
        self.p2l_seconds = 0.
        self.last_p2l = {}

    def match(self, points, position, heading):
        started = time.perf_counter()
        if self.p2l_seconds >= self.MAX_TOTAL_SECONDS:
            offset, info = None, dict(reason='total_budget')
        else:
            try:
                offset, info = point_line_registration(points, self.references, position, heading,
                                                       started+self.MAX_CALL_SECONDS)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                offset, info = None, dict(reason='numerical_failure')
        elapsed = time.perf_counter()-started
        self.p2l_seconds += elapsed
        # A slow final solve must not escape the per-call limit.
        if elapsed > self.MAX_CALL_SECONDS:
            offset, info = None, dict(info, reason='budget')
        self.p2l_stats[info['reason']] += 1
        self.p2l_stats['attempts'] += 1
        self.last_p2l = dict(info,seconds=elapsed)
        if offset is not None:
            self.p2l_stats['accepted'] += 1
            self.p2l_stats['rank_'+str(info['rank'])] += 1
            self.last_info = info
            return offset, info
        self.p2l_stats['fallback_rl11'] += 1
        return super().match(points, position, heading)


class PointLineMapper17(WallLineMapper):
    POINT_LINE_ENABLED = True

    def __init__(self, size):
        super().__init__(size)
        if self.POINT_LINE_ENABLED:
            self.matcher = PointLineMatcher17(size)


class PointLineMapper17Off(PointLineMapper17):
    POINT_LINE_ENABLED = False


class MyDroneRedP2L17(MyDroneRedLines11):
    POINT_LINE_ENABLED = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.POINT_LINE_ENABLED:
            self._submission_mapper = PointLineMapper17(self.size_area)
            self._radio_maps = MappingSnapshots(self._map.shape, self._n, self._submission_mapper)


class MyDroneRedP2L17Off(MyDroneRedP2L17):
    POINT_LINE_ENABLED = False
