"""Observed frontier routing with bounded, current-scan clearance recovery.

Normal routes retain Relay33's physical hull and stopping-sweep checks. A drone
already inside the conservative hull margin may only make a short, slow retreat
whose swept hull never acquires a new unsafe pixel and strictly reduces the
initial unsafe set. Missing/occluded scans cannot authorize that recovery.
Failure is explicit and latched for the current target so the mission controller
can choose another target instead of silently spending the remaining round.
"""
import math

import numpy as np

from swarm_rescue.solutions.red_relay33 import RelayNavigator33


class FrontierNavigator34(RelayNavigator33):
    BLOCKED_LIMIT = 72
    STAGNATION_LIMIT = 150
    RECOVERY_LIMIT = 60
    PROGRESS_DISTANCE = 10.

    def __init__(self, identifier=0, local_factory=None):
        super().__init__(identifier, local_factory)
        self._clear_state34()

    def _clear_state34(self):
        self.failed = False
        self.failure_reason = None
        self.failure_tick = None
        self.consecutive_blocked_ticks = 0
        self.no_progress_ticks = 0
        self._progress_anchor34 = None
        self._progress_tick34 = None
        self._last_call34 = None
        self._recovery_since34 = None
        self._recovery_attempts34 = 0
        self._recovery_details34 = None

    def reset(self):
        super().reset()
        self._clear_state34()

    @classmethod
    def estimate_route(cls, grid, position, target, speed=1.8, reserve_steps=90):
        """Estimate travel only through observed free cells, without mutation.

        An unreachable target has no finite complete-route budget. The proxy
        length/gap are separately exposed for bounded recovery decisions.
        This geometric estimate is not a promise of successful physical travel.
        """
        position, target = np.asarray(position, float), np.asarray(target, float)
        if (position.shape != (2,) or target.shape != (2,)
                or not np.isfinite(position).all() or not np.isfinite(target).all()
                or not math.isfinite(speed) or speed <= 0):
            return dict(reachable=False, exact=False, path_length_px=None, length=None,
                        travel_steps=None, steps=None, route_points=[],
                        proxy_gap_px=None, proxy=None, mode='invalid_input')
        selected = cls._search(grid, position, target, True)
        mode = 'inflated_known_free'
        if not selected['exact']:
            relaxed = cls._search(grid, position, target, False)
            if relaxed['exact'] or relaxed['proxy_distance'] + 1e-6 < selected['proxy_distance']:
                selected, mode = relaxed, 'unexpanded_known_free_fresh_hull_required'
        points = [position.copy()]
        points.extend(np.asarray(grid.point(cell), float) for cell in selected['cells'])
        if selected['exact']:
            points.append(target.copy())
        length = float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())
        budget = int(math.ceil(length / speed)) + int(reserve_steps) if selected['exact'] else None
        return dict(reachable=bool(selected['exact']), exact=bool(selected['exact']),
                    path_length_px=length, length=length, travel_steps=budget, steps=budget,
                    route_points=[point.tolist() for point in points],
                    proxy_gap_px=float(selected['proxy_distance']),
                    proxy=points[-1].tolist(), mode=mode)

    def _fail34(self, reason):
        if not self.failed:
            self.failed = True
            self.failure_reason = reason
            self.failure_tick = self.tick
            self.stats['failed_' + reason] += 1
        return self._brake('frontier_navigation_failed')

    def _monotone_recovery34(self, fine, distances):
        """Retreat using this frame only and exactly the original hull kernel."""
        if (not np.isfinite(distances).all() or not (distances > 3).all()
                or not all(hasattr(fine, key) for key in ('footprint', 'free', 'static', 'cell'))):
            self.stats['recovery_incomplete_observation'] += 1
            return None
        if np.linalg.norm(self._velocity) > 1.:
            self.stats['recovery_wait_velocity'] += 1
            return None
        if self._recovery_since34 is None:
            self._recovery_since34 = self.tick
        if self.tick - self._recovery_since34 >= self.RECOVERY_LIMIT:
            self.stats['recovery_budget_exhausted'] += 1
            return None
        self._recovery_attempts34 += 1
        known = np.asarray(fine.free, bool) & ~np.asarray(fine.static, bool)
        kernel = np.asarray(fine.footprint(), bool)
        offsets = np.argwhere(kernel) - np.asarray(kernel.shape) // 2
        initial = np.asarray(fine.cell(np.zeros(2))) + offsets
        shape = np.asarray(known.shape)
        if np.any(initial < 0) or np.any(initial >= shape):
            return None
        initial_bad = ~known[initial[:, 0], initial[:, 1]]
        count = int(initial_bad.sum())
        if not count:
            return None
        unsafe = np.zeros_like(known)
        unsafe[initial[initial_bad, 0], initial[initial_bad, 1]] = True

        def admissible(vector, strict=False):
            previous = count
            for point in np.linspace(np.zeros(2), vector,
                                     max(2, int(np.linalg.norm(vector)) + 1))[1:]:
                cells = np.asarray(fine.cell(point)) + offsets
                if np.any(cells < 0) or np.any(cells >= shape):
                    return False, previous
                yy, xx = cells[:, 0], cells[:, 1]
                bad = ~known[yy, xx]
                current = int(bad.sum())
                if np.any(bad & ~unsafe[yy, xx]) or current > previous:
                    return False, current
                previous = current
            return (previous < count if strict else True), previous

        choices = []
        for angle in np.linspace(-math.pi, math.pi, 24, endpoint=False):
            direction = np.array([math.cos(angle), math.sin(angle)])
            allowed, remaining = admissible(direction * 10., strict=True)
            if allowed:
                choices.append((count - remaining, angle, direction))
        for _, _, direction in sorted(choices, key=lambda item: (-item[0], item[1])):
            desired = direction * .8
            force = .085 * desired + .65 * (desired - self._velocity)
            force /= max(1., float(np.linalg.norm(force)))
            predicted = .95 * self._velocity + .6 * force
            stopping = predicted * 3 + predicted * np.linalg.norm(predicted) / 1.2
            allowed, remaining = admissible(stopping)
            if allowed:
                self.reason = 'current_scan_monotone_retreat'
                self.stats[self.reason] += 1
                self._recovery_details34 = dict(initial_unsafe_pixels=count,
                    stopping_unsafe_pixels=remaining, stopping_vector=stopping.tolist(),
                    same_hull_kernel=True, scan_tick=self.tick)
                return dict(forward=float(force[0]), lateral=float(force[1]), rotation=0.)
        self.stats['recovery_no_safe_retreat'] += 1
        return None

    def command(self, position, heading, velocity, lidar, angles, semantic, grid, target, tick):
        position_valid = (position is not None and np.shape(position) == (2,)
                          and np.isfinite(position).all())
        target_valid = (target is not None and np.shape(target) == (2,)
                        and np.isfinite(target).all())
        changed = target_valid and (self.target is None
                                    or np.linalg.norm(np.asarray(target) - self.target) > 8.)
        if changed:
            self._clear_state34()
        self.tick = int(tick)
        if (self._last_call34 is None or self.tick - self._last_call34 > 5
                or self._progress_anchor34 is None):
            self._progress_anchor34 = np.asarray(position, float).copy() if position_valid else None
            self._progress_tick34 = self.tick
        self._last_call34 = self.tick
        if velocity is not None and np.shape(velocity) == (2,) and np.isfinite(velocity).all():
            self._velocity = np.asarray(velocity, float).copy()
        if self.failed:
            return self._brake('frontier_navigation_failed')
        # Never let an earlier local scan authorize a recovery after bad input.
        self._last_local = None
        result = super().command(position, heading, velocity, lidar, angles, semantic, grid, target, tick)
        blocked = self.reason in ('current_hull_not_observed_clear', 'scan_unavailable',
            'no_fresh_hull_verified_segment', 'stopping_sweep_blocked',
            'pose_unavailable', 'velocity_unavailable', 'target_unavailable')
        if self.reason == 'current_hull_not_observed_clear' and self._last_local is not None:
            recovered = self._monotone_recovery34(self._last_local, np.asarray(lidar, float).ravel())
            if recovered is not None:
                result = recovered
        else:
            self._recovery_since34 = None
        self.consecutive_blocked_ticks = self.consecutive_blocked_ticks + 1 if blocked else 0
        if position_valid and self._progress_anchor34 is not None:
            if np.linalg.norm(np.asarray(position) - self._progress_anchor34) >= self.PROGRESS_DISTANCE:
                self._progress_anchor34 = np.asarray(position, float).copy()
                self._progress_tick34 = self.tick
            self.no_progress_ticks = self.tick - self._progress_tick34
        if self.arrived:
            self.no_progress_ticks = 0
            self.consecutive_blocked_ticks = 0
        elif self.consecutive_blocked_ticks >= self.BLOCKED_LIMIT:
            return self._fail34('blocked_budget')
        elif self.no_progress_ticks >= self.STAGNATION_LIMIT:
            return self._fail34('no_displacement_budget')
        return result

    step = command

    def status(self):
        value = super().status()
        value.update(failed=self.failed, failure_reason=self.failure_reason,
                     failure_tick=self.failure_tick,
                     consecutive_blocked_ticks=self.consecutive_blocked_ticks,
                     no_progress_ticks=self.no_progress_ticks,
                     recovery_attempts=self._recovery_attempts34,
                     recovery_details=self._recovery_details34)
        return value
