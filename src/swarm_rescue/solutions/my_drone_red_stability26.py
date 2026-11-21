"""RL17-R with a 240-step lifetime for failed-route retirement only.

The original route release still applies its full penalty and avoidance. Only
that call's increment expires; ordinary arrivals and survey retirement remain
permanent. The match clock, navigation, mapping and deployment are inherited.
"""
from collections import Counter, deque

import numpy as np

from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17


class FailureRetirements26:
    """Account for this controller's temporary additions to goal_visits."""

    LIFETIME = 240

    def __init__(self, stats):
        self.stats = stats
        self.values = None
        self.owned = None
        self.entries = deque()
        self.last_step = -1

    def _bind(self, values):
        if self.values is not values:
            if self.values is not None:
                self.stats['array_replacements'] += 1
                self.stats['discarded_events'] += len(self.entries)
            self.values = values
            self.owned = np.zeros(values.shape, dtype=np.float32)
            self.entries.clear()
            self.stats['pending_events'] = 0

    @staticmethod
    def region(shape, cell):
        """The original radius-four-cell disk, clipped before indexing."""
        r, c = cell
        top, bottom = max(0, r - 4), min(shape[0], r + 5)
        left, right = max(0, c - 4), min(shape[1], c + 5)
        yy, xx = np.ogrid[top:bottom, left:right]
        rows, cols = np.nonzero((yy - r) ** 2 + (xx - c) ** 2 <= 16)
        return rows + top, cols + left

    def record(self, values, indices, before, step):
        self._bind(values)
        if step < self.last_step:
            self.stats['clock_regressions'] += 1
            return
        self.last_step = step
        delta = values[indices] - before
        # The frozen parent adds exactly one. An unexpected mutation stays
        # permanent instead of granting an unverified later subtraction.
        if (not np.isfinite(delta).all() or np.any((delta != 0) & (delta != 1))
                or not np.isfinite(values[indices]).all() or np.any(values[indices] < 0)):
            self.stats['unexpected_delta'] += 1
            return
        positive = delta > 0
        if not positive.any():
            return
        indices = tuple(a[positive].copy() for a in indices)
        delta = delta[positive].copy()
        self.owned[indices] += delta
        self.entries.append((step + self.LIFETIME, indices, delta))
        if 'first_failure_step' not in self.stats:
            self.stats['first_failure_step'] = int(step)
        self.stats['failure_events'] += 1
        self.stats['failure_cells'] += len(delta)
        self.stats['pending_events'] = len(self.entries)

    def expire(self, values, step):
        self._bind(values)
        if step < self.last_step:
            self.stats['clock_regressions'] += 1
            return
        self.last_step = step
        while self.entries and self.entries[0][0] <= step:
            _, indices, delta = self.entries.popleft()
            owned = self.owned[indices]
            current = values[indices]
            if not np.isfinite(owned).all() or np.any(owned < delta):
                # Corrupted accounting cannot authorize a map mutation.
                self.stats['ledger_guard_resets'] += 1
                self.stats['discarded_events'] += 1 + len(self.entries)
                self.entries.clear()
                self.owned.fill(0)
                break
            if not np.isfinite(current).all() or np.any(current < owned):
                # A replaced/reset counter must never consume unrelated memory.
                # All overlapping debts are now ambiguous, so retain current
                # counters and discard the entire ledger conservatively.
                self.stats['underflow_guard_events'] += 1
                self.stats['discarded_events'] += 1 + len(self.entries)
                self.entries.clear()
                self.owned.fill(0)
                break
            else:
                values[indices] = current - delta
                if 'first_expiry_step' not in self.stats:
                    self.stats['first_expiry_step'] = int(step)
                self.stats['expired_events'] += 1
                self.stats['expired_cells'] += len(delta)
            self.owned[indices] = owned - delta
        self.stats['pending_events'] = len(self.entries)


class MyDroneRedStability26(MyDroneRedRepair17):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._d26_stats = Counter()
        self._failure_retirements26 = FailureRetirements26(self._d26_stats)

    def _release_stale_route(self, reason):
        route = self._route
        values = self._explore.goal_visits
        if route:
            indices = FailureRetirements26.region(values.shape, route[-1])
            before = values[indices].copy()
            step = self._steps
        result = super()._release_stale_route(reason)
        if route:
            self._failure_retirements26.record(values, indices, before, step)
        return result

    def _navigation(self):
        self._failure_retirements26.expire(self._explore.goal_visits, self._steps)
        return super()._navigation()
