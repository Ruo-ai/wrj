"""Navigation-only removal of obstacle evidence contradicted by legal rays.

The old 20 px grid, free-space sampling, planner, inflation and motion safety
remain unchanged. This adds no wall-shape inference and never reads or writes
the submission mapper. Three recent, spatially separated observations must
contradict an old occupied cell after its latest endpoint observation.
"""
from collections import Counter

import numpy as np

from swarm_rescue.solutions.my_drone_red_advanced2 import (
    _M19_SurveyGrid, _SF_HIT_LIMIT, _SF_LIDAR_RANGE,
)


class EvidenceNavigation32(_M19_SurveyGrid):
    REQUIRED_VIEWS = 3
    VIEW_SEPARATION = 20.0
    HIT_HOLDOFF = 18
    EVIDENCE_WINDOW = 120

    def __init__(self, size, event):
        super().__init__(size, event)
        self.stats = Counter()
        self._last_observe32 = None
        self._last_hit32 = np.full(self.shape, -10**9, dtype=np.int64)
        self._first_view_step32 = np.zeros(self.shape, dtype=np.int64)
        self._views32 = np.zeros(self.shape, dtype=np.uint8)
        self._view_positions32 = np.zeros(self.shape + (2, 2), dtype=float)
        self.cleanup_events32 = []

    def status(self):
        return dict(self.stats, cell_size=self.CELL,
                    required_views=self.REQUIRED_VIEWS,
                    view_separation=self.VIEW_SEPARATION,
                    endpoint_holdoff=self.HIT_HOLDOFF,
                    evidence_window=self.EVIDENCE_WINDOW)

    def observe(self, position, heading, distances, angles, dynamic=None):
        # Preserve the old evidence, free-space, visits and endpoint updates.
        super().observe(position, heading, distances, angles, dynamic)
        step = int(self.step)
        duplicate = self._last_observe32 is not None and step <= self._last_observe32
        if duplicate:
            self.stats['duplicate_or_old_observation'] += 1
        else:
            self._last_observe32 = step
            self.stats['observations'] += 1
        position = np.asarray(position, dtype=float)
        d = np.asarray(distances).ravel()
        a = np.asarray(angles).ravel() + heading
        dyn = np.zeros(len(d), bool) if dynamic is None else np.asarray(dynamic, bool)
        if position.shape != (2,) or not np.isfinite(position).all():
            return
        valid = np.isfinite(d) & np.isfinite(a) & (d > 4) & ~dyn
        if not valid.any():
            return
        d, a = d[valid], a[valid]
        directions = np.column_stack((np.cos(a), np.sin(a)))
        hits = d < _SF_HIT_LIMIT
        endpoint = self.indices(position + directions[hits] * d[hits, None])
        last_hit = self._last_hit32.ravel()
        views = self._views32.ravel()
        first_step = self._first_view_step32.ravel()
        anchors = self._view_positions32.reshape(-1, 2, 2)
        # Any fresh static endpoint wins over all accumulated contradiction.
        last_hit[endpoint] = np.maximum(last_hit[endpoint], step)
        views[endpoint] = 0
        if duplicate:
            return

        # Exactly the parent's 12 px sampling and 26 px endpoint margin;
        # dynamic rays provide no additional cleanup evidence, even in front.
        samples = np.arange(0, _SF_LIDAR_RANGE, 12)
        points = position + directions[:, None, :] * samples[None, :, None]
        inside = samples[None, :] < np.minimum(d[:, None], _SF_LIDAR_RANGE) - 26
        free = self.indices(points[inside])
        free = free[step - last_hit[free] >= self.HIT_HOLDOFF]
        if not len(free):
            return
        expired = (views[free] > 0) & (step - first_step[free] > self.EVIDENCE_WINDOW)
        views[free[expired]] = 0
        self.stats['expired_evidence_cells'] += int(expired.sum())
        empty = free[views[free] == 0]
        anchors[empty, 0] = position
        first_step[empty] = step
        # Each accepted position is at least 20 px from every prior view.
        first = free[views[free] == 1]
        second = free[views[free] == 2]
        first = first[np.linalg.norm(anchors[first, 0] - position, axis=1) >= self.VIEW_SEPARATION]
        second = second[(np.linalg.norm(anchors[second, 0] - position, axis=1) >= self.VIEW_SEPARATION)
                        & (np.linalg.norm(anchors[second, 1] - position, axis=1) >= self.VIEW_SEPARATION)]
        anchors[first, 1] = position
        views[empty] = 1
        views[first] = 2
        views[second] = self.REQUIRED_VIEWS
        self.stats['accepted_spatial_views'] += len(empty) + len(first) + len(second)
        obstacle = self.obstacle.ravel()
        clear = free[(views[free] >= self.REQUIRED_VIEWS) & (obstacle[free] >= 2)]
        if len(clear):
            old = obstacle[clear].copy()
            self.stats['cleanup_frames'] += 1
            self.stats['cleared_occupied_cells'] += len(clear)
            self.stats['removed_obstacle_evidence'] += float(old.sum())
            for index, strength in zip(clear, old):
                if len(self.cleanup_events32) < 128:
                    r, c = np.unravel_index(index, self.shape)
                    self.cleanup_events32.append(dict(step=step, cell=[int(r), int(c)],
                        previous_evidence=float(strength), last_endpoint_step=int(last_hit[index]),
                        view_positions=[*anchors[index].tolist(), position.tolist()]))
            obstacle[clear] = 0.
            views[clear] = 0
