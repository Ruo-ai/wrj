"""Bounded wall-band consolidation from actually received RL17 local maps.

This module has no world truth, map-file, seed, pose or simulator access. A source
is one drone, regardless of how many times another drone relays its snapshot.
Long locally straight bands are represented by their measured medial lines.
Compatible overlapping lines are consolidated across sources and redrawn once.
Unmodelled observations remain in the result; a lone observer is sufficient.
This is a submission renderer, not a modification of collision/navigation maps.
"""
import math

import cv2
import numpy as np


class Fusion27:
    CELL = 3
    MIN_LINE = 12.0  # coarse cells: only observed runs at least 36 px long
    MAX_LINES_PER_SOURCE = 96
    ANGLE_GATE = math.radians(5)
    CROSS_SOURCE_DISTANCE = 4.0  # 12 px; cannot merge a normal drone-size corridor
    SAME_SOURCE_DISTANCE = 1.25
    MAX_HALF_BAND = 10.0  # larger solid interiors are not classified as thin walls
    PIXEL_LINE_THICKNESS = 8  # OpenCV's integer raster produces a 9-px axial band

    def __init__(self, size):
        self.width, self.height = map(int, size)
        if min(self.width, self.height) < 1:
            raise ValueError('Positive output dimensions required')
        self.shape = ((self.height + 2) // 3, (self.width + 2) // 3)
        self.last_stats = {}
        self.last_lines = []

    @staticmethod
    def _direction(vector):
        vector = np.asarray(vector, float)
        length = float(np.linalg.norm(vector))
        if length < 1e-9:
            return None
        vector = vector / length
        if vector[0] < -1e-9 or (abs(vector[0]) <= 1e-9 and vector[1] < 0):
            vector = -vector
        return vector

    def _extract(self, mask, source):
        if int(mask.sum()) < 27:
            return []
        encoded = np.asarray(mask, np.uint8)
        skeleton = cv2.ximgproc.thinning(encoded * 255)
        ys, xs = np.nonzero(skeleton)
        points = np.column_stack((xs, ys)).astype(float)
        if len(points) < self.MIN_LINE:
            return []
        distance = cv2.distanceTransform(encoded, cv2.DIST_L2, 5)
        proposed = cv2.HoughLinesP(skeleton, 1, np.pi / 180, threshold=10,
                                  minLineLength=int(self.MIN_LINE), maxLineGap=2)
        if proposed is None:
            return []
        proposals = np.asarray(proposed).reshape(-1, 4)
        lengths = np.linalg.norm(proposals[:, 2:] - proposals[:, :2], axis=1)
        # Stable ordering plus geometric ties make reconstruction independent of
        # dictionary/packet arrival order; all inputs themselves are immutable.
        order = sorted(range(len(proposals)), key=lambda i: (-lengths[i], *proposals[i]))
        result = []
        for index in order[:self.MAX_LINES_PER_SOURCE * 3]:
            first, last = proposals[index].reshape(2, 2).astype(float)
            direction = self._direction(last - first)
            if direction is None:
                continue
            normal = np.array([-direction[1], direction[0]])
            along = (points - first) @ direction
            low, high = sorted((0., float((last - first) @ direction)))
            select = ((np.abs((points - first) @ normal) <= 1.8)
                      & (along >= low - 2) & (along <= high + 2))
            measured = points[select]
            if len(measured) < 12:
                continue
            center = measured.mean(axis=0)
            centered = measured - center
            values, vectors = np.linalg.eigh(centered.T @ centered / len(measured))
            fitted = self._direction(vectors[:, 1])
            if (fitted is None or float(values[0]) > 1.3 ** 2
                    or abs(float(fitted @ direction)) < math.cos(self.ANGLE_GATE)):
                continue
            normal = np.array([-fitted[1], fitted[0]])
            projected = centered @ fitted
            lo, hi = float(projected.min()), float(projected.max())
            if hi - lo < self.MIN_LINE:
                continue
            # Distance-transform radii distinguish long wall bands from a large
            # filled box. Small boxes have no sufficiently long medial segment.
            radii = distance[measured[:, 1].astype(int), measured[:, 0].astype(int)]
            if float(np.median(radii)) > self.MAX_HALF_BAND:
                continue
            radius = min(self.MAX_HALF_BAND, max(2., float(np.percentile(radii, 90)) + 1.))
            ends = center + np.array([lo, hi])[:, None] * fitted
            duplicate = False
            for previous in result:
                if abs(float(previous['direction'] @ fitted)) < math.cos(self.ANGLE_GATE):
                    continue
                if abs(float((center - previous['center']) @ previous['normal'])) > self.SAME_SOURCE_DISTANCE:
                    continue
                span = (ends - previous['center']) @ previous['direction']
                pspan = (previous['ends'] - previous['center']) @ previous['direction']
                if min(span.max(), pspan.max()) - max(span.min(), pspan.min()) >= .8 * (hi - lo):
                    duplicate = True
                    break
            if duplicate:
                continue
            result.append(dict(source=source, center=center, direction=fitted, normal=normal,
                               ends=ends, length=hi-lo, radius=radius, residual=float(math.sqrt(values[0]))))
            if len(result) >= self.MAX_LINES_PER_SOURCE:
                break
        return result

    def _clusters(self, lines):
        clusters = []
        for line in sorted(lines, key=lambda x: (-x['length'], x['source'], *x['center'])):
            best = None
            best_cost = float('inf')
            for cluster in clusters:
                anchor = cluster[0]
                if abs(float(anchor['direction'] @ line['direction'])) < math.cos(self.ANGLE_GATE):
                    continue
                # No chain drift: every member must agree with this candidate.
                offsets = [abs(float((line['center'] - other['center']) @ other['normal']))
                           for other in cluster]
                if max(offsets) > self.CROSS_SOURCE_DISTANCE:
                    continue
                if any(other['source'] == line['source'] and offset > self.SAME_SOURCE_DISTANCE
                       for other, offset in zip(cluster, offsets)):
                    continue
                span = (line['ends'] - anchor['center']) @ anchor['direction']
                old = np.vstack([other['ends'] for other in cluster])
                old_span = (old - anchor['center']) @ anchor['direction']
                overlap = min(span.max(), old_span.max()) - max(span.min(), old_span.min())
                if overlap < min(6., .25 * min(line['length'], old_span.max() - old_span.min())):
                    continue
                cost = max(offsets) + .5 * np.mean(offsets)
                if cost < best_cost:
                    best, best_cost = cluster, cost
            if best is None:
                clusters.append([line])
            else:
                best.append(line)
        return clusters

    @staticmethod
    def _explain(mask, line):
        ends = line['ends']
        radius = line['radius']
        low = np.floor(ends.min(axis=0) - radius - 3).astype(int)
        high = np.ceil(ends.max(axis=0) + radius + 3).astype(int)
        x0, y0 = np.maximum(low, 0)
        x1, y1 = np.minimum(high + 1, [mask.shape[1], mask.shape[0]])
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - line['center'][0], yy - line['center'][1]
        along = dx * line['direction'][0] + dy * line['direction'][1]
        normal = dx * line['normal'][0] + dy * line['normal'][1]
        extent = (ends - line['center']) @ line['direction']
        explained = ((np.abs(normal) <= radius) & (along >= extent.min() - 2)
                     & (along <= extent.max() + 2))
        view = mask[y0:y1, x0:x1]
        view[explained] = False

    def fuse(self, source_masks, source_weights=None):
        """Return one pixel map; never mutate the maps received over the radio.

        The equal-source policy is deliberate: a newer sequence number means
        freshness, not proof of geometric accuracy. A lone observed line remains.
        """
        if source_weights is not None:
            raise ValueError('Fusion27 uses equal origin weights, not freshness as geometric confidence')
        sources = {}
        for source, mask in sorted(source_masks.items()):
            if not isinstance(source, (int, np.integer)) or isinstance(source, bool) or source < 0:
                raise ValueError('Non-negative integer origin ID required')
            array = np.asarray(mask)
            if array.shape != self.shape or array.dtype != np.bool_:
                raise ValueError('Expected one bool RL17 coarse map per origin')
            sources[int(source)] = array.copy()
        union = np.zeros(self.shape, bool)
        lines = []
        counts = {}
        for source, mask in sources.items():
            union |= mask
            found = self._extract(mask, source)
            counts[str(source)] = len(found)
            lines.extend(found)
        clusters = self._clusters(lines)
        self.last_lines = []
        models = []
        for cluster in clusters:
            by_source = {}
            for line in cluster:
                by_source.setdefault(line['source'], []).append(line)
            # Each source contributes one orientation and one normal position.
            directions = []
            for records in by_source.values():
                reference = records[0]['direction']
                directions.append(self._direction(np.mean([r['direction'] * (1 if r['direction'] @ reference >= 0 else -1)
                                                           for r in records], axis=0)))
            reference = directions[0]
            direction = self._direction(np.mean([d * (1 if d @ reference >= 0 else -1) for d in directions], axis=0))
            normal = np.array([-direction[1], direction[0]])
            positions = [float(np.median([r['center'] @ normal for r in records]))
                         for records in by_source.values()]
            offset = float(np.median(positions))
            observed = np.vstack([line['ends'] for line in cluster])
            projected = observed @ direction
            ends = normal * offset + np.array([projected.min(), projected.max()])[:, None] * direction
            pixel_ends = np.rint(ends * self.CELL + 1).astype(int)
            # Consolidate the explained strip in the common evidence map. If
            # only each source's extracted pixels were removed, a noisy source
            # whose medial segment was fragmented would re-add the thick band.
            # The strip is bounded by observed source radii and offsets, and its
            # length by actual overlapping segments; it never completes a room.
            radius = max(abs(float(line['center'] @ normal) - offset) + line['radius']
                         for line in cluster)
            models.append(dict(center=ends.mean(axis=0), direction=direction,
                               normal=normal, ends=ends, radius=radius,
                               pixel_ends=pixel_ends))
            self.last_lines.append(dict(sources=sorted(by_source),
                                        endpoints=pixel_ends.tolist(),
                                        residual_max=max(line['residual'] * self.CELL for line in cluster),
                                        source_normal_spread=(max(positions)-min(positions))*self.CELL))
        residual = union.copy()
        for model in models:
            self._explain(residual, model)
        pixels = np.repeat(np.repeat(residual, 3, 0), 3, 1)[:self.height, :self.width].astype(np.uint8)
        for model in models:
            pixel_ends = model['pixel_ends']
            cv2.line(pixels, tuple(pixel_ends[0]), tuple(pixel_ends[1]), 1,
                     thickness=self.PIXEL_LINE_THICKNESS, lineType=cv2.LINE_8)
        pixels = cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(pixels, connectivity=8)
        keep = np.zeros(count, np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 27
        result = keep[labels].astype(bool)
        self.last_stats = dict(source_count=len(sources), per_source_lines=counts,
                               extracted_lines=len(lines), fused_lines=len(clusters),
                               multisource_lines=sum(len(line['sources']) > 1 for line in self.last_lines),
                               singleton_lines=sum(len(line['sources']) == 1 for line in self.last_lines),
                               input_union_cells=int(union.sum()), residual_cells=int(residual.sum()),
                               output_pixels=int(result.sum()), ordinary_axial_line_width=9,
                               sources_equal_weight=True, navigation_map_modified=False)
        return result
