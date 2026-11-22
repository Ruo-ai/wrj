"""Complete a rectangular outline inferred from received final-map geometry.

Canvas size is used only for allocation/clipping. No arena dimensions imply a
wall location. All four sides require observed long straight segments; broad
enclosure, interior structure and at least two observed corners are required.
Only gaps/extensions between the inferred side intersections are filled.
This remains a geometric hypothesis about a rectangular unknown map, not proof
that all free/occupied space outside the observations has been discovered.
"""
import itertools

import cv2
import numpy as np

from swarm_rescue.solutions.red_dualmap32 import FinalRenderer32


class FinalBoundary33(FinalRenderer32):
    """Keep 32's received-map rendering; conditionally complete its outline."""

    MIN_STRAIGHT_RUN = 45
    MAX_NORMAL_SPAN = 35
    MERGE_NORMAL_DISTANCE = 12.
    MIN_SIDE_FRACTION = .10
    MIN_RECTANGLE_SPAN = 120.
    MIN_CONTAINMENT = .98
    MIN_INTERIOR_SEGMENTS = 4

    def __init__(self, size, wall_width=11):
        super().__init__(size, wall_width=wall_width)
        self.last_pre_boundary_pixels = None
        self.last_boundary_pixels = None
        self.last_added_boundary_pixels = None

    @staticmethod
    def _intervals(intervals):
        merged = []
        for low, high in sorted(intervals):
            if merged and low <= merged[-1][1]+1:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([float(low), float(high)])
        return merged

    def _segments(self, pixels, vertical):
        kernel = np.ones((self.MIN_STRAIGHT_RUN, 1) if vertical else (1, self.MIN_STRAIGHT_RUN), np.uint8)
        straight = cv2.morphologyEx(pixels.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(straight, connectivity=8)
        records = []
        for label in range(1, count):
            x, y, width, height, _ = map(int, stats[label])
            along_span, normal_span = (height, width) if vertical else (width, height)
            if (along_span < self.MIN_STRAIGHT_RUN or normal_span > self.MAX_NORMAL_SPAN
                    or along_span < 4*normal_span):
                continue
            yy, xx = np.nonzero(labels[y:y+height, x:x+width] == label)
            normal = xx+x if vertical else yy+y
            along = yy+y if vertical else xx+x
            records.append(dict(normal=float(np.median(normal)), low=float(along.min()),
                                high=float(along.max()), vertical=vertical))
        return records

    def _groups(self, segments):
        groups = []
        for segment in sorted(segments, key=lambda s: s['normal']):
            if not groups or segment['normal']-groups[-1]['anchor'] > self.MERGE_NORMAL_DISTANCE:
                groups.append(dict(anchor=segment['normal'], members=[]))
            groups[-1]['members'].append(segment)
        result = []
        for group in groups:
            members = group['members']
            weights = np.asarray([s['high']-s['low']+1 for s in members])
            order = np.argsort([s['normal'] for s in members])
            ordered = weights[order]
            middle = order[np.searchsorted(np.cumsum(ordered), weights.sum()/2)]
            result.append(dict(normal=members[int(middle)]['normal'],
                               intervals=self._intervals([(s['low'], s['high']) for s in members]),
                               segment_count=len(members)))
        return result

    @staticmethod
    def _support(group, low, high):
        intervals = [(max(low, a), min(high, b)) for a, b in group['intervals']]
        return float(sum(max(0., b-a+1) for a, b in intervals))

    @staticmethod
    def _near(group, point, tolerance):
        return any(a-tolerance <= point <= b+tolerance for a, b in group['intervals'])

    def _infer(self, pixels):
        horizontal = self._segments(pixels, False)
        vertical = self._segments(pixels, True)
        hg, vg = self._groups(horizontal), self._groups(vertical)
        diagnostic = dict(horizontal_groups=hg, vertical_groups=vg,
                          horizontal_segments=len(horizontal), vertical_segments=len(vertical),
                          inferred=False, rejection='insufficient_four_observed_sides')
        if len(hg) < 2 or len(vg) < 2:
            return None, diagnostic
        all_segments = horizontal+vertical
        evidence = []
        for segment in all_segments:
            along = np.arange(segment['low'], segment['high']+1, 3.)
            normal = np.full(len(along), segment['normal'])
            evidence.append(np.column_stack((normal, along) if segment['vertical'] else (along, normal)))
        points = np.vstack(evidence)
        tolerance = max(2*self.wall_width, self.MERGE_NORMAL_DISTANCE)
        candidates = []
        # Search only the outermost observed groups, never arbitrary rooms.
        for top, bottom, left, right in itertools.product(hg[:3], hg[-3:], vg[:3], vg[-3:]):
            x0, x1, y0, y1 = left['normal'], right['normal'], top['normal'], bottom['normal']
            width, height = x1-x0, y1-y0
            if min(width, height) < self.MIN_RECTANGLE_SPAN:
                continue
            side_groups = dict(top=top, bottom=bottom, left=left, right=right)
            supports = {name: self._support(group, y0, y1) if name in ('left', 'right')
                        else self._support(group, x0, x1) for name, group in side_groups.items()}
            fractions = {name: support/(height+1 if name in ('left', 'right') else width+1)
                         for name, support in supports.items()}
            if any(value < self.MIN_SIDE_FRACTION for value in fractions.values()):
                continue
            contained = ((points[:, 0] >= x0-tolerance) & (points[:, 0] <= x1+tolerance)
                         & (points[:, 1] >= y0-tolerance) & (points[:, 1] <= y1+tolerance))
            containment = float(contained.mean())
            if containment < self.MIN_CONTAINMENT:
                continue
            corners = dict(top_left=self._near(top, x0, tolerance) and self._near(left, y0, tolerance),
                           top_right=self._near(top, x1, tolerance) and self._near(right, y0, tolerance),
                           bottom_left=self._near(bottom, x0, tolerance) and self._near(left, y1, tolerance),
                           bottom_right=self._near(bottom, x1, tolerance) and self._near(right, y1, tolerance))
            if sum(corners.values()) < 2:
                continue
            interior = 0
            for segment in all_segments:
                if segment['vertical']:
                    inside = x0+tolerance < segment['normal'] < x1-tolerance
                    overlap = min(segment['high'], y1-tolerance)-max(segment['low'], y0+tolerance)
                else:
                    inside = y0+tolerance < segment['normal'] < y1-tolerance
                    overlap = min(segment['high'], x1-tolerance)-max(segment['low'], x0+tolerance)
                interior += bool(inside and overlap >= self.MIN_STRAIGHT_RUN)
            if interior < self.MIN_INTERIOR_SEGMENTS:
                continue
            candidates.append(dict(bounds_pixels=[x0, y0, x1, y1], observed_side_fraction=fractions,
                                   observed_side_support_pixels=supports, observed_corners=corners,
                                   observed_corner_count=int(sum(corners.values())),
                                   long_segment_containment=containment, interior_segments=interior,
                                   sides=side_groups, area=float(width*height)))
        diagnostic['accepted_candidate_count'] = len(candidates)
        if not candidates:
            diagnostic['rejection'] = 'no_four_side_enclosure_with_sufficient_support_and_corners'
            return None, diagnostic
        # Prefer the outer enclosure supported by observations, then coverage.
        best = max(candidates, key=lambda item: (item['area'], item['long_segment_containment'],
                                                sum(item['observed_side_fraction'].values())))
        diagnostic.update(inferred=True, rejection=None, chosen=best)
        return best, diagnostic

    def fuse(self, source_masks, source_weights=None):
        baseline = super().fuse(source_masks, source_weights)
        self.last_pre_boundary_pixels = baseline.copy()
        border = np.zeros_like(baseline)
        rectangle, diagnostic = self._infer(baseline)
        if rectangle is not None:
            x0, y0, x1, y1 = rectangle['bounds_pixels']
            for y in (y0, y1):
                self._draw_flat_band(border, np.array([1., 0.]), np.array([0., 1.]), y, x0, x1)
            for x in (x0, x1):
                self._draw_flat_band(border, np.array([0., 1.]), np.array([1., 0.]), x, y0, y1)
        added = border & ~baseline
        result = baseline | border
        self.last_boundary_pixels = border
        self.last_added_boundary_pixels = added
        self.last_stats.update(
            output_pixels=int(result.sum()), pre_boundary_output_pixels=int(baseline.sum()),
            boundary33=dict(policy='observed_final_map_four_side_rectangular_completion',
                            geometry_source='received_sources_rendered_by_FinalRenderer32',
                            canvas_used_for_geometry_inference=False,
                            required_observed_sides=4, rendered_line_width=self.wall_width,
                            diagnostic=diagnostic, boundary_pixels=int(border.sum()),
                            added_pixels=int(added.sum()), existing_pixels_removed=0,
                            fills_gaps_between_observed_side_intersections=rectangle is not None,
                            guesses_unobserved_side_position=False, truth_adaptive=False,
                            navigation_map_modified=False))
        return result
