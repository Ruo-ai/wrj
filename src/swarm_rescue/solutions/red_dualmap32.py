"""Submission-only regular-wall rendering from actual received source masks.

Eleven pixels is the predeclared ordinary raster-wall prior (a common 7 px
rasterization of a 6 px physical wall, plus 4 px), not an evaluation fit. Separate
short obstacles, boxes, junctions and unresolved residuals may be wider. A twig
attached to a long wall and protruding at most 9 px can be removed by the shape
prior; this is not a guarantee to preserve every real short wall. No navigation writes.
"""
import math

import cv2
import numpy as np

from swarm_rescue.solutions.red_structural28 import Fusion28, quarter_angle


class FinalRenderer32(Fusion28):
    """One origin vote per observed trunk; fixed-width, flat-ended rendering."""
    ORDINARY_WIDTH = 11

    def __init__(self, size, wall_width=11):
        super().__init__(size)
        if type(wall_width) is not int or wall_width not in (7, 9, 11):
            raise ValueError('Predeclared diagnostic widths are 7, 9 or 11 pixels')
        self.wall_width = wall_width

    @staticmethod
    def _runs(values):
        values = np.unique(np.asarray(values, dtype=int))
        return [] if not len(values) else np.split(values, np.flatnonzero(np.diff(values) > 1)+1)

    def _trunk(self, mask, line, axis):
        """Locate a persistent occupied band, not a thick skeleton envelope."""
        direction = line['direction'].copy()
        snapped = False
        if axis is not None:
            angle = math.atan2(direction[1], direction[0])
            correction = quarter_angle(axis[0]-angle)
            if abs(correction) <= self.SNAP_ANGLE:
                direction = self._direction([math.cos(angle+correction), math.sin(angle+correction)])
                snapped = abs(correction) > 1e-8
        normal = np.array([-direction[1], direction[0]])
        center = line['center'].copy()
        extent = (line['ends']-center) @ direction
        radius = min(self.MAX_HALF_BAND+2., line['radius']+2.)
        ends = center + np.array([extent.min(), extent.max()])[:, None]*direction
        low = np.maximum(np.floor(ends.min(axis=0)-radius-2).astype(int), 0)
        high = np.minimum(np.ceil(ends.max(axis=0)+radius+3).astype(int), self.shape[::-1])
        yy, xx = np.nonzero(mask[low[1]:high[1], low[0]:high[0]])
        points = np.column_stack((xx+low[0], yy+low[1])).astype(float)
        along = (points-center) @ direction
        across = (points-center) @ normal
        core = ((along >= extent.min()+2) & (along <= extent.max()-2)
                & (np.abs(across) <= radius))
        if core.sum() < 24:
            return None
        bins = np.rint(across[core]).astype(int)
        first, last = int(bins.min()), int(bins.max())
        histogram = np.bincount(bins-first, minlength=last-first+1)
        persistent = np.flatnonzero(histogram >= max(4., self.PERSISTENCE*histogram.max()))+first
        runs = self._runs(persistent)
        if not runs:
            return None
        # Separate parallel occupied bands: never envelope a gap between them.
        run = min(runs, key=lambda r: (max(0., float(r[0]), float(-r[-1])),
                                      -int(histogram[r-first].sum()), int(r[0])))
        shift = (float(run[0])+float(run[-1]))/2
        half = (float(run[-1])-float(run[0])+1)/2
        centered = center+normal*shift
        support = ((along >= extent.min()-radius) & (along <= extent.max()+radius)
                   & (np.abs(across-shift) <= half+.1))
        observed = points[support]
        if len(observed) < 12:
            return None
        projection = (observed-centered) @ direction
        model = dict(line)
        model.update(center=centered, direction=direction, normal=normal,
                     ends=centered+np.array([projection.min(), projection.max()])[:, None]*direction,
                     length=float(projection.max()-projection.min()),
                     radius=min(self.MAX_HALF_BAND+2., max(half+3., line['radius']+abs(shift))),
                     observed=observed, observed_half_band=half, snapped=snapped)
        return model

    def _protected(self, mask, models):
        """Retain compact boxes and independent short branches before thinning."""
        encoded = mask.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(encoded, connectivity=8)
        keep = np.zeros(count, np.uint8)
        for i in range(1, count):
            width, height, area = map(int, stats[i, 2:5])
            short = max(width, height) < self.MIN_LINE
            compact = max(width, height) <= 3*min(width, height)
            filled = area >= .6*width*height
            if short or (compact and (max(width, height) <= 24 or filled)):
                keep[i] = 1
        protected = keep[labels].astype(bool)
        if not models:
            return mask.copy()
        skeleton = cv2.ximgproc.thinning(encoded*255)
        proposals = cv2.HoughLinesP(skeleton, 1, np.pi/180, threshold=5,
                                   minLineLength=4, maxLineGap=0)
        if proposals is None:
            return protected
        distance = cv2.distanceTransform(encoded, cv2.DIST_L2, 5)
        preserve = np.zeros(self.shape, np.uint8)
        for item in np.asarray(proposals).reshape(-1, 4)[:512]:
            ends = item.reshape(2, 2).astype(float)
            direction = self._direction(ends[1]-ends[0])
            if direction is None:
                continue
            center = ends.mean(axis=0)
            explained = False
            for model in models:
                endpoint_along = (ends-model['center']) @ model['direction']
                endpoint_across = np.abs((ends-model['center']) @ model['normal'])
                model_span = (model['ends']-model['center']) @ model['direction']
                # A skeleton twig can include half the main wall thickness.
                # Protect its independent protrusion, not that inherited half:
                # short fuzz wholly within the bounded cleanup strip is owned
                # by the trunk. Separate small components remain protected.
                if (endpoint_along.min() >= model_span.min()-1
                        and endpoint_along.max() <= model_span.max()+1
                        and endpoint_across.max() <= model['observed_half_band']+3):
                    explained = True
                    break
                if abs(float(direction @ model['direction'])) < math.cos(math.radians(12)):
                    continue
                if abs(float((center-model['center']) @ model['normal'])) > model['radius']:
                    continue
                span = (model['ends']-model['center']) @ model['direction']
                along = float((center-model['center']) @ model['direction'])
                if span.min()-1 <= along <= span.max()+1:
                    explained = True
                    break
            if explained:
                continue
            x, y = np.rint(center).astype(int)
            radius = float(distance[np.clip(y, 0, self.shape[0]-1), np.clip(x, 0, self.shape[1]-1)])
            cv2.line(preserve, tuple(item[:2]), tuple(item[2:]), 1,
                     min(21, max(3, int(math.ceil(2*radius+2)))))
        return protected | ((preserve > 0) & mask)

    def _draw_flat_band(self, pixels, direction, normal, offset, low, high):
        # Pixel coordinates refer to pixel centres. Ordinary axial bands snap
        # by at most half a pixel so their raster width is exactly 7/9/11.
        offset = float(offset)
        if max(abs(direction)) > 1-1e-10:
            offset = float(round(offset))
        ends = normal*offset+np.array([low, high])[:, None]*direction
        margin = self.wall_width/2+1
        lower = np.maximum(np.floor(ends.min(axis=0)-margin).astype(int), 0)
        upper = np.minimum(np.ceil(ends.max(axis=0)+margin+1).astype(int), [self.width, self.height])
        x0, y0 = lower; x1, y1 = upper
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        along = xx*direction[0]+yy*direction[1]
        across = xx*normal[0]+yy*normal[1]-offset
        band = (np.abs(across) <= self.wall_width/2) & (along >= low) & (along <= high)
        pixels[y0:y1, x0:x1] |= band

    def fuse(self, source_masks, source_weights=None):
        if source_weights is not None:
            raise ValueError('One equal geometric vote per origin; no sequence weighting')
        sources = {}
        for source, mask in sorted(source_masks.items()):
            if not isinstance(source, (int, np.integer)) or isinstance(source, bool) or source < 0:
                raise ValueError('Non-negative integer origin ID required')
            value = np.asarray(mask)
            if value.shape != self.shape or value.dtype != np.bool_:
                raise ValueError('Expected one bool RL17 coarse map per origin')
            sources[int(source)] = value.copy()
        extracted = {source: self._extract(mask, source) for source, mask in sources.items()}
        axis = self._observed_axis([line for lines in extracted.values() for line in lines])
        union = np.zeros(self.shape, bool)
        residual = np.zeros(self.shape, bool)
        models = []
        source_stats = {}
        for source, mask in sources.items():
            union |= mask
            own_models = [model for line in extracted[source]
                          if (model := self._trunk(mask, line, axis)) is not None]
            protected = self._protected(mask, own_models)
            own = mask.copy()
            for model in own_models:
                self._explain(own, model)
            own |= protected
            residual |= own
            models.extend(own_models)
            source_stats[str(source)] = dict(input_cells=int(mask.sum()), modeled_lines=len(own_models),
                                            residual_cells=int(own.sum()), protected_cells=int(protected.sum()))
        pixels = np.repeat(np.repeat(residual, 3, 0), 3, 1)[:self.height, :self.width].copy()
        self.last_lines = []
        for cluster in self._clusters(models):
            by_source = {}
            for model in cluster:
                by_source.setdefault(model['source'], []).append(model)
            directions = []
            for records in by_source.values():
                reference = records[0]['direction']
                directions.append(self._direction(np.mean([m['direction']*(1 if m['direction']@reference >= 0 else -1)
                                                           for m in records], axis=0)))
            reference = directions[0]
            direction = self._direction(np.mean([d*(1 if d@reference >= 0 else -1) for d in directions], axis=0))
            normal = np.array([-direction[1], direction[0]])
            offsets = [float(np.median([m['center']@normal for m in records])) for records in by_source.values()]
            offset = float(np.median(offsets))*3+float(np.ones(2)@normal)
            # Any genuinely occupied source run is eligible, including a lone
            # observer. No maxLineGap/closing joins empty along-wall bins.
            observations = np.vstack([m['observed'] for m in cluster])
            bins = np.floor(observations@direction+.5).astype(int)
            intervals = self._runs(bins)
            drawn = []
            for run in intervals:
                low = 3*(float(run[0])-.5)+float(np.ones(2)@direction)
                high = 3*(float(run[-1])+.5)+float(np.ones(2)@direction)
                self._draw_flat_band(pixels, direction, normal, offset, low, high)
                drawn.append([low, high])
            self.last_lines.append(dict(sources=sorted(by_source), direction=direction.tolist(),
                                        normal_offset_pixels=offset, intervals_pixels=drawn,
                                        width_pixels=self.wall_width,
                                        source_normal_spread=3*(max(offsets)-min(offsets)),
                                        participating_bands_enveloped=False))
        self.last_stats = dict(source_count=len(sources), per_source=source_stats,
                               input_union_cells=int(union.sum()), residual_cells=int(residual.sum()),
                               modeled_lines=len(models), fused_lines=len(self.last_lines),
                               output_pixels=int(pixels.sum()), ordinary_axial_line_width=self.wall_width,
                               width_policy='predeclared ordinary-wall prior; not truth-adaptive',
                               width_guarantee_scope='modeled ordinary long walls; residual/boxes/junctions excluded',
                               orthogonal_axis_radians=None if axis is None else axis[0],
                               no_cross_source_residual_erasure=True, flat_ends_no_gap_closing=True,
                               sources_equal_weight=True, navigation_map_modified=False)
        return pixels.astype(bool)
