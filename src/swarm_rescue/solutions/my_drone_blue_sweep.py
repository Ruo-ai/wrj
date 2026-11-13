"""Experimental pressure opponent. Prioritize gaps in the team's search paths."""
import cv2
import numpy as np

from .my_drone_eval_blue import MyDroneBlue, RescueGrid


class SweepGrid(RescueGrid):
    def route(self, position, target=None, identifier=0, number=1, claims=()):
        if target is None:
            allowed, visited = self.layers()
            start = self.cell(position)
            allowed[start] = True
            visited[start] = True
            _, labels = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=4)
            reachable = labels == labels[start]
            gap = cv2.distanceTransform((~visited).astype(np.uint8), cv2.DIST_L2, 3)
            row, col = np.indices(self.shape)
            travel = np.hypot(row - start[0], col - start[1])
            score = np.minimum(gap, 12) - 0.12 * travel
            for point in claims:
                r, c = self.cell(point)
                score -= np.maximum(0., 8 - np.hypot(row-r, col-c))
            score[~reachable | (travel < 3)] = -np.inf
            best = np.unravel_index(np.argmax(score), self.shape)
            if np.isfinite(score[best]) and gap[best] >= 2:
                target = self.point(best)
        return super().route(position, target, identifier, number, claims)


class MyDroneBlueSweep(MyDroneBlue):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grid = SweepGrid((self._grid.width, self._grid.height))
