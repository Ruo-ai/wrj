"""RL23 changes only the raster passed to the final exploration-map submission."""

import cv2
import numpy as np

from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17


def submission_pixels(grid):
    """Vote over one mapper cell (3 px) in each direction, including the centre."""
    return cv2.medianBlur(np.asarray(grid, dtype=np.uint8), 7)


class MyDroneRedMap23(MyDroneRedRepair17):
    def submit_exploration_map(self, grid):
        super().submit_exploration_map(submission_pixels(grid))
