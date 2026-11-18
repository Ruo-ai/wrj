"""Sensor-only mapping fixture for independent, payload-free simulation trials."""

import math

import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR


class PayloadFreeMapperTrial(DroneAbstract):
    """Drive straight, turning on an obstacle, and accumulate measured range endpoints."""

    CELL = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._map_width, self._map_height = self.size_area
        shape = (math.ceil(self._map_height / self.CELL), math.ceil(self._map_width / self.CELL))
        self._hit_scans = np.zeros(shape, dtype=np.uint8)
        self._map_counts = dict(calls=0, mapped_scans=0, missing_pose_scans=0, missing_scan=0)

    def define_message_for_all(self):
        return None

    def control(self) -> CommandsDict:
        self._map_counts["calls"] += 1
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        ranges = self.lidar_values()
        angles = self.lidar_rays_angles()
        if ranges is None or angles is None:
            self._map_counts["missing_scan"] += 1
            return command
        ranges = np.asarray(ranges, dtype=float).reshape(-1)
        angles = np.asarray(angles, dtype=float).reshape(-1)
        if ranges.shape != angles.shape:
            raise ValueError("Lidar range and angle counts differ")
        valid = np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0)
        if not np.any(valid):
            self._map_counts["missing_scan"] += 1
            return command

        position = self.measured_gps_position()
        heading = self.measured_compass_angle()
        if (position is None or np.shape(position) != (2,) or not np.isfinite(position).all()
                or heading is None or not math.isfinite(heading)):
            self._map_counts["missing_pose_scans"] += 1
        else:
            # Like examples/example_mapping.py, omit far-end readings near lidar saturation.
            hits = valid & (ranges < 0.9 * MAX_RANGE_LIDAR_SENSOR)
            theta = angles[hits] + heading
            x = position[0] + ranges[hits] * np.cos(theta)
            y = position[1] + ranges[hits] * np.sin(theta)
            cols = np.floor((x + self._map_width / 2) / self.CELL).astype(int)
            rows = np.floor((self._map_height / 2 - y) / self.CELL).astype(int)
            inside = ((x >= -self._map_width / 2) & (x < self._map_width / 2) &
                      (y > -self._map_height / 2) & (y <= self._map_height / 2))
            # An occupied cell requires endpoints in two scans. Multiple rays landing
            # in the same cell within a single scan advance its counter only once.
            cells = np.unique(rows[inside] * self._hit_scans.shape[1] + cols[inside])
            flat = self._hit_scans.ravel()
            flat[cells] = np.minimum(2, flat[cells] + 1)
            self._map_counts["mapped_scans"] += 1

        front = np.abs(angles) <= math.pi / 6
        if front.any() and valid[front].all():
            if np.min(ranges[front]) < 55.0:
                command["rotation"] = 0.35
            else:
                command["forward"] = 0.35
        return command

    def estimated_walls(self) -> np.ndarray:
        """Return a new binary H x W image, with column=x+W/2 and row=H/2-y."""
        occupied = (self._hit_scans >= 2).astype(np.uint8)
        return occupied.repeat(self.CELL, axis=0).repeat(self.CELL, axis=1)[:self._map_height, :self._map_width]

    def mapping_stats(self) -> dict:
        """Counts of callbacks and scans projected, missing a pose, or missing lidar."""
        return self._map_counts.copy()
