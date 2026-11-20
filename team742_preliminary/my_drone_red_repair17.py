"""RL17-R: preserve every accepted RL11 decision; fill rejected matches only.

Independent ablation after all-front-end RL17 regressed in same-input replay.
No threshold tuning, navigation change, peer validation or pending-map gating.
"""
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper, WallLineMatcher, MyDroneRedLines11
from swarm_rescue.solutions.my_drone_red_mapping import MappingSnapshots
from swarm_rescue.solutions.my_drone_red_p2l17 import PointLineMatcher17


class RepairMatcher17(PointLineMatcher17):
    def match(self, points, position, heading):
        original = WallLineMatcher.match(self, points, position, heading)
        self.last_original = original[1]
        if original[0] is not None:
            self.last_p2l = dict(reason='kept_rl11',original=original[1])
            self.p2l_stats['kept_rl11'] += 1
            return original
        # Parent may evaluate RL11 again on rejection (bounded, read-only).
        # Reference insertion still happens exactly once, in the mapper.
        result = super().match(points, position, heading)
        self.last_p2l['original'] = original[1]
        return result


class RepairMapper17(WallLineMapper):
    def __init__(self, size):
        super().__init__(size)
        self.matcher = RepairMatcher17(size)


class MyDroneRedRepair17(MyDroneRedLines11):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submission_mapper = RepairMapper17(self.size_area)
        self._radio_maps = MappingSnapshots(self._map.shape, self._n, self._submission_mapper)
