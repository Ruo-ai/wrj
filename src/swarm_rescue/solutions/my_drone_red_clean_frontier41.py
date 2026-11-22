"""Clean + observed boundary, with only Delivery39's pending-frontier planner.

The experiment intentionally retains Clean's pose, mapping, deployment, radio,
gathering and submission deadlines. It does not inherit Delivery39's return
controller or Continuous38. Empty-inventory frontiers are available before
Clean's existing gathering cutoff; no extra 60% activation delay is added.
"""
from collections import Counter

from swarm_rescue.solutions.my_drone_red_frontier34 import MyDroneRedCleanBoundary34
from swarm_rescue.solutions.red_delivery39 import FrontierQueue39


class MyDroneRedCleanFrontier41(MyDroneRedCleanBoundary34):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._explore = FrontierQueue39(self.size_area, self._event)
        self._cf41_stats = Counter()

    def _navigation(self):
        self._explore.extension_enabled = bool(
            self._position is not None
            and self.carried_bombs_count() == 0
            and not self._gathering
            and self._progress() < self.GATHER_AT
        )
        if self._explore.extension_enabled:
            self._cf41_stats['frontier_enabled_ticks'] += 1
        # In particular, gathering still runs through the original Clean chain.
        return super()._navigation()
