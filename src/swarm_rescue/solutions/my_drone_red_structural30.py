"""Structural30: preserve local route targets and necessary return corners.

Mapping, goal selection, deployment, clock, cutoff and mutual receipt protocol
are inherited. Local passage repairs also isolate mutable direction/target
arguments; observed-route return retains necessary short corners.
"""
from swarm_rescue.solutions.my_drone_red_structural29 import MyDroneRedStructural29
from swarm_rescue.solutions.red_return30 import ReturnPlanner30
from swarm_rescue.solutions.red_passage30 import PassageMotion30


class MyDroneRedStructural30(MyDroneRedStructural29):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._s28_return_plan = ReturnPlanner30(int(self.identifier or 0))
        self._rn7 = PassageMotion30(int(self.identifier or 0))
