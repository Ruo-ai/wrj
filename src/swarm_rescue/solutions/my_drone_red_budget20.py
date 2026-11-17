"""R20-B: two bounded renewals of RN7's exhausted eight-task allowance.

Single factor on RL17-R: grant at most eight further local passage episodes
after step 1200, and eight after 2400. The original clearance, velocity,
task duration, cooldown, peer brakes and mission preemption stay inherited.
No map identity/geometry/truth, new messages, or navigation-map edits.
"""
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.my_drone_red_rn7 import PassageMotion


class RenewedPassage20(PassageMotion):
    def __init__(self, identifier=0):
        super().__init__(identifier)
        self.renewals = 0

    @property
    def lifetime_episodes(self):
        return 8*self.renewals+self.episodes

    def command(self, d, a, coarse, position, heading, desired, objects=(), allow=True):
        if (allow and self.valid_odom and d is not None and a is not None
                and not self.path and self.episodes >= 8 and self.renewals < 2
                and self.tick >= 1200*(self.renewals+1) and self.tick >= self.cooldown):
            self.episodes = 0
            self.renewals += 1
            self.event('r20_budget_renewed')
        return super().command(d, a, coarse, position, heading, desired, objects, allow)


class MyDroneRedBudget20(MyDroneRedRepair17):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rn7 = RenewedPassage20(int(self.identifier or 0))
