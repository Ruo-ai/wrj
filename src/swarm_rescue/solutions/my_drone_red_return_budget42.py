"""Schedule40R1 with only the travel-time term of its return budget changed.

Use each drone's legal odometry, including stopping/turning samples. The
observed return graph, executor, recovery/exchange reserves, clocks, hard
cutoff, source sealing, radio, pose, navigation and renderer are inherited.
"""
from collections import Counter, deque
import math
import numpy as np

from swarm_rescue.solutions.my_drone_red_schedule40r1 import MyDroneRedSchedule40R1


class MyDroneRedReturnBudget42(MyDroneRedSchedule40R1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rb42_motion = deque(maxlen=240)
        self._rb42_sample_step = -1
        self._rb42_stats = Counter()

    def _budget39(self):
        # Retain both native clocks and all fixed delivery margins verbatim.
        legacy_due = super()._budget39()
        budget = self._d39_budget
        if self._steps != self._rb42_sample_step:
            if self._steps-self._rb42_sample_step > 1:
                self._rb42_motion.clear()
            value = self.odometer_values()
            valid = bool(value is not None and np.shape(value) == (3,)
                         and np.isfinite(value).all())
            speed = float(np.clip(abs(value[0]), 0., 8.)) if valid else 0.
            self._rb42_motion.append((speed, valid))
            self._rb42_sample_step = self._steps
        samples = list(self._rb42_motion)
        count = len(samples)
        mean_long = sum(s for s, _ in samples)/max(1, count)
        recent = samples[-60:]
        mean_recent = sum(s for s, _ in recent)/max(1, len(recent))
        valid_fraction = sum(v for _, v in samples)/max(1, count)
        known_route = bool(self._d39_route and self._d39_cost is not None
                           and math.isfinite(self._d39_cost))
        # Do not infer a faster return from a single burst or an unknown path.
        effective_speed = 2.1
        if known_route and count >= 120 and valid_fraction >= .9:
            effective_speed = float(np.clip(.8*min(mean_long, mean_recent), 2.1, 3.8))
        legacy_needed = budget['needed']
        needed = budget['route_pixels']/effective_speed+180+100
        due = budget['remaining'] <= needed or self._progress() >= .84
        budget.update(needed=float(needed), legacy_needed=legacy_needed,
            effective_speed=effective_speed, mean_odometry_long=mean_long,
            mean_odometry_recent=mean_recent, speed_samples=count,
            valid_fraction=valid_fraction, known_route=known_route,
            legacy_due=bool(legacy_due), budget_due=bool(due),
            travel_steps=float(budget['route_pixels']/effective_speed),
            recovery_reserve=180, exchange_reserve=100)
        if (self._d39_mode == 'explore' and self.carried_bombs_count() == 0
                and legacy_due and not due):
            self._rb42_stats['early_return_deferred_ticks'] += 1
        return due
