"""Structural31: complete legal deployment when the current return budget is due.

Only pending placement selection is preempted. Placement confirmations, retry
limits, original release times, separation, physics and all mapping/navigation
remain inherited. A historical latched return request is never sufficient.
"""
import math

from swarm_rescue.solutions.my_drone_red_structural30 import MyDroneRedStructural30
from swarm_rescue.solutions.my_drone_red_advanced2 import _M06_MyDroneRedV3


class MyDroneRedStructural31(MyDroneRedStructural30):
    PRESSURE_CONFIRM_STEPS = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._s31_pressure_frames = 0
        self._s31_pressure_step = None

    def _deployment(self, command):
        inventory = self.carried_bombs_count()
        plan = self._s28_return_plan
        budget = plan.last_budget
        required = budget.get('required_steps')
        remaining = budget.get('effective_remaining_steps')
        current = (plan.step == self._steps and not plan.returning
                   and isinstance(required, (int, float)) and math.isfinite(required)
                   and isinstance(remaining, (int, float)) and math.isfinite(remaining))
        pressure = bool(inventory > 0 and current and required >= remaining)
        if self._s31_pressure_step != self._steps:
            consecutive = self._s31_pressure_step == self._steps-1
            self._s31_pressure_frames = (self._s31_pressure_frames+1 if consecutive else 1) if pressure else 0
            self._s31_pressure_step = self._steps
        confirming = (self._attempt_step is not None or
                      (self._last_inventory is not None and inventory < self._last_inventory))
        if confirming or not pressure or self._s31_pressure_frames < self.PRESSURE_CONFIRM_STEPS:
            return super()._deployment(command)
        # Use the existing fallback unchanged, without pretending the match
        # clock is later or inventing an optimised target at the current pose.
        self._s28_stats['budget_due_placement_fallback_ticks'] += 1
        before = self._total_attempts
        _M06_MyDroneRedV3._deployment(self, command)
        if self._total_attempts > before:
            self._event('s31_budget_due_placement', required_steps=float(required),
                        effective_remaining_steps=float(remaining), inventory=inventory)
