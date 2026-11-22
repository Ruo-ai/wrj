"""Structural29: retain Structural28 geometry and repair mission arbitration.

Original placement targets remain active until inventory is empty. Return
requests never clear an unfinished deployment. ReturnPlanner29 handles fresh
observed paths and dynamic peer yielding. Frozen Structural28 evidence remains
unchanged; _s28 receipt field names are retained for identical passive capture.
"""
import hashlib
import numpy as np
from swarm_rescue.solutions.my_drone_red_structural28 import MyDroneRedStructural28
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.red_return29 import ReturnPlanner29


class MyDroneRedStructural29(MyDroneRedStructural28):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._s28_return_plan = ReturnPlanner29(int(self.identifier or 0))

    def _navigation(self):
        planner = self._s28_return_plan
        limits = self._misc_data
        step_limit = getattr(limits, 'max_timestep_limit', None)
        time_limit = getattr(limits, 'max_walltime_limit', None)
        remaining_steps = None if not step_limit else .94*step_limit-self.elapsed_timestep
        remaining_seconds = None if not time_limit else .94*time_limit-self.elapsed_walltime
        planner.should_return(remaining_steps, remaining_seconds, self.carried_bombs_count())
        if self._progress() >= self.RETURN_LATEST:
            if self.carried_bombs_count() == 0:
                planner.start_return('latest_return_start')
            else:
                planner.request_return('latest_return_waiting_for_inventory')
        if planner.returning:
            self._gathering = True
            self._velocity = planner.velocity.copy()
            self._route = []
            self._bomb_target = None
            if not self._s28_return_started:
                self._s28_return_started = True
                self._s28_stats['return_started_step'] = self._steps
                self._event('s29_return_started', return_state=planner.status())
                if self._rn7.path:
                    self._rn7.release('return_priority')
                motion = getattr(self, '_segment_motion', None)
                if motion is not None and motion.state not in ('NORMAL', 'COOLDOWN'):
                    motion.release('return_priority')
            command = planner.command(self._position, self._heading, self._velocity,
                                      self.lidar_values(), self.lidar_rays_angles(),
                                      self.semantic_values(), peers=self._peers)
            self._s28_return = planner.status()
            self._slow_ticks = 0
            return self._empty_command() if command is None else dict(self._empty_command(), **command)
        # Preserve original deployment/goal logic, but skip Fusion27's fixed .65
        # gather switch. Return uses the verified path and native remaining budget.
        if self._rn7.path:
            motion = getattr(self, '_segment_motion', None)
            if motion is not None and motion.state not in ('NORMAL', 'COOLDOWN'):
                motion.release('s28_passage')
        original = MyDroneRedRepair17._navigation(self)
        near_bomb = (self._bomb_target is not None and self._position is not None
                     and np.linalg.norm(self._bomb_target-self._position) < 55)
        allow = (self._position is not None and self.drone_health > 10
                 and not self._gathering and not self._exploration_submitted
                 and self._attempt_step is None and not near_bomb
                 and self._steps >= self._yield_until
                 and (bool(self._rn7.path) or abs(original.get('rotation', 0.)) < .02))
        desired = self._last_motion_angle if self._last_motion_angle is not None else 0.
        command = self._rn7.command(self.lidar_values(), self.lidar_rays_angles(), self._explore,
                                    self._position, self._heading, desired, self.semantic_values(), allow)
        if command is not None:
            motion = getattr(self, '_segment_motion', None)
            if motion is not None and motion.state not in ('NORMAL', 'COOLDOWN'):
                motion.release('s28_passage')
            original.update(command)
        self._s28_return = planner.status()
        return original

    def _submit(self):
        if self._exploration_submitted:
            return
        state = self._exchange_state()
        self._s28_delivery = self._f27_delivery = state
        own = int(self.identifier or 0)
        deadline = self.DEADLINE_AT if own == 0 else min(.995, .980+.003*own)
        complete = state['final_complete']
        if not ((own == 0 and complete) or self._progress() >= deadline):
            self._s28_stats['submission_deferred'] += 1
            return
        pixels = self._fusion27.fuse(self._radio_maps.mapping_masks)
        receipt = dict(state, submitter=own, submit_control_step=self._steps,
            submit_elapsed_step=int(self.elapsed_timestep), submit_walltime=float(self.elapsed_walltime),
            reason='all_final_maps_and_acknowledgements_received' if complete else 'deadline_incomplete_fallback',
            exploration_cutoff_policy=self.FREEZE_AT, return_policy='complete deployment before return; observed route budget; latest .70 request',
            fusion_stats=dict(self._fusion27.last_stats),
            return_state=self._s28_return_plan.status(), passage_stats=dict(self._rn7.stats),
            submitted_pixels_sha256=hashlib.sha256(np.asarray(pixels, dtype=bool).tobytes()).hexdigest(),
            scope='Final maps at explicit cutoff; every source acknowledges the same complete set; not post-cutoff scans')
        self._s28_submission = self._f27_submission = receipt
        self.submit_exploration_map(np.asarray(pixels, dtype=np.uint8))
        self._exploration_submitted = True
        self._event('s29_submitted', complete=complete, missing=state['missing'],
                    acknowledgements=state['acknowledged_sources'], sources=state['received_versions'],
                    fusion=dict(self._fusion27.last_stats))

