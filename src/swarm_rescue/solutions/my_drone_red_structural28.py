"""Structural28: legal regular-wall evidence, observed passages and final exchange.

The fixed exploration cutoff is explicit. Each origin acknowledges the same
received final-map set; leader submits only after all five receipts, except a
deadline fallback which remains explicitly incomplete in the actual receipt.
"""
from collections import Counter
import hashlib
from types import SimpleNamespace

import numpy as np

from swarm_rescue.solutions.my_drone_red_fusion27 import MyDroneRedFusion27, FinalSnapshots27
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.red_structural28 import StructuralMapper28, Fusion28
from swarm_rescue.solutions.red_passage28 import PassageMotion28
from swarm_rescue.solutions.red_return28 import ReturnPlanner28


class MyDroneRedStructural28(MyDroneRedFusion27):
    NARROW_ENABLED = False  # The new passage controller is called explicitly.
    FREEZE_AT = .86
    RETURN_LATEST = .70

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submission_mapper = StructuralMapper28(self.size_area)
        self._radio_maps = FinalSnapshots27(self._map.shape, self._n, self._submission_mapper,
                                            lambda: int(self.elapsed_timestep))
        self._fusion27 = Fusion28(self.size_area)
        self._rn7 = PassageMotion28(int(self.identifier or 0))
        self._s28_return_plan = ReturnPlanner28(int(self.identifier or 0))
        self._s28_stats = Counter()
        self._s28_return = {}
        self._s28_delivery = {}
        self._s28_submission = None
        self._s28_receipts = {}
        self._s28_return_started = False

    def _exchange_state(self):
        state = self._radio_maps.delivery()
        payload_complete = state['final_complete']
        signature = None
        if payload_complete:
            manifest = tuple((i, state['received_versions'][str(i)], state['received_digests'][str(i)])
                             for i in range(self._n))
            signature = hashlib.sha256(repr(manifest).encode('ascii')).hexdigest()
            own = int(self.identifier or 0)
            self._s28_receipts[own] = (own, state['epoch_elapsed'], signature)
        acknowledged = [i for i, ack in self._s28_receipts.items()
                        if signature is not None and ack[1] == state['epoch_elapsed'] and ack[2] == signature]
        state.update(source_maps_complete=payload_complete, final_set_sha256=signature,
                     acknowledged_sources=sorted(acknowledged),
                     missing_acknowledgements=[i for i in range(self._n) if i not in acknowledged],
                     final_complete=payload_complete and len(acknowledged) == self._n)
        return state

    def define_message_for_all(self):
        self._exchange_state()
        original = super().define_message_for_all()
        return ('structural28-v1', original,
                tuple(self._s28_receipts[i] for i in sorted(self._s28_receipts)))

    def _receive(self):
        messages = []
        for sender, message in self.communicator.received_messages:
            if not isinstance(message, tuple) or len(message) != 3 or message[0] != 'structural28-v1':
                continue
            messages.append((sender, message[1]))
            receipts = message[2]
            if not isinstance(receipts, tuple) or len(receipts) > self._n:
                continue
            for ack in receipts:
                if (not isinstance(ack, tuple) or len(ack) != 3 or type(ack[0]) is not int
                        or not 0 <= ack[0] < self._n or type(ack[1]) is not int
                        or not 0 < ack[1] <= self.elapsed_timestep or not isinstance(ack[2], str)
                        or len(ack[2]) != 64 or any(c not in '0123456789abcdef' for c in ack[2])):
                    self._s28_stats['invalid_receipt'] += 1
                    continue
                previous = self._s28_receipts.get(ack[0])
                if previous is None or previous == ack:
                    self._s28_receipts[ack[0]] = ack
                else:
                    self._s28_stats['conflicting_receipt'] += 1
        proxy = SimpleNamespace(_n=self._n, elapsed_timestep=self.elapsed_timestep,
            _status=self._status, _peers=self._peers, _radio_maps=self._radio_maps,
            _explore=self._explore, _valid_peer=self._valid_peer,
            _bomb_claims=self._bomb_claims, _bomb_records=self._bomb_records,
            _f27_home=self._f27_home, communicator=SimpleNamespace(received_messages=messages))
        MyDroneRedFusion27._receive(proxy)
        self._f27_home = proxy._f27_home

    def _update_observation(self):
        super()._update_observation()
        planner = self._s28_return_plan
        planner.identifier = int(self.identifier or 0)
        if self._f27_home is not None:
            planner.set_home(self._f27_home[2])
        elif self._home is not None:
            planner.set_home(self._home)
        planner.observe(self._steps, self._position, self._heading, self.odometer_values(),
                        self.lidar_values(), self.lidar_rays_angles(), self.semantic_values(),
                        elapsed_seconds=float(self.elapsed_walltime))

    def _navigation(self):
        planner = self._s28_return_plan
        limits = self._misc_data
        step_limit = getattr(limits, 'max_timestep_limit', None)
        time_limit = getattr(limits, 'max_walltime_limit', None)
        remaining_steps = None if not step_limit else .94*step_limit-self.elapsed_timestep
        remaining_seconds = None if not time_limit else .94*time_limit-self.elapsed_walltime
        planner.should_return(remaining_steps, remaining_seconds, self.carried_bombs_count())
        if self._progress() >= self.RETURN_LATEST:
            planner.start_return('latest_return_start')
        if planner.returning:
            self._gathering = True
            self._velocity = planner.velocity.copy()
            self._route = []
            self._bomb_target = None
            if not self._s28_return_started:
                self._s28_return_started = True
                self._s28_stats['return_started_step'] = self._steps
                self._event('s28_return_started', return_state=planner.status())
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
            exploration_cutoff_policy=self.FREEZE_AT, return_policy='observed route budget; latest .70',
            fusion_stats=dict(self._fusion27.last_stats),
            return_state=self._s28_return_plan.status(), passage_stats=dict(self._rn7.stats),
            submitted_pixels_sha256=hashlib.sha256(np.asarray(pixels, dtype=bool).tobytes()).hexdigest(),
            scope='Final maps at explicit cutoff; every source acknowledges the same complete set; not post-cutoff scans')
        self._s28_submission = self._f27_submission = receipt
        self.submit_exploration_map(np.asarray(pixels, dtype=np.uint8))
        self._exploration_submitted = True
        self._event('s28_submitted', complete=complete, missing=state['missing'],
                    acknowledgements=state['acknowledged_sources'], sources=state['received_versions'],
                    fusion=dict(self._fusion27.last_stats))

    def control(self):
        command = super().control()
        self._s28_return = self._s28_return_plan.status()
        self._s28_delivery = self._exchange_state()
        return command
