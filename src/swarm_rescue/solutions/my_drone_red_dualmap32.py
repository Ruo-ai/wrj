"""DualMap32: Fusion27 flight policy and an independent submission renderer.

Control retains Fusion27 navigation/deployment/clock cutoffs exactly. Clean
changes only the exploration grid's treatment of repeatedly contradicted hits.
Submission maps never become navigation occupancy. Delivery is radio-only;
an incomplete deadline fallback is recorded rather than called complete.
"""
from collections import Counter
import hashlib

import numpy as np

from swarm_rescue.solutions.my_drone_red_fusion27 import MyDroneRedFusion27, FinalSnapshots27
from swarm_rescue.solutions.my_drone_red_structural28 import MyDroneRedStructural28
from swarm_rescue.solutions.red_structural28 import StructuralMapper28
from swarm_rescue.solutions.red_dualmap32 import FinalRenderer32


class MyDroneRedDualMap32(MyDroneRedFusion27):
    """Submission-only control arm; original Fusion27 flight policy."""
    CLEAN_NAVIGATION = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submission_mapper = StructuralMapper28(self.size_area)
        self._radio_maps = FinalSnapshots27(self._map.shape, self._n,
            self._submission_mapper, lambda: int(self.elapsed_timestep))
        self._fusion27 = FinalRenderer32(self.size_area, wall_width=11)
        self._s28_stats = Counter()
        self._s28_receipts = {}
        self._s28_delivery = {}
        self._s28_submission = None
        self._s28_return = {'policy': 'original_fusion27', 'gather_at': self.GATHER_AT}
        if self.CLEAN_NAVIGATION:
            from swarm_rescue.solutions.red_navigation32 import EvidenceNavigation32
            self._explore = EvidenceNavigation32(self.size_area, self._event)

    def _exchange_state(self):
        return MyDroneRedStructural28._exchange_state(self)

    def _receive(self):
        # This method decodes actual radio packets; it does not access any
        # other drone or replace the navigation policy with Structural28.
        return MyDroneRedStructural28._receive(self)

    def define_message_for_all(self):
        self._exchange_state()
        original = super().define_message_for_all()
        return ('structural28-v1', original,
                tuple(self._s28_receipts[i] for i in sorted(self._s28_receipts)))

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
            exploration_cutoff_policy=self.FREEZE_AT, gathering_policy=self.GATHER_AT,
            navigation_policy='fusion27_repeat_free_cleanup' if self.CLEAN_NAVIGATION else 'original_fusion27',
            submission_policy='structural28_local_mapper; independent_11px_supported_long_walls',
            navigation_stats=dict(getattr(self._explore, 'stats', {})),
            fusion_stats=dict(self._fusion27.last_stats),
            submitted_pixels_sha256=hashlib.sha256(np.asarray(pixels, dtype=bool).tobytes()).hexdigest(),
            scope='Received source maps sealed at explicit cutoff; all-source ACK required; not post-cutoff scans')
        self._s28_submission = self._f27_submission = receipt
        self.submit_exploration_map(np.asarray(pixels, dtype=np.uint8))
        self._exploration_submitted = True
        self._event('dualmap32_submitted', complete=complete, missing=state['missing'],
            acknowledgements=state['acknowledged_sources'], sources=state['received_versions'],
            fusion=dict(self._fusion27.last_stats))

    def control(self):
        command = super().control()
        self._s28_delivery = self._exchange_state()
        return command


class MyDroneRedDualMap32Clean(MyDroneRedDualMap32):
    """Paired arm: the only strategy delta is repeated-free navigation cleanup."""
    CLEAN_NAVIGATION = True
