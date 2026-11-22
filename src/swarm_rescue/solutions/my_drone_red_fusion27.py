"""RL17-R with evidence fusion and a radio-only final-map delivery barrier.

The exploration cutoff is explicit. After it, each drone's submission map and
published final payload are immutable; ordinary navigation sensing continues.
No shared team state, map identity, true pose, or simulator clock override.
"""
from collections import Counter
import hashlib
import math
from types import SimpleNamespace

import numpy as np

from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17, RepairMapper17
from swarm_rescue.solutions.my_drone_red_mapping import MappingSnapshots
from swarm_rescue.solutions.my_drone_red_advanced2 import _M08_MyDroneRedV4
from swarm_rescue.solutions.red_fusion27 import Fusion27


class FinalizableRepairMapper17(RepairMapper17):
    def __init__(self, size):
        super().__init__(size)
        self.final_mask = None
        self.final_mapper_step = None

    def update(self, *args, **kwargs):
        if self.final_mask is None:
            return super().update(*args, **kwargs)
        self.stats['after_cutoff_observations_not_mapped'] = self.stats.get('after_cutoff_observations_not_mapped', 0) + 1

    def mask(self):
        if self.final_mask is not None:
            return self.final_mask.copy()
        return super().mask()

    def seal(self):
        if self.final_mask is None:
            self.final_mask = super().mask().copy()
            self.final_mask.setflags(write=False)
            self.final_mapper_step = self.step


class FinalSnapshots27(MappingSnapshots):
    """Immutable source finals, authenticated by their actual received bytes.

    Digests detect accidental/inconsistent payloads; they are not signatures.
    Manifests alone never satisfy the delivery barrier.
    """
    def __init__(self, shape, number_drones, mapper, clock):
        super().__init__(shape, number_drones, mapper)
        self.clock = clock
        self.epoch = None
        self.own_final = None
        self.manifests = {}
        self.protocol_stats = Counter()
        self._digests = {}

    def accept(self, snapshot):
        if not isinstance(snapshot, tuple) or len(snapshot) != 3:
            return False
        source, sequence, payload = snapshot
        if (type(source) is not int or not 0 <= source < self.number_drones or
                type(sequence) is not int or not 0 <= sequence <= self.clock() or
                not isinstance(payload, bytes)):
            self.protocol_stats['invalid_snapshot'] += 1
            return False
        previous = self.snapshots.get(source)
        if previous is not None and sequence == previous[1] and payload != previous[2]:
            self.protocol_stats['conflicting_payload'] += 1
            return False
        final = self.manifests.get(source)
        if final is not None and sequence > final[2]:
            self.protocol_stats['post_final_snapshot_rejected'] += 1
            return False
        accepted = super().accept(snapshot)
        if accepted:
            self._digests.pop(source, None)
        return accepted

    def publish(self, source, sequence, mask):
        if self.own_final is not None and source == self.own_final[0]:
            self.protocol_stats['immutable_republish'] += 1
            return
        return super().publish(source, sequence, mask)

    def payload_digest(self, source):
        snapshot = self.snapshots.get(source)
        if snapshot is None:
            return None
        if source not in self._digests:
            self._digests[source] = hashlib.sha256(snapshot[2]).hexdigest()
        return self._digests[source]

    def freeze(self, source, sequence, epoch, nav_mask, inventory):
        if self.own_final is not None:
            return self.own_final
        if (type(epoch) is not int or epoch != self.clock() or type(sequence) is not int
                or not 0 <= sequence <= epoch):
            raise ValueError('Final epoch must be the current public elapsed step')
        self.mapper.seal()
        self.epoch = epoch
        self.manifests = {i: m for i, m in self.manifests.items() if m[1] == epoch}
        super().publish(source, sequence, nav_mask)
        self.own_final = self.snapshots[source]
        if self.own_final[1] != sequence:
            raise ValueError('Final version was not published')
        self.manifests[source] = (source, epoch, sequence, self.payload_digest(source), int(inventory))
        self.protocol_stats['own_finalized'] += 1
        return self.own_final

    def accept_manifest(self, manifest):
        if not isinstance(manifest, tuple) or len(manifest) != 5:
            return False
        source, epoch, sequence, digest, inventory = manifest
        if (type(source) is not int or not 0 <= source < self.number_drones or
                type(epoch) is not int or not 0 < epoch <= self.clock() or
                type(sequence) is not int or not 0 <= sequence <= epoch or
                type(inventory) is not int or inventory < 0 or
                not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest) or
                (self.epoch is not None and epoch != self.epoch)):
            self.protocol_stats['invalid_manifest'] += 1
            return False
        previous = self.manifests.get(source)
        if previous is not None and previous != manifest:
            self.protocol_stats['conflicting_manifest'] += 1
            return False
        self.manifests[source] = manifest
        return True

    def delivery(self):
        missing = {}
        for source in range(self.number_drones):
            manifest = self.manifests.get(source)
            snapshot = self.snapshots.get(source)
            reason = None
            if self.epoch is None:
                reason = 'own_cutoff_not_reached'
            elif manifest is None:
                reason = 'missing_final_manifest'
            elif manifest[1] != self.epoch:
                reason = 'different_cutoff_epoch'
            elif snapshot is None:
                reason = 'missing_payload'
            elif snapshot[1] != manifest[2]:
                reason = 'final_version_not_received'
            elif self.payload_digest(source) != manifest[3]:
                reason = 'final_digest_mismatch'
            elif manifest[4] != 0:
                reason = 'inventory_remaining_at_cutoff'
            if reason:
                missing[str(source)] = reason
        return dict(final_complete=not missing, epoch_elapsed=self.epoch,
            missing=missing, required_sources=list(range(self.number_drones)),
            manifests={str(i): list(m) for i, m in self.manifests.items()},
            received_versions={str(i): s[1] for i, s in self.snapshots.items()},
            received_digests={str(i): self.payload_digest(i) for i in self.snapshots},
            protocol_stats=dict(self.protocol_stats))


class MyDroneRedFusion27(MyDroneRedRepair17):
    GATHER_AT = .65
    FREEZE_AT = .86
    DEADLINE_AT = .975

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submission_mapper = FinalizableRepairMapper17(self.size_area)
        self._radio_maps = FinalSnapshots27(self._map.shape, self._n, self._submission_mapper,
                                            lambda: int(self.elapsed_timestep))
        self._fusion27 = Fusion27(self.size_area)
        self._f27_stats = Counter()
        self._f27_delivery = {}
        self._f27_submission = None
        self._f27_home = None

    def define_message_for_all(self):
        base = super().define_message_for_all()
        store = self._radio_maps
        extra = None
        if store.snapshots:
            sources = sorted(store.snapshots)
            own = int(self.identifier or 0)
            source = own if self._steps % 2 == 0 and own in store.snapshots else sources[(self._steps // 2) % len(sources)]
            extra = store.snapshots[source]
        return ('fusion27-v1', base, extra, tuple(store.manifests[i] for i in sorted(store.manifests)), self._f27_home)

    def _receive(self):
        stripped = []
        for sender, message in self.communicator.received_messages:
            if not isinstance(message, tuple) or len(message) != 5 or message[0] != 'fusion27-v1':
                continue
            _, base, extra, manifests, home = message
            stripped.append((sender, base))
            if extra is not None:
                self._radio_maps.accept(extra)
            if isinstance(manifests, tuple) and len(manifests) <= self._n:
                for manifest in manifests:
                    self._radio_maps.accept_manifest(manifest)
            if self._f27_home is None and isinstance(home, tuple) and len(home) == 3:
                source, step, position = home
                if (source == 0 and type(source) is int and type(step) is int and
                        0 <= step <= self.elapsed_timestep and isinstance(position, tuple) and
                        len(position) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in position)):
                    self._f27_home = home
        proxy = SimpleNamespace(_n=self._n, elapsed_timestep=self.elapsed_timestep,
            _status=self._status, _peers=self._peers, _radio_maps=self._radio_maps,
            _explore=self._explore, _valid_peer=self._valid_peer,
            _bomb_claims=self._bomb_claims, _bomb_records=self._bomb_records,
            communicator=SimpleNamespace(received_messages=stripped))
        _M08_MyDroneRedV4._receive(proxy)

    def _update_observation(self):
        super()._update_observation()
        own = int(self.identifier or 0)
        if own == 0 and self._f27_home is None and self._home is not None:
            self._f27_home = (0, int(self.elapsed_timestep), tuple(map(float, self._home)))
        if self._progress() >= self.FREEZE_AT and self._radio_maps.own_final is None:
            self._radio_maps.freeze(own, self._steps, int(self.elapsed_timestep),
                                   self._map.mask(), self.carried_bombs_count())
            self._f27_stats['cutoff_elapsed'] = int(self.elapsed_timestep)
            self._f27_stats['cutoff_control'] = self._steps
            self._f27_stats['cutoff_walltime'] = float(self.elapsed_walltime)
            self._event('f27_finalized', epoch=int(self.elapsed_timestep), sequence=self._steps,
                        inventory=int(self.carried_bombs_count()))

    def _navigation(self):
        if self._progress() >= self.GATHER_AT:
            if not self._gathering:
                self._gathering = True
                self._route = []
                self._f27_stats['gather_started_control'] = self._steps
                self._event('f27_gather_started')
            if self._f27_home is not None:
                self._home = np.asarray(self._f27_home[2], dtype=float)
        return super()._navigation()

    def _try_hold(self, distances, dynamic, command):
        if self._position is None or self._home is None:
            return False
        peers = [p[1] for i, p in self._peers.items()
                 if i != int(self.identifier or 0) and self._steps - p[0] < 20]
        separation = min((np.linalg.norm(self._position - p) for p in peers), default=float('inf'))
        static = distances[np.isfinite(distances) & (distances > 0) & ~dynamic]
        clearance = float(np.min(static)) if len(static) >= 12 else 0.
        home_distance = float(np.linalg.norm(self._position - self._home))
        eligible = self._gathering and self.carried_bombs_count() == 0
        if self._holding and (not eligible or home_distance > 85 or separation < 40 or clearance < 23):
            self._holding = False
        if not self._holding and eligible and home_distance < 70 and separation > 50 and clearance > 38:
            self._holding = True
        if not self._holding:
            return False
        self._route = []
        self._bomb_target = None
        self._slow_ticks = 0
        self._motion_anchor = self._position.copy()
        if self._steps % 20 == 0:
            self._radio_maps.publish(int(self.identifier or 0), self._steps, self._map.mask())
        move = np.clip(-.7 * self._velocity, -1, 1)
        command.update(forward=float(move[0]), lateral=float(move[1]), rotation=0.)
        return True

    def _submit(self):
        if self._exploration_submitted:
            return
        state = self._radio_maps.delivery()
        self._f27_delivery = state
        own = int(self.identifier or 0)
        deadline = self.DEADLINE_AT if own == 0 else min(.995, .980 + .003 * own)
        complete = state['final_complete']
        if not ((own == 0 and complete) or self._progress() >= deadline):
            self._f27_stats['parent_submit_deferred'] += 1
            return
        pixels = self._fusion27.fuse(self._radio_maps.mapping_masks)
        self._f27_submission = dict(state, submitter=own, submit_control_step=self._steps,
            submit_elapsed_step=int(self.elapsed_timestep), submit_walltime=float(self.elapsed_walltime),
            reason='all_final_maps_received' if complete else 'deadline_incomplete_fallback',
            exploration_cutoff_policy=self.FREEZE_AT, gathering_policy=self.GATHER_AT,
            fusion_stats=dict(self._fusion27.last_stats),
            submitted_pixels_sha256=hashlib.sha256(np.asarray(pixels, dtype=bool).tobytes()).hexdigest(),
            scope='Final maps at the explicit exploration cutoff, not post-cutoff navigation scans')
        self.submit_exploration_map(np.asarray(pixels, dtype=np.uint8))
        self._exploration_submitted = True
        self._event('f27_submitted', complete=complete, missing=state['missing'],
                    sources=state['received_versions'], fusion=dict(self._fusion27.last_stats))

    def control(self):
        command = super().control()
        self._f27_delivery = self._radio_maps.delivery()
        if self._radio_maps.own_final is not None and not self._exploration_submitted:
            self._submit()
        return command
