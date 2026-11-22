"""Mesh36: participant-only, radio-carried final-map delivery.

Retains the observed Boundary33 renderer and 86% collection cutoff. All peer
state comes from actual messages. No map names, seeds, simulator peer objects,
truth geometry, shared memory, or engine modifications are used.
"""
from collections import Counter, deque
from types import SimpleNamespace
import hashlib
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_rendezvous35 import MyDroneRedRendezvous35
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.my_drone_red_advanced2 import _M06_MyDroneRedV3
from swarm_rescue.solutions.my_drone_red_rn7 import rotate
from swarm_rescue.solutions.red_rendezvous35 import RendezvousNavigator35
from swarm_rescue.solutions.red_egress36 import Egress36
from swarm_rescue.solutions.red_mesh36 import MeshBook36


class MyDroneRedMesh36(MyDroneRedRendezvous35):
    """Exchange through fresh multi-hop links; repair before waiting for ACKs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._m36_init()

    def _m36_init(self):
        self._m36_book = MeshBook36(int(self.identifier or 0), self._n,
                                   edge_ttl=18, report_ttl=90)
        self._m36_egress = Egress36(int(self.identifier or 0))
        self._m36_final_pixels = None
        self._m36_render_key = None
        self._m36_stats = Counter()
        self._m36_events = deque(maxlen=100)
        self._m36_previous_direct = set()
        self._m36_lost = {}
        self._m36_contact_points = {}
        self._m36_repair = None
        self._m36_repair_cooldown = {}
        self._m36_target_cache = None
        self._m36_target_tick = -1000
        self._m36_detour = None
        self._m36_failed_points = deque(maxlen=12)
        self._m36_route_kind = 'none'
        self._m36_nav_target = None
        self._m36_nav_pause = -1
        self._m36_exchange_until = -1
        self._m36_last_versions = None
        self._m36_delivery_due = False
        self._m36_phase = 'explore'
        self._m36_connected_since = None
        self._m36_last_state = {}
        self._m36_egress_previous = 'idle'

    def _m36_event(self, reason, **fields):
        self._m36_stats[reason] += 1
        self._m36_events.append(dict(control_step=int(self._steps), event=reason, **fields))
        self._event('mesh36_'+reason, **fields)

    def _exchange_state(self):
        # During initialization the parent's constructor has no Mesh36 state.
        if not hasattr(self, '_m36_book'):
            return super()._exchange_state()
        state = self._radio_maps.delivery()
        payload_complete = state['final_complete']
        signature = None
        if payload_complete:
            manifest = tuple((i, state['received_versions'][str(i)], state['received_digests'][str(i)])
                             for i in range(self._n))
            signature = hashlib.sha256(repr(manifest).encode('ascii')).hexdigest()
            key = (state['epoch_elapsed'], manifest)
            if key != self._m36_render_key:
                # Stable source order and the unchanged observed-only renderer.
                masks = {i: self._radio_maps.mapping_masks[i] for i in sorted(self._radio_maps.mapping_masks)}
                pixels = self._fusion27.fuse(masks)
                ack = self._m36_book.confirm_render(self._radio_maps, pixels, int(self.elapsed_timestep))
                if ack is not None:
                    self._m36_final_pixels = self._m36_book.final_pixels
                    self._m36_render_key = key
                    self._m36_event('final_render_verified', pixel_sha256=hashlib.sha256(
                        np.asarray(self._m36_final_pixels, dtype=bool).tobytes()).hexdigest())
            # Preserve old audit compatibility, but submission additionally
            # requires all peers' matching ACTUAL rendered-map acknowledgments.
            own = int(self.identifier or 0)
            self._s28_receipts[own] = (own, state['epoch_elapsed'], signature)
        mesh = self._m36_book.status(int(self.elapsed_timestep), self._radio_maps)
        acknowledged = mesh.get('render_ack_sources', [])
        state.update(source_maps_complete=payload_complete, final_set_sha256=signature,
                     acknowledged_sources=list(acknowledged),
                     missing_acknowledgements=[i for i in range(self._n) if i not in acknowledged],
                     final_complete=bool(payload_complete and mesh.get('render_complete', False)),
                     rendered_map_sha256=mesh.get('render_hash'), mesh36=mesh)
        self._m36_last_state = mesh
        return state

    def _receive(self):
        inner = []
        now = int(self.elapsed_timestep)
        for sender, packet in self.communicator.received_messages:
            if not (isinstance(packet, tuple) and len(packet) == 5 and packet[0] == 'mesh36-v1'):
                continue
            _, origin, issued, original, mesh = packet
            if (type(origin) is not int or not 0 <= origin < self._n
                    or origin == int(self.identifier or 0) or type(issued) is not int
                    or not 0 <= now-issued <= 3):
                continue
            self._m36_book.ingest(origin, mesh, now, self._radio_maps)
            inner.append((sender, original))
        # Adapt only OUR received data, never replace a simulator communicator.
        names = ('_n', '_steps', 'elapsed_timestep', '_status', '_peers', '_radio_maps',
                 '_explore', '_valid_peer', '_bomb_claims', '_bomb_records', '_f27_home',
                 '_s28_stats', '_s28_receipts', '_f34_reports', '_f34_direct',
                 '_d32_stats', '_f34_upload_acks', '_f34_event', 'identifier',
                 '_r35_book', '_r35_receipts', '_r35_last_rx', '_r35_record_receipts')
        proxy = SimpleNamespace(**{name: getattr(self, name) for name in names},
                                communicator=SimpleNamespace(received_messages=inner))
        MyDroneRedRendezvous35._receive(proxy)
        self._f27_home = proxy._f27_home
        self._m36_track_links()

    def define_message_for_all(self):
        self._exchange_state()
        original = super().define_message_for_all()
        # The Mesh36 protocol owns the ONE extra source-map payload slot.
        frontier = original[1]
        structural = frontier[1]
        fusion = structural[1]
        red4, red3 = fusion[1], fusion[1][1]
        red1 = red3[1]
        red1 = (red1[0], red1[1], None)
        red3 = (red3[0], red1, red3[2], red3[3])
        red4 = (red4[0], red3, red4[2], red4[3])
        fusion = (fusion[0], red4, None, fusion[3], fusion[4])
        structural = (structural[0], fusion, structural[2])
        frontier = (frontier[0], structural, frontier[2])
        original = (original[0], frontier, original[2], original[3])
        return ('mesh36-v1', int(self.identifier or 0), int(self.elapsed_timestep), original,
                self._m36_book.packet(int(self.elapsed_timestep), self._radio_maps))

    def _m36_track_links(self):
        # A previously heard peer is only a short-lived REPAIR destination;
        # it is never asserted to remain connected after messages disappear.
        direct = {i for i, step in self._r35_last_rx.items() if self._steps-step <= 8}
        for i in direct:
            report = self._f34_reports.get(i)
            if report is not None and self._steps-report[1] <= 10:
                self._m36_contact_points[i] = (self._steps, np.asarray(report[2], float).copy())
            if i in self._m36_lost:
                self._m36_event('direct_link_restored', peer=i)
                self._m36_lost.pop(i, None)
        for i in self._m36_previous_direct-direct:
            contact = self._m36_contact_points.get(i)
            if contact is not None:
                self._m36_lost[i] = contact
                self._m36_event('direct_link_lost', peer=i, observed_control_step=contact[0])
        self._m36_previous_direct = direct

    def _r35_target(self):
        # The shared fallback changes only on the agreed clock, not on a local
        # failure unknown to disconnected teammates. Local repairs are separate.
        book = self._r35_book
        if book.commit is None:
            return self._f34_home()
        fallback = self._progress() >= book.proposal[6]
        if fallback != self._r35_fallback:
            self._r35_fallback = fallback
            self._m36_target_tick = -1000
            self._m36_nav_target = None
            self._f34_budget_tick = -1000
            self._m36_event('common_fallback_window', active=fallback)
        return np.asarray(book.proposal[4] if fallback else book.proposal[3], float)

    def _m36_install_route(self, target):
        if self._m36_route_kind == 'own_trace_fresh_hull':
            # A reset must never resurrect already consumed return waypoints.
            self._f34_return_points = self._f34_nav.remaining_observed_route()
        self._f34_nav.reset()
        route = self._f34_nav.estimate_route(self._explore, self._position, target,
                                            speed=self.RETURN_SPEED34, reserve_steps=0)
        if route['exact']:
            # Known-grid waypoints still require fresh physical hull and
            # stopping-sweep checks on every movement; no occupancy is edited.
            self._m36_route_kind = 'known_grid_fresh_hull'
        else:
            self._f34_nav.set_observed_route(self._f34_return_points)
            self._m36_route_kind = 'own_trace_fresh_hull'
        self._m36_nav_target = np.asarray(target, float).copy()
        self._m36_event('delivery_route_installed', route_kind=self._m36_route_kind,
                        target=list(map(float, target)), route_exact=bool(route['exact']))

    def _f34_start_return(self, reason):
        was_returning = self._f34_mode in ('return', 'exchange')
        super()._f34_start_return(reason)
        if not was_returning:
            self._m36_install_route(self._r35_target())

    def _f34_budget_update(self):
        if self._position is None or self._home is None:
            return
        if self._steps-self._f34_budget_tick < 30 and self._f34_budget:
            return
        self._f34_budget_tick = self._steps
        target = self._r35_target()
        route = self._f34_nav.estimate_route(self._explore, self._position, target,
                                            speed=self.RETURN_SPEED34, reserve_steps=0)
        trace = [self._position, *reversed(self._f34_crumbs), target]
        trace_length = sum(float(np.linalg.norm(b-a)) for a, b in zip(trace, trace[1:]))
        travel = (float(route['steps']) if route['exact'] else
                  trace_length/self.RETURN_SPEED34+6*RendezvousNavigator35._turns35(trace))
        inventory = int(self.carried_bombs_count())
        # Include unfinished placement and its safe exit BEFORE releasing more
        # exploration. These are reserves, not a claim of a guaranteed route.
        deployment = inventory*180.
        active_egress = self._m36_egress.status()
        egress = inventory*90.+(90. if active_egress['active'] else 0.)
        onward = 0.
        if self._r35_book.commit is not None and not self._r35_fallback:
            fallback = np.asarray(self._r35_book.proposal[4], float)
            leg = self._f34_nav.estimate_route(self._explore, target, fallback,
                                              speed=self.RETURN_SPEED34, reserve_steps=0)
            onward = (float(leg['steps']) if leg['exact'] else
                      float(np.linalg.norm(fallback-target))/self.RETURN_SPEED34+240.)
        exchange, render_confirm, recovery = 150., 150., 180.
        remaining, timing = self._f34_remaining(.965)
        arrival_left, arrival_timing = self._f34_remaining(.88)
        required = travel+onward+deployment+egress+exchange+render_confirm+recovery
        self._f34_budget = dict(route_exact=bool(route['exact']), travel_steps=travel,
            traced_length_px=trace_length, deployment_reserve_steps=deployment,
            egress_reserve_steps=egress, exchange_steps=exchange,
            agreed_fallback_reserve_steps=onward,
            confirmation_steps=render_confirm, recovery_steps=recovery,
            required_steps=required, remaining_steps=remaining,
            arrival_remaining_steps=arrival_left,
            arrival_required_steps=travel+deployment+egress+recovery,
            target=target.tolist(), arrival_timing=arrival_timing, **timing)
        self._m36_delivery_due = (remaining <= required or
                                 arrival_left <= self._f34_budget['arrival_required_steps'])

    def _m36_choose_repair(self, component):
        own = int(self.identifier or 0)
        if self._m36_repair is not None:
            peer, point, until = self._m36_repair
            if (peer in self._m36_previous_direct or self._steps >= until
                    or self._steps-self._m36_contact_points.get(peer, (-10000, None))[0] > 150):
                self._m36_event('link_repair_ended', peer=peer,
                                restored=peer in self._m36_previous_direct)
                self._m36_repair_cooldown[peer] = self._steps+150
                self._m36_repair = None
            else:
                return point.copy()
        # One local task-complete member approaches the last ACTUAL contact;
        # the rest continue towards the jointly agreed point. No old global
        # position report can initiate a long search.
        for peer, (seen, point) in sorted(self._m36_lost.items()):
            if (peer in component or self._steps-seen > 90
                    or self._m36_repair_cooldown.get(peer, -1) > self._steps):
                continue
            contenders = []
            for member in component:
                row = self._f34_reports.get(member)
                if row is None or self._steps-row[1] > 30 or row[8] != 0:
                    continue
                contenders.append((float(np.linalg.norm(np.asarray(row[2])-point)), member))
            if contenders and min(contenders)[1] != own:
                continue
            if np.linalg.norm(point-self._position) > 200:
                continue
            route = self._f34_nav.estimate_route(self._explore, self._position, point,
                                                speed=self.RETURN_SPEED34, reserve_steps=0)
            if not route['exact'] or route['steps']+90 > self._f34_budget['remaining_steps']:
                continue
            self._m36_repair = (peer, point.copy(), self._steps+90)
            self._m36_event('link_repair_started', peer=peer, contact_age=self._steps-seen,
                            target=point.tolist(), expires_control_step=self._steps+90)
            return point.copy()
        return None

    def _m36_fresh_detour(self, target):
        fine = self._f34_nav._last_local
        if fine is None or self._f34_nav.tick != self._steps-1:
            return None
        candidates = []
        for distance in (40., 65.):
            for angle in np.linspace(-math.pi, math.pi, 24, endpoint=False):
                body = distance*np.array([math.cos(angle), math.sin(angle)])
                if not fine.line_clear(np.zeros(2), body):
                    continue
                point = self._position+rotate(body, self._heading)
                if any(np.linalg.norm(point-old) < 30 for old in self._m36_failed_points):
                    continue
                candidates.append((float(np.linalg.norm(point-target)), float(angle), point))
        return None if not candidates else min(candidates, key=lambda x: x[:2])[2]

    def _f34_return_motion(self):
        mesh = self._m36_book.status(int(self.elapsed_timestep), self._radio_maps)
        component = set(mesh.get('component', [int(self.identifier or 0)]))
        connected = bool(mesh.get('all_bidirectionally_connected', False))
        if connected:
            if self._m36_connected_since is None:
                self._m36_connected_since = self._steps
                self._m36_event('whole_team_connected')
            self._f34_mode, self._m36_phase = 'exchange', 'whole_team_exchange'
            self._m36_stats['connected_exchange_ticks'] += 1
            return self._r35_brake()
        if self._m36_connected_since is not None:
            self._m36_event('whole_team_link_broken')
            self._m36_connected_since = None
        self._f34_mode = 'return'
        self._m36_phase = 'rejoin_mesh'
        target = self._r35_target()
        repair = self._m36_choose_repair(component)
        if repair is not None:
            target, self._m36_phase = repair, 'repair_recent_direct_link'
        # An actual payload improvement permits a SHORT local exchange, never
        # an unlimited hold merely because one neighbour is present.
        versions = tuple(self._radio_maps.snapshots[i][1] if i in self._radio_maps.snapshots else -1
                         for i in range(self._n))
        if versions != self._m36_last_versions:
            previous = self._m36_last_versions
            self._m36_last_versions = versions
            if (previous is not None and self._radio_maps.own_final is not None
                    and any(a > b for a, b in zip(versions, previous))):
                self._m36_exchange_until = self._steps+12
        if self._steps < self._m36_exchange_until and self._m36_previous_direct:
            self._m36_phase = 'bounded_payload_exchange'
            return self._r35_brake()
        if self._m36_detour is not None:
            point, until = self._m36_detour
            if self._steps >= until or np.linalg.norm(point-self._position) < 18:
                self._m36_detour = None
                self._m36_nav_target = None
            else:
                target, self._m36_phase = point.copy(), 'fresh_observed_detour'
        if self._f34_nav.failed:
            self._r35_failures += 1
            alternative = self._m36_fresh_detour(target)
            self._m36_failed_points.append(np.asarray(target).copy())
            if alternative is not None:
                self._m36_detour = (alternative.copy(), self._steps+90)
                target = alternative
                self._m36_event('blocked_route_changed', target=target.tolist())
            else:
                self._m36_nav_pause = self._steps+12
                self._m36_event('blocked_route_reobserve', failure=self._f34_nav.failure_reason)
            self._m36_nav_target = None
        if self._m36_nav_target is None or np.linalg.norm(target-self._m36_nav_target) > 30:
            self._m36_install_route(target)
        if self._steps < self._m36_nav_pause:
            return self._r35_brake()
        command = self._f34_nav.command(self._position, self._heading,
            self._velocity if self._r35_velocity_valid else None,
            self.lidar_values(), self.lidar_rays_angles(), self.semantic_values(),
            self._explore, target, self._steps)
        self._m36_stats['return_executor_calls'] += 1
        return command

    def _m36_sensor_args(self):
        return (self._position, self._heading, self.odometer_values(), self.lidar_values(),
                self.lidar_rays_angles(), self.semantic_values(), int(self.carried_bombs_count()),
                int(self._steps))

    def _navigation(self):
        egress_command = self._m36_egress.update(*self._m36_sensor_args())
        self._f34_policy()
        if egress_command is not None:
            self._r35_refresh_delivery_observation()
            command = dict(self._empty_command(), **egress_command)
            self._m36_phase = 'bomb_egress'
        elif (self._f34_mode in ('return', 'exchange') and self._position is not None
              and self.carried_bombs_count() == 0 and self._attempt_step is None):
            self._r35_refresh_delivery_observation()
            command = dict(self._empty_command(), **self._f34_return_motion())
        else:
            command = MyDroneRedRepair17._navigation(self)
            self._m36_phase = self._f34_mode
        self._f34_report()
        self._d32_stats[self._f34_mode+'_ticks'] += 1
        self._m36_snapshot_state()
        return command

    def _deployment(self, command):
        before = int(self.carried_bombs_count())
        attempts_before = (self._attempts, self._total_attempts)
        confirming = (self._attempt_step is not None or
                      (self._last_inventory is not None and before < self._last_inventory))
        if self._m36_egress.command() is not None and not confirming:
            command['place_bomb'] = 0
            return
        if self._m36_delivery_due and before > 0:
            # Stop spending an unbounded detour on a high-scoring bomb target.
            # The original local placement restrictions and native inventory
            # confirmation remain, and the pre-drop exit must also be proved.
            _M06_MyDroneRedV3._deployment(self, command)
        else:
            super()._deployment(command)
        if command.get('place_bomb', 0):
            if self._m36_egress.prepare(*self._m36_sensor_args()):
                command.update(self._r35_brake())
                command['rotation'] = 0.
                self._m36_event('drop_exit_prevalidated', inventory=before)
            else:
                command['place_bomb'] = 0
                # Preflight refusal never reached the native bomb actuator.
                # It must not consume the real deployment retry allowance.
                self._attempts, self._total_attempts = attempts_before
                self._attempt_step = None
                self._attempt_position = None
                self._next_attempt = self._steps+12
                self._bomb_target = None
                self._route = []
                self._m36_event('drop_deferred_no_exit', inventory=before)

    def _m36_snapshot_state(self):
        mesh = self._m36_book.status(int(self.elapsed_timestep), self._radio_maps)
        self._s28_return = dict(policy='mesh36_radio_multihop_final_render_confirmation',
            mode=self._f34_mode, phase=self._m36_phase, budget=dict(self._f34_budget),
            return_reason=self._f34_return_reason, rendezvous=self._r35_book.status(),
            last_direct_rx_steps=dict(self._r35_last_rx),
            locally_verified_payload_receipts=list(self._r35_local_receipts.values()),
            reported_payload_receipts=list(self._r35_receipts.values()),
            fallback_active=self._r35_fallback, blocked_task_exits=self._r35_failures,
            navigator=self._f34_nav.status(), stats=dict(self._d32_stats),
            reports={str(i): list(r) for i, r in self._f34_reports.items()},
            mesh36=dict(mesh, phase=self._m36_phase, egress=self._m36_egress.status(),
                route_kind=self._m36_route_kind, statistics=dict(self._m36_stats),
                events=list(self._m36_events)))

    def _submit(self):
        if self._exploration_submitted:
            return
        state = self._exchange_state()
        self._s28_delivery = self._f27_delivery = state
        complete = state['final_complete']
        own = int(self.identifier or 0)
        winner = (min(self._m36_book.acks.values(), key=lambda ack: (ack[4], ack[0]))[0]
                  if complete else None)
        deadline = self.DEADLINE_AT if own == 0 else min(.995, .980+.003*own)
        if not (complete and own == winner) and self._progress() < deadline:
            self._s28_stats['submission_deferred'] += 1
            return
        # The five matching ACKs deterministically select their earliest local
        # renderer (ID breaks ties). This may be a nonzero drone and avoids
        # several native submissions during the same simulator control round.
        pixels = (self._m36_final_pixels if self._m36_final_pixels is not None else
                  self._fusion27.fuse({i: self._radio_maps.mapping_masks[i]
                                      for i in sorted(self._radio_maps.mapping_masks)}))
        self._m36_snapshot_state()
        self._s28_submission = self._f27_submission = dict(state, submitter=own,
            submit_control_step=int(self._steps), submit_elapsed_step=int(self.elapsed_timestep),
            render_selected_submitter=winner,
            submit_walltime=float(self.elapsed_walltime),
            reason='all_sources_and_actual_render_acknowledged' if complete else 'deadline_incomplete_fallback',
            exploration_cutoff_policy=self.FREEZE_AT,
            gathering_policy='remaining_task_budget; actual_radio_links; agreed_observed_point',
            navigation_policy='mesh36_safe_bomb_egress_and_fresh_link_repair',
            submission_policy='unchanged_observed_boundary33',
            fusion_stats=dict(self._fusion27.last_stats),
            submitted_pixels_sha256=hashlib.sha256(np.asarray(pixels, dtype=bool).tobytes()).hexdigest(),
            mesh36=self._s28_return['mesh36'],
            scope='Five source maps at explicit collection cutoff; final rendered pixels checked by every source; not post-cutoff scans')
        self.submit_exploration_map(np.asarray(pixels, dtype=np.uint8))
        self._exploration_submitted = True
        self._m36_event('submitted', complete=complete, submitter=own,
                        acknowledgements=state['acknowledged_sources'], missing=state['missing'])
