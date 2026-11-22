"""Agreed, deadline-budgeted final-map exchange over actual radio packets.

Keep the Clean mapper, observation-derived outer boundary, native clock,
inventory confirmation, 86% seal and strict five-source final-set ACK barrier.
No map identity, seed, truth geometry or private peer state is consulted.
"""
from collections import deque
from types import SimpleNamespace
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_frontier34r1 import MyDroneRedFrontier34R1
from swarm_rescue.solutions.my_drone_red_frontier34 import MyDroneRedFrontier34
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.red_rendezvous35_protocol import RendezvousBook35
from swarm_rescue.solutions.red_rendezvous35 import RendezvousNavigator35


class MyDroneRedRendezvous35(MyDroneRedFrontier34R1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._r35_init()

    def _r35_init(self):
        self._r35_book = RendezvousBook35(int(self.identifier or 0), self._n)
        self._f34_nav = RendezvousNavigator35(int(self.identifier or 0))
        self._r35_events = deque(maxlen=120)
        self._r35_phase = 'explore'
        self._r35_fallback = False
        self._r35_primary_failed = False
        self._r35_retry_until = -1
        self._r35_failures = 0
        self._r35_local_receipts = {}
        self._r35_receipts = {}
        self._r35_last_rx = {}
        self._r35_packet_source = 0
        self._r35_plan_checked = -1000
        self._r35_had_commit = False
        self._r35_velocity_valid = True

    def _r35_event(self, name, **data):
        self._d32_stats['r35_'+name] += 1
        self._r35_events.append(dict(control_step=int(self._steps), event=name, **data))
        self._event('rendezvous35_'+name, **data)

    def _r35_record_receipts(self):
        state = self._radio_maps.delivery()
        own = int(self.identifier or 0)
        for source in range(self._n):
            if str(source) in state['missing']:
                continue
            snapshot = self._radio_maps.snapshots[source]
            digest = self._radio_maps.payload_digest(source)
            key = (own, source)
            if key in self._r35_local_receipts:
                continue
            receipt = (own, source, snapshot[1], digest, int(self.elapsed_timestep))
            self._r35_local_receipts[key] = receipt
            self._r35_receipts[key] = receipt
            self._r35_event('payload_verified_here', observer=own, origin=source,
                            version=snapshot[1], digest=digest,
                            receipt_stage='submitter_received' if own == 0 else 'peer_received')

    def _receive(self):
        inner = []
        for sender, packet in self.communicator.received_messages:
            if not isinstance(packet, tuple) or len(packet) != 4 or packet[0] != 'rendezvous35-v1':
                continue
            inner.append((sender, packet[1]))
            self._r35_book.ingest(packet[2], int(self.elapsed_timestep))
            receipts = packet[3]
            if isinstance(receipts, tuple) and len(receipts) <= self._n*self._n:
                for row in receipts:
                    if (isinstance(row, tuple) and len(row) == 5
                            and type(row[0]) is int and 0 <= row[0] < self._n
                            and type(row[1]) is int and 0 <= row[1] < self._n
                            and type(row[2]) is int and 0 <= row[2] <= self.elapsed_timestep
                            and isinstance(row[3], str) and len(row[3]) == 64
                            and all(v in '0123456789abcdef' for v in row[3])
                            and type(row[4]) is int and 0 <= row[4] <= self.elapsed_timestep
                            and row[0] != int(self.identifier or 0)):
                        self._r35_receipts.setdefault((row[0], row[1]), row)
        names = ('_n', '_steps', 'elapsed_timestep', '_status', '_peers', '_radio_maps',
                 '_explore', '_valid_peer', '_bomb_claims', '_bomb_records', '_f27_home',
                 '_s28_stats', '_s28_receipts', '_f34_reports', '_f34_direct',
                 '_d32_stats', '_f34_upload_acks', '_f34_event', 'identifier')
        proxy = SimpleNamespace(**{name: getattr(self, name) for name in names},
                                communicator=SimpleNamespace(received_messages=inner))
        MyDroneRedFrontier34._receive(proxy)
        self._f27_home = proxy._f27_home
        for source, last in self._f34_direct.items():
            if self._steps-last <= 3:
                self._r35_last_rx[source] = int(self._steps)
        self._r35_record_receipts()

    def define_message_for_all(self):
        self._r35_record_receipts()
        self._r35_book.advance(int(self.elapsed_timestep))
        packet = super().define_message_for_all()
        # Prefer actual final bytes missing at a recently heard direct peer.
        # Every third packet retains original round-robin to avoid starvation.
        if self._steps % 3 != 0:
            neighbors = [r for i, r in self._f34_reports.items()
                         if i != int(self.identifier or 0) and self._steps-r[1] <= 15
                         and self._steps-self._r35_last_rx.get(i, -10000) <= 8]
            needed = [i for i, snap in sorted(self._radio_maps.snapshots.items())
                      if any(r[4][i] < snap[1] for r in neighbors)]
            if needed:
                origin = needed[self._r35_packet_source % len(needed)]
                self._r35_packet_source += 1
                structural = packet[1]
                fusion = structural[1]
                revised = (fusion[0], fusion[1], self._radio_maps.snapshots[origin], fusion[3], fusion[4])
                structural = (structural[0], revised, structural[2])
                packet = (packet[0], structural, packet[2])
                self._d32_stats['r35_missing_bytes_forwarded_packets'] += 1
        return ('rendezvous35-v1', packet, self._r35_book.packet(),
                tuple(self._r35_receipts[k] for k in sorted(self._r35_receipts)))

    def _r35_agreement(self):
        if self._position is None or self._home is None:
            return
        book = self._r35_book
        if book.source == 0 and book.proposal is None and self._steps >= 60:
            # This is the proposer's observed position at proposal time, not
            # an arena centre or a teammate's old reported location.
            meeting = np.asarray(self._position, float).copy()
            fallback = self._f34_home().copy()
            if book.propose(int(self.elapsed_timestep), meeting, fallback):
                self._r35_event('plan_proposed', meeting=meeting.tolist(), fallback=fallback.tolist())
        if book.proposal is not None and not hasattr(book, '_local_accept') and self._steps-self._r35_plan_checked >= 30:
            self._r35_plan_checked = self._steps
            routes = [self._f34_nav.estimate_route(self._explore, self._position, p,
                       speed=self.RETURN_SPEED34, reserve_steps=0) for p in book.proposal[3:5]]
            if book.accept(int(self.elapsed_timestep), all(r['exact'] for r in routes)):
                self._r35_event('plan_accepted', plan_id=book.proposal[8], route_exact=[r['exact'] for r in routes])
        book.advance(int(self.elapsed_timestep))
        if book.commit is not None and not self._r35_had_commit:
            self._r35_had_commit = True
            self._f34_budget_tick = -1000
            self._r35_event('plan_committed_here', plan_id=book.proposal[8],
                            accepted_sources=sorted(book.accepts))

    def _r35_target(self):
        book = self._r35_book
        if book.commit is None:
            return self._f34_home()
        p = book.proposal
        use_fallback = self._r35_primary_failed or self._progress() >= p[6]
        if use_fallback and not self._r35_fallback:
            self._r35_fallback = True
            self._f34_return_points = self._f34_nav.remaining_observed_route()
            self._f34_nav.reset()
            self._f34_nav.set_observed_route(self._f34_return_points)
            self._f34_budget_tick = -1000
            self._r35_event('agreed_fallback_activated',
                            trigger='blocked_primary' if self._r35_primary_failed else 'common_window_expired')
        return np.asarray(p[4] if self._r35_fallback else p[3], float)

    def _f34_budget_update(self):
        if self._position is None or self._home is None:
            return
        if self._steps-self._f34_budget_tick < 30 and self._f34_budget:
            return
        self._f34_budget_tick = self._steps
        target = self._r35_target()
        estimate = self._f34_nav.estimate_route(self._explore, self._position, target,
                                               speed=self.RETURN_SPEED34, reserve_steps=0)
        remaining_trace = (self._f34_nav.remaining_observed_route()
                           if self._f34_mode in ('return', 'exchange')
                           else list(reversed(self._f34_crumbs)))
        trace = [self._position, *remaining_trace]
        traced = sum(float(np.linalg.norm(np.asarray(b)-a)) for a, b in zip(trace, trace[1:]))
        # This navigator executes its own recorded itinerary first. Budget
        # that itinerary (including turns) even if a shorter coarse route is
        # known; a geometric shortcut is not the route actually installed.
        turns = self._f34_nav._turns35(trace)
        terminal_start = np.asarray(trace[-1], float)
        terminal = self._f34_nav.estimate_route(self._explore, terminal_start, target,
                                               speed=self.RETURN_SPEED34, reserve_steps=0)
        terminal_steps = (terminal['steps'] if terminal['exact']
                          else float(np.linalg.norm(terminal_start-target))/self.RETURN_SPEED34+240.)
        trace_steps = traced/self.RETURN_SPEED34+6*turns+terminal_steps
        travel = max(float(estimate['steps'] or 0.), trace_steps)
        p = self._r35_book.proposal if self._r35_book.commit is not None else None
        onward = 0.
        onward_exact = True
        if p is not None and not self._r35_fallback:
            route = self._f34_nav.estimate_route(self._explore, target, p[4],
                                                speed=self.RETURN_SPEED34, reserve_steps=0)
            onward_exact = bool(route['exact'])
            onward = route['steps'] if onward_exact else traced/self.RETURN_SPEED34+240.
        arrival = p[5] if p is not None else .88
        finish = p[7] if p is not None else .965
        remaining, timing = self._f34_remaining(finish)
        arrival_remaining, arrival_timing = self._f34_remaining(arrival)
        # All four stages are charged, even when a prior hop has some data.
        exchange, confirm, recovery = 120., 120., 180.
        self._f34_budget = dict(route_exact=bool(estimate['exact']), route_mode=estimate['mode'],
            travel_steps=float(travel), exchange_steps=exchange, return_to_fallback_steps=float(onward),
            return_to_fallback_exact=onward_exact, confirmation_steps=confirm, recovery_steps=recovery,
            required_steps=float(travel+exchange+onward+confirm+recovery),
            remaining_steps=float(remaining), arrival_remaining_steps=float(arrival_remaining),
            arrival_required_steps=float(travel+recovery), traced_length_px=traced,
            executed_route_policy='own_recorded_route_with_currently_visible_shortcuts',
            traced_turn_count=turns, traced_route_steps=float(trace_steps),
            terminal_route_exact=bool(terminal['exact']),
            target=target.tolist(), plan_committed=self._r35_book.commit is not None,
            arrival_timing=arrival_timing, **timing)

    def _f34_start_return(self, reason):
        previous = self._f34_mode
        super()._f34_start_return(reason)
        if previous not in ('return', 'exchange'):
            self._r35_phase = 'to_primary' if not self._r35_fallback else 'to_fallback'
            self._f34_nav.set_observed_route(self._f34_return_points)
            self._r35_event('delivery_preempted_search', trigger=reason, budget=dict(self._f34_budget))

    def _f34_policy(self):
        if self._position is None or self._home is None:
            return
        if self._steps % 30 == 0:
            self._f34_wall_samples.append((self._steps, float(self.elapsed_walltime)))
        if self._f34_mode not in ('return', 'exchange'):
            self._f34_trace()
        self._r35_agreement()
        self._f34_budget_update()
        empty = self.carried_bombs_count() == 0 and self._attempt_step is None
        budget = self._f34_budget
        overdue = (budget['remaining_steps'] <= budget['required_steps']
                   or budget['arrival_remaining_steps'] <= budget['arrival_required_steps'])
        if self._f34_mode not in ('return', 'exchange') and empty:
            if self._radio_maps.own_final is not None:
                self._f34_start_return('final_epoch_preemption')
            elif overdue:
                self._f34_start_return('meeting_exchange_return_confirmation_budget')
        if overdue and not empty:
            self._d32_stats['r35_delivery_waiting_inventory_confirmation'] += 1
        if self._f34_mode in ('return', 'exchange'):
            self._gathering = True
            return
        self._gathering = False
        # Preserve the prior unknown-frontier policy while there is budget.
        self._explore.radio_claims34 = [np.asarray(r[3], float) for i, r in self._f34_reports.items()
            if i != int(self.identifier or 0) and r[3] is not None and self._steps-r[1] <= 90
            and r[7] in ('explore', 'support')]
        if self._steps % 6 == 0:
            self._f34_samples.append((self._steps, int(self._explore.free.sum()), int(np.count_nonzero(self._explore.visits))))
        if len(self._f34_samples) >= 31 and empty:
            first, last = self._f34_samples[0], self._f34_samples[-1]
            self._f34_gain = dict(window_steps=last[0]-first[0], new_free=last[1]-first[1], new_visited=last[2]-first[2])
            if last[0]-first[0] >= 180 and last[1]-first[1] <= 3 and last[2]-first[2] <= 3:
                if self._f34_mode == 'explore':
                    self._f34_mode = 'support'
                    self._explore.support_mode34 = True
                    self._route = []
                    self._last_route_step = self._steps-12
                    self._r35_event('local_complete_support', gain=dict(self._f34_gain))
            elif self._f34_mode == 'support' and last[1]-first[1] >= 12:
                self._f34_mode = 'explore'
                self._explore.support_mode34 = False

    def _r35_brake(self):
        if not self._r35_velocity_valid:
            return dict(forward=0., lateral=0., rotation=0.)
        force = np.clip(-.7*np.asarray(self._velocity), -1., 1.)
        return dict(forward=float(force[0]), lateral=float(force[1]), rotation=0.)

    def _r35_refresh_delivery_observation(self):
        """Keep native sensing/publication when skipping cancelled planning."""
        odom = self.odometer_values()
        self._r35_velocity_valid = bool(odom is not None and np.shape(odom) == (3,) and np.isfinite(odom).all())
        if self._r35_velocity_valid:
            direction = odom[1]-odom[2]
            measured = odom[0]*np.array([math.cos(direction), math.sin(direction)])
            self._velocity = .4*self._velocity+.6*measured
        self._explore.step = self._steps
        values, angles = self.lidar_values(), self.lidar_rays_angles()
        if values is not None and angles is not None:
            distances, angles = np.asarray(values).ravel(), np.asarray(angles).ravel()
            dynamic = np.zeros(len(distances), bool)
            if len(distances) == len(angles):
                for obj in self.semantic_values() or []:
                    if getattr(obj.entity_type, 'name', None) in ('DRONE', 'BOMB'):
                        delta = np.abs((angles-obj.angle+math.pi) % (2*math.pi)-math.pi)
                        dynamic |= (delta < .19) & (np.abs(distances-obj.distance) < 25)
                if self._steps % 6 == 0:
                    self._explore.observe(self._position, self._heading, distances, angles, dynamic)
        if self._steps % 30 == 0:
            self._visited_snapshot = np.packbits(self._explore.shared_visited | (self._explore.visits > 0)).tobytes()
            self._radio_maps.publish(int(self.identifier or 0), self._steps, self._map.mask())

    def _f34_return_motion(self):
        target = self._r35_target()
        own = int(self.identifier or 0)
        peers = [i for i, last in self._r35_last_rx.items() if i != own and self._steps-last <= 8]
        close = float(np.linalg.norm(self._position-target)) <= 115.
        # Exchange is evidenced by actual messages, not geometric distance.
        # The submitter waits at its contract; peers can wait at distinct nearby
        # observed positions instead of converging on the same occupied pixel.
        if close and (own == 0 or peers):
            if self._f34_mode != 'exchange':
                self._r35_event('exchange_wait_started', direct_peers=peers, target=target.tolist())
            self._f34_mode = 'exchange'
            self._r35_phase = 'fallback_exchange' if self._r35_fallback else 'primary_exchange'
            self._d32_stats['r35_exchange_ticks'] += 1
            self._r35_record_receipts()
            return self._r35_brake()
        self._f34_mode = 'return'
        if self._steps < self._r35_retry_until:
            self._r35_phase = 'fresh_observation_pause'
            self._d32_stats['r35_rescan_pause_ticks'] += 1
            return self._r35_brake()
        if self._f34_nav.failed:
            self._r35_failures += 1
            self._r35_event('blocked_task_exited', navigator=self._f34_nav.status(),
                            failure_number=self._r35_failures, next_action='agreed_fallback_and_rescan')
            self._r35_primary_failed = True
            self._r35_retry_until = self._steps+20
            self._r35_phase = 'fresh_observation_pause'
            self._f34_return_points = self._f34_nav.remaining_observed_route()
            self._f34_nav.reset()
            self._f34_nav.set_observed_route(self._f34_return_points)
            self._r35_target()
            return self._r35_brake()
        self._r35_phase = 'to_fallback' if self._r35_fallback else 'to_primary'
        command = self._f34_nav.command(self._position, self._heading,
            self._velocity if self._r35_velocity_valid else None,
            self.lidar_values(), self.lidar_rays_angles(), self.semantic_values(),
            self._explore, target, self._steps)
        self._d32_stats['r35_return_motion_ticks'] += 1
        return command

    def _navigation(self):
        self._f34_policy()
        # Do not spend a second route search on an exploration mission which
        # has already been cancelled by the final-delivery deadline.
        delivering = (self._f34_mode in ('return', 'exchange') and self._position is not None
                      and self.carried_bombs_count() == 0 and self._attempt_step is None)
        if delivering:
            self._r35_refresh_delivery_observation()
            command = dict(self._empty_command(), **self._f34_return_motion())
        else:
            command = MyDroneRedRepair17._navigation(self)
        self._f34_report()
        self._d32_stats[self._f34_mode+'_ticks'] += 1
        self._s28_return = dict(policy='rendezvous35_agreed_deadline_delivery',
            mode=self._f34_mode, phase=self._r35_phase, budget=dict(self._f34_budget),
            return_reason=self._f34_return_reason, rendezvous=self._r35_book.status(),
            last_direct_rx_steps=dict(self._r35_last_rx),
            locally_verified_payload_receipts=list(self._r35_local_receipts.values()),
            reported_payload_receipts=list(self._r35_receipts.values()),
            fallback_active=self._r35_fallback, blocked_task_exits=self._r35_failures,
            navigator=self._f34_nav.status(), stats=dict(self._d32_stats),
            events=list(self._r35_events), reports={str(i): list(r) for i, r in self._f34_reports.items()})
        return command

    def _submit(self):
        super()._submit()
        if self._s28_submission is not None:
            self._s28_submission.update(navigation_policy='rendezvous35_agreed_deadline_delivery',
                gathering_policy='observed_contract; route_exchange_return_confirmation_budget',
                rendezvous35=self._s28_return)
