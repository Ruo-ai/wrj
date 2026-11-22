"""One radio-elected mobile courier plus observation-derived final boundaries.

No peer objects, map files, seed or true positions are available to this policy.
The original Clean exploration/deployment and final epoch remain unchanged;
only a granted, empty-inventory courier overrides its navigation command.
"""
from collections import Counter, deque
import math
from types import SimpleNamespace

import numpy as np

from swarm_rescue.solutions.my_drone_red_dualmap32 import MyDroneRedDualMap32Clean
from swarm_rescue.solutions.red_boundary33 import FinalBoundary33
from swarm_rescue.solutions.red_relay33 import RelayNavigator33


def valid_report33(row, number, clock):
    if not isinstance(row, tuple) or len(row) != 10:
        return False
    source, step, point, inventory, depth, health, versions, missing, acks, mode = row
    return (type(source) is int and 0 <= source < number and type(step) is int
            and 0 <= step <= clock and isinstance(point, tuple) and len(point) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)
            and type(inventory) is int and inventory >= 0
            and isinstance(depth, (int, float)) and math.isfinite(depth) and depth >= 0
            and isinstance(health, (int, float)) and math.isfinite(health) and health >= 0
            and isinstance(versions, tuple) and len(versions) == number
            and all(type(v) is int and -1 <= v <= clock for v in versions)
            and all(isinstance(ids, tuple) and len(ids) <= number
                    and all(type(i) is int and 0 <= i < number for i in ids)
                    for ids in (missing, acks))
            and mode in ('exploring', 'gathering', 'courier'))


def choose_courier33(reports, clock, center):
    """Coordinator chooses one fresh, task-complete, shallow candidate."""
    eligible = [row for row in reports.values()
                if 0 <= clock-row[1] <= 60 and row[3] == 0 and row[5] > 10]
    if not eligible:
        return None
    def cost(row):
        return (row[4] + .65*float(np.linalg.norm(np.asarray(row[2])-center)), row[0])
    return min(eligible, key=cost)[0]


class MyDroneRedRelay33(MyDroneRedDualMap32Clean):
    COURIER_AT = .55
    SEARCH_BUDGET = 240
    TARGET_HOLD = 18

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fusion27 = FinalBoundary33(self.size_area, wall_width=11)
        self._relay33_nav = RelayNavigator33(int(self.identifier or 0))
        self._relay33_reports = {}
        self._relay33_previous = {}
        self._relay33_grant = None
        self._relay33_depth = 0.
        self._relay33_direct = {}
        self._relay33_serviced = {}
        self._relay33_searches = Counter()
        self._relay33_task = None
        self._relay33_task_since = 0
        self._relay33_contact_since = None
        self._relay33_events = deque(maxlen=160)
        self._d32_stats = Counter()

    def _relay33_event(self, reason, **fields):
        self._d32_stats[reason] += 1
        self._relay33_events.append(dict(step=self._steps, reason=reason, **fields))
        self._event('relay33_'+reason, **fields)

    def _relay33_is_courier(self):
        return (self._relay33_grant is not None
                and self._relay33_grant[2] == int(self.identifier or 0)
                and self._progress() >= self.COURIER_AT
                and self.carried_bombs_count() == 0 and self.drone_health > 10
                and self._attempt_step is None and not self._exploration_submitted)

    def _relay33_refresh_report(self):
        if self._position is None or not np.isfinite(self._position).all():
            return
        own = int(self.identifier or 0)
        home = self._f27_home[2] if self._f27_home is not None else self._home
        if home is not None:
            self._relay33_depth = max(self._relay33_depth, float(np.linalg.norm(self._position-home)))
        state = self._exchange_state()
        versions = tuple(self._radio_maps.snapshots[i][1] if i in self._radio_maps.snapshots else -1
                         for i in range(self._n))
        mode = 'courier' if self._relay33_is_courier() else ('gathering' if self._gathering else 'exploring')
        row = (own, int(self._steps), tuple(map(float, self._position)),
               int(self.carried_bombs_count()), self._relay33_depth, float(self.drone_health),
               versions, tuple(sorted(map(int, state['missing']))),
               tuple(state['acknowledged_sources']), mode)
        self._relay33_reports[own] = row

    def _relay33_accept(self, metadata):
        if not isinstance(metadata, tuple) or len(metadata) != 2:
            self._d32_stats['invalid_metadata'] += 1
            return
        reports, grant = metadata
        if isinstance(reports, tuple) and len(reports) <= self._n:
            for row in reports:
                if not valid_report33(row, self._n, int(self.elapsed_timestep)):
                    self._d32_stats['invalid_report'] += 1
                    continue
                source = row[0]
                if source == int(self.identifier or 0):
                    continue
                old = self._relay33_reports.get(source)
                if old is None or row[1] > old[1]:
                    previous = self._relay33_previous.get(source)
                    if old is not None and (previous is None or old[1]-previous[1] >= 20):
                        self._relay33_previous[source] = old
                    self._relay33_reports[source] = row
        if (isinstance(grant, tuple) and len(grant) == 3 and type(grant[0]) is int
                and grant[0] == 0 and type(grant[1]) is int
                and 0 <= grant[1] <= self.elapsed_timestep
                and type(grant[2]) is int and 0 <= grant[2] < self._n):
            # The coordinator grants once. Relay forwarding cannot create a
            # new election, fresh timestamp or an independent partition leader.
            if self._relay33_grant is None:
                self._relay33_grant = grant
                self._relay33_event('grant_received', courier=grant[2], origin_step=grant[1])
            elif self._relay33_grant != grant:
                self._d32_stats['conflicting_grant_ignored'] += 1

    def _receive(self):
        messages = []
        for sender, message in self.communicator.received_messages:
            if not isinstance(message, tuple) or len(message) != 3 or message[0] != 'relay33-v1':
                continue
            messages.append((sender, message[1]))
            self._relay33_accept(message[2])
            # The original reports still populate _peers only for direct
            # communication; relayed remote poses never enter collision logic.
        proxy = SimpleNamespace(_n=self._n, elapsed_timestep=self.elapsed_timestep,
            _status=self._status, _peers=self._peers, _radio_maps=self._radio_maps,
            _explore=self._explore, _valid_peer=self._valid_peer,
            _bomb_claims=self._bomb_claims, _bomb_records=self._bomb_records,
            _f27_home=self._f27_home, _s28_stats=self._s28_stats,
            _s28_receipts=self._s28_receipts,
            communicator=SimpleNamespace(received_messages=messages))
        MyDroneRedDualMap32Clean._receive(proxy)
        self._f27_home = proxy._f27_home
        for source, peer in self._peers.items():
            if 0 <= self._steps-peer[0] <= 3:
                self._relay33_direct[source] = peer[0]

    def define_message_for_all(self):
        self._relay33_refresh_report()
        base = super().define_message_for_all()
        metadata = (tuple(self._relay33_reports[i] for i in sorted(self._relay33_reports)),
                    self._relay33_grant)
        return ('relay33-v1', base, metadata)

    def _relay33_elect(self):
        if (int(self.identifier or 0) != 0 or self._relay33_grant is not None
                or self._progress() < self.COURIER_AT or not self._relay33_reports):
            return
        # Centre of actual reported team positions, not a true-map centre.
        points = [row[2] for row in self._relay33_reports.values()]
        center = np.median(np.asarray(points, float), axis=0)
        # Grant only to the coordinator itself or a current direct neighbour
        # with radio margin. A fresh relayed report does not prove that the
        # remote drone can receive this new assignment.
        nearby = {i: row for i, row in self._relay33_reports.items()
                  if i == 0 or (self._steps-self._relay33_direct.get(i, -10000) <= 3
                      and np.linalg.norm(np.asarray(row[2])-self._position) < 150.)}
        selected = choose_courier33(nearby, self._steps, center)
        if selected is not None:
            self._relay33_grant = (0, int(self._steps), selected)
            self._relay33_event('grant_created', courier=selected, center=center.tolist())

    def _relay33_target_for(self, source):
        report = self._relay33_reports.get(source)
        if report is None:
            return None
        position = np.asarray(report[2], float).copy()
        age = max(0, self._steps-report[1])
        previous = self._relay33_previous.get(source)
        # Old positions define a bounded search region, not an asserted pose.
        # The first search follows the last observed travel direction; later
        # searches fan out around the original report. All motion is planned
        # on this courier's observed grid and checked with its fresh scan.
        if age > 180:
            radius = min(420., 90.+.15*age)
            phase = self._relay33_searches[source] % 5
            direction = None
            if previous is not None:
                delta = position-np.asarray(previous[2])
                if np.linalg.norm(delta) > 4:
                    direction = delta/np.linalg.norm(delta)
            if phase == 0 and direction is not None:
                position += radius*direction
            elif phase:
                angle = (phase-1)*math.pi/2
                position += radius*np.array([math.cos(angle), math.sin(angle)])
        return position

    def _relay33_pick_task(self):
        own = int(self.identifier or 0)
        state = self._exchange_state()
        versions = self._relay33_reports[own][6]
        leader = self._relay33_reports.get(0)
        candidates = []
        sealed = self._radio_maps.own_final is not None
        for source in range(self._n):
            if source == own or source not in self._relay33_reports:
                continue
            row = self._relay33_reports[source]
            since_service = self._steps-self._relay33_serviced.get(source, -10000)
            if since_service < 120:
                continue
            missing_final = str(source) in state['missing']
            need_ack = sealed and state['source_maps_complete'] and source not in state['acknowledged_sources']
            stale = max(0, self._steps-versions[source])
            deliver = (source == 0 and any(versions[i] > row[6][i]+(0 if sealed else 240)
                                          for i in range(self._n)))
            deliver_ack = source == 0 and bool(set(state['acknowledged_sources'])-set(row[8]))
            if sealed:
                priority = 1600 if deliver or deliver_ack else (1300 if need_ack else (1100 if missing_final else 0))
            else:
                priority = (800 if deliver else min(700, stale)) if stale > 180 or deliver else 0
            if priority <= 0:
                continue
            target = self._relay33_target_for(source)
            distance = float(np.linalg.norm(target-self._position))
            candidates.append((priority-.22*distance+min(240, since_service*.08), -source, target,
                               'deliver' if deliver or deliver_ack else ('confirm' if need_ack else 'collect')))
        if not candidates:
            # Rendezvous with the freshest reported team centre when all local
            # data is current; the leader can still have an incomplete cache.
            target = np.median(np.asarray([r[2] for r in self._relay33_reports.values()]), axis=0)
            return dict(source=None, point=target.tolist(), purpose='rendezvous')
        _, negative_source, target, purpose = max(candidates, key=lambda item:(item[0], item[1]))
        return dict(source=-negative_source, point=target.tolist(), purpose=purpose)

    def _relay33_task_target(self):
        task = self._relay33_task
        if task is not None:
            source = task['source']
            contacted = source is not None and self._steps-self._relay33_direct.get(source, -10000) <= 3
            if contacted:
                if self._relay33_contact_since is None:
                    self._relay33_contact_since = self._steps
                elapsed_contact = self._steps-self._relay33_contact_since
                if elapsed_contact >= self.TARGET_HOLD and self._relay33_task_confirmed(task):
                    self._relay33_serviced[source] = self._steps
                    self._relay33_event('contact_exchange_confirmed', source=source, purpose=task['purpose'],
                        received_versions=self._exchange_state()['received_versions'])
                    task = None
                elif elapsed_contact >= 60:
                    self._relay33_serviced[source] = self._steps-80
                    self._relay33_event('contact_without_task_confirmation', source=source, purpose=task['purpose'])
                    task = None
            else:
                self._relay33_contact_since = None
            if task is not None and self._steps-self._relay33_task_since >= task['budget_steps']:
                if source is not None:
                    self._relay33_searches[source] += 1
                    self._relay33_serviced[source] = self._steps
                self._relay33_event('search_budget', source=source)
                task = None
        if task is None:
            task = self._relay33_pick_task()
            distance = float(np.linalg.norm(np.asarray(task['point'])-self._position))
            task['budget_steps'] = max(self.SEARCH_BUDGET, min(900, int(distance/1.5+120)))
            state = self._exchange_state()
            task['required_versions'] = list(self._relay33_reports[int(self.identifier or 0)][6])
            task['required_final_sources'] = [i for i in range(self._n) if str(i) not in state['missing']]
            task['required_acks'] = list(state['acknowledged_sources'])
            self._relay33_task_since = self._steps
            self._relay33_contact_since = None
            self._relay33_event('task_selected', **task)
        self._relay33_task = task
        source = task['source']
        if source is not None and self._steps-self._relay33_reports[source][1] < 90:
            return np.asarray(self._relay33_reports[source][2], float).copy()
        return np.asarray(task['point'], float).copy()

    def _relay33_task_confirmed(self, task):
        source = task['source']
        peer = self._relay33_reports.get(source)
        if peer is None or self._steps-peer[1] > 10:
            return False
        state = self._exchange_state()
        if task['purpose'] == 'deliver':
            return (all(got >= wanted for got, wanted in zip(peer[6], task['required_versions']))
                    and not set(task['required_final_sources']).intersection(peer[7])
                    and set(task['required_acks']).issubset(peer[8]))
        if task['purpose'] == 'confirm':
            return source in state['acknowledged_sources']
        if self._radio_maps.own_final is not None:
            return str(source) not in state['missing']
        got = self._radio_maps.snapshots.get(source)
        return (got is not None and got[1] > task['required_versions'][source]
                and got[1] >= peer[6][source])

    def _try_hold(self, distances, dynamic, command):
        if self._relay33_is_courier():
            self._holding = False
            return False
        return super()._try_hold(distances, dynamic, command)

    def _navigation(self):
        original = super()._navigation()
        self._relay33_refresh_report()
        self._relay33_elect()
        active = self._relay33_is_courier()
        if active and self._position is not None:
            target = self._relay33_task_target()
            # Leave enough separation for collision avoidance while retaining
            # the whole actual radio exchange for several rotation cycles.
            contact = (self._relay33_task['source'] is not None
                       and self._steps-self._relay33_direct.get(self._relay33_task['source'], -10000) <= 3)
            if contact:
                move = np.clip(-.7*self._velocity, -1, 1)
                motion = dict(forward=float(move[0]), lateral=float(move[1]), rotation=0.)
            else:
                motion = self._relay33_nav.command(self._position, self._heading, self._velocity,
                    self.lidar_values(), self.lidar_rays_angles(), self.semantic_values(),
                    self._explore, target, self._steps)
            original.update(motion)
            self._d32_stats['courier_ticks'] += 1
        self._s28_return = dict(policy='single_radio_granted_mobile_courier', active=active,
            grant=self._relay33_grant, task=self._relay33_task,
            reports={str(i):list(row) for i,row in self._relay33_reports.items()},
            navigator=self._relay33_nav.status(), stats=dict(self._d32_stats),
            events=list(self._relay33_events))
        return original

    def _submit(self):
        super()._submit()
        if self._s28_submission is not None:
            self._s28_submission.update(
                navigation_policy='clean_plus_radio_elected_mobile_courier',
                submission_policy='dualmap32_plus_observed_rectangular_boundary_completion',
                relay33=dict(grant=self._relay33_grant, stats=dict(self._d32_stats),
                             task=self._relay33_task))
