"""Exploration-led frontier support, bounded return, unchanged observed border.

All decisions use local sensing and actual radio packets. There is no map/seed
identity, forced courier election, private peer access or synthetic receipt.
The common 86% final epoch and all-source byte/digest ACK barrier are retained.
"""
from collections import Counter, deque
import math
from types import SimpleNamespace

import cv2
import numpy as np

from swarm_rescue.solutions.my_drone_red_dualmap32 import MyDroneRedDualMap32Clean
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.red_boundary33 import FinalBoundary33
from swarm_rescue.solutions.red_frontier34 import FrontierNavigator34
from swarm_rescue.solutions.red_navigation32 import EvidenceNavigation32
from swarm_rescue.solutions.my_drone_red_advanced2 import (
    _M13_obstructed_route, _M11_safe_velocity, _M01_wrap_angle,
)


class FrontierSurvey34(EvidenceNavigation32):
    """Keep Clean evidence and normal targets; reassign only idle exploration."""
    def __init__(self, size, event):
        super().__init__(size, event)
        self.support_mode34 = False
        self.radio_claims34 = []
        self.last_frontier34 = None

    def plan(self, position, home, identifier, number, claims, blocked_goals=(), rendezvous=False):
        if not self.support_mode34 or rendezvous:
            return super().plan(position, home, identifier, number, claims, blocked_goals, rendezvous)
        wall = cv2.dilate((self.obstacle >= 2).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))).astype(bool)
        allowed = self.free & ~wall
        start = self.cell(position)
        allowed[start] = True
        distance, parents, queue = {start: 0}, {}, deque([start])
        while queue:
            r, c = queue.popleft()
            for other in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if (0 <= other[0] < self.shape[0] and 0 <= other[1] < self.shape[1]
                        and allowed[other] and other not in distance):
                    distance[other] = distance[(r,c)]+1
                    parents[other] = (r,c)
                    queue.append(other)
        unknown = cv2.filter2D((~self.free & ~wall).astype(np.float32), -1,
                              np.ones((5,5), np.float32), borderType=cv2.BORDER_CONSTANT)
        visited = cv2.filter2D((self.shared_visited | (self.visits > 0)).astype(np.float32),
                              -1, np.ones((5,5), np.float32), borderType=cv2.BORDER_CONSTANT)
        candidates = []
        for cell, steps in distance.items():
            if steps < 4 or cell in blocked_goals or unknown[cell] < 3:
                continue
            point = self.point(cell)
            # An occupied claim is a temporary allocation, not a wall.
            all_claims = [*claims, *self.radio_claims34]
            if any(np.linalg.norm(point-np.asarray(p)) < 110 for p in all_claims):
                continue
            score = 12.*min(16., float(unknown[cell])) + 3.*(25.-visited[cell]) - .6*steps
            score -= min(30., float(self.goal_visits[cell])*8.) + min(40., float(self.visits[cell])*2.)
            score += min(240., np.linalg.norm(point-position))*.12
            candidates.append((float(score), cell))
        if not candidates:
            self.stats['support_no_reachable_frontier'] += 1
            return super().plan(position, home, identifier, number, claims, blocked_goals, False)
        score, cell = max(candidates)
        path = [cell]
        while path[-1] != start:
            path.append(parents[path[-1]])
        self.last_frontier34 = dict(step=int(self.step), point=self.point(cell).tolist(),
                                   unknown=int(unknown[cell]), route_cells=len(path), score=score)
        self.stats['support_frontier_selected'] += 1
        self.event('frontier34_target', **self.last_frontier34)
        return path[::-1]


def valid_report34(row, number, clock):
    if not isinstance(row, tuple) or len(row) != 9:
        return False
    source, step, point, goal, versions, finals, acks, mode, inventory = row
    def point_ok(p):
        return (isinstance(p, tuple) and len(p) == 2
                and all(type(x) in (int,float) and math.isfinite(x) for x in p))
    return (type(source) is int and 0 <= source < number and type(step) is int
            and 0 <= step <= clock and point_ok(point) and (goal is None or point_ok(goal))
            and isinstance(versions, tuple) and len(versions) == number
            and all(type(v) is int and -1 <= v <= clock for v in versions)
            and all(isinstance(ids, tuple) and len(ids) <= number
                    and all(type(i) is int and 0 <= i < number for i in ids) for ids in (finals,acks))
            and mode in ('explore','support','return','exchange')
            and type(inventory) is int and inventory >= 0)


class MyDroneRedCleanBoundary34(MyDroneRedDualMap32Clean):
    """Control: exactly Clean flight and delivery, with the retained renderer."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fusion27 = FinalBoundary33(self.size_area, wall_width=11)

    def _submit(self):
        super()._submit()
        if self._s28_submission is not None:
            self._s28_submission['submission_policy'] = 'unchanged_observation_only_boundary33'


class MyDroneRedFrontier34(MyDroneRedCleanBoundary34):
    """Local exploration first; shared frontier claims; route-budget delivery."""
    ARRIVAL_PROGRESS34 = .94
    EXCHANGE_RESERVE34 = 180
    RETURN_SPEED34 = 1.8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._explore = FrontierSurvey34(self.size_area, self._event)
        self._f34_init()

    def _f34_init(self):
        self._f34_mode = 'explore'
        self._f34_reports = {}
        self._f34_direct = {}
        self._f34_samples = deque(maxlen=41)
        self._f34_events = deque(maxlen=160)
        self._f34_crumbs = []
        self._f34_return_points = []
        self._f34_budget = {}
        self._f34_budget_tick = -1000
        self._f34_support_since = None
        self._f34_gain = {}
        self._f34_return_reason = None
        self._f34_wall_samples = deque(maxlen=16)
        self._f34_upload_acks = {}
        self._f34_packet_cursor = 0
        self._f34_return_estimate = None
        self._f34_return_estimate_tick = -1000
        self._f34_return_estimate_target = None
        self._f34_trace_until = -1
        self._f34_station = None
        self._f34_station_probe_tick = -1000
        self._f34_detour = None
        self._f34_failed_targets = []
        self._f34_retry_after = -1
        self._f34_nav = FrontierNavigator34(int(self.identifier or 0))
        self._d32_stats = Counter()

    def _f34_event(self, reason, **data):
        self._d32_stats[reason] += 1
        self._f34_events.append(dict(step=int(self._steps), reason=reason, **data))
        self._event('frontier34_'+reason, **data)

    def _f34_report(self):
        if self._position is None or not np.isfinite(self._position).all():
            return
        own = int(self.identifier or 0)
        state = self._exchange_state()
        goal = tuple(map(float, self._explore.point(self._route[-1]))) if self._route else None
        versions = tuple(self._radio_maps.snapshots[i][1] if i in self._radio_maps.snapshots else -1
                         for i in range(self._n))
        self._f34_reports[own] = (own, int(self._steps), tuple(map(float,self._position)), goal,
            versions, tuple(i for i in range(self._n) if str(i) not in state['missing']),
            tuple(state['acknowledged_sources']), self._f34_mode, int(self.carried_bombs_count()))

    def _receive(self):
        messages = []
        for sender, packet in self.communicator.received_messages:
            if not isinstance(packet, tuple) or len(packet) != 3 or packet[0] != 'frontier34-v1':
                continue
            messages.append((sender,packet[1]))
            reports = packet[2]
            if not isinstance(reports,tuple) or len(reports) > self._n:
                continue
            for row in reports:
                if not valid_report34(row,self._n,int(self.elapsed_timestep)):
                    self._d32_stats['invalid_report'] += 1
                    continue
                source = row[0]
                if source == int(self.identifier or 0):
                    continue
                old = self._f34_reports.get(source)
                if old is None or row[1] > old[1]:
                    self._f34_reports[source] = row
        proxy = SimpleNamespace(_n=self._n, elapsed_timestep=self.elapsed_timestep,
            _status=self._status, _peers=self._peers, _radio_maps=self._radio_maps,
            _explore=self._explore, _valid_peer=self._valid_peer,
            _bomb_claims=self._bomb_claims, _bomb_records=self._bomb_records,
            _f27_home=self._f27_home, _s28_stats=self._s28_stats, _s28_receipts=self._s28_receipts,
            communicator=SimpleNamespace(received_messages=messages))
        MyDroneRedDualMap32Clean._receive(proxy)
        self._f27_home = proxy._f27_home
        for source, peer in self._peers.items():
            if 0 <= self._steps-peer[0] <= 3:
                self._f34_direct[source] = peer[0]
        own = int(self.identifier or 0)
        leader = self._f34_reports.get(0)
        final = self._radio_maps.own_final
        if (leader is not None and final is not None and leader[4][own] == final[1]
                and own in leader[5] and own not in self._f34_upload_acks):
            self._f34_upload_acks[own] = dict(received_step=self._steps, leader_report_step=leader[1],
                                            version=final[1], kind='leader_reported_payload_manifest_receipt')
            self._f34_event('own_final_at_leader', **self._f34_upload_acks[own])

    def define_message_for_all(self):
        self._f34_report()
        packet = super().define_message_for_all()
        # Retain one snapshot per packet. Missing neighbours get priority on
        # alternate packets; regular round-robin still forwards all origins.
        if self._steps % 2 == 0 and self._radio_maps.snapshots:
            peers = [row for i,row in self._f34_reports.items()
                     if i != int(self.identifier or 0) and self._steps-row[1] <= 20
                     and self._steps-self._f34_direct.get(i,-10000) <= 10]
            needed = [i for i,s in sorted(self._radio_maps.snapshots.items())
                      if any(row[4][i] < s[1] for row in peers)]
            if needed:
                source = needed[self._f34_packet_cursor % len(needed)]
                self._f34_packet_cursor += 1
                fusion = packet[1]
                packet = (packet[0], (fusion[0],fusion[1],self._radio_maps.snapshots[source],
                                     fusion[3],fusion[4]), packet[2])
                self._d32_stats['missing_peer_payload_prioritized'] += 1
        return ('frontier34-v1',packet,tuple(self._f34_reports[i] for i in sorted(self._f34_reports)))

    def _f34_trace(self):
        position = np.asarray(self._position,float)
        if not self._f34_crumbs:
            self._f34_crumbs.append(position.copy())
            return
        # A physical re-visit removes a loop; no unobserved shortcut is added.
        near = [i for i,p in enumerate(self._f34_crumbs[:-3])
                if np.linalg.norm(p-position) < 22. and self._explore.line_clear(position,p)]
        if near:
            self._f34_crumbs = self._f34_crumbs[:near[0]+1]
        if np.linalg.norm(self._f34_crumbs[-1]-position) >= 35.:
            self._f34_crumbs.append(position.copy())

    def _f34_home(self):
        return np.asarray(self._f27_home[2] if self._f27_home is not None else self._home,float)

    def _f34_remaining(self, progress):
        limits = self._misc_data
        step_limit = getattr(limits,'max_timestep_limit',None)
        seconds_limit = getattr(limits,'max_walltime_limit',None)
        step_left = float('inf') if not step_limit else progress*step_limit-self.elapsed_timestep
        seconds_left = float('inf') if not seconds_limit else progress*seconds_limit-self.elapsed_walltime
        rate = 25.
        if len(self._f34_wall_samples) >= 2:
            s0,t0 = self._f34_wall_samples[0]
            s1,t1 = self._f34_wall_samples[-1]
            if t1-t0 > .01:
                rate = min(60., max(1., .75*(s1-s0)/(t1-t0)))
        return min(step_left, seconds_left*rate), dict(step_remaining=step_left,
            wall_seconds_remaining=seconds_left, conservative_steps_per_second=rate)

    def _f34_budget_update(self):
        if self._steps-self._f34_budget_tick < 30 and self._f34_budget:
            return
        self._f34_budget_tick = self._steps
        home = self._f34_home()
        estimate = self._f34_nav.estimate_route(self._explore,self._position,home,
                                               speed=self.RETURN_SPEED34,reserve_steps=0)
        trace = [self._position,*reversed(self._f34_crumbs),home]
        traced_length = sum(float(np.linalg.norm(np.asarray(b)-a)) for a,b in zip(trace,trace[1:]))
        straight = float(np.linalg.norm(home-self._position))
        length = max(straight, estimate['length'] if estimate['exact'] else traced_length)
        travel = length/self.RETURN_SPEED34
        recovery = 160. if estimate['exact'] else 300.
        remaining, timing = self._f34_remaining(self.ARRIVAL_PROGRESS34)
        self._f34_budget = dict(route_exact=bool(estimate['exact']), length_px=length,
            traced_length_px=traced_length, route_length_px=estimate['length'],
            travel_steps=travel, recovery_steps=recovery, exchange_reserve_steps=self.EXCHANGE_RESERVE34,
            required_steps=travel+recovery+self.EXCHANGE_RESERVE34,
            arrival_remaining_steps=remaining, **timing)

    def _f34_start_return(self, reason):
        if self._f34_mode in ('return','exchange'):
            return
        self._f34_mode = 'return'
        self._f34_return_reason = reason
        self._f34_return_points = [p.copy() for p in reversed(self._f34_crumbs)]
        self._gathering = True
        self._holding = False
        self._route = []
        self._bomb_target = None
        self._explore.support_mode34 = False
        self._f34_nav.reset()
        self._f34_event('return_started', trigger=reason, budget=dict(self._f34_budget))

    def _f34_policy(self):
        if self._position is None or self._home is None:
            return
        if self._steps % 30 == 0:
            self._f34_wall_samples.append((self._steps,float(self.elapsed_walltime)))
        if self._f34_mode not in ('return','exchange'):
            self._f34_trace()
        self._f34_budget_update()
        self._explore.radio_claims34 = [np.asarray(row[3],float) for i,row in self._f34_reports.items()
            if i != int(self.identifier or 0) and row[3] is not None and self._steps-row[1] <= 90
            and row[7] in ('explore','support')]
        empty = self.carried_bombs_count() == 0 and self._attempt_step is None
        if empty and self._f34_mode not in ('return','exchange'):
            if self._radio_maps.own_final is not None:
                self._f34_start_return('final_epoch_preemption')
            elif self._f34_budget['arrival_remaining_steps'] <= self._f34_budget['required_steps']:
                self._f34_start_return('route_and_exchange_budget')
        if self._f34_mode in ('return','exchange'):
            self._gathering = True
            return
        self._gathering = False
        if self._steps % 6 == 0:
            self._f34_samples.append((self._steps,int(self._explore.free.sum()),
                                      int(np.count_nonzero(self._explore.visits))))
        if len(self._f34_samples) < 31 or not empty:
            return
        first,last = self._f34_samples[0],self._f34_samples[-1]
        self._f34_gain = dict(window_steps=last[0]-first[0], new_free=last[1]-first[1],
                              new_visited=last[2]-first[2])
        stagnant = last[0]-first[0] >= 180 and last[1]-first[1] <= 3 and last[2]-first[2] <= 3
        if stagnant and self._f34_mode == 'explore':
            self._f34_mode = 'support'
            self._f34_support_since = self._steps
            self._explore.support_mode34 = True
            self._route = []
            self._last_route_step = self._steps-12
            self._f34_event('local_complete_support', gain=dict(self._f34_gain))
        elif self._f34_mode == 'support' and last[1]-first[1] >= 12:
            self._f34_mode = 'explore'
            self._explore.support_mode34 = False
            self._f34_event('productive_exploration_resumed', gain=dict(self._f34_gain))

    def _f34_safe_station(self, center, radius=120., detour=False):
        """Choose an observed free station; navigation still checks each move."""
        walls = cv2.dilate((self._explore.obstacle >= 2).astype(np.uint8),
                          np.ones((3,3),np.uint8)).astype(bool)
        cells = np.argwhere(self._explore.free & ~walls)
        peers = [p[1] for i,p in self._peers.items()
                 if i != int(self.identifier or 0) and self._steps-p[0] <= 10]
        self._f34_failed_targets = [(p,t) for p,t in self._f34_failed_targets if t > self._steps]
        candidates = []
        for row in cells:
            point = self._explore.point(tuple(row))
            distance = float(np.linalg.norm(point-center))
            travel = float(np.linalg.norm(point-self._position))
            if distance > radius or travel < 32:
                continue
            if any(np.linalg.norm(point-p) < 58 for p in peers):
                continue
            if any(np.linalg.norm(point-p) < 35 for p,t in self._f34_failed_targets):
                continue
            if detour and travel > 130:
                continue
            angle = math.atan2(*(point-center)[::-1])
            preferred = 2*math.pi*int(self.identifier or 0)/max(1,self._n)
            score = travel + .15*distance - 18*math.cos(angle-preferred)
            candidates.append((score,tuple(row),point))
        for _,_,point in sorted(candidates,key=lambda x:(x[0],x[1]))[:4]:
            route = self._f34_nav.estimate_route(self._explore,self._position,point,
                                                speed=self.RETURN_SPEED34,reserve_steps=0)
            if route['exact']:
                return point.copy()
        return None

    def _f34_return_motion(self):
        home = self._f34_home()
        own = int(self.identifier or 0)
        # A common, actually radio-announced home is the final rendezvous. The
        # return starts from a route budget, never from an early fixed fraction.
        direct_leader = own == 0 or self._steps-self._f34_direct.get(0,-10000) <= 6
        distance_home = float(np.linalg.norm(self._position-home))
        leader = self._f34_reports.get(0)
        target = home.copy()
        if own != 0 and leader is not None and self._steps-leader[1] <= 20:
            target = np.asarray(leader[2],float)
        # Stay spread inside the proven radio neighbourhood; close peers must
        # still pass the fresh scan collision checks before a station is held.
        close_peer = any(self._steps-p[0] <= 8 and np.linalg.norm(self._position-p[1]) < 48
                         for i,p in self._peers.items() if i != own)
        scan = self.lidar_values()
        ranges = np.asarray([] if scan is None else scan,float).ravel()
        ranges = ranges[np.isfinite(ranges) & (ranges > 0)]
        hold_clear = len(ranges) >= 24 and float(ranges.min()) > 38.
        if direct_leader and distance_home < 120 and not close_peer and hold_clear:
            if self._f34_mode != 'exchange':
                self._f34_mode = 'exchange'
                self._f34_event('exchange_station_reached')
            self._d32_stats['exchange_hold_ticks'] += 1
            force = np.clip(-.7*self._velocity,-1,1)
            return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)
        self._f34_mode = 'return'
        if direct_leader and distance_home < 170 and (close_peer or not hold_clear):
            if ((self._f34_station is None and self._steps-self._f34_station_probe_tick >= 30)
                    or (self._f34_station is not None and self._steps-self._f34_station[1] >= 90)):
                self._f34_station_probe_tick = self._steps
                station = self._f34_safe_station(home)
                self._f34_station = None if station is None else (station,self._steps)
                if station is not None:
                    self._f34_event('separated_exchange_station',point=station.tolist())
            if self._f34_station is not None:
                target = self._f34_station[0].copy()
        if self._f34_detour is not None:
            point,until = self._f34_detour
            if self._steps < until and np.linalg.norm(point-self._position) >= 30:
                target = point.copy()
            else:
                self._f34_detour = None
        if (self._f34_return_estimate is None or self._steps-self._f34_return_estimate_tick >= 30
                or np.linalg.norm(target-self._f34_return_estimate_target) > 60.):
            self._f34_return_estimate = self._f34_nav.estimate_route(self._explore,self._position,target,
                                                   speed=self.RETURN_SPEED34,reserve_steps=0)
            self._f34_return_estimate_tick = self._steps
            self._f34_return_estimate_target = target.copy()
        estimate = self._f34_return_estimate
        while self._f34_return_points and np.linalg.norm(self._f34_return_points[0]-self._position) < 35:
            self._f34_return_points.pop(0)
        if self._f34_nav.failed:
            self._f34_trace_until = self._steps+180
        use_trace = not estimate['exact'] or self._steps < self._f34_trace_until
        if use_trace and self._f34_return_points:
            target = self._f34_return_points[0].copy()
            if self._f34_nav.failed:
                self._f34_event('return_target_blocked', navigator=self._f34_nav.status())
                # Step to a different earlier visited waypoint rather than
                # repeatedly pursuing the same failed global proxy target.
                if len(self._f34_return_points) > 1:
                    self._f34_return_points.pop(0)
                    target = self._f34_return_points[0].copy()
                self._f34_nav.reset()
            self._d32_stats['breadcrumb_return_ticks'] += 1
        elif self._f34_nav.failed:
            # Once the recorded path is exhausted, a failed target must not
            # silently hold the controller forever. Retry only after a bounded
            # observation pause, preferably at a distinct known safe station.
            if self._steps >= self._f34_retry_after:
                failed_point = self._f34_nav.target
                if failed_point is not None:
                    self._f34_failed_targets.append((np.asarray(failed_point).copy(),self._steps+240))
                alternative = self._f34_safe_station(self._position,130.,detour=True)
                self._f34_retry_after = self._steps+60
                if alternative is not None:
                    self._f34_detour = (alternative,self._steps+120)
                    target = alternative.copy()
                    self._f34_event('exhausted_trace_detour',point=target.tolist())
                else:
                    self._f34_event('exhausted_trace_rescan_retry')
                self._f34_nav.reset()
        motion = self._f34_nav.command(self._position,self._heading,self._velocity,
            self.lidar_values(),self.lidar_rays_angles(),self.semantic_values(),
            self._explore,target,self._steps)
        self._d32_stats['return_motion_ticks'] += 1
        return motion

    def _navigation(self):
        self._f34_policy()
        # Bypass only Fusion27's .65 gather switch. The copied motion method
        # below also removes its independent .72 switch without changing time.
        original = MyDroneRedRepair17._navigation(self)
        if (self._f34_mode in ('return','exchange') and self._position is not None
                and self.carried_bombs_count() == 0 and self._attempt_step is None):
            original.update(self._f34_return_motion())
        self._f34_report()
        self._d32_stats[self._f34_mode+'_ticks'] += 1
        self._s28_return = dict(policy='exploration_led_frontier_and_budget_delivery',
            mode=self._f34_mode, role=self._f34_mode, local_gain=dict(self._f34_gain),
            frontier=self._explore.last_frontier34, budget=dict(self._f34_budget),
            return_reason=self._f34_return_reason, return_waypoints=len(self._f34_return_points),
            home=None if self._home is None else self._f34_home().tolist(),
            reports={str(i):list(r) for i,r in self._f34_reports.items()},
            upload_receipts=dict(self._f34_upload_acks), navigator=self._f34_nav.status(),
            stats=dict(self._d32_stats), events=list(self._f34_events))
        return original

    def _try_hold(self, distances, dynamic, command):
        if self._f34_mode in ('return','exchange'):
            self._holding = False
            return False
        return super()._try_hold(distances,dynamic,command)

    def _submit(self):
        super()._submit()
        if self._s28_submission is not None:
            self._s28_submission.update(navigation_policy='frontier34_local_completion_support',
                gathering_policy='known_route_budget_or_final_epoch_preemption',
                frontier34=dict(mode=self._f34_mode, stats=dict(self._d32_stats),
                    return_reason=self._f34_return_reason,budget=dict(self._f34_budget),
                    upload_receipts=dict(self._f34_upload_acks)))

    def _navigate_motion(self):
        command = self._empty_command()
        if self._position is None:
            return command
        values = self.lidar_values()
        if values is None:
            command.update(forward=float(-np.clip(self._velocity[0], -1, 1)), lateral=float(-np.clip(self._velocity[1], -1, 1)))
            return command
        d = np.asarray(values).ravel()
        angles = np.asarray(self.lidar_rays_angles())
        bomb_rays = np.zeros(len(d), bool)
        dynamic = np.zeros(len(d), bool)
        for obj in self.semantic_values() or []:
            kind = getattr(obj.entity_type, 'name', None)
            if kind in ('DRONE', 'BOMB'):
                delta = np.abs((angles - obj.angle + math.pi) % (2 * math.pi) - math.pi)
                matched = (delta < 0.19) & (np.abs(d - obj.distance) < 25)
                dynamic |= matched
                if kind == 'BOMB':
                    bomb_rays |= matched
        if self._launch_open is None:
            static = d[np.isfinite(d) & (d > 0) & ~dynamic]
            self._launch_open = bool(len(static) >= 12 and np.min(static) > 90)
        odom = self.odometer_values()
        if odom is not None and np.shape(odom) == (3,) and np.isfinite(odom).all():
            direction = odom[1] - odom[2]
            measured = odom[0] * np.array([math.cos(direction), math.sin(direction)])
            self._velocity = 0.4 * self._velocity + 0.6 * measured
        self._explore.step = self._steps
        # Gathering is set explicitly by _f34_policy using the return budget.
        if self._try_hold(d, dynamic, command):
            return command
        if not self._gathering and self._loop_watch.update(self._steps, self._position, float(np.linalg.norm(self._velocity)), int(self._explore.free.sum())):
            self._explore.avoid_region(self._position, self._steps + 480)
            self._escape_until = 0
            self._start_recovery(d, angles, 'local_loop')
            self._escape_until = self._steps + 55
            self._last_bomb_plan = self._steps + 80
            self._event('local_loop', position=list(map(float, self._position)))
        self._slow_ticks = self._slow_ticks + 1 if np.linalg.norm(self._velocity) < 0.8 else max(0, self._slow_ticks - 2)
        if self._slow_ticks >= 24 and self._steps - self._last_motion_recovery >= 40:
            self._start_recovery(d, angles, 'no_progress')
            self._last_motion_recovery = self._steps
            self._slow_ticks = 0
        if self._steps % 6 == 0 or self._steps == 1:
            self._explore.observe(self._position, self._heading, d, angles, dynamic)
        if self._steps % 6 == 0 and self._route and _M13_obstructed_route(self._explore, self._position, self._route):
            self._release_stale_route('new_wall')
        if self._steps % 30 == 0:
            self._visited_snapshot = np.packbits(self._explore.shared_visited | (self._explore.visits > 0)).tobytes()
            self._radio_maps.publish(int(self.identifier or 0), self._steps, self._map.mask())
        if self._direction_anchor is None:
            self._direction_anchor = self._position.copy()
        delta = self._position - self._direction_anchor
        if np.linalg.norm(delta) > 75:
            self._explore.forward_direction = delta / np.linalg.norm(delta)
            self._direction_anchor = self._position.copy()
        fresh = {i: p for (i, p) in self._peers.items() if self._steps - p[0] < 90}
        claims = [p[2] for (i, p) in fresh.items() if p[2] is not None and i != int(self.identifier or 0)]
        if self._steps % 60 == 0:
            if self._motion_anchor is not None and np.linalg.norm(self._position - self._motion_anchor) < 25:
                self._stalled_steps += 60
                self._start_recovery(d, angles, 'no_progress')
            self._motion_anchor = self._position.copy()
        self._blocked_goals = {g: t for (g, t) in self._blocked_goals.items() if t > self._steps}
        if self._launch_open and self._steps < 65 and (np.linalg.norm(self._position - self._home) < 180):
            self._explore.forward_direction = np.array([math.cos(self._launch_angle), math.sin(self._launch_angle)])
        if self._route and np.linalg.norm(self._position - self._explore.point(self._route[-1])) < 22:
            self._explore.retire_goal(self._route[-1])
            self._route = []
        if self._route and self._steps - self._goal_step > 600:
            self._explore.retire_goal(self._route[-1])
            self._route = []
        if not self._route and self._steps >= self._escape_until and (self._steps - self._last_route_step >= 12):
            self._route = self._explore.plan(self._position, self._home, int(self.identifier or 0), self._n, claims, self._blocked_goals, self._gathering)
            self._last_route_step = self._steps
            self._goal_step = self._steps
            self._replans += 1
        while len(self._route) > 1 and np.linalg.norm(self._position - self._explore.point(self._route[0])) < 22:
            self._route.pop(0)
        target_distance = 160.0
        if self._route:
            index = 0
            for (i, cell) in enumerate(self._route[:9]):
                point = self._explore.point(cell)
                if np.linalg.norm(point - self._position) > 130:
                    break
                if self._explore.line_clear(self._position, point):
                    index = i
                else:
                    break
            if index:
                self._route = self._route[index:]
            target = self._explore.point(self._route[0]) - self._position
            target_distance = float(np.linalg.norm(target))
            target_angle = math.atan2(target[1], target[0]) - self._heading
        else:
            if self._escape_heading is None or self._steps >= self._escape_until:
                self._start_recovery(d, angles, 'disconnected')
            target_angle = self._escape_heading - self._heading
        if self._steps < self._escape_until and self._escape_heading is not None:
            target_angle = self._escape_heading - self._heading
            target_distance = 160.0
        if self._launch_open and self._steps < 45:
            target_angle = self._launch_angle - self._heading
        if self._steps >= self._yield_until:
            for (i, (seq, position, goal)) in fresh.items():
                delta = position - self._position
                if self._steps - seq < 8 and i < int(self.identifier or 0) and (np.linalg.norm(delta) < 75):
                    angle = math.atan2(delta[1], delta[0]) - self._heading
                    if math.cos(angle - target_angle) > 0.4:
                        self._yield_heading = self._heading + angle - math.pi / 5
                        self._yield_until = self._steps + 18
                        break
        if self._steps < self._yield_until:
            target_angle = self._yield_heading - self._heading
            self._yield_steps += 1
        if self._route and self._steps >= self._escape_until and (self._steps >= self._yield_until):
            target_point = self._explore.point(self._route[0])
            if self._waypoint_progress.update(self._steps, self._position, target_point):
                self._release_stale_route('waypoint_stall')
                self._explore.avoid_region(self._position, self._steps + 240)
                self._start_recovery(d, angles, 'waypoint_stall')
                target_angle = self._escape_heading - self._heading
                target_distance = 160.0
        else:
            self._waypoint_progress.reset()
        steering_d = d.copy()
        steering_d[bomb_rays] = 80
        (move, self._last_motion_angle) = _M11_safe_velocity(steering_d, angles, _M01_wrap_angle(target_angle), self._velocity, self._last_motion_angle, target_distance)
        command.update(forward=float(move[0]), lateral=float(move[1]), rotation=0.0)
        return command
