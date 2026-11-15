"""B8 independent empty-speed / payload-route candidates on unchanged B0.

No B7 relay, B6 pose matcher, truth access, layout files or online bomb internals.
Terminal disposal behavior is deliberately the stable B0 behavior in both arms.
"""
from collections import Counter, deque
import math
import time
import numpy as np
from swarm_rescue.solutions.my_drone_eval_blue import MyDroneBlueAdapted
from swarm_rescue.solutions.blue_nav_b8 import PayloadPlanner, SpeedGovernor, wrap


class MyDroneBlueNavB8(MyDroneBlueAdapted):
    EMPTY_SPEED = False
    PAYLOAD_NAV = False
    PLAN_PERIOD = 36

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._b8_governor = SpeedGovernor()
        self._b8_planner = PayloadPlanner(self._grid)
        self._b8_stats = Counter()
        self._b8_events = deque(maxlen=512)
        self._b8_info = {}
        self._b8_plan_tick = -100
        self._b8_carry = False
        self._b8_trace = deque(maxlen=100)
        self._b8_anchor = None
        self._b8_progress_tick = 0
        self._b8_recovery_until = 0
        self._b8_cooldown_until = 0
        self._b8_recovery = None
        self._b8_goal = None
        self._b8_health = float(self.drone_health)
        self._b8_caution_until = 0
        self._b8_seconds = 0.
        self._b8_previous_reason = None

    def _b8_event(self, kind, **fields):
        if len(self._b8_events) == self._b8_events.maxlen:
            self._b8_stats['event_overflow'] += 1
        self._b8_events.append(dict(step=int(self._tick), kind=kind, **fields))
        self._b8_stats[kind] += 1

    def _navigate(self, goal, distances, angles, carrying=False):
        if not self.PAYLOAD_NAV or not carrying:
            return super()._navigate(goal, distances, angles, carrying)
        # Preserve B0's centre approach/release logic. Its output after this
        # method is not subsequently overwritten by the cruise governor.
        hit = self._center_hit
        if hit is not None and np.isfinite(hit.distance) and hit.distance < 105:
            self._b8_info = dict(reason='b0_terminal_preserved')
            return super()._navigate(goal, distances, angles, carrying)
        if goal is None:
            cmd = super()._navigate(goal, distances, angles, carrying)
            self._b8_info = dict(reason='unknown_center_b0_search')
            return cmd
        if self._b8_goal is None or np.linalg.norm(goal-self._b8_goal) > 24:
            self._b8_plan_tick = -100
            self._b8_goal = np.array(goal).copy()
        if (self._b8_anchor is None or np.linalg.norm(self._position-self._b8_anchor) > 16):
            self._b8_anchor = self._position.copy()
            self._b8_progress_tick = self._tick
        if self._tick-self._b8_plan_tick >= self.PLAN_PERIOD:
            plan = self._b8_planner.plan(self._position, goal)
            self._b8_plan_tick = self._tick
            self._route = [p.copy() for p in plan.points]
            self._route_step = self._tick
            self._b8_stats['replans'] += 1
            if plan.reason != self._b8_previous_reason:
                self._b8_event('route_state', reason=plan.reason, complete=plan.complete)
                self._b8_previous_reason = plan.reason
        plan = self._b8_planner.last
        while self._route and np.linalg.norm(self._route[0]-self._position) < 12:
            self._route.pop(0)
        # Only shortcut segments checked with the full payload envelope.
        for k in range(min(8, len(self._route))-1, 0, -1):
            if self._b8_planner.segment(self._position, self._route[k]):
                self._route = self._route[k:]
                break
        waypoint = self._route[0] if self._route else None
        if waypoint is not None and not self._b8_planner.segment(self._position, waypoint,
                egress=plan.reason=='observed_clearance_egress'):
            waypoint = None
            self._route = []
            self._b8_stats['segment_rejected'] += 1
        stalled = self._tick-self._b8_progress_tick >= 100
        if stalled and self._tick >= self._b8_cooldown_until:
            self._b8_recovery = None
            # Previously traversed alone is insufficient: CURRENT known space
            # must accommodate the payload disk over the whole return segment.
            for _, p in reversed(self._b8_trace):
                distance = float(np.linalg.norm(p-self._position))
                if 24 < distance < 90 and self._b8_planner.segment(self._position, p):
                    self._b8_recovery = p.copy(); break
            self._b8_recovery_until = self._tick+45
            self._b8_cooldown_until = self._tick+180
            self._b8_progress_tick = self._tick
            self._b8_plan_tick = -100
            self._b8_event('stagnation', recovery=bool(self._b8_recovery is not None))
        recovery = False
        if self._b8_recovery is not None and self._tick < self._b8_recovery_until:
            if (np.linalg.norm(self._b8_recovery-self._position) > 14
                    and self._b8_planner.segment(self._position, self._b8_recovery)):
                waypoint = self._b8_recovery; recovery = True
            else: self._b8_recovery = None
        cmd = dict(forward=0., lateral=0., rotation=0., grasper=1)
        reason = 'validated_trace_recovery' if recovery else plan.reason
        if waypoint is not None:
            delta = waypoint-self._position
            angle = wrap(math.atan2(delta[1], delta[0])-self._heading)
            cmd.update(forward=.62*math.cos(angle), lateral=.62*math.sin(angle),
                       rotation=float(np.clip(angle*1.25, -.45, .45)))
        else:
            self._b8_stats['hold_no_route_steps'] += 1
            reason = 'approach_occluded_replan' if plan.complete else plan.reason
        self._b8_info = dict(reason=reason, complete=plan.complete, route_choice=plan.choice,
            route_length=plan.length, estimated_steps=plan.estimated_steps if math.isfinite(plan.estimated_steps) else None,
            minimum_clearance=plan.min_clearance, stagnating=bool(stalled),
            waypoint=None if waypoint is None else waypoint.tolist())
        return cmd

    def control(self):
        if not self.EMPTY_SPEED and not self.PAYLOAD_NAV:
            return super().control()
        started = time.perf_counter()
        carrying_before = bool(self.grasped_bombs())
        if carrying_before != self._b8_carry:
            self._tick = int(self.elapsed_timestep)
            self._b8_event('pickup_observed' if carrying_before else 'possession_ended_unconfirmed')
            self._b8_governor = SpeedGovernor()
            self._b8_plan_tick = -100; self._b8_goal = None; self._b8_anchor = None
            self._b8_recovery = None
            self._b8_carry = carrying_before
        command = super().control()
        if self._position is None:
            self._b8_seconds += time.perf_counter()-started
            return command
        health = float(self.drone_health)
        if health < self._b8_health:
            self._b8_event('health_loss', amount=self._b8_health-health)
            self._b8_caution_until=self._tick+60; self._b8_plan_tick=-100
        self._b8_health = health
        if not self._b8_trace or np.linalg.norm(self._position-self._b8_trace[-1][1])>20:
            self._b8_trace.append((self._tick, self._position.copy()))
        carrying = bool(self.grasped_bombs())
        hit = self._center_hit
        terminal = carrying and hit is not None and np.isfinite(hit.distance) and hit.distance < 105
        enabled = ((self.EMPTY_SPEED and not carrying and self._mode=='search' and not command.get('grasper'))
                   or (self.PAYLOAD_NAV and carrying and not terminal))
        if enabled:
            raw=self.lidar_values(); distances=np.asarray(raw if raw is not None else [], float)
            angles=np.asarray(self.lidar_rays_angles(), float)[:len(distances)]
            objects=self.semantic_values() or ()
            dynamic=any(getattr(o.entity_type, 'name', '')=='DRONE' and np.isfinite(o.distance) and o.distance<110 for o in objects)
            compass=self.measured_compass_angle(); gps=self.measured_gps_position()
            uncertain=(gps is None or compass is None or not np.isfinite(gps).all() or not np.isfinite(compass)
                       or np.linalg.norm(gps-self._position)>25 or self._tick<self._b8_caution_until)
            turn = 0.; distance = None
            if self._route:
                delta=self._route[0]-self._position
                distance=float(np.linalg.norm(delta))
                if len(self._route)>1:
                    next_delta=self._route[1]-self._route[0]
                    turn=wrap(math.atan2(next_delta[1], next_delta[0])-math.atan2(delta[1], delta[0]))
                    # Anticipate a turn but do not brake to every point on a straight road.
                    if abs(turn)<.2 and len(self._route)>2:
                        distance=float(np.linalg.norm(self._route[min(4,len(self._route)-1)]-self._position))
            command=self._b8_governor.update(command, distances, angles, self.odometer_values(),
                carrying=carrying, turn=turn, waypoint_distance=distance, dynamic=dynamic, pose_uncertain=uncertain)
            self._b8_info={**self._b8_info, 'speed':dict(self._b8_governor.last),
                           'mode':'payload' if carrying else 'empty_search'}
            self._b8_stats['payload_regulated_steps' if carrying else 'empty_regulated_steps']+=1
        elif terminal:
            self._b8_info=dict(reason='b0_terminal_preserved', mode='terminal')
        else:
            self._b8_info=dict(reason='b0_task_preserved', mode=self._mode)
        self._b8_seconds += time.perf_counter()-started
        return command


class MyDroneBlueB8Empty(MyDroneBlueNavB8):
    EMPTY_SPEED = True


class MyDroneBlueB8Payload(MyDroneBlueNavB8):
    PAYLOAD_NAV = True


class MyDroneBlueB8Off(MyDroneBlueNavB8):
    pass
