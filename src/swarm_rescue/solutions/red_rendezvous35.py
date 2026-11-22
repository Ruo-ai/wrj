"""Current-scan return along recorded own routes, with directional recovery.

The 181 native lidar distances are pixel ranges, not a boolean full-scan flag.
Invalid bearings remain unknown. Normal moves use the unchanged physical hull
and stopping sweep. When the initial conservative hull includes unknown/margin
pixels, recovery may only remove that same unsafe set; it cannot acquire any
new unknown/occupied pixel. This permits a visible retreat sector without
pretending an unrelated missing sector was observed.
"""
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_rn7 import rotate
from swarm_rescue.solutions.red_frontier34 import FrontierNavigator34
from swarm_rescue.solutions.red_relay33 import RelayNavigator33


class RendezvousNavigator35(FrontierNavigator34):
    BLOCKED_LIMIT = 84
    STAGNATION_LIMIT = 150
    RECOVERY_LIMIT = 60
    REOBSERVE_TICKS = 3
    CORNER_REACHED = 10.

    def __init__(self, identifier=0, local_factory=None):
        super().__init__(identifier, local_factory)
        self._observed_route35 = []
        self._observed_cursor35 = 0
        self._scan_status35 = {}
        self._reobserve_since35 = None
        self._last_target35 = None

    def reset(self):
        super().reset()
        self._observed_route35 = []
        self._observed_cursor35 = 0
        self._scan_status35 = {}
        self._reobserve_since35 = None
        self._last_target35 = None

    def set_observed_route(self, points):
        """Copy own already-travelled points in RETURN order, current end first.

        No new shortcuts or safety claims are inferred from proximity. This is
        an itinerary; every executed move still needs the current sensor frame.
        Calling this method does not release a latched navigation failure.
        """
        parsed = []
        for point in points:
            p = np.asarray(point, float)
            if p.shape != (2,) or not np.isfinite(p).all():
                raise ValueError('Observed route must contain finite 2D points')
            if not parsed or np.linalg.norm(p - parsed[-1]) > .25:
                parsed.append(p.copy())
        if (len(parsed) == len(self._observed_route35)
                and all(np.array_equal(a, b) for a, b in zip(parsed, self._observed_route35))):
            return
        self._observed_route35 = parsed
        self._observed_cursor35 = 0
        self.route = []
        self._repair = []
        self._plan_tick = -10**9
        self.stats['own_route_installed'] += 1

    def remaining_observed_route(self):
        """Return the unconsumed itinerary for an explicit mission reset."""
        return [p.copy() for p in self._observed_route35[self._observed_cursor35:]]

    @staticmethod
    def _turns35(points):
        points = np.asarray(points, float)
        if len(points) < 3:
            return 0
        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        directions = vectors[lengths > .5] / lengths[lengths > .5, None]
        if len(directions) < 2:
            return 0
        return int(np.count_nonzero(np.sum(directions[:-1]*directions[1:], axis=1) < .8))

    @classmethod
    def estimate_route(cls, grid, position, target, speed=1.8, reserve_steps=120):
        result = super().estimate_route(grid, position, target, speed=speed, reserve_steps=0)
        corners = cls._turns35(result['route_points'])
        extra = int(reserve_steps) + 6*corners
        result.update(corner_count=corners, corner_reserve_steps=6*corners,
                      reserve_steps=int(reserve_steps), budget_scope='currently_known_grid_route',
                      unknown_gap=not result['exact'])
        if result['exact']:
            result['steps'] += extra
            result['travel_steps'] = result['steps']
        return result

    def estimate_observed_route(self, position, target, speed=1.8, reserve_steps=120):
        position, target = np.asarray(position, float), np.asarray(target, float)
        if (position.shape != (2,) or target.shape != (2,)
                or not np.isfinite(position).all() or not np.isfinite(target).all()
                or not math.isfinite(speed) or speed <= 0):
            raise ValueError('Observed route budget needs finite poses and positive speed')
        route = [position.copy(), *self._observed_route35[self._observed_cursor35:]]
        length = float(sum(np.linalg.norm(b-a) for a, b in zip(route, route[1:])))
        gap = float(np.linalg.norm(route[-1]-target))
        corners = self._turns35(route)
        terminal_covered = bool(self._observed_route35 and gap <= self.ARRIVAL_RADIUS)
        steps = (int(math.ceil(length/speed))+int(reserve_steps)+6*corners
                 if terminal_covered else None)
        return dict(exact=False, reachable=False, observed_route=bool(self._observed_route35),
                    terminal_covered=terminal_covered, unknown_gap=not terminal_covered,
                    length=length, path_length_px=length, steps=steps, travel_steps=steps,
                    terminal_gap_px=gap, corner_count=corners, route_points=[p.tolist() for p in route],
                    budget_scope='own_recorded_route_not_current_clearance')

    def _plan(self, grid, position, target):
        if (not self._observed_route35
                or self._observed_cursor35 >= len(self._observed_route35)):
            return super()._plan(grid, position, target)
        fine = self._last_local
        points = self._observed_route35
        cursor = self._observed_cursor35
        # A near corner is removed only if the successor can be reached by the
        # current physical hull. A 35px distance threshold alone lost corners.
        while cursor < len(points)-1:
            next_body = rotate(points[cursor+1]-position, -self._heading)
            if (np.linalg.norm(points[cursor]-position) <= self.CORNER_REACHED
                    and fine.line_clear(np.zeros(2), next_body)):
                cursor += 1
                self.stats['observed_corner_passed'] += 1
            else:
                break
        # Current visibility can prove a shortcut; historical proximity cannot.
        best = cursor
        for index in range(cursor, min(len(points), cursor+8)):
            body = rotate(points[index]-position, -self._heading)
            if np.linalg.norm(body) > self.LOOKAHEAD:
                break
            if fine.line_clear(np.zeros(2), body):
                best = index
        if best > cursor:
            self.stats['freshly_visible_own_route_shortcut'] += best-cursor
            cursor = best
        self._observed_cursor35 = cursor
        terminal_body = rotate(points[-1]-position, -self._heading)
        if (cursor == len(points)-1 and np.linalg.norm(terminal_body) <= self.CORNER_REACHED
                and fine.line_clear(np.zeros(2), terminal_body)):
            # Own travel may end at a different spawn/home point from the
            # jointly agreed rendezvous. Only after actually reaching this
            # observed terminal do we plan the remaining known-grid leg.
            # Unknown final gaps remain unknown; execution still uses the same
            # current-frame hull and stopping-sweep checks.
            self._observed_cursor35 = len(points)
            self.stats['observed_terminal_reached_grid_leg'] += 1
            return super()._plan(grid, position, target)
        self.route = [p.copy() for p in points[cursor:]]
        direct = rotate(target-position, -self._heading)
        direct_seen = (np.linalg.norm(direct) <= self.LOOKAHEAD
                       and fine.line_clear(np.zeros(2), direct))
        terminal_matches = np.linalg.norm(points[-1]-target) <= 1.
        if direct_seen:
            self.route = [target.copy()]
        self.target_reachable = bool(terminal_matches or direct_seen)
        self.inflated_target_reachable = False
        self.proxy_distance = float(np.linalg.norm(self.route[-1]-target))
        self.reachable_cells = 0
        self.planning_mode = 'own_recorded_route_fresh_hull_required'
        self._plan_tick = self.tick
        self.stats['own_route_plans'] += 1

    def _monotone_recovery34(self, fine, distances):
        # Unlike34, no requirement that every one of the 181 directions is
        # valid. The swept new footprint itself has to be observed free.
        if not all(hasattr(fine, key) for key in ('footprint','free','static','cell','indices')):
            return None
        if self._reobserve_since35 is None:
            self._reobserve_since35 = self.tick
        if self.tick-self._reobserve_since35 < self.REOBSERVE_TICKS:
            self.stats['directional_reobserve_hold'] += 1
            return None
        if np.linalg.norm(self._velocity) > 1.:
            self.stats['directional_recovery_wait_velocity'] += 1
            return None
        if self._recovery_since34 is None:
            self._recovery_since34 = self.tick
        if self.tick-self._recovery_since34 >= self.RECOVERY_LIMIT:
            self.stats['directional_recovery_budget'] += 1
            return None
        self._recovery_attempts34 += 1
        known = np.asarray(fine.free,bool) & ~np.asarray(fine.static,bool)
        kernel = np.asarray(fine.footprint(),bool)
        offsets = np.argwhere(kernel)-np.asarray(kernel.shape)//2
        initial = np.asarray(fine.cell(np.zeros(2)))+offsets
        shape = np.asarray(known.shape)
        if np.any(initial < 0) or np.any(initial >= shape):
            return None
        bad = ~known[initial[:,0],initial[:,1]]
        count = int(bad.sum())
        if not count:
            return None
        unsafe = np.zeros_like(known)
        unsafe[initial[bad,0],initial[bad,1]] = True

        def admissible(vector, strict=False):
            samples = np.linspace(np.zeros(2), vector,
                                  max(3,int(math.ceil(np.linalg.norm(vector)))+1))[1:]
            centers = fine.indices(samples)[:,::-1]
            cells = centers[:,None,:]+offsets[None,:,:]
            if np.any(cells < 0) or np.any(cells >= shape):
                return False, count
            yy,xx = cells[:,:,0],cells[:,:,1]
            occupied = ~known[yy,xx]
            totals = occupied.sum(axis=1)
            if (np.any(occupied & ~unsafe[yy,xx]) or totals[0] > count
                    or np.any(np.diff(totals) > 0)):
                return False,int(totals[-1])
            remaining = int(totals[-1])
            return (remaining < count if strict else True),remaining

        choices = []
        for angle in np.linspace(-math.pi,math.pi,24,endpoint=False):
            direction = np.array([math.cos(angle),math.sin(angle)])
            allowed, remaining = admissible(direction*10., strict=True)
            if allowed:
                choices.append((count-remaining,angle,direction))
        for _,_,direction in sorted(choices,key=lambda item:(-item[0],item[1])):
            desired = direction*.8
            force = .085*desired+.65*(desired-self._velocity)
            force /= max(1.,float(np.linalg.norm(force)))
            predicted = .95*self._velocity+.6*force
            stopping = predicted*3+predicted*np.linalg.norm(predicted)/1.2
            allowed, remaining = admissible(stopping)
            if allowed:
                self.reason = 'current_sector_monotone_retreat'
                self.stats[self.reason] += 1
                self._recovery_details34 = dict(initial_unsafe_pixels=count,
                    stopping_unsafe_pixels=remaining, stopping_vector=stopping.tolist(),
                    same_hull_kernel=True, scan_tick=self.tick,
                    partial_scan_allowed=True, new_unobserved_pixels=0)
                return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)
        self.stats['directional_recovery_no_safe_sector'] += 1
        return None

    def command(self, position, heading, velocity, lidar, angles, semantic, grid, target, tick):
        self.tick = int(tick)
        position_valid = (position is not None and np.shape(position)==(2,)
                          and np.isfinite(position).all())
        if velocity is not None and np.shape(velocity)==(2,) and np.isfinite(velocity).all():
            self._velocity = np.asarray(velocity,float).copy()
        if self.failed:
            # Failure belongs to this mission episode, not rounded target
            # coordinates. Controller must explicitly reset on a new plan.
            return self._brake('rendezvous_navigation_failed')
        if (self._last_call34 is None or self.tick-self._last_call34 > 5
                or self._progress_anchor34 is None):
            self._progress_anchor34 = np.asarray(position,float).copy() if position_valid else None
            self._progress_tick34 = self.tick
        self._last_call34 = self.tick
        self._last_local = None
        self._scan_status35 = dict(ray_count=0,valid_ray_count=0,long_ray_count=0,
                                  scan_tick=self.tick,partial=True)
        if lidar is not None:
            distances = np.asarray(lidar,float).ravel()
            valid = np.isfinite(distances) & (distances > 3)
            self._scan_status35.update(ray_count=len(distances),valid_ray_count=int(valid.sum()),
                long_ray_count=int(np.count_nonzero(np.isfinite(distances) & (distances > 25))),
                partial=not bool(valid.all()))
        result = RelayNavigator33.command(self,position,heading,velocity,lidar,angles,
                                          semantic,grid,target,tick)
        blocked = self.reason in ('current_hull_not_observed_clear','scan_unavailable',
            'no_fresh_hull_verified_segment','stopping_sweep_blocked','pose_unavailable',
            'velocity_unavailable','target_unavailable')
        if self.reason=='current_hull_not_observed_clear' and self._last_local is not None:
            recovered = self._monotone_recovery34(self._last_local,distances)
            if recovered is not None:
                result = recovered
        elif self.reason not in ('scan_unavailable','pose_unavailable','velocity_unavailable'):
            self._reobserve_since35 = None
            self._recovery_since34 = None
        moved = False
        if position_valid and self._progress_anchor34 is not None:
            moved = np.linalg.norm(np.asarray(position)-self._progress_anchor34) >= self.PROGRESS_DISTANCE
            if moved:
                self._progress_anchor34 = np.asarray(position,float).copy()
                self._progress_tick34 = self.tick
            self.no_progress_ticks = self.tick-self._progress_tick34
        self.consecutive_blocked_ticks = self.consecutive_blocked_ticks+1 if blocked and not moved else 0
        if self.arrived:
            self.no_progress_ticks = self.consecutive_blocked_ticks = 0
            self._progress_tick34 = self.tick
            self._progress_anchor34 = np.asarray(position,float).copy()
        elif self.consecutive_blocked_ticks >= self.BLOCKED_LIMIT:
            return self._fail34('blocked_budget')
        elif self.no_progress_ticks >= self.STAGNATION_LIMIT:
            return self._fail34('no_displacement_budget')
        return result

    step = command

    def status(self):
        value = super().status()
        value.update(scan_status=dict(self._scan_status35),
            observed_route_points=len(self._observed_route35),observed_route_cursor=self._observed_cursor35,
            observed_route_remaining=max(0,len(self._observed_route35)-self._observed_cursor35),
            failure_reset_policy='explicit_reset_for_new_mission_not_target_jitter')
        return value


# Short public name retained for lightweight navigation-only callers/tests.
Navigator35 = RendezvousNavigator35
