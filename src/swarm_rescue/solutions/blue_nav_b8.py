"""B8 independent navigation primitives. Only supplied observations, never truth.

Units are pixels / playground step (one step contains ten physics substeps).
The full-swing payload disk is conservative, not a measured bomb position.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


BODY_RADIUS = math.hypot(13, 10)  # audited DroneBase hull circumradius
BOMB_RADIUS = 12.0  # native bomb texture hull fits this circle
# Grasper expands the hull by 5. A pivot at .7 of the centre separation
# preserves anchor lengths .7*d + .3*d, though neither body orientation is known.
PAYLOAD_RADIUS = BODY_RADIUS + 5 + 2 * BOMB_RADIUS


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def body_velocity(odometry):
    """Previous displacement expressed in the CURRENT body frame; no compass."""
    if odometry is None or np.shape(odometry) != (3,) or not np.isfinite(odometry).all():
        return None
    distance, alpha, theta = odometry
    if abs(distance) > 40 or abs(theta) > 1:
        return None
    return max(0., float(distance)) * np.array([math.cos(alpha-theta), math.sin(alpha-theta)])


def stopping_speed(room, deceleration, latency=3.):
    room = max(0., float(room))
    return math.sqrt((deceleration*latency)**2 + 2*deceleration*room) - deceleration*latency


def scan_geometry(distances, angles, bearing, radius, margin=7.):
    """Observed capsule clearance and visibility, including missing angular rays.

    Existing overlaps of a conservative disk do not by themselves prove physical
    contact. Egress is permitted only if their distances do not decrease, with
    fixed attitude and slow motion. All NEW overlaps remain forbidden.
    """
    d, a = np.asarray(distances, float).ravel(), np.asarray(angles, float).ravel()
    if d.shape != a.shape or len(d) < 8:
        return dict(room=0., minimum=0., visible=False, egress=False, sideways=0.)
    valid = np.isfinite(d) & np.isfinite(a) & (d > 0)
    relative = wrap(a-bearing)
    cone = np.abs(relative) < .28
    visible = bool(cone.sum() >= 3 and np.all(valid[cone]))
    if not visible or valid.sum() < 8:
        return dict(room=0., minimum=0., visible=False, egress=False, sideways=0.)
    d = np.minimum(d[valid], 230.); relative = relative[valid]
    x, y = d*np.cos(relative), d*np.sin(relative)
    effective = radius + margin
    intrusion = d < effective
    # A wall ahead or beside the disk prevents a monotone egress. A wall behind
    # can recede without rotating the unobserved articulated payload into it.
    egress = bool(intrusion.any() and np.all(x[intrusion] < -1.))
    ahead = (x > 0) & (np.abs(y) < effective)
    contact = x[ahead] - np.sqrt(np.maximum(0., effective**2-y[ahead]**2))
    room = min(float(np.min(contact)) if contact.size else 230.-effective,
               float(np.min(d[np.abs(relative)<.28]))-effective)
    if intrusion.any() and not egress:
        room = 0.
    sideways = float(np.min(d[np.abs(np.sin(relative))>.8])) if np.any(np.abs(np.sin(relative))>.8) else 0.
    return dict(room=max(0., room), minimum=float(d.min()), visible=visible,
                egress=egress, sideways=sideways)


class SpeedGovernor:
    """Closed-loop speed demand, bounded acceleration, velocity-aware braking.

    .6/.2 are approximate force impulses per empty/loaded step, because the
    stock force is applied once before ten substeps. Tests do not certify a
    universal stopping guarantee in collisions or uncertain sensor geometry.
    """
    def __init__(self):
        self.velocity = np.zeros(2)
        self.ready = False
        self.reference = 0.
        self.last = {}

    def update(self, command, distances, angles, odometry, *, carrying=False,
               turn=0., waypoint_distance=None, dynamic=False, pose_uncertain=False):
        cmd = dict(command)
        measured = body_velocity(odometry)
        if measured is None:
            self.reference = 0.
            cmd.update(forward=0., lateral=0.)
            self.last = dict(reason='missing_odometry_coast', target_speed=0., velocity=None)
            return cmd
        self.velocity = measured.copy() if not self.ready else .55*measured+.45*self.velocity
        self.ready = True
        wanted = np.array([cmd.get('forward', 0.), cmd.get('lateral', 0.)])
        magnitude = float(np.linalg.norm(wanted))
        direction = wanted / magnitude if magnitude > .001 else np.zeros(2)
        angle = math.atan2(direction[1], direction[0]) if magnitude else 0.
        radius = PAYLOAD_RADIUS if carrying else BODY_RADIUS
        geo = scan_geometry(distances, angles, angle, radius)
        decel, impulse = (.10, .20) if carrying else (.35, .60)
        vmax = 3.05 if carrying else 9.2
        safe = stopping_speed(geo['room'], decel)
        turn_size = max(abs(turn), abs(float(cmd.get('rotation', 0.)))*.8)
        if turn_size > .18:
            safe = min(safe, max(.55 if carrying else 1.1, vmax/(1+5*turn_size)))
        # Slow for narrow sides/intersections and unknown near lookahead, rather
        # than use a constant high throttle everywhere.
        if geo['sideways'] < radius+24:
            safe = min(safe, 1.15 if carrying else 3.)
        if waypoint_distance is not None:
            safe = min(safe, stopping_speed(max(0., waypoint_distance-8.), decel)+.4)
        if dynamic:
            safe = min(safe, .85 if carrying else 2.0)
        if pose_uncertain:
            safe = min(safe, 1.4 if carrying else 3.5)
        if geo['egress']:
            safe = min(safe, .55)
            cmd['rotation'] = 0.
        if carrying and geo['minimum'] < radius+7:
            cmd['rotation'] = 0.  # do not swing an unseen bomb in tight space
        desired = min(vmax, safe) if magnitude > .001 else 0.
        self.reference = min(desired, self.reference + (.14 if carrying else .35))
        target = direction*self.reference
        # Feed-forward drag and proportional velocity feedback. Actual measured
        # speed can trigger immediate braking even while the reference ramps up.
        force = .05*target/impulse + (.48 if carrying else .30)*(target-self.velocity)
        norm = float(np.linalg.norm(force))
        if norm > 1: force /= norm
        braking = float(np.dot(force, self.velocity)) < 0
        # Do not command translation into an unobserved reverse sector. Brake
        # against existing momentum only; after nearly stopped, coast instead.
        if braking and np.linalg.norm(self.velocity) < .18:
            force[:] = 0.
        cmd.update(forward=float(force[0]), lateral=float(force[1]))
        self.last = dict(reason='brake' if braking else 'egress' if geo['egress'] else 'regulated',
                         target_speed=self.reference, velocity=self.velocity.tolist(),
                         speed=float(np.linalg.norm(self.velocity)), desired_speed=desired,
                         room=geo['room'], radius=radius, dynamic=bool(dynamic),
                         turn=float(turn_size), scan_visible=geo['visible'])
        return cmd


@dataclass
class RoutePlan:
    points: list
    complete: bool
    reason: str
    length: float = 0.
    estimated_steps: float = math.inf
    min_clearance: float = 0.
    choice: str = 'none'


class PayloadPlanner:
    """Observed-space routes; compare shortest and clearance/time candidates.

    No path ever means 'drive directly at the goal'. Unknown space is blocked.
    The centre itself is solid: route to a reachable observed approach within
    105px, then leave the existing relative-observation terminal controller intact.
    """
    def __init__(self, grid):
        self.grid = grid
        self.allowed = None
        self.clearance = None
        self.last = RoutePlan([], False, 'not_planned')

    def layers(self):
        g = self.grid
        free, walls = g.free.copy(), g.evidence >= 2
        for _, (f, w, _) in g.remote.values():
            free |= f
            walls |= w & ~(g.free & (g.evidence < .5))
        known = free & ~walls
        self.known = known
        # Account for the cell footprint, not just obstacle-cell centres.
        clearance = distance_transform_edt(np.pad(known, 1))[1:-1, 1:-1]*g.CELL-g.CELL/math.sqrt(2)
        self.clearance = clearance
        self.allowed = known & (clearance >= PAYLOAD_RADIUS+3.)
        return self.allowed

    def segment(self, start, end, *, egress=False):
        if self.allowed is None: self.layers()
        distance = float(np.linalg.norm(np.asarray(end)-start))
        points = np.linspace(start, end, max(2, int(distance/5)+2))
        cells = [self.grid.cell(p) for p in points]
        if all(self.allowed[c] for c in cells): return True
        if not egress: return False
        # A short initial extraction can use previously observed free cells,
        # only along non-decreasing clearance, terminating in full-envelope space.
        clear = np.array([self.clearance[c] for c in cells])
        return bool(distance <= 80 and self.allowed[cells[-1]] and clear.min()>0
                    and np.all(np.diff(clear) >= -.01))

    def _graph(self, allowed, weighted):
        h, w = allowed.shape; n = h*w
        idx = np.arange(n); r, c = idx//w, idx%w
        neighbours = idx[:, None]+np.array([w, -w, 1, -1])
        valid = np.column_stack((r<h-1, r>0, c<w-1, c>0))
        valid &= allowed.ravel()[:, None] & allowed.ravel()[np.clip(neighbours, 0, n-1)]
        starts = np.broadcast_to(idx[:, None], valid.shape)[valid]
        ends = neighbours[valid]
        clear = np.minimum(self.clearance.ravel()[starts], self.clearance.ravel()[ends])
        speed = np.clip((clear-PAYLOAD_RADIUS)/16, .65, 3.05)
        costs = self.grid.CELL/speed if weighted else np.full(len(ends), self.grid.CELL)
        return csr_matrix((costs, (starts, ends)), shape=(n, n))

    def _evaluate(self, points):
        path = np.asarray(points)
        if len(path)<2: return 0., 0., float(self.clearance[self.grid.cell(path[0])])
        vectors = np.diff(path, axis=0); lengths = np.linalg.norm(vectors, axis=1)
        clear = np.array([self.clearance[self.grid.cell(p)] for p in path])
        speeds = np.clip((np.minimum(clear[:-1], clear[1:])-PAYLOAD_RADIUS)/16, .65, 3.05)
        headings = np.arctan2(vectors[:, 1], vectors[:, 0])
        turns = np.abs(wrap(np.diff(headings)))
        seconds = np.sum(lengths/speeds) + 12*np.sum(turns/(math.pi/2))
        return float(np.sum(lengths)), float(seconds), float(clear.min())

    def plan(self, position, goal):
        allowed = self.layers(); g = self.grid; start = g.cell(position)
        if goal is None:
            self.last = RoutePlan([], False, 'unknown_center'); return self.last
        if not allowed[start]:
            cells = np.argwhere(allowed)
            points = [g.point(tuple(c)) for c in cells]
            candidates = sorted((p for p in points if np.linalg.norm(p-position)<=80),
                                key=lambda p: np.linalg.norm(p-position))[:16]
            for p in candidates:
                if self.segment(position, p, egress=True):
                    self.last=RoutePlan([p], False, 'observed_clearance_egress', float(np.linalg.norm(p-position)))
                    return self.last
            self.last=RoutePlan([], False, 'payload_start_not_clear'); return self.last
        coords=np.argwhere(allowed)
        if not len(coords):
            self.last=RoutePlan([], False, 'no_payload_space'); return self.last
        points=np.array([g.point(tuple(c)) for c in coords])
        # Centre approach is provisional, not disposal confirmation or proof of
        # semantic line of sight. If centre is still occluded, stop and replan.
        goal_mask=np.linalg.norm(points-goal, axis=1)<=100
        # Do not choose a spatially close approach behind a known wall. The
        # final 24px may be the physical centre itself; all earlier ray samples
        # must be observed free. This is visibility support, not proof of pickup
        # alignment, contact success, or the absence of localization drift.
        for index in np.flatnonzero(goal_mask):
            p=points[index];length=float(np.linalg.norm(goal-p))
            samples=np.linspace(p,goal,max(2,int(length/5)+2))
            goal_mask[index]=all(self.known[g.cell(q)] for q in samples if np.linalg.norm(q-goal)>24)
        goals=np.ravel_multi_index(coords[goal_mask].T, g.shape)
        source=np.ravel_multi_index(start, g.shape); candidates=[]
        for weighted in (False, True):
            distance, parents=dijkstra(self._graph(allowed, weighted), directed=True,
                                       indices=source, return_predecessors=True)
            reachable=goals[np.isfinite(distance[goals])]
            if not len(reachable): continue
            endpoint=int(reachable[np.argmin(distance[reachable])]); path=[]; node=endpoint
            while node != source:
                path.append(g.point(np.unravel_index(node, g.shape)))
                node=int(parents[node])
                if node < 0: raise RuntimeError('Invalid predecessor chain')
            path=list(reversed(path)); length, cost, clearance=self._evaluate([position]+path)
            candidates.append(RoutePlan(path, True, 'observed_approach_route', length, cost, clearance,
                                        'clearance_time' if weighted else 'distance'))
        if candidates:
            self.last=min(candidates, key=lambda p:p.estimated_steps)
        else:
            self.last=RoutePlan([], False, 'return_disconnected')
        return self.last
