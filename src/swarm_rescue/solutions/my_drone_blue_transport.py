"""B3: bounded routing and contact delivery, isolated from the B0 reference.

Only legal sensor/history/message data are used online. No map/layout files,
true poses, bomb internals or hidden disposal counters are read here.
SciPy 1.15's BSD-licensed sparse shortest-path API is used, not vendored code.
"""
import json
import math
import time
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from swarm_rescue.solutions.my_drone_eval_blue import MyDroneBlueAdapted, RescueGrid, wrap


class TransportGrid(RescueGrid):
    """Native shortest paths on observed free space, with explicit partial ends."""
    def __init__(self, size):
        super().__init__(size)
        rr, cc = np.indices(self.shape)
        self._points = np.column_stack(((cc.ravel()+.5)*self.CELL-self.width/2,
                                       self.height/2-(rr.ravel()+.5)*self.CELL))
        self.last_complete = False
        self.last_endpoint = None
        self.last_route_length = 0.
        self.route_seconds = 0.
        self.route_calls = 0
        self.payload = False
        self.avoid = []

    def _shortest(self, allowed, position):
        allowed = allowed.copy()
        start = np.ravel_multi_index(self.cell(position), self.shape)
        allowed.ravel()[start] = True
        h, w = self.shape
        index = np.arange(h*w).reshape(h, w)
        vertical = allowed[:-1] & allowed[1:]
        horizontal = allowed[:, :-1] & allowed[:, 1:]
        a = np.r_[index[:-1][vertical], index[:, :-1][horizontal]]
        b = np.r_[index[1:][vertical], index[:, 1:][horizontal]]
        graph = csr_matrix((np.ones(2*len(a)), (np.r_[a,b],np.r_[b,a])), shape=(h*w,h*w))
        lengths, parents = dijkstra(graph, directed=False, indices=int(start),
                                    unweighted=True, return_predecessors=True)
        return int(start), lengths, parents

    def _narrow_layer(self):
        # This is a fallback on observed free cells, never unknown space.
        free, walls = self.free.copy(), self.evidence >= 2
        for _, (other_free, other_walls, _) in self.remote.values():
            free |= other_free
            walls |= other_walls & ~(self.free & (self.evidence < .5))
        distance = np.minimum(cv2.distanceTransform((~walls).astype(np.uint8), cv2.DIST_L2, 3),
                              max(self.shape))*self.CELL
        return free & (distance >= (22 if self.payload else 16))

    def route(self, position, target=None, identifier=0, number=1, claims=()):
        started = time.perf_counter()
        allowed, seen = self.layers()
        start, lengths, parents = self._shortest(allowed, position)
        if target is not None:
            target = np.asarray(target)
            target_distance = np.linalg.norm(self._points-target, axis=1)
            if np.min(target_distance[np.isfinite(lengths)]) > 64:
                ns, nl, npred = self._shortest(self._narrow_layer(), position)
                if np.min(target_distance[np.isfinite(nl)])+16 < np.min(target_distance[np.isfinite(lengths)]):
                    start, lengths, parents = ns, nl, npred
        finite = np.isfinite(lengths)
        cells = np.flatnonzero(finite)
        points = self._points[cells]
        if target is not None:
            values = -np.linalg.norm(points-target, axis=1)-.15*lengths[cells]
            for point in self.avoid:
                values -= 180*np.maximum(0,1-np.linalg.norm(points-point,axis=1)/64)
        else:
            unknown = ~self.free & ~(self.evidence >= 2)
            for _, (remote_free, remote_walls, _) in self.remote.values():
                unknown &= ~remote_free & ~remote_walls
            novelty = cv2.boxFilter(unknown.astype(np.float32), -1, (5,5), normalize=False,
                                   borderType=cv2.BORDER_CONSTANT).ravel()
            delta = points-position
            values = 5*novelty[cells]-.4*lengths[cells]-4*self.visits.ravel()[cells]
            values += 15*(~seen.ravel()[cells])+12*np.cos(np.arctan2(delta[:,1],delta[:,0])-identifier*2*math.pi/max(1,number))
            for point in claims:
                values -= 65*np.maximum(0,1-np.linalg.norm(points-point,axis=1)/160)
            values[lengths[cells]<3] = -1e6
        end = int(cells[np.argmax(values)])
        self.last_endpoint = self._points[end].copy()
        self.last_complete = bool(target is not None and np.linalg.norm(self.last_endpoint-target)<48)
        self.last_route_length = float(lengths[end]*self.CELL)
        path=[]; node=end
        while node != start and node >= 0:
            path.append(self._points[node].copy()); node=int(parents[node])
        self.route_calls += 1
        self.route_seconds += time.perf_counter()-started
        return list(reversed(path))


def terminal_command(hit, distances, angles, turn, direction):
    """Surface-relative approach; keep possession until an actual angle sweep."""
    distance, angle = float(hit.distance), float(hit.angle)
    cmd=dict(forward=0., lateral=0., rotation=float(np.clip(angle*1.5,-.65,.65)), grasper=1)
    if distance >= 35:
        cone=np.abs((angles-angle+math.pi)%(2*math.pi)-math.pi)<.25
        blockers=distances[cone & np.isfinite(distances) & (distances>0)]
        if len(blockers) and blockers.min() >= distance-12:
            speed=min(.32,max(.12,(distance-20)/200))
            cmd.update(forward=float(speed*math.cos(angle)),lateral=float(speed*math.sin(angle)))
    else:
        cmd['rotation']=.65*direction
        # Close the last small gap while the held object sweeps the surface.
        if distance>23:
            cmd.update(forward=.12*math.cos(angle),lateral=.12*math.sin(angle))
        if distance<25 and turn >= math.pi:
            cmd['grasper']=0
    return cmd


class MyDroneBlueTransport(MyDroneBlueAdapted):
    """Keep B0's sensing/search; isolate planning and delivery changes."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grid=TransportGrid(self.size_area)
        self._terminal_turn=0.
        self._terminal_active=False
        self._carry_started=None
        self._carried_previous=False
        self._release_requested=False
        self._partial_end=None
        self._partial_since=0
        self._avoid_until=0
        self._control_seconds=0.
        self._control_calls=0
        self._last_diagnostic=-1000

    def _navigate(self, goal, distances, angles, carrying=False):
        self._grid.payload=bool(carrying)
        if carrying and goal is not None and self._tick-self._route_step>=24:
            if self._tick>=self._avoid_until:self._grid.avoid=[]
            self._route=self._grid.route(self._position,goal,int(self.identifier),self._number,())
            self._route_step=self._tick
            end=self._grid.last_endpoint
            if not self._grid.last_complete and end is not None:
                if self._partial_end is None or np.linalg.norm(end-self._partial_end)>48:
                    self._partial_end=end.copy();self._partial_since=self._tick
                elif self._tick-self._partial_since>180 and np.linalg.norm(end-self._position)<64:
                    self._grid.avoid=[end.copy()];self._avoid_until=self._tick+180
                    self._route=self._grid.route(self._position,goal,int(self.identifier),self._number,())
                    self._partial_since=self._tick
                if not self._route:
                    self._route=self._grid.route(self._position,None,int(self.identifier),self._number,())
            else:self._partial_end=None
        nav_goal=goal if not carrying or self._route else None
        cmd=super()._navigate(nav_goal,distances,angles,carrying)
        if carrying and goal is not None:self._target=np.asarray(goal).copy()
        return cmd

    def control(self):
        started=time.perf_counter()
        cmd=super().control()
        carrying=bool(self.grasped_bombs())
        if carrying and not self._carried_previous:
            self._carry_started=self._tick;self._event('pickup_observed')
        elif not carrying and self._carried_previous:
            self._event('grasp_ended',released=bool(self._release_requested),
                        duration=int(self._tick-(self._carry_started or self._tick)))
            self._release_requested=False
        hit=self._center_hit
        if carrying and hit is not None and np.isfinite([hit.distance,hit.angle]).all() and hit.distance<125:
            if hit.distance<35:
                odo=self.odometer_values()
                if self._terminal_active and odo is not None and np.shape(odo)==(3,) and np.isfinite(odo).all():
                    self._terminal_turn=max(0.,self._terminal_turn+float(np.clip(odo[2],-.4,.4))*self._escape_sign)
                self._terminal_active=True
            else:self._terminal_active=False;self._terminal_turn=0.
            distances=np.asarray(self.lidar_values() if self.lidar_values() is not None else [],float)
            angles=np.asarray(self.lidar_rays_angles(),float)[:len(distances)]
            cmd=terminal_command(hit,distances,angles,self._terminal_turn,self._escape_sign)
            if cmd['grasper']==0 and not self._release_requested:
                self._release_requested=True;self._event('release_requested',angle=float(self._terminal_turn))
        else:
            self._terminal_active=False;self._terminal_turn=0.
            # A recovery or inherited tick counter must never drop a payload.
            if carrying:cmd['grasper']=1
        self._carried_previous=carrying
        self._control_calls+=1;self._control_seconds+=time.perf_counter()-started
        if self._tick-self._last_diagnostic>=240:
            self._last_diagnostic=self._tick
            self._event('state',mode=self._mode,carrying=carrying,route_points=len(self._route),
                partial=not self._grid.last_complete,health=int(self.drone_health),
                center_distance=float(hit.distance) if hit is not None else None,
                control_seconds=self._control_seconds,route_seconds=self._grid.route_seconds,
                calls=self._control_calls,walltime=float(self.elapsed_walltime))
        return cmd

    def _event(self,event,**data):
        print('B3_EVENT '+json.dumps(dict(event=event,drone=int(self.identifier),step=int(self._tick),**data)))
