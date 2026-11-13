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
from scipy.sparse.csgraph import breadth_first_order
from swarm_rescue.solutions.my_drone_eval_blue import MyDroneBlueAdapted, RescueGrid, wrap
from swarm_rescue.solutions.my_drone_eval_blue import LIDAR_RANGE


class SteeringTable:
    """B0's exact angular choices with cached ray/candidate trigonometry."""
    def __init__(self):
        self.angles=None
        self.candidates=np.linspace(-math.pi,math.pi,49,endpoint=False)

    def choose(self,distances,angles,desired,carrying=False):
        valid=np.isfinite(distances)&np.isfinite(angles)&(distances>0)
        d,a=np.minimum(distances[valid],LIDAR_RANGE),angles[valid]
        if not len(d):return None
        if self.angles is None or not np.array_equal(a,self.angles):
            self.angles=a.copy()
            relative=(a[None,:]-self.candidates[:,None]+math.pi)%(2*math.pi)-math.pi
            self.cosine=np.cos(relative);self.sine=np.sin(relative);self.cone=np.abs(relative)<.65
        clearance=43 if carrying else 31
        longitudinal=d[None,:]*self.cosine
        corridor=self.cone&(np.abs(d[None,:]*self.sine)<clearance)
        gaps=np.min(np.where(corridor,longitudinal,np.inf),axis=1)
        gaps[~np.isfinite(gaps)]=LIDAR_RANGE
        scores=-2.8*np.abs((self.candidates-desired+math.pi)%(2*math.pi)-math.pi)
        scores+=np.minimum(gaps,140)/100-8*(gaps<clearance)
        best=int(np.argmax(scores))
        return float(self.candidates[best]),float(gaps[best])


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
        self.experimental_fallback = False

    def _shortest(self, allowed, position):
        allowed = allowed.copy()
        start = np.ravel_multi_index(self.cell(position), self.shape)
        allowed.ravel()[start] = True
        h, w = self.shape; size=h*w
        index=np.arange(size);r,c=index//w,index%w
        adjacent=index[:,None]+np.array([w,-w,1,-1])
        valid=np.column_stack((r+1<h,r>0,c+1<w,c>0))
        valid &= allowed.ravel()[:,None] & allowed.ravel()[np.clip(adjacent,0,size-1)]
        # Preserve B0's down/up/right/left CSR neighbor order. Sorting the CSR
        # columns changes tie-breaking and can change the entire exploration.
        indptr=np.r_[0,np.cumsum(valid.sum(axis=1))].astype(np.int32)
        graph=csr_matrix((np.ones(int(valid.sum())),adjacent[valid].astype(np.int32),indptr),shape=(size,size))
        order,parents=breadth_first_order(graph,int(start),directed=True)
        lengths=np.full(size,np.inf);lengths[start]=0
        for node in order[1:]:lengths[node]=lengths[parents[node]]+1
        self._bfs_order=order
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
        if target is not None and self.experimental_fallback:
            target = np.asarray(target)
            target_distance = np.linalg.norm(self._points-target, axis=1)
            if np.min(target_distance[np.isfinite(lengths)]) > 64:
                original_order=self._bfs_order
                ns, nl, npred = self._shortest(self._narrow_layer(), position)
                if np.min(target_distance[np.isfinite(nl)])+16 < np.min(target_distance[np.isfinite(lengths)]):
                    start, lengths, parents = ns, nl, npred
                else:self._bfs_order=original_order
        finite = np.isfinite(lengths)
        cells = self._bfs_order
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


def terminal_command(hit, distances, angles, turn, direction, fallback=None):
    """Surface-relative approach; keep possession until an actual angle sweep."""
    distance, angle = float(hit.distance), float(hit.angle)
    cmd=dict(forward=0., lateral=0., rotation=float(np.clip(angle*1.5,-.65,.65)), grasper=1)
    if distance >= 35:
        cone=np.abs((angles-angle+math.pi)%(2*math.pi)-math.pi)<.25
        blockers=distances[cone & np.isfinite(distances) & (distances>0)]
        if len(blockers) and blockers.min() >= distance-12:
            speed=min(.32,max(.12,(distance-20)/200))
            cmd.update(forward=float(speed*math.cos(angle)),lateral=float(speed*math.sin(angle)))
        elif fallback is not None:
            # A visible center behind another obstacle is not a direct path.
            # Preserve the obstacle-avoiding action instead of freezing here.
            return dict(fallback,grasper=1)
    else:
        cmd['rotation']=.65*direction
        # Close the last small gap while the held object sweeps the surface.
        if distance>23:
            cmd.update(forward=.12*math.cos(angle),lateral=.12*math.sin(angle))
        if distance<25 and turn >= math.pi:
            cmd['grasper']=0
    return cmd


def trace_waypoint(trace, cursor, position, heading, distances, angles):
    """Earlier visited node, admitted only by current measured corridor clearance."""
    if not trace or cursor is None:return None
    for i in range(max(0,cursor-100),min(cursor+1,len(trace))):
        delta=trace[i]-position;length=float(np.linalg.norm(delta))
        if not 32<length<140:continue
        bearing=wrap(math.atan2(delta[1],delta[0])-heading)
        relative=(angles-bearing+math.pi)%(2*math.pi)-math.pi
        corridor=(np.abs(relative)<.65)&(np.abs(distances*np.sin(relative))<28)
        forward=distances*np.cos(relative)
        valid=corridor & np.isfinite(forward) & (distances>0)
        if valid.any() and float(forward[valid].min())>length+24:
            return i,trace[i].copy()
    return None


class MyDroneBlueTransport(MyDroneBlueAdapted):
    """Keep B0's sensing/search; isolate planning and delivery changes."""
    PAYLOAD_RECOVERY=False  # v1/v2 failed full checks; retain only as explicit experiments
    TERMINAL_CONTROL=True
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grid=TransportGrid(self.size_area)
        self._grid.experimental_fallback=self.PAYLOAD_RECOVERY
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
        self._last_navigation_cmd=dict(forward=0.,lateral=0.,rotation=0.,grasper=0)
        self._trace=[]
        self._return_trace=None
        self._trace_until=0
        self._steering=SteeringTable()

    def _sense(self,distances,angles):
        # WALL observations need no semantic world point: lidar already maps
        # them. Preserve ordered bomb/center updates and the exact exclusion.
        hits=[];self._center_hit=None
        dynamic=np.zeros(len(distances),bool)
        for obj in self.semantic_values() or ():
            kind=getattr(obj.entity_type,'name','')
            if kind not in ('BOMB','DRONE','DISPOSAL_CENTER'):continue
            if not math.isfinite(obj.distance) or not math.isfinite(obj.angle) or obj.distance<=0:continue
            delta=np.abs((angles-obj.angle+math.pi)%(2*math.pi)-math.pi)
            dynamic |= (delta<math.atan2(24,max(1,obj.distance)))&(np.abs(distances-obj.distance)<35)
            if kind=='DRONE':continue
            bearing=self._heading+obj.angle
            point=self._position+obj.distance*np.array([math.cos(bearing),math.sin(bearing)])
            if kind=='BOMB':
                self._record_bomb(point,self._tick,not obj.grasped,create=not obj.grasped)
                if not obj.grasped:hits.append((obj,point))
            else:
                if self._center_hit is None or obj.distance<self._center_hit.distance:self._center_hit=obj
                if self._center is None:self._center=point
                else:self._center+=.15*(point-self._center)
                self._center_step=self._tick
        if self._tick%3==0:self._grid.observe(self._position,self._heading,distances,angles,dynamic)
        if self._tick%18==int(self.identifier)%18 or self._snapshot is None:self._snapshot=(self._tick,self._grid.snapshot())
        return hits

    def _navigate(self, goal, distances, angles, carrying=False):
        self._grid.payload=bool(carrying)
        if self.PAYLOAD_RECOVERY and carrying and goal is not None and self._tick-self._route_step>=24:
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
                    self._trace_until=self._tick+180
                if not self._route:
                    self._route=self._grid.route(self._position,None,int(self.identifier),self._number,())
            else:self._partial_end=None
            if self._tick<self._trace_until:
                recovery=trace_waypoint(self._trace,self._return_trace,self._position,self._heading,distances,angles)
                if recovery is not None:
                    self._return_trace,point=recovery;self._route=[point]
        nav_goal=goal if not self.PAYLOAD_RECOVERY or not carrying or self._route else None
        cmd=dict(forward=0.,lateral=0.,rotation=0.,grasper=int(carrying))
        if self._tick-self._route_step>=24:
            claims=[np.asarray(p[3]) for p in self._peers.values() if p[3] is not None and p[2]=='search']
            self._route=self._grid.route(self._position,nav_goal,int(self.identifier),self._number,claims)
            self._route_step=self._tick
        while self._route and np.linalg.norm(self._route[0]-self._position)<20:self._route.pop(0)
        self._route=self._grid.shortcut(self._position,self._route)
        if nav_goal is None:self._target=self._route[-1].copy() if self._route else None
        waypoint=self._route[0] if self._route else nav_goal
        if waypoint is None:desired=wrap(int(self.identifier)*2.399+self._tick/100-self._heading)
        else:
            delta=waypoint-self._position
            desired=wrap(math.atan2(delta[1],delta[0])-self._heading)
        choice=self._steering.choose(distances,angles,desired,carrying)
        if choice is not None:
            angle,gap=choice;clearance=43 if carrying else 31
            cmd['rotation']=float(np.clip(angle*1.5,-.85,.85))
            speed=(.62 if carrying else .72)*np.clip((gap-clearance)/65,0,1)
            if abs(angle)>1.2:speed=0.
            cmd['forward']=float(speed*math.cos(angle));cmd['lateral']=float(speed*math.sin(angle))
            if self._anchor is None or np.linalg.norm(self._position-self._anchor)>24:
                self._anchor,self._anchor_step=self._position.copy(),self._tick
            elif self._tick-self._anchor_step>85:
                self._escape_until=self._tick+24;self._escape_sign=1 if int(self.identifier)%2==0 else -1
                self._anchor_step=self._tick;self._route=[];self._search_goal=None
                self._grid.visits[self._grid.cell(self._position)]+=30
            if self._tick<self._escape_until:
                valid=np.isfinite(distances)&np.isfinite(angles)&(distances>0)
                d,a=distances[valid],angles[valid];rear=d[np.abs(a)>2.5]
                cmd.update(forward=-.3 if len(rear) and rear.min()>55 else 0.,lateral=0.,rotation=.65*self._escape_sign)
        if carrying and goal is not None:self._target=np.asarray(goal).copy()
        self._last_navigation_cmd=cmd.copy()
        return cmd

    def control(self):
        started=time.perf_counter()
        cmd=super().control()
        carrying=bool(self.grasped_bombs())
        if self.PAYLOAD_RECOVERY and self._position is not None and (not self._trace or np.linalg.norm(self._position-self._trace[-1])>24):
            self._trace.append(self._position.copy());self._trace=self._trace[-500:]
        if carrying and not self._carried_previous:
            self._carry_started=self._tick;self._event('pickup_observed')
            self._return_trace=len(self._trace)-1;self._trace_until=0
        elif not carrying and self._carried_previous:
            self._event('grasp_ended',released=bool(self._release_requested),
                        duration=int(self._tick-(self._carry_started or self._tick)))
            self._release_requested=False
        hit=self._center_hit
        if self.TERMINAL_CONTROL and carrying and hit is not None and np.isfinite([hit.distance,hit.angle]).all() and hit.distance<105:
            if hit.distance<35:
                odo=self.odometer_values()
                if self._terminal_active and odo is not None and np.shape(odo)==(3,) and np.isfinite(odo).all():
                    self._terminal_turn=max(0.,self._terminal_turn+float(np.clip(odo[2],-.4,.4))*self._escape_sign)
                self._terminal_active=True
            else:self._terminal_active=False;self._terminal_turn=0.
            distances=np.asarray(self.lidar_values() if self.lidar_values() is not None else [],float)
            angles=np.asarray(self.lidar_rays_angles(),float)[:len(distances)]
            cmd=terminal_command(hit,distances,angles,self._terminal_turn,self._escape_sign,self._last_navigation_cmd)
            if cmd['grasper']==0 and not self._release_requested:
                self._release_requested=True;self._event('release_requested',angle=float(self._terminal_turn))
        else:
            self._terminal_active=False;self._terminal_turn=0.
            # A recovery or inherited tick counter must never drop a payload.
            if self.TERMINAL_CONTROL and carrying:cmd['grasper']=1
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


class MyDroneBlueTransportCompute(MyDroneBlueTransport):
    """B0 policy with equivalent computation only; terminal control ablation."""
    TERMINAL_CONTROL=False
