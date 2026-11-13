"""B2: isolated regional candidate from Adapted, not the failed enhanced bundle."""
import math
import json
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_eval_blue import MyDroneBlueAdapted,RescueGrid,owner_for
from swarm_rescue.solutions.semifinal_localization import PoseGuard,scan_points,wrap


class TrustedRescueGrid(RescueGrid):
    def __init__(self,size,guard):
        super().__init__(size);self.guard=guard;self.payload=False

    def layers(self):
        allowed,seen=super().layers()
        if self.payload:
            # An extra cell at corners, including diagonal corners. This is a
            # conservative grid margin, not an exact articulated-payload proof.
            blocked=(self.evidence>=2).copy()
            for _,(_,walls,_) in self.remote.values():
                blocked |= walls & ~(self.free & (self.evidence<.5))
            clearance=cv2.distanceTransform((~blocked).astype(np.uint8),cv2.DIST_L2,3)*self.CELL
            allowed &= clearance>=24
        return allowed,seen

    def observe(self,*args,**kwargs):
        if self.guard.trusted:super().observe(*args,**kwargs)

    def segment_known(self,start,end):
        allowed,_=self.layers()
        points=np.linspace(start,end,max(2,int(np.linalg.norm(end-start)/6)+1))
        return all(allowed[self.cell(p)] for p in points[1:])


class MyDroneBlueSemifinal(MyDroneBlueAdapted):
    LOCAL_MATCHING=True
    REGIONAL_TASKS=True

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._pose_guard=PoseGuard(self.size_area)
        self._pose_guard.enabled=self.LOCAL_MATCHING
        self._grid=TrustedRescueGrid(self.size_area,self._pose_guard)
        self._breadcrumbs=[];self._turn_progress=0.;self._last_delivery_heading=None
        self._last_region_log=-100;self._visible_bombs=[]
        self._transport_carrying=False;self._transport_start=None;self._release_requested=False
        self._narrow_return=False

    def _pose(self):
        g=self._pose_guard
        g.scan=scan_points(self.lidar_values(),self.lidar_rays_angles(),self.semantic_values())
        pose=g.update(self.measured_gps_position(),self.measured_compass_angle(),self.odometer_values())
        if pose is None:return False
        self._position,self._heading=pose
        if self._home is None:self._home=self._position.copy()
        if not g.trusted or g.relocked:self._pickup=None
        if g.relocked:
            self._route=[];self._route_step=-100;self._target=None;self._anchor=None
            self._search_goal=None;self._deferred=[];self._visible_bombs=[]
            self._turn_progress=0.;self._last_delivery_heading=None
        if g.trusted and (not self._breadcrumbs or np.linalg.norm(self._position-self._breadcrumbs[-1])>24):
            self._breadcrumbs.append(self._position.copy());self._breadcrumbs=self._breadcrumbs[-160:]
        if self._tick-self._last_region_log>=240:
            print('REGION_POSE '+json.dumps(dict(team='blue',drone=int(self.identifier),step=self._tick,
                missing=g.absolute_missing_steps,trusted=g.trusted,**g.diagnostic)))
            self._last_region_log=self._tick
        return True

    def define_message_for_all(self):
        return super().define_message_for_all() if self._pose_guard.trusted else None

    def _record_bomb(self,*args,**kwargs):
        if self._pose_guard.trusted:super()._record_bomb(*args,**kwargs)

    def _sense(self,distances,angles):
        trusted=self._pose_guard.trusted
        bombs=[list(b) for b in self._bombs];center=None if self._center is None else self._center.copy()
        center_step=self._center_step;snapshot=self._snapshot
        hits=super()._sense(distances,angles)
        self._visible_bombs=hits
        if not trusted:
            # Relative near-field semantics remain usable; uncertain world
            # coordinates never become persistent map, bomb or center reports.
            self._bombs=bombs;self._center=center;self._center_step=center_step;self._snapshot=snapshot
        return hits

    def _choose_bomb(self):
        if not self.REGIONAL_TASKS:return super()._choose_bomb()
        if not self._pose_guard.trusted:
            return min(self._visible_bombs,key=lambda v:v[0].distance)[1].copy() if self._visible_bombs else None
        options=[]
        for x,y,stamp,available in self._bombs:
            point=np.array([x,y])
            if not available or self._tick-stamp>self.BOMB_TTL:continue
            if any(until>self._tick and np.linalg.norm(point-p)<40 for p,until in self._deferred):continue
            # Only fresh commitments to THIS task can suppress a local bid.
            # A busy peer claiming another bomb is not a universal owner.
            bidders={int(self.identifier):self._position};committed={}
            for i,(seq,pos,mode,target) in self._peers.items():
                if self._tick-seq>45:continue
                if mode=='collect' and target is not None and np.linalg.norm(point-target)<40:
                    committed[i]=pos
                elif mode=='search':bidders[i]=pos
            if self._mode=='collect' and self._target is not None and np.linalg.norm(point-self._target)<35:
                committed[int(self.identifier)]=self._position
            if owner_for(point,committed or bidders)!=int(self.identifier):continue
            cost=float(np.linalg.norm(point-self._position))
            if self._target is not None and np.linalg.norm(point-self._target)<35:cost-=35
            options.append((cost,x,y))
        return np.array(min(options)[1:]) if options else None

    def _navigate(self,goal,distances,angles,carrying=False):
        self._grid.payload=bool(carrying)
        if carrying and goal is not None and self._tick-self._route_step>=24 and self._pose_guard.trusted:
            route=self._grid.route(self._position,goal,int(self.identifier),self._number,())
            self._narrow_return=False
            if not route or np.linalg.norm(route[-1]-goal)>64:
                self._grid.payload=False
                narrow=self._grid.route(self._position,goal,int(self.identifier),self._number,())
                if narrow and (not route or np.linalg.norm(narrow[-1]-goal)+32<np.linalg.norm(route[-1]-goal)):
                    route=narrow;self._narrow_return=True
                self._grid.payload=not self._narrow_return
            if (not route or (len(route)<=1 and np.linalg.norm(route[-1]-goal)>64)):
                # Reuse only a locally checked segment of an actually visited
                # trace; never call an empty route a successful return path.
                usable=[p for p in self._breadcrumbs if np.linalg.norm(p-self._position)>24
                        and np.linalg.norm(p-self._position)<120 and self._grid.segment_known(self._position,p)]
                if usable:route=[min(usable,key=lambda p:np.linalg.norm(p-goal)).copy()]
            self._route=route;self._route_step=self._tick
        if carrying:self._grid.payload=not self._narrow_return
        # Do not interpret a disconnected return map as a direct route through
        # unseen walls; local exploration can still discover a connection.
        nav_goal=None if carrying and goal is not None and not self._route else goal
        cmd=super()._navigate(nav_goal,distances,angles,carrying)
        if carrying and goal is not None:self._target=np.asarray(goal).copy()
        if carrying and self._narrow_return:
            cmd['forward']*=.6;cmd['lateral']*=.6
        if not self._pose_guard.trusted:
            cmd['forward']*=.55;cmd['lateral']*=.55
        return cmd

    def control(self):
        cmd=super().control()
        carrying=bool(self.grasped_bombs())
        hit=self._center_hit
        if carrying and not self._transport_carrying:
            self._transport_start=self._tick
            self._transport_event('pickup_observed')
        elif not carrying and self._transport_carrying:
            self._transport_event('grasp_ended',near_center=bool(hit is not None and hit.distance<35),
                                  released=self._release_requested,
                                  duration=self._tick-(self._transport_start or self._tick))
            self._release_requested=False
        self._transport_carrying=carrying
        if carrying and hit is not None and hit.distance<18:
            if self._last_delivery_heading is not None:
                odo=self.odometer_values()
                delta=float(odo[2]) if odo is not None and np.shape(odo)==(3,) and np.isfinite(odo).all() else 0.
                # Actual signed odometry, not noisy absolute heading fusion or
                # a tick counter. Reverse motion cancels earlier progress.
                if abs(delta)<.5:self._turn_progress=max(0.,self._turn_progress+delta*self._escape_sign)
            self._last_delivery_heading=self._heading
            cmd.update(forward=0.,lateral=0.,rotation=.65*self._escape_sign)
            # Override the old 28-tick release: rotation must actually happen.
            cmd['grasper']=0 if self._turn_progress>=1.2 else 1
            if cmd['grasper']==0 and not self._release_requested:
                self._release_requested=True;self._transport_event('release_requested')
        else:
            self._last_delivery_heading=None;self._turn_progress=0.
        # Apply confidence speed bound AFTER all approach/recovery overrides.
        if not self._pose_guard.trusted:
            cmd['forward']=float(np.clip(cmd.get('forward',0),-.25,.25))
            cmd['lateral']=float(np.clip(cmd.get('lateral',0),-.25,.25))
        return cmd

    def _transport_event(self,event,**data):
        # No release/disappearance is called a successful disposal here.
        # The offline evaluator measures actual disposal from engine events.
        print('REGION_TRANSPORT '+json.dumps(dict(event=event,drone=int(self.identifier),
              step=self._tick,**data)))


class MyDroneBlueSemifinalPose(MyDroneBlueSemifinal):
    """Localization ablation; retain original task/navigation/delivery behavior."""
    REGIONAL_TASKS=False
    _navigate=MyDroneBlueAdapted._navigate
    control=MyDroneBlueAdapted.control
