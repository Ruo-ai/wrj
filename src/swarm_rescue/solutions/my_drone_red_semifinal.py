"""A2: legal online regional red, retaining the A0 baseline and FS1 ablation."""
import numpy as np
from swarm_rescue.solutions.my_drone_red_fs1 import MyDroneRedFS1Gain,AreaCoverageField,supported_area
from swarm_rescue.solutions.my_drone_red_advanced2 import (
    _M18_ObservationMapRefined,_M19_SurveyGrid,_M16_reachable,_M16_path_to,_M16_PlacementPlan)
from swarm_rescue.solutions.semifinal_localization import PoseGuard,scan_points


class TrustedObservation(_M18_ObservationMapRefined):
    def __init__(self,size,guard):
        super().__init__(size);self.guard=guard
    def observe(self,*args,**kwargs):
        if self.guard.trusted:super().observe(*args,**kwargs)
    def observe_nonwalls(self,*args,**kwargs):
        if self.guard.trusted:super().observe_nonwalls(*args,**kwargs)


class TrustedSurveyGrid(_M19_SurveyGrid):
    def __init__(self,size,event,guard):
        super().__init__(size,event);self.guard=guard
    def observe(self,*args,**kwargs):
        if self.guard.trusted:super().observe(*args,**kwargs)


def choose_area_placement(grid,position,field,travel_budget):
    """Observed marginal wall area per remaining travel cost; no difficulty proxy.

    The 65 px separation retains the inherited deployer's 55 px own-placement
    and 45 px semantic guards, with estimation slack. It is much smaller than
    a blast diameter: overlapping circles can still cover different walls.
    """
    distances,parents=_M16_reachable(grid,position,50)
    candidates=[]
    for cell,steps in distances.items():
        if steps*grid.CELL+32>travel_budget:continue
        point=grid.point(cell)
        if any(np.linalg.norm(point-p)<65 for p in field.placed):continue
        value=field.sampled_value(point)
        if value>0:candidates.append((value/(1+.015*steps),cell,steps))
    best=None
    for _,cell,steps in sorted(candidates,reverse=True)[:8]:
        point=grid.point(cell)
        for dx in (-6.,0.,6.):
            for dy in (-6.,0.,6.):
                target=point+np.array([dx,dy])
                if not grid.line_clear(point,target):continue
                if any(np.linalg.norm(target-p)<65 for p in field.placed):continue
                value=field.value(target)
                cost=steps+np.linalg.norm(target-point)/grid.CELL
                if cost*grid.CELL+32>travel_budget:continue
                score=value/(1+.015*cost)
                if value>0 and (best is None or score>best[0]):best=(score,cell,target,value)
    if best is None:return None
    return _M16_PlacementPlan(_M16_path_to(parents,best[1]),best[2],best[3])


class MyDroneRedSemifinal(MyDroneRedFS1Gain):
    LOCAL_MATCHING=True
    REGIONAL_CLAIMS=True

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._pose_filter=PoseGuard(self.size_area,.12,.15)
        self._pose_filter.enabled=self.LOCAL_MATCHING
        self._map=TrustedObservation(self.size_area,self._pose_filter)
        self._explore=TrustedSurveyGrid(self.size_area,self._event,self._pose_filter)
        self._region_task=None;self._region_remaining=None;self._region_progress=0
        self._region_plan_step=-100

    def _update_observation(self):
        self._pose_filter.scan=scan_points(self.lidar_values(),self.lidar_rays_angles(),self.semantic_values())
        super()._update_observation()
        if self._pose_filter.relocked:
            self._bomb_target=None;self._route=[];self._region_task=None
            self._motion_anchor=None;self._direction_anchor=None;self._advanced_field=None
        if self._steps%240==0:
            self._event('region_pose',trusted=self._pose_filter.trusted,
                missing=self._pose_filter.absolute_missing_steps,diagnostic=self._pose_filter.diagnostic)

    def define_message_for_all(self):
        return super().define_message_for_all() if self._pose_filter.trusted else None

    def _coverage_field(self,force=False):
        if not self.REGIONAL_CLAIMS:return super()._coverage_field(force)
        placed=self._known_bombs()
        signature=(tuple(tuple(p) for p in placed),tuple((i,seq,tuple(p)) for i,(seq,p)
            in sorted(self._bomb_claims.items()) if p is not None and 0<=self._steps-seq<90))
        # Facts have no TTL. Promises decay rather than immediately erasing 95%
        # of every wall in a region. Apply each source once, via inherited store.
        if force or self._advanced_field is None or self._steps-self._field_step>=12 or signature!=self._fs1_signature:
            remote=np.zeros(self._map.shape,bool)
            for source,mask in self._radio_maps.masks.items():
                if source!=int(self.identifier or 0):remote|=mask
            field=AreaCoverageField(supported_area(self._map,remote),self._map.CELL,self.size_area,
                                   placed,[],round(.12*min(self.size_area)))
            for source,(seq,point) in self._bomb_claims.items():
                age=self._steps-seq
                if point is not None and source!=int(self.identifier or 0) and 0<=age<90:
                    priority=.65 if source<int(self.identifier or 0) else .3
                    field.weight[field.disk(point)]*=1-priority*(1-age/90)
            # Rebuild the same convolution after soft promises; factual residual
            # is retained separately for deployment verification.
            import cv2
            r=int(np.ceil(field.radius/field.cell));y,x=np.ogrid[-r:r+1,-r:r+1]
            kernel=((x*field.cell)**2+(y*field.cell)**2<=field.radius**2).astype(np.float32)
            field.gain=cv2.filter2D(field.weight,-1,kernel,borderType=cv2.BORDER_CONSTANT)
            self._advanced_field=field;self._field_step=self._steps
            self._fs1_signature=signature
        return self._advanced_field

    def _navigation(self):
        target=self._bomb_target
        if target is not None and self._position is not None:
            # Only extend commitment when the remaining current route decreases.
            points=[self._position]+[self._explore.point(c) for c in self._route]+[target]
            remaining=sum(np.linalg.norm(b-a) for a,b in zip(points,points[1:]))
            same=self._region_task is not None and np.linalg.norm(target-self._region_task)<1
            if not same:
                self._region_task=target.copy();self._region_remaining=remaining;self._region_progress=self._steps
            elif remaining<self._region_remaining-16:
                self._region_remaining=remaining;self._region_progress=self._steps
            if self._steps-self._region_progress<60:
                self._bomb_target_step=max(self._bomb_target_step,self._steps-120)
        previous_center=None if self._disposal_observed is None else self._disposal_observed.copy()
        # Keep the inherited execution and safety checks, but own candidate
        # selection. The old transport-distance difficulty bonus is uncalibrated
        # against B2, so it must not decide this candidate's placement locations.
        if self._pose_filter.trusted and self._position is not None and self.carried_bombs_count()>0 and self._progress()<.68:
            initial=self._initial_inventory or self.carried_bombs_count()
            placed=initial-self.carried_bombs_count()
            due=.1+.36*(placed+1)/(initial+1)+.008*int(self.identifier or 0)
            committed=self._route and self._steps-self._goal_step<100 and self._progress()<.55
            if not committed and self._bomb_target is None and self._progress()>=due and self._steps-self._region_plan_step>=45:
                limit=getattr(self._misc_data,'max_timestep_limit',None) or 5400
                speed=float(np.clip(np.linalg.norm(self._velocity),.8,2.))
                budget=max(0.,(.68-self._progress())*limit-40)*speed
                plan=choose_area_placement(self._explore,self._position,self._coverage_field(force=True),budget)
                self._region_plan_step=self._steps
                if plan is not None:
                    if self._explore.survey is not None:self._explore.finish_survey('placement_priority')
                    self._route=plan.route;self._goal_step=self._steps
                    self._bomb_target=plan.target;self._bomb_target_step=self._steps;self._bomb_gain=plan.gain
                    self._event('bomb_target',target=plan.target.tolist(),estimated_gain=plan.gain,
                                travel_budget=budget,difficulty_proxy_enabled=False)
        self._last_bomb_plan=self._steps  # suppress only inherited candidate selection
        cmd=super()._navigation()
        if not self._pose_filter.trusted:
            self._disposal_observed=previous_center
            cmd['forward']=float(np.clip(cmd.get('forward',0),-.35,.35))
            cmd['lateral']=float(np.clip(cmd.get('lateral',0),-.35,.35))
        return cmd

    def _release_stale_route(self,reason):
        # A newly seen obstacle invalidates the local route, not automatically
        # the placement task. Retain it only if a fresh observed route exists.
        target=self._bomb_target
        if target is not None and self._pose_filter.trusted and self._steps-self._region_progress<60:
            distances,parents=_M16_reachable(self._explore,self._position,80)
            nearby=[c for c in distances if np.linalg.norm(self._explore.point(c)-target)<24
                    and self._explore.line_clear(self._explore.point(c),target)]
            if nearby and self._coverage_field(force=True).marginal(target)>0:
                end=min(nearby,key=lambda c:distances[c])
                self._route=_M16_path_to(parents,end);self._waypoint_progress.reset()
                self._last_route_step=self._steps
                self._event('placement_route_replanned',reason=reason,remaining_cells=len(self._route))
                return
        super()._release_stale_route(reason)

    def _deployment(self,command):
        pending=self._attempt_step is not None or (self._last_inventory is not None and
                  self.carried_bombs_count()<self._last_inventory)
        if not pending and not self._pose_filter.trusted and self._progress()<.76:
            return
        super()._deployment(command)
