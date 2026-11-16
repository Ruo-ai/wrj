"""RC15: RL11 mapper/bombs, additive observed hull-clearance navigation.

No truth/map identifiers. The distant 20px graph remains conservative; a bounded
body-frame graph supplies reachable local goals even when that graph rejects a
door. Selection, smoothing, following and stopping use the same configuration
space. Current orientation is retained (holonomic lateral motion is legal).
"""
from functools import lru_cache
import math
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_red_lines11 import MyDroneRedLines11
from swarm_rescue.solutions.my_drone_red_rn7 import HULL, CoarseView, rotate
from swarm_rescue.solutions.red_footprint8 import FootprintMap, FootprintMotion
from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid


@lru_cache(maxsize=96)
def swept_kernel(margin=2., angle0=0., angle1=0., cell=2.):
    """Full hull swept through rotation, with explicit discretization bounds.

    2px extra clearance is NOT a noise-confidence bound. Half-cell diagonal
    bounds raster quantization; .5px bounds 1px translation samples; angular
    interpolation has its own circumradius * half-step bound.
    """
    if not 2 <= margin <= 4: raise ValueError('margin must be 2..4 px per side')
    radius=float(np.linalg.norm(HULL,axis=1).max())
    delta=(angle1-angle0+math.pi)%(2*math.pi)-math.pi
    count=max(1,int(math.ceil(abs(delta)/math.radians(3))))
    padding=margin+cell/math.sqrt(2)+.5+radius*abs(delta)/(2*count)
    extent=int(math.ceil((radius+padding)/cell))
    kernel=np.zeros((2*extent+1,2*extent+1),np.uint8)
    polygons=[rotate(HULL,t).astype(np.float32) for t in np.linspace(angle0,angle0+delta,count+1)]
    for r in range(-extent,extent+1):
        for c in range(-extent,extent+1):
            if any(cv2.pointPolygonTest(p,(c*cell,-r*cell),True)>=-padding for p in polygons):
                kernel[r+extent,c+extent]=1
    return kernel


class ClearanceMap(FootprintMap):
    def __init__(self,d,a,objects=(),margin=2.):
        # Copy: the retained fitter operates in-place on its temporary ray array.
        super().__init__(np.array(d,dtype=float,copy=True),np.array(a,dtype=float,copy=True),objects,margin)
        self.configuration=cv2.erode(self.free.astype(np.uint8),swept_kernel(margin),
            borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        self._clearance=cv2.distanceTransform(np.pad(self.configuration.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]*self.CELL
        self.search_nodes=0

    def swept_clear(self,start,end,angle0=0.,angle1=0.):
        start=np.asarray(start,float);end=np.asarray(end,float)
        if not np.isfinite(np.r_[start,end,angle0,angle1]).all():return False
        pts=np.linspace(start,end,max(2,int(math.ceil(np.linalg.norm(end-start)))+1))
        # Bounds are checked before cell() can clip them to the map border.
        if np.any(np.abs(pts[:,0])>=self.width/2-1) or np.any(np.abs(pts[:,1])>=self.height/2-1):return False
        config=self.configuration if angle0==angle1==0 else cv2.erode(
            self.free.astype(np.uint8),swept_kernel(self.margin,angle0,angle1),
            borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        ix=self.indices(pts)
        return bool(config[ix[:,1],ix[:,0]].all())

    def line_clear(self,start,end):return self.swept_clear(start,end)

    def route(self,position,preferred,coarse,avoid=(),target=None,limit=165):
        # Existing bounded search: <=6000 nodes / 165px net / 247.5px path.
        # Its eligible endpoints, graph edges and smoothing all use this hull
        # C-space, not the coarse-grid inflation or the old 22/25px discs.
        return super().route(position,preferred,coarse,avoid,target,limit)


class ClearanceGrid(_M19_SurveyGrid):
    """Local certificates supplement distant coarse line checks, never free-fill."""
    live=None
    def line_clear(self,position,target):
        if self.live is not None:
            fine,origin,heading=self.live
            a=rotate(np.asarray(position)-origin,-heading);b=rotate(np.asarray(target)-origin,-heading)
            if max(np.linalg.norm(a),np.linalg.norm(b))<=165:
                return fine.line_clear(a,b)
        return super().line_clear(position,target)


class ClearanceMotion(FootprintMotion):
    SAFETY_MARGIN=2.
    REFRESH_BUDGET=True
    MAX_TASK_TICKS=240
    def __init__(self,identifier=0):
        super().__init__(identifier);self.factory=self.make_map
        self.live=None;self.cached_tick=-1;self.audit=[];self.context={}
        self.failed=[];self.last_reason='idle';self.close_wall_ticks=0

    def make_map(self,d,a,objects=(),margin=2.):
        if self.cached_tick!=self.tick:
            self.live=ClearanceMap(d,a,objects,margin);self.cached_tick=self.tick
        return self.live

    def event(self,reason):
        super().event(reason);self.last_reason=reason
        self.audit.append(dict(tick=self.tick,reason=reason,**self.context,
            goal_body=None if not self.path else self.path[-1].tolist(),
            path=[p.tolist() for p in self.path],velocity=self.velocity.tolist()))
        if len(self.audit)>900:self.audit.pop(0)

    def release(self,reason):
        if self.path and reason in ('no_progress','fresh_scan_obstructed','task_or_active_budget','peer_wait_budget'):
            self.failed.append((self.tick,self.path[-1].copy()))
        super().release(reason)

    def update(self,odometry):
        super().update(odometry)
        if self.valid_odom:
            distance,alpha,theta=map(float,odometry);shift=distance*np.array([math.cos(alpha),math.sin(alpha)])
            self.failed=[(t,rotate(p-shift,-theta)) for t,p in self.failed if self.tick-t<180][-6:]

    def command(self,d,a,coarse,position,heading,desired,objects=(),allow=True):
        if d is None or a is None:return super().command(d,a,coarse,position,heading,desired,objects,False)
        self.context=dict(position=np.asarray(position).tolist(),heading=float(heading),margin=self.SAFETY_MARGIN,
            desired=float(desired),min_ray=float(np.nanmin(d)) if np.isfinite(d).any() else None)
        before=self.episodes
        command=super().command(d,a,coarse,position,heading,desired,objects,allow)
        if self.episodes>before and self.path:
            end=self.path[-1];world=np.asarray(position)+rotate(end,heading)
            # Compare unmodified coarse predicate for diagnostics only.
            coarse_ok=_M19_SurveyGrid.line_clear(coarse,position,world)
            self.context['coarse_clear']=bool(coarse_ok)
            self.event('fine_goal_selected')
            if not coarse_ok:self.stats['coarse_rejected_fine_accepted']+=1
            if any(np.linalg.norm(end-p)<35 for _,p in self.failed):
                self.release('recent_failed_goal');return self.brake()
        if self.live is not None and self.cached_tick==self.tick:
            slack=float(self.live.clearance[self.live.cell((0,0))])
            self.context['residual_clearance']=slack
            if slack<=6:self.close_wall_ticks+=1
        return command


class MyDroneRedClearance15(MyDroneRedLines11):
    MIN_CLEARANCE_ENABLED=True
    SAFETY_MARGIN=2.
    # RN7's old overlay is replaced, never stacked with the new C-space filter.
    NARROW_ENABLED=False
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.MIN_CLEARANCE_ENABLED:
            self._rn7=ClearanceMotion(int(self.identifier or 0));self._rn7.SAFETY_MARGIN=self.SAFETY_MARGIN
            grid=ClearanceGrid(self.size_area,self._event);grid.__dict__.update(self._explore.__dict__)
            self._explore=grid

    def _navigation(self):
        if not self.MIN_CLEARANCE_ENABLED:return super()._navigation()
        self._explore.live=None
        n=self._rn7
        # Current body-frame geometry also certifies local shortcuts used by the
        # global route follower. No persistent/global free-space is overwritten.
        if n.path and self._position is not None and self.lidar_values() is not None:
            f=n.make_map(self.lidar_values(),self.lidar_rays_angles(),self.semantic_values(),n.SAFETY_MARGIN)
            self._explore.live=(f,self._position.copy(),self._heading)
        original=super()._navigation()
        if self._position is None:return original
        near_bomb=self._bomb_target is not None and np.linalg.norm(self._bomb_target-self._position)<55
        allow=(self.drone_health>10 and self._progress()<.68 and not self._gathering
            and not self._exploration_submitted and self._attempt_step is None and not near_bomb
            and self._steps>=self._yield_until)
        # Use the actual navigation goal before 22/25px avoidance deflects it.
        desired=self._last_motion_angle if self._last_motion_angle is not None else 0.
        for cell in self._route:
            delta=self._explore.point(cell)-self._position
            if np.linalg.norm(delta)>28:
                desired=math.atan2(delta[1],delta[0])-self._heading;break
        command=n.command(self.lidar_values(),self.lidar_rays_angles(),self._explore,
            self._position,self._heading,desired,self.semantic_values(),allow)
        if command is not None:
            m=self._segment_motion
            if m.state not in ('NORMAL','COOLDOWN'):m.release('rc15_local_passage')
            original.update(command)
        self._explore.live=None
        return original


class MyDroneRedClearance15Off(MyDroneRedClearance15):
    MIN_CLEARANCE_ENABLED=False
    NARROW_ENABLED=True
