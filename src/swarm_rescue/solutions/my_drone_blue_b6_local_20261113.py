"""B6 independent blue experiments; existing B0/B2/B5 files remain read-only.

Local consistency and portal linking are separate selectors, never combined.
Only legal sensors, private history and received messages enter these policies.
"""
import copy
import math
from collections import Counter
import numpy as np
from swarm_rescue.solutions.my_drone_eval_blue import MyDroneBlueAdapted,wrap
from swarm_rescue.solutions.semifinal_localization import LocalMatcher,scan_points
from swarm_rescue.solutions.my_drone_blue_narrow import MyDroneBlueNarrow,corridor_geometry


class PrincipalMatcher(LocalMatcher):
    """Reuse bounded correlation, but test translation's principal directions.

    Axis-only observability overstates certainty in a diagonal straight hall.
    Finite difference curvature is a quality indicator, not a covariance proof.
    """
    def match(self,points,position,heading):
        if self.distance is None or len(points)<12:return None,dict(reason='insufficient_support',rank=0)
        coarse=np.array([(x,y,t) for x in (-6,0,6) for y in (-6,0,6) for t in (-.06,-.03,0,.03,.06)])
        raw,_=self.costs(points,position,heading,coarse)
        best=coarse[np.argmin(raw+.007*np.sum(coarse[:,:2]**2,axis=1)+4*coarse[:,2]**2)]
        fine=best+np.array([(x,y,t) for x in (-2,0,2) for y in (-2,0,2) for t in (-.01,0,.01)])
        values,_=self.costs(points,position,heading,fine)
        best=fine[np.argmin(values+.007*np.sum(fine[:,:2]**2,axis=1)+4*fine[:,2]**2)].copy()
        probes=np.array([(0,0,0),(8,0,0),(-8,0,0),(0,8,0),(0,-8,0),
                         (8,8,0),(-8,-8,0),(8,-8,0),(-8,8,0),(0,0,.04),(0,0,-.04)])
        costs,_=self.costs(points,position,heading,best+probes)
        cross=(costs[5]+costs[6]-costs[7]-costs[8])/256
        h=np.array([[(costs[1]+costs[2]-2*costs[0])/64,cross],
                    [cross,(costs[3]+costs[4]-2*costs[0])/64]])
        eigen,vectors=np.linalg.eigh(h)
        # Gate raster curvature with normals of locally straight scan segments.
        # Parallel walls cannot constrain motion along their tangent, regardless
        # of stair-stepping artifacts in the distance image.
        segments=points[2:]-points[:-2];length=np.linalg.norm(segments,axis=1)
        midpoint=points[1:-1]-points[:-2]
        deviation=np.abs(segments[:,0]*midpoint[:,1]-segments[:,1]*midpoint[:,0])/np.maximum(length,1e-6)
        straight=(length>12)&(length<80)&(deviation<4)
        geometric_rank=0
        if straight.sum()>=6:
            tangents=segments[straight]/length[straight,None]
            normals=np.column_stack((-tangents[:,1],tangents[:,0]))
            c,s=math.cos(heading+best[2]),math.sin(heading+best[2])
            normals=normals@np.array([[c,s],[-s,c]])
            ge,gv=np.linalg.eigh(normals.T@normals/len(normals))
            geometric_rank=int((ge>.15*ge.max()).sum())
            if geometric_rank==1:
                vectors=gv;eigen=np.array([0.,max(float(eigen.max()),.005)])
        # The raster distance field is piecewise linear, so a Cartesian
        # Hessian alone can falsely give a diagonal corridor rank two.
        # Check both signs along its principal directions against the prior.
        principal=np.zeros((4,3))
        for k in range(2):
            principal[2*k,:2]=8*vectors[:,k];principal[2*k+1,:2]=-8*vectors[:,k]
        principal_cost,_=self.costs(points,position,heading,best+principal)
        observable=(eigen>max(.004,float(eigen.max())*.25)) & (principal_cost.reshape(2,2).min(axis=1)-costs[0]>.3)
        projection=(vectors*observable)@vectors.T
        best[:2]=projection@best[:2]
        yaw_observable=bool(min(costs[9:])-costs[0]>.18)
        if not yaw_observable:best[2]=0.
        cost,support=self.costs(points,position,heading,best[None,:])
        prior=float(self.costs(points,position,heading,np.zeros((1,3)))[0][0])
        # Do not let a weak direction inherit a confident displacement from a
        # repetitive distant minimum. The prior remains odometry in that axis.
        projected=(coarse[:,:2]-best[:2])@projection
        alternative=np.linalg.norm(projected,axis=1)>8
        ambiguous=bool(alternative.any() and raw[alternative].min()<cost[0]-.1)
        accept=bool(cost[0]<=6.5 and support[0]>=.65 and cost[0]<=prior+.1
                    and np.linalg.norm(best[:2])<=10 and abs(best[2])<=.071 and not ambiguous
                    and (observable.any() or yaw_observable))
        return (best if accept else None),dict(reason='matched' if accept else 'rejected',rank=int(observable.sum()),
            eigenvalues=eigen.tolist(),geometric_rank=geometric_rank,projection=projection.tolist(),yaw_observable=yaw_observable,
            residual=float(cost[0]),prior_residual=prior,support=float(support[0]),candidates=89)


class BlueLocalWindow:
    PERIOD=6
    MAX_STEPS=180
    MAX_TRAVEL=100.
    def __init__(self,matching=True):
        self.matcher=PrincipalMatcher((768,768));self.matching=matching
        self.origin=None;self.started=0;self.failures=0;self.missing=0;self.good=8
        self.ready=False;self.local_rank=0;self.info=dict(reason='initial');self.stats=Counter()

    def update(self,position,heading,points,gps,compass,tick):
        position=np.asarray(position,float).copy()
        good_gps=gps is not None and np.shape(gps)==(2,) and np.isfinite(gps).all()
        good_compass=compass is not None and np.ndim(compass)==0 and np.isfinite(compass)
        absolute=bool(good_gps and good_compass)
        was_missing=self.missing
        self.missing=0 if absolute else self.missing+1
        self.good=self.good+1 if absolute else 0
        self.ready=bool(absolute and self.good>=8 and np.linalg.norm(np.asarray(gps)-position)<15
                        and abs(wrap(float(compass)-heading))<.12)
        self.info=dict(reason='period_wait',rank=self.local_rank,absolute_ready=self.ready,missing=self.missing)
        if not self.matching:return position,heading
        if absolute and was_missing:
            self.matcher.frames.clear();self.matcher.distance=None;self.origin=None;self.local_rank=0
            self.stats['absolute_recovery_reset']+=1
        if tick%self.PERIOD:return position,heading
        reset=(self.origin is None or tick-self.started>=self.MAX_STEPS
               or np.linalg.norm(position-self.origin)>self.MAX_TRAVEL or self.failures>=5)
        if reset:
            self.origin=position.copy();self.started=tick;self.failures=0;self.local_rank=0
            self.matcher.frames.clear();self.matcher.distance=None;self.stats['window_reset']+=1
        local=position-self.origin
        if absolute:
            if self.ready:self.matcher.add(points,local,heading)
            self.info=dict(reason='absolute_reference' if self.ready else 'absolute_recovery_wait',rank=2 if self.ready else 0,absolute_ready=self.ready)
            return position,heading
        # Match existing history first. A new local frame is allowed to warm up
        # without claiming any global trust; this avoids B2's permanent freeze.
        offset,info=self.matcher.match(points,local,heading)
        self.info=dict(info,absolute_ready=False,missing=self.missing,window_age=tick-self.started)
        if offset is not None:
            if not good_gps:position+=.75*offset[:2]
            if not good_compass:heading=wrap(heading+.65*offset[2])
            self.local_rank=info['rank'];self.failures=0;self.stats['accepted']+=1
            self.matcher.add(points,position-self.origin,heading)
        elif self.matcher.distance is None:
            self.matcher.add(points,local,heading);self.stats['warmup']+=1
        else:self.failures+=1;self.stats['rejected']+=1
        return position,heading


class MyDroneBlueB6Local(MyDroneBlueAdapted):
    LOCAL_CONSISTENCY=True
    LOCAL_MATCHING=True
    PRIVATE_MAP=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._b6_window=BlueLocalWindow(self.LOCAL_MATCHING)
        self._b6_book=None;self._b6_events=Counter();self._b6_pose_info={}

    def _pose(self):
        good=super()._pose()
        if not good or not self.LOCAL_CONSISTENCY:return good
        before=self._position.copy();old_heading=self._heading
        points=scan_points(self.lidar_values(),self.lidar_rays_angles(),self.semantic_values())
        self._position,self._heading=self._b6_window.update(self._position,self._heading,points,
            self.measured_gps_position(),self.measured_compass_angle(),self._tick)
        self._b6_pose_info=dict(self._b6_window.info,correction=(self._position-before).tolist(),
            heading_correction=wrap(self._heading-old_heading),global_trusted=self._b6_window.ready)
        if self.PRIVATE_MAP and not self._b6_window.ready and self._b6_book is None:
            self._b6_book=dict(grid=self._grid,bombs=copy.deepcopy(self._bombs),
                center=None if self._center is None else self._center.copy(),center_step=self._center_step)
            self._grid=copy.deepcopy(self._grid)
            self._b6_events['private_enter']+=1
        elif self.PRIVATE_MAP and self._b6_window.ready and self._b6_book is not None:
            book=self._b6_book;self._grid=book['grid'];self._bombs=book['bombs']
            self._center=book['center'];self._center_step=book['center_step']
            self._b6_book=None;self._snapshot=None;self._route=[];self._route_step=-100
            self._target=None;self._search_goal=None;self._anchor=None
            # Possession remains the actual grasper observation. Do not publish
            # a tombstone at an untrusted temporary pickup coordinate.
            self._pickup=None;self._b6_events['private_discard']+=1
        return True

    def define_message_for_all(self):
        if self.LOCAL_CONSISTENCY and self.PRIVATE_MAP and self._b6_book is not None:
            return None
        return super().define_message_for_all()


class MyDroneBlueB6Off(MyDroneBlueB6Local):
    """Exact B0 policy path for equivalence checks, apart from inactive fields."""
    LOCAL_CONSISTENCY=False


class MyDroneBlueB6Private(MyDroneBlueB6Local):
    """Private map ablation: odometry unchanged, no scan correction."""
    LOCAL_MATCHING=False


class PortalObservation:
    """Propose a safe observed approach; never turn an unknown cell into free."""
    def __init__(self):
        self.previous=None;self.confirmations=0;self.reason='initial';self.point=None

    def propose(self,grid,position,heading,distances,angles,carrying=False):
        self.point=None
        if carrying:self.reason='payload_not_validated';return None
        geo=corridor_geometry(distances,angles)
        if geo is None:self.reason='no_sensor_supported_opening';self.confirmations=0;return None
        if self.previous and abs(geo['width']-self.previous['width'])<4 and abs(geo['angle']-self.previous['angle'])<.08:
            self.confirmations+=1
        else:self.confirmations=1
        self.previous=geo
        if self.confirmations<3:self.reason='waiting_multiframe';return None
        if geo['width']-geo['uncertainty']<26:self.reason='uncertain_width';return None
        bearing=heading+geo['angle'];direction=np.array([math.cos(bearing),math.sin(bearing)])
        allowed,_=grid.layers();probe=position+direction*min(100,max(40,geo['end']))
        if allowed[grid.cell(probe)]:self.reason='coarse_route_open';return None
        points=[]
        # Enumerate only existing allowed cells near the entrance. Require the
        # complete conservative body ring AND approach segment to be observed.
        for r,c in np.argwhere(allowed):
            q=grid.point((r,c));delta=q-position;forward=float(delta@direction)
            if not 4<forward<max(5,geo['start']-22) or np.linalg.norm(delta)>85:continue
            if abs(direction[0]*delta[1]-direction[1]*delta[0])>20:continue
            ring=q+18*np.column_stack((np.cos(np.arange(8)*math.pi/4),np.sin(np.arange(8)*math.pi/4)))
            if np.any(np.abs(ring[:,0])>=grid.width/2) or np.any(np.abs(ring[:,1])>=grid.height/2):continue
            if not all(grid.free[grid.cell(v)] and grid.evidence[grid.cell(v)]<2 for v in ring):continue
            line=np.linspace(position,q,max(2,int(np.linalg.norm(delta)/6)+1))
            if not all(allowed[grid.cell(v)] for v in line):continue
            points.append((forward,q))
        if not points:self.reason='no_known_safe_viewpoint';return None
        self.point=max(points,key=lambda item:item[0])[1].copy();self.reason='pending_observation'
        return self.point


class MyDroneBlueB6Portal(MyDroneBlueNarrow):
    """Separate B5 portal-link experiment; does NOT inherit B6 localization."""
    PORTAL_LINK=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self._b6_portal=PortalObservation()
        self._b6_portal_point=None;self._b6_portal_until=0;self._b6_portal_cooldown=0

    def _navigate(self,goal,distances,angles,carrying=False):
        if not self.PORTAL_LINK or carrying or self._mode!='search':
            if self._b6_portal_point is not None:self._route_step=-100
            self._b6_portal_point=None
            return super()._navigate(goal,distances,angles,carrying)
        q=self._b6_portal_point
        if q is not None:
            allowed,_=self._grid.layers()
            line=np.linspace(self._position,q,max(2,int(np.linalg.norm(q-self._position)/6)+1))
            blocked=not all(allowed[self._grid.cell(v)] for v in line)
            reached=np.linalg.norm(q-self._position)<14
            expired=self._tick>=self._b6_portal_until
            if blocked or reached or expired:
                self._b6_portal.reason='approach_blocked' if blocked else 'observation_reached' if reached else 'observation_timeout'
                self._b6_portal_point=None;self._b6_portal_cooldown=self._tick+90;self._route_step=-100
        if self._b6_portal_point is None and self._tick%12==0 and self._tick>=self._b6_portal_cooldown:
            self._b6_portal_point=self._b6_portal.propose(self._grid,self._position,self._heading,distances,angles)
            if self._b6_portal_point is not None:
                self._b6_portal_until=self._tick+90;self._route_step=-100
        q=self._b6_portal_point
        # Pass the observation goal BEFORE ordinary route recomputation. A
        # second navigation call would either reuse the wrong route or double
        # the planning work. B5 still independently owns local confirmation.
        action=super()._navigate(q if q is not None else goal,distances,angles,False)
        if q is not None and self._narrow_action is None:self._command_source='portal_observation'
        return action
