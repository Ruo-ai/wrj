"""B4 independent, bounded payload boundary following. Legal observations only.

Does not enable B3 PAYLOAD_RECOVERY. A stalled local endpoint may require moving
away from the destination; a fixed side prevents frame-to-frame reversals.
This is a local heuristic, not a complete-motion-planning guarantee.
"""
import math
import numpy as np
from swarm_rescue.solutions.my_drone_blue_transport import MyDroneBlueTransportCompute,SteeringTable
from swarm_rescue.solutions.my_drone_eval_blue import wrap


def corridor_gap(distances,angles,bearing,radius=43.):
    valid=np.isfinite(distances)&np.isfinite(angles)&(distances>0)
    if not valid.any():return 0.
    d=distances[valid];rel=(angles[valid]-bearing+math.pi)%(2*math.pi)-math.pi
    front=d*np.cos(rel)
    corridor=(front>0)&(np.abs(d*np.sin(rel))<radius)&(np.abs(rel)<1.4)
    return float(np.min(front[corridor])) if corridor.any() else 230.


class BoundaryRecovery:
    def __init__(self,identifier):
        self.side=1 if identifier%2==0 else -1
        self.started=None;self.anchor=None;self.anchor_step=0;self.cooldown=0
        self.entry_distance=0.;self.last_direction=None;self.reason='inactive'
        self.steering=SteeringTable()

    def reset(self):
        self.started=None;self.anchor=None;self.last_direction=None;self.reason='inactive'

    def command(self,step,position,heading,target,distances,angles,partial,active=True):
        if not active or target is None:
            self.reset();return None
        delta=target-position;remaining=float(np.linalg.norm(delta))
        bearing=wrap(math.atan2(delta[1],delta[0])-heading)
        if self.anchor is None or np.linalg.norm(position-self.anchor)>36:
            self.anchor=position.copy();self.anchor_step=step
        stalled=step-self.anchor_step>120
        if self.started is None:
            if not partial or not stalled or step<self.cooldown:return None
            self.started=step;self.entry_distance=remaining;self.reason='partial_endpoint_stall'
        if step-self.started>=720:
            self.started=None;self.anchor=position.copy();self.anchor_step=step
            self.cooldown=step+60;self.side*=-1;self.reason='bounded_timeout'
            return None
        if step-self.started>45 and remaining<self.entry_distance-40 and corridor_gap(distances,angles,bearing)>min(180,remaining+43):
            self.started=None;self.anchor=position.copy();self.anchor_step=step
            self.cooldown=step+60;self.reason='observed_goal_corridor';return None
        valid=np.isfinite(distances)&np.isfinite(angles)&(distances>2)&(distances<120)
        if valid.any():
            d=distances[valid];a=angles[valid];i=int(np.argmin(d))
            normal=np.array([math.cos(a[i]),math.sin(a[i])])
            tangent=self.side*np.array([-normal[1],normal[0]])
            # Follow at a conservative payload margin; never traverse unknown
            # space because every translation is gated by the current lidar.
            direction=tangent+np.clip((float(d[i])-62)/30,-1.3,.6)*normal
            desired=math.atan2(direction[1],direction[0])
            self.last_direction=wrap(desired+heading)
        elif self.last_direction is not None:desired=wrap(self.last_direction-heading)
        else:desired=bearing
        choice=self.steering.choose(distances,angles,desired,True)
        cmd=dict(forward=0.,lateral=0.,rotation=0.,grasper=1)
        if choice is None:self.reason='missing_lidar';return cmd
        angle,gap=choice
        speed=.52*float(np.clip((gap-43)/70,0,1))
        if abs(angle)>1.2:speed=0.
        cmd.update(forward=speed*math.cos(angle),lateral=speed*math.sin(angle),
                   rotation=float(np.clip(angle*1.5,-.75,.75)))
        self.reason='boundary_follow'
        return cmd


class MyDroneBlueBoundary(MyDroneBlueTransportCompute):
    """Only payload stagnation recovery changes; B0 search and terminal retained."""
    BOUNDARY_RECOVERY=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._boundary=BoundaryRecovery(int(self.identifier));self._command_source=None

    def _navigate(self,goal,distances,angles,carrying=False):
        original=super()._navigate(goal,distances,angles,carrying)
        active=bool(self.BOUNDARY_RECOVERY and carrying and self._center_hit is None)
        end=self._route[-1] if self._route else None
        partial=goal is not None and (end is None or np.linalg.norm(end-goal)>96)
        cmd=self._boundary.command(self._tick,self._position,self._heading,goal,
                                   distances,angles,partial,active)
        self._command_source='payload_boundary:'+self._boundary.reason if cmd is not None else None
        return original if cmd is None else cmd

    def control(self):
        self._command_source=None
        return super().control()
