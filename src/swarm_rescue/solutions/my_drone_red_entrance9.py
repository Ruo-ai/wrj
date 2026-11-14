"""E9: preserve intended entrance goals before coarse avoidance turns away.

Only red subclass hooks are used. A recent coarse rejection is a REQUEST for
fresh fine observation, never evidence that unknown space is traversable.
"""
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_r8 import MyDroneRedR8Narrow
from swarm_rescue.solutions.my_drone_red_rn7 import NarrowMixin


class EntranceHint:
    def __init__(self):self.goal=None;self.until=-1
    def remember(self,point,tick):self.goal=np.asarray(point,float).copy();self.until=tick+150
    def direction(self,position,heading,tick):
        if self.goal is None or tick>self.until or position is None:return None
        delta=self.goal-position
        if np.linalg.norm(delta)<12:return None
        return math.atan2(delta[1],delta[0])-heading


def needs_precision(distances,angles,direction):
    if distances is None or angles is None:return False
    d=np.asarray(distances,float);a=np.asarray(angles,float)-direction
    valid=np.isfinite(d)&np.isfinite(a)&(d>0)&(d<280)
    along=d*np.cos(a);across=d*np.sin(a)
    hit=valid&(along>0)&(np.abs(across)<25.)
    if not hit.any():return False
    entry=along[hit]-np.sqrt(np.maximum(0.,25.**2-across[hit]**2))
    return bool(entry.min()<55.)


class MyDroneRedEntrance9(MyDroneRedR8Narrow):
    ENTRANCE_HINTS=True
    LATE_ENTRANCES=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self._entrance9=EntranceHint()
        self._extra_probes9=0;self._last_extra_probe9=-99
        self._extra_probe_window9=[]

    def _release_stale_route(self,reason):
        if reason=='new_wall' and self._route and self._position is not None:
            # Save an intended nearby waypoint, not the steering angle already
            # changed by the circular local avoider. Actual traversal still
            # requires a completely observed, hull-safe fine path.
            point=self._explore.point(self._route[0])
            if np.linalg.norm(point-self._position)>12:
                self._entrance9.remember(point,self._steps)
        return super()._release_stale_route(reason)

    def _navigation(self):
        narrow=self._rn7
        if narrow.path:
            self._slow_ticks=0;self._waypoint_progress.last_advance=self._steps
            if self._steps%60==0:self._motion_anchor=None
        # Skip ONLY the inherited RN7 motion overlay; S6, A3 navigation,
        # avoidance, deployment and submission remain in the original chain.
        original=super(NarrowMixin,self)._navigation()
        hint=self._entrance9.direction(self._position,self._heading,self._steps) if self.ENTRANCE_HINTS else None
        desired=self._last_motion_angle if self._last_motion_angle is not None else 0.
        if hint is not None:
            desired=hint;narrow.stats['intended_entrance_ticks']+=1
            self._extra_probe_window9=[t for t in self._extra_probe_window9 if narrow.tick-t<600]
            if (not narrow.path and self._extra_probes9<48 and len(self._extra_probe_window9)<6
                    and narrow.tick-self._last_extra_probe9>=12):
                narrow.cooldown=min(narrow.cooldown,narrow.tick)
                self._last_extra_probe9=narrow.tick;self._extra_probes9+=1
                self._extra_probe_window9.append(narrow.tick)
        elif self._route and self._position is not None and not self._gathering:
            delta=self._explore.point(self._route[0])-self._position
            if np.linalg.norm(delta)>12:desired=math.atan2(delta[1],delta[0])-self._heading
        near_bomb=(self._bomb_target is not None and self._position is not None and
                   np.linalg.norm(self._bomb_target-self._position)<55)
        late=(self.LATE_ENTRANCES and self.carried_bombs_count()==0 and
              (hint is not None or bool(narrow.path)) and self._progress()<.79)
        allow=(self._position is not None and self.drone_health>10 and
               ((self._progress()<.68 and not self._gathering) or late) and
               not self._exploration_submitted and self._attempt_step is None and not near_bomb and
               self._steps>=self._yield_until)
        # Do not replace ordinary open-corridor flight with a slower fine task.
        # Precision is requested by a coarse rejection or a near circular-sweep
        # conflict, then the real hull decides whether an observed route exists.
        allow=allow and (bool(narrow.path) or hint is not None or
                         needs_precision(self.lidar_values(),self.lidar_rays_angles(),desired))
        command=narrow.command(self.lidar_values(),self.lidar_rays_angles(),self._explore,
                              self._position,self._heading,desired,self.semantic_values(),allow)
        if command is not None:
            motion=self._segment_motion
            if motion.state not in ('NORMAL','COOLDOWN'):motion.release('entrance9_local_passage')
            original.update(command)
        return original
