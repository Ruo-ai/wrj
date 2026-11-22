"""Repair-route target fidelity and observed corner execution, over frozen 29."""
import math
import numpy as np

from swarm_rescue.solutions.red_return29 import ReturnPlanner29
from swarm_rescue.solutions.red_passage28 import HullPassage28
from swarm_rescue.solutions.my_drone_red_rn7 import rotate


class ReturnLocal30(HullPassage28):
    def route(self, position, preferred, coarse, avoid=(), target=None, limit=220):
        origin=np.asarray(position,float).copy()
        direction=np.asarray(preferred,float).copy()
        destination=None if target is None else np.asarray(target,float).copy()
        # The inherited search normalizes preferred in place. Inputs and target
        # must never alias, including when a caller supplies the same array.
        if destination is not None and np.linalg.norm(destination-origin)<=limit and self.line_clear(origin,destination):
            self.last_reason='observed_direct_target'
            return [origin,destination]
        return super().route(origin,direction,coarse,avoid=avoid,target=destination,limit=limit)


class ReturnPlanner30(ReturnPlanner29):
    def __init__(self, identifier=0, local_factory=None):
        super().__init__(identifier,ReturnLocal30 if local_factory is None else local_factory)
        self._repair_goal_index30=None

    def _clear_repair30(self):
        self._repair=[]
        self._repair_goal_index30=None

    def _safe_reached30(self,index,fine,position,heading):
        body=rotate(self.trail[index]-position,-heading)
        return np.linalg.norm(body)<self.REACHED and fine.line_clear(np.zeros(2),body)

    def command(self,position,heading,velocity,lidar,angles,semantic=(),peers=None):
        if not self.returning:return None
        if position is None or heading is None or np.shape(position)!=(2,) or not np.isfinite(position).all() or not math.isfinite(heading):
            return self._brake('return_pose_unavailable')
        position=np.asarray(position,float);self.position=position.copy();self.heading=float(heading)
        if not self.trail:return self._brake('return_no_recorded_route')
        if not self.valid_odometry:
            if velocity is not None and np.shape(velocity)==(2,) and np.isfinite(velocity).all():self.velocity=np.asarray(velocity,float).copy()
            return self._brake('return_odometry_unavailable')
        if self.step<self._angular_hold_until:return self._brake('return_wait_fixed_heading')
        fine=self._scan(lidar,angles,semantic)
        if fine is None:return self._brake('return_scan_unavailable')
        yielded=self._yield_move(fine,semantic,peers)
        if yielded is not None:return yielded
        if self.home is not None and np.linalg.norm(position-self.home)<=self.HOME_RADIUS:
            self.arrived=True;return self._brake('at_rendezvous')
        if self._home_relay(peers):return self._brake('at_near_home_relay')
        self.arrived=False
        if not fine.line_clear(np.zeros(2),np.zeros(2)):
            recovered=self._clearance_recovery(fine)
            return recovered if recovered is not None else self._brake('origin_clearance_no_safe_retreat')
        self._recovery_since=None
        if self._target_index is not None and self._target_index<=self.cursor and self._safe_reached30(self._target_index,fine,position,heading):
            self.cursor=max(0,self._target_index-1);self._target_index=None
            self._last_advance=self.step;self._clear_repair30()
        while self.cursor>0 and self._safe_reached30(self.cursor,fine,position,heading):
            self.cursor-=1;self._last_advance=self.step;self._clear_repair30()
        if self.step%6==0:
            old=np.asarray(self.trail[:self.cursor+1])
            candidates=np.flatnonzero(np.linalg.norm(old-position,axis=1)<=(40 if self._cache_used29 else self.LOOKAHEAD))
            for index in candidates:
                if index>=max(0,self.cursor-12):break
                if fine.line_clear(np.zeros(2),rotate(old[index]-position,-heading)):
                    self.cursor=int(index)+1;self.stats['visible_return_shortcut']+=1;self._clear_repair30();break
        target=None;chosen=None
        for index in range(max(0,self.cursor-12),self.cursor+1):
            body=rotate(self.trail[index]-position,-heading)
            if np.linalg.norm(body)<=self.LOOKAHEAD and fine.line_clear(np.zeros(2),body):
                target=body;chosen=index;break
        if self.cursor==0 and self.home is not None:
            body=rotate(self.home-position,-heading)
            if np.linalg.norm(body)<=self.LOOKAHEAD and fine.line_clear(np.zeros(2),body):target=body;chosen=0
        repair_motion=False
        if target is None:
            self._blocked_ticks+=1
            if self.step-self._repair_step>=12:
                self._repair_step=self.step
                for index in range(max(0,self.cursor-6),self.cursor+1):
                    body=rotate(self.trail[index]-position,-heading)
                    if not 15<np.linalg.norm(body)<=160:continue
                    path=fine.route(np.zeros(2),body.copy(),None,target=body.copy(),limit=165)
                    if path:
                        # A repair is successful only when it reaches the
                        # requested target, within the existing raster cell.
                        if np.linalg.norm(path[-1]-body)>fine.CELL*math.sqrt(2)+1e-8:
                            self.stats['repair_target_mismatch_rejected']+=1;continue
                        self._repair=[position+rotate(point,heading) for point in path[1:]]
                        self._repair_goal_index30=index
                        self.stats['observed_local_repair']+=1;break
            # A near corner is removed only when the next segment is itself
            # visible from the current position. Distance alone is insufficient.
            while len(self._repair)>1 and np.linalg.norm(self._repair[0]-position)<8:
                next_body=rotate(self._repair[1]-position,-heading)
                if not fine.line_clear(np.zeros(2),next_body):
                    self.stats['necessary_near_corner_retained']+=1;break
                self._repair.pop(0)
            goal=self._repair_goal_index30
            if goal is not None and self._safe_reached30(goal,fine,position,heading):
                self.cursor=max(0,goal-1);self._target_index=None;self._clear_repair30()
                self._last_advance=self.step
                return self._brake('return_repair_goal_reached')
            if self._repair:
                body=rotate(self._repair[0]-position,-heading)
                if fine.line_clear(np.zeros(2),body):target=body;chosen=goal;repair_motion=True
            if target is None:return self._brake('recorded_route_currently_blocked')
        else:
            self._blocked_ticks=0;self._clear_repair30()
            if chosen is not None and np.linalg.norm(target)<self.REACHED:
                self.cursor=max(0,chosen-1);self._last_advance=self.step
                return self._brake('return_waypoint_reached')
        self.last_target=(position+rotate(target,heading)).tolist();self._target_index=chosen
        distance=float(np.linalg.norm(target))
        if distance<=.25:return self._brake('return_subpixel_waypoint')
        clearance=float(fine.sample(np.linspace(np.zeros(2),target,8)).min())
        stop_margin=.25 if repair_motion else 4.
        speed=min(1.2 if self._cache_used29 else self.MAX_SPEED,
                  max(.7,.28*(clearance-fine.TRACK_RADIUS)),math.sqrt(max(0.,.8*(distance-stop_margin))))
        desired=target/distance*speed
        force=.085*desired+.65*(desired-self.velocity);force/=max(1.,float(np.linalg.norm(force)))
        predicted=.95*self.velocity+.6*force
        stopping=predicted*3+predicted*np.linalg.norm(predicted)/1.2
        if not fine.line_clear(np.zeros(2),stopping):return self._brake('return_inertial_clearance')
        self.last_reason='following_observed_repair_route' if repair_motion else 'following_observed_return_route'
        self.stats['return_motion_ticks']+=1
        if repair_motion:self.stats['repair_motion_ticks']+=1
        return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)

    def status(self):
        value=super().status();value.update(repair_goal_index=self._repair_goal_index30,repair_waypoints=len(self._repair))
        return value
