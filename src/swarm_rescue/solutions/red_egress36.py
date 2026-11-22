"""Bounded departure from an own deployment, using a pre-drop legal corridor.

Bombs ignore drone collision but remain ray-visible. An inventory decrement
plus local BOMB semantics may authorize a short previously observed corridor;
it never turns the occluded current scan into free space. Cached geometry is
private to this one departure and is never published to mapping/navigation.
"""
from collections import Counter, deque
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_rn7 import HULL, rotate
from swarm_rescue.solutions.red_return30 import ReturnLocal30


class Egress36:
    PREPARE_MAX_AGE = 4
    MAX_TICKS = 70
    MAX_DISTANCE = 60.
    CORRIDOR_LENGTH = 48.
    EXTRA_MARGIN = 4.
    MAX_SPEED = 1.2
    MAX_YAW = .10
    MAX_STEP_DISTANCE = 5.
    POSITION_CHECK = 24.
    NO_PROGRESS_TICKS = 24
    CLEAR_FRAMES = 2
    BOMB_RADIUS = 12.

    def __init__(self, identifier=0):
        self.identifier = int(identifier)
        self.stats = Counter()
        self.events = deque(maxlen=80)
        self.state = 'idle'
        self.reason = 'inactive'
        self.active = self.failed = False
        self._plan = None
        self._last_step = None
        self._command = None
        self._offset = np.zeros(2)
        self._yaw = 0.
        self._velocity = np.zeros(2)
        self._distance = 0.
        self._clear_frames = 0
        self._confirmed = None
        self._started = None
        self._progress_tick = None
        self._progress_position = np.zeros(2)
        self._last_scan = {}
        self._last_stop = None

    def _event(self, reason, step):
        self.reason = reason
        self.stats[reason] += 1
        self.events.append(dict(step=int(step), reason=reason))

    @staticmethod
    def _valid_pose(position, heading, odometry):
        return (position is not None and np.shape(position)==(2,)
                and np.isfinite(position).all() and heading is not None
                and math.isfinite(heading) and odometry is not None
                and np.shape(odometry)==(3,) and np.isfinite(odometry).all())

    @staticmethod
    def _scan(lidar, angles, semantic):
        if lidar is None or angles is None:
            return None, None, None
        d, a = np.asarray(lidar,float).ravel(), np.asarray(angles,float).ravel()
        if len(d)<24 or d.shape!=a.shape or not np.isfinite(a).all():
            return None, None, None
        if np.count_nonzero(np.isfinite(d)) < .9*len(d):
            return None, d, a
        return ReturnLocal30(d,a,semantic), d, a

    @staticmethod
    def _body_velocity(odometry):
        distance, alpha, theta = map(float,odometry)
        return distance*np.array([math.cos(alpha-theta), math.sin(alpha-theta)])

    def prepare(self, position, heading, odometry, lidar, angles, semantic, inventory, step):
        """Call on the *requested place_bomb command*, before engine deployment.

        False means no safe departure was certified: caller should cancel this
        placement and clear its pending placement request. No scan is modified.
        The accepted placement frame must keep rotation=0.
        """
        step = int(step)
        if self.active or self.failed:
            self._event('prepare_rejected_busy',step)
            return False
        if (not self._valid_pose(position,heading,odometry)
                or inventory is None or int(inventory)<=0
                or abs(float(odometry[2]))>.035
                or abs(float(odometry[0]))>2.):
            self._event('prepare_invalid_or_unsettled',step)
            return False
        fine,d,a = self._scan(lidar,angles,semantic)
        if fine is None:
            self._event('prepare_no_scan',step)
            return False
        velocity = self._body_velocity(odometry)
        preferred = -velocity if np.linalg.norm(velocity)>.1 else np.array([-1.,0.])
        preferred /= np.linalg.norm(preferred)
        choices=[]
        for angle in np.arange(24)*math.pi/12:
            direction=np.array([math.cos(angle),math.sin(angle)])
            end=direction*self.CORRIDOR_LENGTH
            points=np.linspace(np.zeros(2),end,33)
            clearance=float(fine.sample(points).min())
            if (clearance >= fine.PLAN_RADIUS+self.EXTRA_MARGIN
                    and fine.line_clear(np.zeros(2),end)):
                choices.append((float(np.dot(direction,preferred))*5+min(clearance,40),angle,end))
        if not choices:
            self._event('prepare_no_verified_corridor',step)
            return False
        _,_,endpoint=max(choices,key=lambda v:(v[0],-v[1]))
        self._plan=dict(tick=step,inventory=int(inventory),position=np.asarray(position,float).copy(),
            heading=float(heading),endpoint=endpoint.copy(),fine=fine,lidar=d.copy(),angles=a.copy())
        self._offset=np.zeros(2);self._yaw=0.;self._velocity=velocity.copy();self._distance=0.
        self._clear_frames=0;self._confirmed=None;self._started=None
        self._progress_tick=step;self._progress_position=np.zeros(2)
        self._last_step=step;self._last_stop=None;self._command=None
        self.active=False;self.failed=False;self.state='prepared'
        self._event('corridor_prepared',step)
        return True

    def _brake(self):
        force=np.clip(-.65*self._velocity,-1.,1.)
        return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)

    def _fail(self, reason, step):
        self.failed=True;self.active=False;self.state='failed'
        self._event(reason,step)
        self._command=self._brake()
        return self.command()

    def _release(self, reason, step):
        self.active=False;self.failed=False;self.state='cleared'
        self._event(reason,step)
        self._command=None
        return None

    def _own_bomb_mask(self, distances, angles, objects):
        """Associate current rays with the just-deployed bomb, without filling."""
        mask=np.zeros(len(distances),bool)
        close=[]
        local_bomb=rotate(-self._offset,-self._yaw)
        center_distance=float(np.linalg.norm(local_bomb))
        expected_distance=max(0.,center_distance-self.BOMB_RADIUS)
        expected_angle=math.atan2(local_bomb[1],local_bomb[0])
        half_angle=(math.pi if center_distance<=self.BOMB_RADIUS+3 else
            math.asin(min(1.,(self.BOMB_RADIUS+3)/center_distance))+.15)
        for obj in objects or ():
            if getattr(obj.entity_type,'name','')!='BOMB':
                continue
            if not math.isfinite(obj.distance) or not math.isfinite(obj.angle):
                continue
            bearing=abs((float(obj.angle)-expected_angle+math.pi)%(2*math.pi)-math.pi)
            if obj.distance<0 or abs(float(obj.distance)-expected_distance)>10 or bearing>half_angle:
                continue
            close.append(obj)
            delta=np.abs((angles-float(obj.angle)+math.pi)%(2*math.pi)-math.pi)
            mask |= ((delta<.20)&np.isfinite(distances)
                     &(np.abs(distances-float(obj.distance))<=14))
        return mask,close

    @staticmethod
    def _point_to_segment(points, end):
        points=np.asarray(points,float)
        scale=np.clip(points@np.asarray(end)/max(1e-9,float(np.dot(end,end))),0.,1.)
        return np.linalg.norm(points-scale[:,None]*end,axis=1)

    def _fresh_obstruction(self, d, a, objects, own_mask, stopping):
        # Use a conservative circle only for detecting a NEW current obstacle;
        # the cached route still uses the original physical convex footprint.
        radius=float(np.linalg.norm(HULL,axis=1).max())+self.EXTRA_MARGIN
        hit=np.isfinite(d)&~own_mask&(d<210.)
        if np.any(hit):
            endpoints=np.maximum(d[hit],0.)[:,None]*np.c_[np.cos(a[hit]),np.sin(a[hit])]
            if np.any(self._point_to_segment(endpoints,stopping)<radius):
                return 'current_nonself_ray_obstruction'
        for obj in objects or ():
            if getattr(obj.entity_type,'name','')=='BOMB':
                continue  # Non-self BOMB rays were retained above.
            if not math.isfinite(obj.distance) or not math.isfinite(obj.angle):
                return 'current_object_invalid'
            point=max(0.,float(obj.distance))*np.array([[math.cos(obj.angle),math.sin(obj.angle)]])
            if self._point_to_segment(point,stopping)[0]<radius:
                return 'current_nonbomb_object_obstruction'
        return None

    def update(self, position, heading, odometry, lidar, angles, semantic, inventory, step):
        """One call per control step; return an override dict or None to hand off.

        A failed episode remains latched. Caller must explicitly acknowledge it;
        repeated calls do not refresh the old scan or renew its motion budget.
        """
        step=int(step)
        if self._last_step==step:
            return self.command()
        previous=self._last_step;self._last_step=step
        if self.failed:
            # No retry of the expired corridor. A genuinely fresh complete
            # hull observation can nevertheless return control to navigation.
            if self._valid_pose(position,heading,odometry):
                self._velocity=self._body_velocity(odometry)
                fresh,_,_=self._scan(lidar,angles,semantic)
                if fresh is not None and fresh.line_clear(np.zeros(2),np.zeros(2)):
                    self._plan=None
                    return self._release('late_scan_recovered',step)
            self._command=self._brake()
            return self.command()
        if self.state not in ('prepared','active') or self._plan is None:
            self._command=None
            return None
        if not self._valid_pose(position,heading,odometry):
            self._velocity=np.zeros(2)
            return self._fail('egress_invalid_pose_or_odometry',step)
        if previous is None or step-previous!=1:
            return self._fail('egress_missing_odometry_interval',step)
        distance,alpha,theta=map(float,odometry)
        if abs(distance)>self.MAX_STEP_DISTANCE:
            return self._fail('egress_odometry_jump',step)
        self._offset += rotate(distance*np.array([math.cos(alpha),math.sin(alpha)]),self._yaw)
        self._yaw += theta
        measured=self._body_velocity(odometry)
        self._velocity=.4*rotate(self._velocity,-theta)+.6*measured
        self._distance+=abs(distance)
        gps_shift=rotate(np.asarray(position,float)-self._plan['position'],-self._plan['heading'])
        if np.linalg.norm(gps_shift-self._offset)>self.POSITION_CHECK:
            return self._fail('egress_localization_disagreement',step)
        if abs(self._yaw)>self.MAX_YAW:
            return self._fail('egress_heading_budget',step)
        if np.linalg.norm(self._offset-self._progress_position)>=2.:
            self._progress_position=self._offset.copy();self._progress_tick=step
        fine,d,a=self._scan(lidar,angles,semantic)
        if self.state=='prepared':
            if int(inventory)<self._plan['inventory']:
                self._confirmed=step
                if step-self._plan['tick']>self.PREPARE_MAX_AGE:
                    return self._fail('egress_confirmation_after_cache_expiry',step)
                if fine is None:
                    return self._fail('egress_confirmation_without_scan',step)
                own_mask,objects=self._own_bomb_mask(d,a,semantic)
                if len(objects)>=4 and np.count_nonzero(own_mask)>=.20*len(d):
                    self.active=True;self.state='active';self._started=step
                    self._event('own_drop_and_bomb_confirmed',step)
                elif fine.line_clear(np.zeros(2),np.zeros(2)):
                    return self._release('drop_already_clear',step)
                else:
                    return self._fail('egress_unconfirmed_occluder',step)
            elif step-self._plan['tick']>self.PREPARE_MAX_AGE:
                return self._release('placement_not_confirmed',step)
            else:
                self._command=None
                return None
        if step-self._started>self.MAX_TICKS or self._distance>self.MAX_DISTANCE:
            return self._fail('egress_motion_budget',step)
        if step-self._progress_tick>self.NO_PROGRESS_TICKS:
            return self._fail('egress_no_odometry_progress',step)
        if fine is None:
            self._event('egress_current_scan_missing',step)
            self._command=self._brake()
            return self.command()
        own_mask,bombs=self._own_bomb_mask(d,a,semantic)
        self._last_scan=dict(step=step,rays=len(d),own_bomb_rays=int(own_mask.sum()),
            long_rays=int(np.count_nonzero(np.isfinite(d)&(d>25))),near_bomb_semantics=len(bombs))
        if fine.line_clear(np.zeros(2),np.zeros(2)):
            self._clear_frames+=1
            if self._clear_frames>=self.CLEAR_FRAMES:
                return self._release('fresh_hull_cleared',step)
        else:
            self._clear_frames=0
        if not np.isfinite(d).all():
            # Missing measurements are not attributed to the new bomb merely
            # because it was recently deployed.
            self._event('egress_unattributed_missing_rays',step)
            self._command=self._brake()
            return self.command()
        target=self._plan['endpoint']-self._offset
        remaining=float(np.linalg.norm(target))
        if remaining<4.:
            return self._fail('egress_corridor_exhausted_without_clear_scan',step)
        desired=rotate(target/max(remaining,1e-9),-self._yaw)*min(self.MAX_SPEED,math.sqrt(.7*max(0.,remaining-3.)))
        force=.085*desired+.65*(desired-self._velocity)
        force/=max(1.,float(np.linalg.norm(force)))
        predicted=.95*self._velocity+.6*force
        stopping=3*predicted+predicted*np.linalg.norm(predicted)/1.2
        end=self._offset+rotate(stopping,self._yaw)
        cached=self._plan['fine']
        if (not cached.line_clear(self._offset,end)
                or float(cached.sample(np.linspace(self._offset,end,12)).min())<cached.PLAN_RADIUS+self.EXTRA_MARGIN
                or np.linalg.norm(end)>self.CORRIDOR_LENGTH+2):
            return self._fail('egress_cached_stopping_sweep_rejected',step)
        obstruction=self._fresh_obstruction(d,a,semantic,own_mask,stopping)
        if obstruction:
            self._event(obstruction,step)
            self._command=self._brake()
            return self.command()
        self._last_stop=dict(current_body_vector=stopping.tolist(),cached_start=self._offset.tolist(),
            cached_end=end.tolist(),cache_control_step=self._plan['tick'],fresh_control_step=step,
            physical_hull_unchanged=True,extra_margin=self.EXTRA_MARGIN)
        self._event('certified_egress_motion_command',step)
        self.stats['motion_command_ticks']+=1
        self._command=dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)
        return self.command()

    def command(self):
        return None if self._command is None else dict(self._command)

    def acknowledge_failure(self):
        """Release control without authorizing another move through this cache."""
        self._plan=None;self._command=None;self.active=False;self.failed=False;self.state='idle'

    def status(self):
        return dict(state=self.state,reason=self.reason,active=self.active,failed=self.failed,
            prepared_control_step=None if self._plan is None else self._plan['tick'],
            confirmed_control_step=self._confirmed,step=self._last_step,
            endpoint_prepared_frame=None if self._plan is None else self._plan['endpoint'].tolist(),
            odometry_offset=self._offset.tolist(),odometry_distance_px=float(self._distance),
            odometry_yaw=float(self._yaw),clear_frames=self._clear_frames,
            current_scan=dict(self._last_scan),last_stopping_certificate=self._last_stop,
            stats=dict(self.stats),events=list(self.events),
            movement_metric='motion_command_ticks is command count; actual displacement requires odometry/telemetry',
            observation_scope='one pre-deployment observed corridor; no fabricated current free rays')
