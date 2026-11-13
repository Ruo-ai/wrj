"""Independent B4 contact hysteresis; no claim that release equals disposal."""
import math
import numpy as np
from swarm_rescue.solutions.my_drone_blue_transport import MyDroneBlueTransportCompute


class ContactDelivery:
    def __init__(self):self.reset()
    def reset(self):
        self.active=False;self.last_seen=-1000;self.turn=0.;self.started=0
        self.direction=1;self.state='inactive';self.stable=0;self.last_distance=None;self.retry_until=0

    def command(self,step,carrying,hit,odo,distances,angles,fallback,direction=1):
        if not carrying:self.reset();return None
        if not len(distances) or not (np.isfinite(distances)&(distances>0)).any():
            self.state='missing_lidar_hold'
            return dict(forward=0.,lateral=0.,rotation=0.,grasper=1)
        if step<self.retry_until:
            self.state='retry_navigation';return dict(fallback,grasper=1)
        valid=hit is not None and math.isfinite(hit.distance) and math.isfinite(hit.angle) and hit.distance>0
        if not valid:
            if self.active and step-self.last_seen<=6:
                self.state='short_loss_hold'
                # Preserve turn history but do not move or release blind.
                return dict(forward=0.,lateral=0.,rotation=0.,grasper=1)
            self.active=False;self.turn=0.;self.stable=0;self.state='center_unseen'
            return dict(fallback,grasper=1)
        distance=float(hit.distance);angle=float(hit.angle)
        self.last_seen=step;self.last_distance=distance
        if distance>=105:
            self.active=False;self.turn=0.;self.state='navigate'
            return dict(fallback,grasper=1)
        if distance<18 and not self.active:
            self.active=True;self.started=step;self.turn=0.;self.direction=direction;self.stable=0
        if self.active and distance<=26:
            if odo is not None and np.shape(odo)==(3,) and np.isfinite(odo).all() and abs(odo[2])<.4:
                self.turn=max(0.,self.turn+float(odo[2])*self.direction)
            self.stable=self.stable+1 if distance<18 else 0
            self.state='contact_turn'
            cmd=dict(forward=0.,lateral=0.,rotation=.65*self.direction,grasper=1)
            # Physical contact normally disposes while still held. A late
            # bounded release is only a REQUEST, never a success transition.
            if self.turn>=2*math.pi and self.stable>=8:
                cmd['grasper']=0;self.state='release_request_unconfirmed'
            elif step-self.started>=600:
                self.active=False;self.turn=0.;self.state='contact_retry'
                self.retry_until=step+45
                return dict(fallback,grasper=1)
            return cmd
        if self.active:
            self.active=False;self.turn=0.;self.stable=0
        cone=np.abs((angles-angle+math.pi)%(2*math.pi)-math.pi)<.3
        blockers=distances[cone&np.isfinite(distances)&(distances>0)]
        if not len(blockers) or blockers.min()<distance-12:
            self.state='obstructed_use_navigation'
            return dict(fallback,grasper=1)
        self.state='relative_approach'
        return dict(forward=.3*math.cos(angle),lateral=.3*math.sin(angle),
                    rotation=float(np.clip(angle*1.5,-.65,.65)),grasper=1)


class MyDroneBlueContact(MyDroneBlueTransportCompute):
    CONTACT_DELIVERY=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self._contact=ContactDelivery()
        self._contact_state='inactive';self._command_source=None

    def control(self):
        self._command_source=None
        command=super().control()
        if not self.CONTACT_DELIVERY:return command
        raw=self.lidar_values();distances=np.asarray(raw if raw is not None else [],float)
        angles=np.asarray(self.lidar_rays_angles(),float)[:len(distances)]
        if len(angles)!=len(distances):return command
        carrying=bool(self.grasped_bombs())
        fallback=self._last_navigation_cmd if carrying else command
        result=self._contact.command(self._tick,carrying,self._center_hit,self.odometer_values(),
                                      distances,angles,fallback,self._escape_sign)
        self._contact_state=self._contact.state
        if result is not None:self._command_source='contact:'+self._contact.state
        return command if result is None else result
