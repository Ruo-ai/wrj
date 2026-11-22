"""Delivery39R1: advance reached breadcrumb nodes before live look-ahead.

Only the return executor is changed; the failed initial trial remains frozen.
Continuous38 estimation/rendering and the Delivery39 schedule/protocol persist.
"""
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_delivery39 import MyDroneRedDelivery39
from swarm_rescue.solutions.my_drone_red_stopturn import body_motion
from swarm_rescue.solutions.my_drone_red_advanced2 import _M11_safe_velocity


class MyDroneRedDelivery39R1(MyDroneRedDelivery39):
    def _return_motion39(self):
        command = self._empty_command(); self._gathering = True
        d = self.lidar_values(); angles = self.lidar_rays_angles()
        measured = body_motion(self.odometer_values())
        if measured is not None:self._velocity = .4*self._velocity+.6*measured
        if d is None:
            v=np.clip(-.7*self._velocity,-1,1)
            command.update(forward=float(v[0]),lateral=float(v[1]));return command
        d=np.asarray(d).ravel();angles=np.asarray(angles).ravel()
        dynamic=np.zeros(len(d),bool);bombs=np.zeros(len(d),bool)
        for obj in self.semantic_values() or ():
            kind=getattr(obj.entity_type,'name',None)
            if kind in ('DRONE','BOMB'):
                delta=np.abs((angles-obj.angle+math.pi)%(2*math.pi)-math.pi)
                matched=(delta<.19)&(np.abs(d-obj.distance)<25)
                dynamic|=matched
                if kind=='BOMB':bombs|=matched
        self._explore.step=self._steps
        if self._steps%6==0:self._explore.observe(self._position,self._heading,d,angles,dynamic)
        if self._steps%30==0:
            self._visited_snapshot=np.packbits(self._explore.shared_visited|(self._explore.visits>0)).tobytes()
            self._radio_maps.publish(int(self.identifier or 0),self._steps,self._map.mask())
        root=self._walk39.points[0]
        # Keep a live radio footprint at the agreed observed launch area.
        home_distance=float(np.linalg.norm(self._position-root))
        if home_distance < 30:
            if self._d39_mode!='exchange':self._note39('exchange_arrived',position=self._position.tolist())
            self._d39_mode='exchange';self._holding=True;self._d39_stats['exchange_ticks']+=1
            v=np.clip(-.7*self._velocity,-1,1)
            command.update(forward=float(v[0]),lateral=float(v[1]));return command
        self._holding=False;self._d39_mode='return'
        if self._steps-self._d39_plan_step>=12 or not self._d39_route:self._path39()
        route=list(self._d39_route)
        # Consume reached vertices before testing look-ahead. A corner that
        # cannot be string-pulled is still the next walked edge, not a request
        # to keep steering at the current position. Live avoidance below owns
        # all translation and braking; there is no bypass of sensor safety.
        while len(route)>1 and np.linalg.norm(self._walk39.points[route[0]]-self._position)<18:
            route.pop(0)
            self._d39_stats['reached_nodes_consumed']+=1
        self._d39_route=route
        target=self._walk39.points[route[0]] if route else root
        node=route[0] if route else 0
        for candidate in route[:8]:
            point=self._walk39.points[candidate]
            if np.linalg.norm(point-self._position)>110:break
            if self._walk39.visible(self._position,point,self._heading,d,angles):
                target=point;node=candidate
            else:break
        distance=float(np.linalg.norm(target-self._position))
        if self._d39_target!=node:
            self._d39_target=node;self._d39_best=distance;self._d39_advance=self._steps
        elif distance<self._d39_best-6:
            self._d39_best=distance;self._d39_advance=self._steps
        direction=math.atan2(*(target-self._position)[::-1])-self._heading
        if self._steps-self._d39_advance>65:
            self._d39_stats['bounded_recovery']+=1
            if len(route)>1:self._walk39.blocked[tuple(sorted(route[:2]))]=self._steps+180
            # Brief live-scan escape, then retry graph routing; never wait forever.
            headings=np.linspace(-math.pi,math.pi,24,endpoint=False)
            from swarm_rescue.solutions.my_drone_red_stopturn import ray_clearance
            clear=np.array([ray_clearance(d,angles,a,22) for a in headings])
            scores=np.minimum(clear,100)+25*np.cos(headings-direction)
            self._d39_escape_angle=float(headings[int(np.argmax(scores))]+self._heading)
            self._d39_escape_until=self._steps+18;self._d39_advance=self._steps
            self._d39_plan_step=-100
            self._note39('return_recovery',target=target.tolist(),position=self._position.tolist())
        if self._steps<self._d39_escape_until:
            direction=self._d39_escape_angle-self._heading;distance=65.
        steering=d.copy();steering[bombs]=80 # Retain existing movement-only bomb-ray handling.
        move,self._last_motion_angle=_M11_safe_velocity(steering,angles,
            (direction+math.pi)%(2*math.pi)-math.pi,self._velocity,self._last_motion_angle,max(18.,distance))
        command.update(forward=float(move[0]),lateral=float(move[1]),rotation=0.)
        self._d39_stats['return_ticks']+=1
        return command

