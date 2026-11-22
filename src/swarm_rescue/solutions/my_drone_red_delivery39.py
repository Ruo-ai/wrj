"""Continuous38 unchanged mapping, with walked-path delivery and pending frontiers.

    Communication contracts, receipts and positions come only from real radio
    messages. No simulator truth, map identity, seed or zone geometry is read.
"""
from collections import Counter, deque
import math
from types import SimpleNamespace
import numpy as np

from swarm_rescue.solutions.my_drone_red_continuous38 import MyDroneRedContinuous38
from swarm_rescue.solutions.my_drone_red_advanced2 import _M11_safe_velocity
from swarm_rescue.solutions.my_drone_red_stopturn import body_motion
from swarm_rescue.solutions.red_delivery39 import WalkGraph39, FrontierQueue39


class MyDroneRedDelivery39(MyDroneRedContinuous38):
    GATHER_AT = 1.1  # Replaced by route + measured wall-clock delivery budget.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._explore = FrontierQueue39(self.size_area, self._event)
        self._walk39 = WalkGraph39()
        self._d39_stats = Counter(); self._d39_events = []; self._d39_samples = []
        self._d39_mode = 'explore'; self._d39_route = []; self._d39_cost = None
        self._d39_plan_step = -100; self._d39_target = None; self._d39_best = float('inf')
        self._d39_advance = 0; self._d39_clock = deque(maxlen=120)
        self._d39_contract = None; self._d39_accepts = set(); self._d39_contacts = {}
        self._d39_rx_step = -1; self._d39_reports = {}; self._d39_budget = {}
        self._d39_escape_until = 0; self._d39_escape_angle = 0.

    def _note39(self, kind, **data):
        row = dict(step=self._steps,kind=kind,**data)
        if len(self._d39_events) < 250:
            self._d39_events.append(row)
        self._event('d39_'+kind,**data)

    def _update_observation(self):
        super()._update_observation()
        self._walk39.observe(self._position,self._heading,self.lidar_values(),
                             self.lidar_rays_angles(),self._steps)
        self._d39_clock.append((self._steps,float(self.elapsed_walltime)))
        if int(self.identifier or 0) == 0 and self._d39_contract is None and self._f27_home is not None:
            # Initially connected team agrees on one observed final meeting point.
            # Each robot may use its verified launch node as the local fallback.
            self._d39_contract = (int(self.elapsed_timestep),tuple(self._f27_home[2]),.86,.95)
            self._d39_accepts.add(0)

    @staticmethod
    def _valid_contract39(value, clock):
        return (isinstance(value,tuple) and len(value)==4 and type(value[0]) is int
            and 0 <= value[0] <= clock and isinstance(value[1],tuple) and len(value[1])==2
            and all(isinstance(v,(int,float)) and math.isfinite(v) for v in value[1])
            and value[2:]==(.86,.95))

    def define_message_for_all(self):
        base = super().define_message_for_all()
        versions = tuple((i,s[1]) for i,s in sorted(self._radio_maps.snapshots.items()))
        report = (int(self.identifier or 0),int(self.elapsed_timestep),self._d39_mode,versions)
        return ('delivery39-v1',base,self._d39_contract,tuple(sorted(self._d39_accepts)),report)

    def _receive(self):
        stripped = []; contacts = []
        for sender,message in self.communicator.received_messages:
            if not isinstance(message,tuple) or len(message)!=5 or message[0]!='delivery39-v1':
                continue
            _,base,contract,accepts,report = message
            stripped.append((sender,base))
            if self._valid_contract39(contract,self.elapsed_timestep):
                if self._d39_contract is None:
                    self._d39_contract = contract
                    self._d39_accepts.add(int(self.identifier or 0))
                if contract == self._d39_contract and isinstance(accepts,tuple) and len(accepts)<=self._n:
                    self._d39_accepts.update(i for i in accepts if type(i)is int and 0<=i<self._n)
            if (isinstance(report,tuple) and len(report)==4 and type(report[0])is int
                    and 0<=report[0]<self._n and type(report[1])is int
                    and 0<=report[1]<=self.elapsed_timestep and report[2] in ('explore','return','exchange')
                    and isinstance(report[3],tuple) and len(report[3])<=self._n
                    and all(isinstance(v,tuple) and len(v)==2 and type(v[0])is int
                            and 0<=v[0]<self._n and type(v[1])is int and 0<=v[1]<=report[1] for v in report[3])):
                self._d39_reports[report[0]] = report
                contacts.append(report[0])
        # Proxy unwraps only this candidate envelope. Keep the real validated
        # snapshot/manifest/ACK decoder intact, without modifying the device.
        proxy = SimpleNamespace(**self.__dict__)
        proxy.elapsed_timestep = self.elapsed_timestep
        proxy._valid_peer = self._valid_peer
        proxy.communicator = SimpleNamespace(received_messages=stripped)
        MyDroneRedContinuous38._receive(proxy)
        self._f27_home = proxy._f27_home
        if contacts and self._position is not None:
            if self._steps-self._d39_rx_step > 120:
                self._note39('contact_restored',peers=contacts,position=self._position.tolist())
            self._d39_rx_step = self._steps
            for i in contacts:
                self._d39_contacts[i] = (self._steps,self._position.copy())

    def _try_hold(self, distances, dynamic, command):
        if self._d39_mode == 'explore':
            # Legacy motion sets .72 gathering before this virtual dispatch.
            # It must not override this candidate's explicit delivery budget.
            self._gathering = False; self._holding = False
            return False
        return super()._try_hold(distances,dynamic,command)

    def _path39(self):
        self._d39_route,self._d39_cost = self._walk39.path(self._position,self._heading,
            self.lidar_values(),self.lidar_rays_angles(),self._steps)
        self._d39_plan_step = self._steps

    def _budget39(self):
        limits = self._misc_data
        step_limit = getattr(limits,'max_timestep_limit',None) or 5400
        seconds_limit = getattr(limits,'max_walltime_limit',None) or 180
        rate = .04
        if len(self._d39_clock)>20:
            a,b = self._d39_clock[0],self._d39_clock[-1]
            rate = max(.02,(b[1]-a[1])/max(1,b[0]-a[0]))
        remaining = min(.94*step_limit-self.elapsed_timestep,
                        (.94*seconds_limit-self.elapsed_walltime)/rate)
        cost = self._d39_cost if self._d39_cost is not None and math.isfinite(self._d39_cost) else 2400.
        needed = cost/2.1+180+100 # travel, recovery allowance, payload/ACK exchange
        self._d39_budget = dict(remaining=float(remaining),needed=float(needed),
            route_pixels=float(cost),seconds_per_step=float(rate))
        return remaining <= needed or self._progress() >= .84

    def _navigation(self):
        if self._position is None:
            return super()._navigation()
        if self._steps-self._d39_plan_step >= 24:
            self._path39()
        due = self._budget39()
        if self._d39_mode == 'explore' and due and self.carried_bombs_count()==0:
            self._d39_mode = 'return'; self._gathering = True; self._holding = False
            self._route = []; self._bomb_target = None
            self._d39_advance = self._steps
            if self._entry37.active:self._entry37.release('delivery39_return')
            self._segment_motion.release('delivery39_return',cooldown=1)
            if self._rn7.path:self._rn7.release('delivery39_return')
            self._note39('return_started',budget=self._d39_budget,
                contract_accepted=sorted(self._d39_accepts),position=self._position.tolist())
            self._f27_stats['gather_started_control'] = self._steps
        self._explore.extension_enabled = (self._d39_mode=='explore'
            and self.carried_bombs_count()==0 and self._progress()>=.60 and not due)
        if self._d39_mode == 'explore':
            self._gathering = False
            return super()._navigation()
        return self._return_motion39()

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
        route=self._d39_route
        # Look ahead only along the return path and with fresh radar clearance.
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

    def control(self):
        command=super().control()
        if self._steps%30==0:
            self._d39_samples.append(dict(step=self._steps,mode=self._d39_mode,
                budget=dict(self._d39_budget),last_rx=self._d39_rx_step,
                nodes=len(self._walk39.points),route=self._d39_route[:12],
                versions={str(i):s[1] for i,s in self._radio_maps.snapshots.items()},
                accepted=sorted(self._d39_accepts)))
        return command
