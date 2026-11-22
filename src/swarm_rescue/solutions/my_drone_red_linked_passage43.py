"""ReturnBudget42 pose/rendering with delivery-aware contact and fine passages.

Only legal own sensors, public clocks and received messages enter decisions.
No map identity, seed, truth, disabler geometry or shared team memory.
"""
from collections import Counter
from types import SimpleNamespace
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_return_budget42 import MyDroneRedReturnBudget42
from swarm_rescue.solutions.my_drone_red_rn7 import PassageMotion
from swarm_rescue.solutions.my_drone_red_stopturn import body_motion
from swarm_rescue.solutions.my_drone_red_advanced2 import _M11_safe_velocity
from swarm_rescue.solutions.red_linked_passage43 import valid_receipt, receipt_delivered, observed_approach


class MyDroneRedLinkedPassage43(MyDroneRedReturnBudget42):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lp43_stats=Counter();self._lp43_events=[];self._lp43_samples=[]
        self._lp43_receipt=None;self._lp43_link={};self._lp43_stable=0
        self._lp43_contact_node=None;self._lp43_target_node=0
        self._lp43_contact_wait=None;self._lp43_fallback=False
        self._fine43=PassageMotion(int(self.identifier or 0))
        self._fine43_goal=None;self._fine43_start=0;self._fine43_retry=0
        self._lp43_default_narrow=self.NARROW_ENABLED

    def _note43(self,kind,**data):
        self._lp43_stats[kind]+=1
        if len(self._lp43_events)<400:
            self._lp43_events.append(dict(step=self._steps,kind=kind,**data))

    def _update_observation(self):
        super()._update_observation()
        self._fine43.update(self.odometer_values())

    def define_message_for_all(self):
        base=super().define_message_for_all()
        if int(self.identifier or 0)==0:
            self._lp43_receipt=(int(self.elapsed_timestep),tuple((i,s[1],self._radio_maps.payload_digest(i))
                for i,s in sorted(self._radio_maps.snapshots.items())))
        return ('linked43-v1',base,self._lp43_receipt)

    def _receive(self):
        stripped=[]
        for sender,message in self.communicator.received_messages:
            if not (isinstance(message,tuple) and len(message)==3 and message[0]=='linked43-v1'):continue
            stripped.append((sender,message[1]));receipt=message[2]
            if valid_receipt(receipt,self._n,int(self.elapsed_timestep)):
                if self._lp43_receipt is None or receipt[0]>self._lp43_receipt[0]:
                    self._lp43_receipt=receipt
                    self._lp43_stats['received_leader_receipt']+=1
        proxy=SimpleNamespace(**self.__dict__)
        proxy.elapsed_timestep=self.elapsed_timestep
        proxy._valid_peer=self._valid_peer;proxy._valid_contract39=self._valid_contract39
        proxy._note39=self._note39
        proxy.communicator=SimpleNamespace(received_messages=stripped)
        MyDroneRedReturnBudget42._receive(proxy)
        for name in ('_f27_home','_d39_contract','_d39_rx_step'):
            setattr(self,name,getattr(proxy,name))

    def _link_state43(self):
        own=int(self.identifier or 0);clock=int(self.elapsed_timestep);store=self._radio_maps
        sealed=store.own_final is not None
        if own==0:
            missing=[]
            for i in range(self._n):
                snap=store.snapshots.get(i)
                final=store.manifests.get(i)
                if sealed:
                    good=bool(snap and final and snap[1]==final[2] and store.payload_digest(i)==final[3])
                else:good=bool(snap and (i==own or clock-snap[1]<=120))
                if not good:missing.append(i)
            healthy=not missing
        else:
            snap=store.snapshots.get(own)
            healthy=receipt_delivered(self._lp43_receipt,own,snap,store.payload_digest(own),clock,sealed)
            missing=[] if healthy else [own]
        self._lp43_link=dict(healthy=healthy,sealed=sealed,missing_sources=missing,
            leader_receipt_step=None if self._lp43_receipt is None else self._lp43_receipt[0])
        return healthy

    def _path39(self):
        graph=self._walk39
        node=self._lp43_contact_node if not self._lp43_fallback else None
        if node is None or node>=len(graph.points):node=0
        self._lp43_target_node=node
        target=graph.points[node] if graph.points else None
        self._d39_route,self._d39_cost=graph.path(self._position,self._heading,
            self.lidar_values(),self.lidar_rays_angles(),self._steps,target=target)
        self._d39_plan_step=self._steps

    def _budget39(self):
        super()._budget39()
        healthy=self._link_state43()
        # A recent end-to-end receipt permits exploration even beyond old .84.
        # An immutable final still requires exact source sequence AND digest.
        due=(not healthy and (self._d39_budget['remaining']<=self._d39_budget['needed']
                             or self._radio_maps.own_final is not None))
        self._d39_budget.update(link_healthy=healthy,budget_due=bool(due),
            reconnect_target_node=self._lp43_target_node,forced_fraction_removed=True)
        if healthy and self._d39_budget['legacy_due']:self._lp43_stats['connected_budget_deferred']+=1
        return due

    def _navigation(self):
        healthy=self._link_state43()
        self._lp43_stable=self._lp43_stable+1 if healthy else 0
        if healthy and self._walk39.last is not None:
            # CorrectedWalkGraph maintains this observed node through pose recovery.
            self._lp43_contact_node=self._walk39.last
            self._lp43_fallback=False;self._lp43_contact_wait=None
        if healthy and self._radio_maps.own_final is not None:
            self._finish_fine43('final_exchange')
            self._d39_mode='exchange';self._gathering=True;self._holding=True
            self._lp43_stats['connected_final_exchange_ticks']+=1
            command=self._empty_command();v=np.clip(-.7*self._velocity,-1,1)
            command.update(forward=float(v[0]),lateral=float(v[1]),rotation=0.)
            return command
        if self._d39_mode!='explore' and self._lp43_stable>=30 and self._radio_maps.own_final is None:
            self._d39_mode='explore';self._gathering=False;self._holding=False
            self._route=[];self._d39_target=None
            self._note43('exploration_resumed_after_delivery')
        if self._entry37.active or self._d39_mode!='explore':self._finish_fine43('priority_preempted')
        self.NARROW_ENABLED=self._lp43_default_narrow and self._fine43_goal is None
        return super()._navigation()

    def _release_stale_route(self,reason):
        eligible=(reason=='new_wall' and self._route and self._d39_mode=='explore'
            and self._bomb_target is None and self.carried_bombs_count()==0
            and self._steps>=self._fine43_retry and self._radio_maps.own_final is None
            and not self._entry37.active)
        if eligible:
            goal=self._explore.point(self._route[-1])
            if self._explore.unknown_near(self._route[-1])>=3 and self._start_fine43(goal):
                self._note43('coarse_rejection_rescued',goal=goal.tolist(),position=self._position.tolist())
                return
        return super()._release_stale_route(reason)

    def _start_fine43(self,goal,continuation=False):
        d,a=self.lidar_values(),self.lidar_rays_angles()
        if d is None or a is None:return False
        fine,path=observed_approach(d,a,self.semantic_values(),self._position,self._heading,goal)
        self._lp43_stats['fine_probes']+=1
        if not path:
            self._lp43_stats['fine_probe_rejected']+=1;self._fine43_retry=self._steps+30
            return False
        if self._rn7.path:self._rn7.release('fine43_goal_priority')
        self._segment_motion.release('fine43_goal_priority',cooldown=1)
        task=self._fine43;task.path=[p.copy() for p in path[1:]]
        task.since=task.tick;task.last_progress=task.tick;task.best_remaining=float('inf')
        task.snapshot=fine;task.last_refresh=task.tick;task.blocked_ticks=0;task.wait=0
        task.event('targeted_fine_leg')
        self._fine43_goal=np.asarray(goal).copy()
        if not continuation:self._fine43_start=self._steps
        self.NARROW_ENABLED=False
        return True

    def _finish_fine43(self,reason):
        if self._fine43_goal is None:return
        goal=self._fine43_goal.copy()
        self._fine43.release(reason);self._fine43_goal=None
        self._route=[];self._fine43_retry=self._steps+60
        self.NARROW_ENABLED=self._lp43_default_narrow
        self._note43('fine_finished',reason=reason,goal=goal.tolist(),position=self._position.tolist())

    def _fine_command43(self):
        goal=self._fine43_goal
        if goal is None:return None
        if self._steps-self._fine43_start>230 or np.linalg.norm(goal-self._position)<20:
            self._finish_fine43('target_reached' if np.linalg.norm(goal-self._position)<20 else 'bounded_timeout')
            return None
        if not self._fine43.path and not self._start_fine43(goal,True):
            self._finish_fine43('fresh_scan_no_safe_leg');return None
        cmd=self._fine43.command(self.lidar_values(),self.lidar_rays_angles(),self._explore,
            self._position,self._heading,0.,self.semantic_values(),True)
        self._slow_ticks=0;self._motion_anchor=None;self._waypoint_progress.reset();self._loop_watch.samples.clear()
        self._lp43_stats['fine_task_ticks']+=1
        return cmd

    def _navigate_motion(self):
        if self._fine43_goal is None:
            command=super()._navigate_motion()
        else:
            command=self._empty_command()
            d=np.asarray(self.lidar_values()).ravel();a=np.asarray(self.lidar_rays_angles());dynamic=np.zeros(len(d),bool)
            for obj in self.semantic_values() or ():
                if getattr(obj.entity_type,'name',None) in ('DRONE','BOMB'):
                    dynamic|=(np.abs((a-obj.angle+math.pi)%(2*math.pi)-math.pi)<.19)&(np.abs(d-obj.distance)<25)
            measured=body_motion(self.odometer_values())
            if measured is not None:self._velocity=.4*self._velocity+.6*measured
            self._explore.step=self._steps
            if self._steps%6==0:self._explore.observe(self._position,self._heading,d,a,dynamic)
            if self._steps%30==0:
                self._visited_snapshot=np.packbits(self._explore.shared_visited|(self._explore.visits>0)).tobytes()
                self._radio_maps.publish(int(self.identifier or 0),self._steps,self._map.mask())
        fine=self._fine_command43()
        if fine is not None:command.update(fine)
        return command

    def control(self):
        command=super().control()
        if self._steps%30==0:
            self._lp43_samples.append(dict(step=self._steps,mode=self._d39_mode,link=dict(self._lp43_link),
                receipt=self._lp43_receipt,contact_node=self._lp43_contact_node,target_node=self._lp43_target_node,
                fine_goal=None if self._fine43_goal is None else self._fine43_goal.tolist(),budget=dict(self._d39_budget)))
        return command

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
        root=self._walk39.points[self._lp43_target_node]
        # Keep a live radio footprint at the agreed observed launch area.
        home_distance=float(np.linalg.norm(self._position-root))
        if home_distance < 30 and not self._link_state43() and self._lp43_target_node!=0:
            if self._lp43_contact_wait is None:self._lp43_contact_wait=self._steps
            if self._steps-self._lp43_contact_wait>=120:
                self._lp43_fallback=True;self._lp43_contact_wait=None
                self._path39();root=self._walk39.points[self._lp43_target_node]
                home_distance=float(np.linalg.norm(self._position-root))
                self._note43('contact_point_expired_fallback')
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

