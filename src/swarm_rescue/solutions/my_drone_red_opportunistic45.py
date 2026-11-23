"""Exploration first; placement only observes the current position until all-map closure.

No placement route is installed during exploration or communication. Existing
inventory-gated phase checks are split into map-work timing and map receipt.
"""
from collections import deque
from types import SimpleNamespace
import math
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_red_startup44_delivery import MyDroneRedStartup44Delivery
from swarm_rescue.solutions.my_drone_red_schedule40 import MyDroneRedSchedule40
from swarm_rescue.solutions.my_drone_red_schedule40r1 import ExplicitMotionDispatch40
from swarm_rescue.solutions.my_drone_red_delivery39 import MyDroneRedDelivery39
from swarm_rescue.solutions.red_deep_bomb45 import walked_costs, difficulty_features, choose_deep_placement
from swarm_rescue.solutions.red_opportunistic45 import (
    MapOnlyFinal45, exploration_extension_due, release_decision, valid_report,
    post_exploration_allowed, movement_snapshot,
)


class ExplorationPrioritySchedule45(MyDroneRedSchedule40):
    def _navigation(self):
        if self._position is None:
            return super(MyDroneRedDelivery39,self)._navigation()
        if self._steps-self._d39_plan_step >= 24:
            self._path39()
        due = self._budget39()
        if self._d39_mode == 'explore' and due:
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
            and exploration_extension_due(self._progress(), int(self.identifier or 0)) and not due)
        if self._explore.extension_enabled:
            self._schedule40_stats['frontier_enabled_ticks'] += 1
            if self._progress() < .60:self._schedule40_stats['early_frontier_enabled_ticks'] += 1
        if self._d39_mode == 'explore':
            self._gathering = False
            return super(MyDroneRedDelivery39,self)._navigation()
        return self._return_motion39()


class OpportunisticDispatch45(ExplicitMotionDispatch40):
    def _navigation(self):
        grid = self._explore
        grid.step = self._steps
        if self._position is not None:
            for obj in self.semantic_values() or []:
                if getattr(obj.entity_type, 'name', None) == 'DISPOSAL_CENTER' and np.isfinite([obj.distance, obj.angle]).all() and (obj.distance > 0):
                    angle = self._heading + obj.angle
                    point = self._position + obj.distance * np.array([math.cos(angle), math.sin(angle)])
                    if self._disposal_observed is None:
                        self._disposal_observed = point
                    else:
                        self._disposal_observed += 0.15 * (point - self._disposal_observed)
        task = grid.survey
        if task is not None:
            matching = bool(self._route and self._route[-1] == task['cell'])
            resolved = task['unknown'] - grid.unknown_near(task['cell']) >= 3
            expired = self._steps - task['start'] >= 90
            if not matching or resolved or expired or (self._progress() >= 0.6):
                grid.finish_survey('observed' if resolved else 'budget' if expired else 'preempted')
                if matching and self._bomb_target is None:
                    self._route = []
        grid.survey_enabled = 0.18 < self._progress() < 0.6 and self._bomb_target is None
        command = self._navigate_motion()
        return command


class MyDroneRedOpportunistic45(MyDroneRedStartup44Delivery, ExplorationPrioritySchedule45,
                                OpportunisticDispatch45):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        old = self._radio_maps
        self._radio_maps = MapOnlyFinal45(old.shape,old.number_drones,old.mapper,old.clock)
        self._op45_scores = deque(maxlen=60)
        self._op45_reports = {}; self._op45_frontiers = 1
        self._op45_empty_since = None; self._op45_explored = False
        self._op45_checked = 0; self._op45_violations = 0
        self._op45_active = False; self._op45_samples = []
        self._db45_events = []; self._db45_first = {}; self._db45_reserve_ticks = 0

    def _event(self,kind,**data):
        if kind.startswith('op45_') or kind in ('placement_requested','placement_confirmed','placement_retry'):
            self._db45_events.append(dict(step=int(self._steps),kind=kind,**data))
        return super()._event(kind,**data)

    def _completion_evidence45(self):
        # Freeze does not mean explored. Unfinished/blocked unknowns remain work.
        if self._radio_maps.own_final is None and self._steps%30==0 and self._position is not None:
            g=self._explore
            walls=cv2.dilate(np.uint8(g.obstacle>=2),cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))).astype(bool)
            allowed=g.free & ~walls; cell=g.cell(self._position); allowed[cell]=True
            _,labels=cv2.connectedComponents(np.uint8(allowed),connectivity=4)
            reachable=labels==labels[cell]
            near=cv2.dilate(np.uint8(reachable),np.ones((3,3),np.uint8)).astype(bool)
            count=int((near & ~g.free & ~walls).sum())
            pending=sum(g.unknown_near(c)>=3 for c in g.pending39)
            self._op45_frontiers=count+pending
            busy=bool(self._entry37.active or self._fine43_goal is not None or self._rn7.path)
            if self._op45_frontiers==0 and not busy:
                if self._op45_empty_since is None:self._op45_empty_since=self._steps
                self._op45_explored=self._steps-self._op45_empty_since>=120
            else:self._op45_empty_since=None;self._op45_explored=False
        i=int(self.identifier or 0)
        self._op45_reports[i]=(i,int(self.elapsed_timestep),self._op45_explored,
                              int(self._op45_frontiers),int(self.carried_bombs_count()))

    def define_message_for_all(self):
        self._completion_evidence45()
        return ('opportunistic45-v1',super().define_message_for_all(),
                tuple(self._op45_reports[i] for i in sorted(self._op45_reports)))

    def _receive(self):
        stripped=[]
        for sender,packet in self.communicator.received_messages:
            if not (isinstance(packet,tuple) and len(packet)==3 and packet[0]=='opportunistic45-v1'):continue
            stripped.append((sender,packet[1]));rows=packet[2]
            if isinstance(rows,tuple) and len(rows)<=self._n:
                for row in rows:
                    if valid_report(row,self._n,int(self.elapsed_timestep)):
                        old=self._op45_reports.get(row[0])
                        if old is None or row[1]>old[1]:self._op45_reports[row[0]]=row
        proxy=SimpleNamespace(**self.__dict__)
        proxy.elapsed_timestep=self.elapsed_timestep
        proxy.communicator=SimpleNamespace(received_messages=stripped)
        for name in ('_valid_peer','_valid_contract39','_note39','_note44','_delivery_note44'):
            setattr(proxy,name,getattr(self,name))
        MyDroneRedStartup44Delivery._receive(proxy)
        for name in ('_f27_home','_d39_contract','_d39_rx_step','_lp43_receipt','_st44_plan'):
            setattr(self,name,getattr(proxy,name))

    def _remaining45(self):
        m=self._misc_data;limit=getattr(m,'max_timestep_limit',5400) or 5400
        step_left=limit-self.elapsed_timestep
        wall_limit=getattr(m,'max_walltime_limit',180) or 180
        rate=max(self.elapsed_walltime/max(1,self.elapsed_timestep),.03)
        return max(0.,min(step_left,(wall_limit-self.elapsed_walltime)/rate))

    def _active_allowed45(self):
        return post_exploration_allowed(self._op45_reports,int(self.elapsed_timestep),self._n,
                                        self._exchange_state(),self._remaining45())

    def _deployment(self,command):
        before=movement_snapshot(self,command)
        try:self._release_at_current_position45(command)
        finally:
            self._op45_checked+=1
            if movement_snapshot(self,command)!=before:
                self._op45_violations+=1
                raise RuntimeError('Placement changed exploration, delivery or movement state')

    def _release_at_current_position45(self,command):
        inventory=self.carried_bombs_count()
        if self._initial_inventory is None:self._initial_inventory=self._last_inventory=inventory
        if inventory<self._last_inventory:
            point=self._attempt_position if self._attempt_position is not None else self._position
            self._placements.append(tuple(map(float,point)))
            self._event('placement_confirmed',inventory=inventory,attempted=self._attempt_step is not None)
            self._attempt_step=None;self._attempts=0;self._failed_place=None
            self._next_attempt=self._steps+35
            self._op45_scores.clear()
        self._last_inventory=inventory
        if inventory<=0 or self._position is None:return
        if self._attempt_step is not None:
            if self._steps-self._attempt_step<4:return
            self._failed_place=self._attempt_position.copy();self._attempt_step=None
            self._next_attempt=self._steps+30
            self._event('placement_retry',inventory=inventory,attempts=self._attempts)
            # Failure only retries along the continuing route; no recovery turn.
            return
        if (not self._st44_done or self._entry37.active or self._steps<self._next_attempt
                or self._total_attempts>=60 or self._steps%12):return
        if self._failed_place is not None and np.linalg.norm(self._position-self._failed_place)<55:return
        if any(np.linalg.norm(self._position-p)<85 for p in self._known_bombs()):return
        values=self.lidar_values()
        if values is None:return
        values=np.asarray(values);valid=values[np.isfinite(values)&(values>0)]
        if len(valid)<12 or np.min(valid)<30:return
        for o in self.semantic_values() or ():
            if getattr(o.entity_type,'name',None)=='BOMB' and o.distance<60:return
            if getattr(o.entity_type,'name',None)=='DISPOSAL_CENTER' and o.distance<80:return
        field=self._coverage_field(force=True);gain=field.value(self._position)
        graph=self._walk39;costs=walked_costs(graph,self._steps)
        node=graph.last;cost=costs.get(node)
        root=graph.points[0] if graph.points else None
        f=difficulty_features(self._position,cost,root,self.size_area)
        score=gain*(1+.8*f['depth']+.4*f['detour'])
        self._op45_scores.append(float(score));best=max(self._op45_scores,default=0.)
        placed=self._initial_inventory-inventory
        if not release_decision(self._progress(),int(self.identifier or 0),placed,gain,f['depth'],f['detour'],best):return
        command.update(grasper=0,place_bomb=1)
        self._attempt_step=self._steps;self._attempt_position=self._position.copy()
        self._attempts+=1;self._total_attempts+=1
        self._event('op45_release',gain=gain,depth=f['depth'],detour=f['detour'],
                    recent_best=best,score=score,position=self._position.tolist(),
                    navigation_unchanged=True,mode=self._d39_mode)
        self._event('placement_requested',inventory=inventory,attempt=self._attempts)

    def _release_stale_route(self,reason):
        # Same LP43 observation/safety test; actual remaining ammo cannot block it.
        eligible=(reason=='new_wall' and self._route and self._d39_mode=='explore'
            and self._bomb_target is None and self.carried_bombs_count()>0
            and exploration_extension_due(self._progress(),int(self.identifier or 0))
            and self._steps>=self._fine43_retry and self._radio_maps.own_final is None
            and not self._entry37.active)
        if eligible:
            goal=self._explore.point(self._route[-1])
            if self._explore.unknown_near(self._route[-1])>=3 and self._start_fine43(goal):
                self._note43('coarse_rejection_rescued',goal=goal.tolist(),position=self._position.tolist())
                return
        return super()._release_stale_route(reason)

    def _submit(self):
        # Map exchange must not wait for ammunition; ending the flight may.
        own=int(self.identifier or 0)
        remaining=self.carried_bombs_count()>0 or any(
            i!=own and self._status.get(i,(-1,None))[1]!=0 for i in range(self._n))
        if remaining and self._progress()<self.DEADLINE_AT:return
        return super()._submit()

    def control(self):
        command=super().control();self._completion_evidence45()
        allowed=self.carried_bombs_count()>0 and self._active_allowed45()
        self._op45_active=bool(allowed)
        if allowed:
            # This branch is forbidden until ALL observed frontiers and final
            # source-set receipts are complete. No time-based substitute.
            if self._bomb_target is None or self._steps-self._bomb_target_step>120:
                plan=choose_deep_placement(self._explore,self._position,self._coverage_field(force=True),
                    self._disposal_observed,self._walk39,{},self._steps,
                    getattr(self._misc_data,'max_timestep_limit',5400) or 5400,self._event)
                if plan is not None:
                    self._route=plan.route;self._goal_step=self._steps
                    self._bomb_target=plan.target;self._bomb_target_step=self._steps
                    self._event('op45_post_completion_route',target=plan.target.tolist())
            if self._bomb_target is not None:
                gathering=self._gathering
                try:
                    self._gathering=False
                    movement=self._navigate_motion()
                    command.update({k:movement[k] for k in ('forward','lateral','rotation')})
                finally:self._gathering=gathering
        if self._steps%30==0:
            self._op45_samples.append(dict(step=self._steps,active_search=bool(allowed),
                explored=self._op45_explored,frontiers=self._op45_frontiers,
                mode=self._d39_mode,inventory=self.carried_bombs_count(),
                placement_checks=self._op45_checked,violations=self._op45_violations))
        return command
