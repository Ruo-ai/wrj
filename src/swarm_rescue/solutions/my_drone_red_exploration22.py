"""RL22: RL17-R with an owned direction buffer during RN7 route repair.

One bug fix, no new frontier policy. The inherited route normalizes preferred
in place; RN7 passes the same array as preferred and target when repairing.
Copy only the direction argument so the destination keeps its original length.
Safety, task budgets, mapper, radio, and simulated payload policy are inherited.

PassageMotion22.command is the retained RN7 executor with only its two map
constructors replaced. Keeping this local copy avoids patching a shared class
or changing RL11 / the archived controller. 'completed' still denotes local
waypoint exhaustion, never a claim of physical passage crossing.
"""
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.my_drone_red_rn7 import CoarseView, LocalPassage, PassageMotion


class LocalPassage22(LocalPassage):
    def route(self, position, preferred, coarse, avoid=(), target=None, limit=220):
        return super().route(position, np.array(preferred, dtype=float, copy=True),
                             coarse, avoid=avoid, target=target, limit=limit)


class PassageMotion22(PassageMotion):
    def command(self,d,a,coarse,position,heading,desired,objects=(),allow=True):
        if not allow or not self.valid_odom or d is None or a is None:
            if self.path:self.release('mission_preempted')
            return None
        if not self.path:
            if self.tick<self.cooldown or self.episodes>=8:return None
            self.cooldown=self.tick+40+self.identifier*3
            fine=LocalPassage22(d,a,objects);self.stats['probes']+=1
            preferred=np.array([math.cos(desired),math.sin(desired)])
            path=fine.route(np.zeros(2),preferred,CoarseView(coarse,position,heading),limit=165)
            if not path:
                self.stats['reject_'+fine.last_reason]+=1;return None
            self.path=path[1:];self.since=self.tick;self.last_progress=self.tick
            self.best_remaining=float('inf');self.episodes+=1
            self.blocked_ticks=0
            self.snapshot=fine;self.last_refresh=self.tick;self.event('entered')
        if self.tick-self.since>170:
            self.release('task_timeout');return self.brake()
        while self.path and np.linalg.norm(self.path[0])<9:self.path.pop(0)
        if not self.path:
            self.release('completed');return self.brake()
        remaining=float(sum(np.linalg.norm(v) for v in np.diff([np.zeros(2),*self.path],axis=0)))
        if remaining<self.best_remaining-6:
            self.best_remaining=remaining;self.last_progress=self.tick
        if self.tick-self.last_progress>35:
            self.release('no_progress');return self.brake()
        peers=[o for o in objects or () if getattr(o.entity_type,'name','')=='DRONE'
               and o.distance<60 and abs((o.angle-math.atan2(self.path[0][1],self.path[0][0])+math.pi)%(2*math.pi)-math.pi)<.8]
        if peers:
            self.wait+=1;self.stats['peer_brake_ticks']+=1
            if self.wait>7+3*self.identifier:self.release('peer_wait_budget')
            return self.brake()
        self.wait=0
        fine=LocalPassage22(d,a,objects)
        target=self.path[0].copy()
        if not fine.line_clear(np.zeros(2),target):
            self.blocked_ticks+=1
            if self.tick-self.repair_tick>=6:
                self.repair_tick=self.tick
                repaired=[]
                for endpoint in reversed(self.path):
                    if np.linalg.norm(endpoint)<25:continue
                    normal=np.array([-endpoint[1],endpoint[0]])/np.linalg.norm(endpoint)
                    choices=endpoint+np.arange(-12.,13.,2.)[:,None]*normal
                    clear=fine.sample(choices)
                    ix=int(np.argmax(clear-.04*np.abs(np.arange(-12.,13.,2.))))
                    if clear[ix]<fine.PLAN_RADIUS:continue
                    endpoint=choices[ix]
                    repaired=fine.route(np.zeros(2),endpoint,
                        CoarseView(coarse,position,heading),target=endpoint,limit=165)
                    break
                if repaired:
                    self.path=repaired[1:];self.blocked_ticks=0
                    self.event('local_replan')
            if self.blocked_ticks>=12:self.release('fresh_scan_obstructed')
            return self.brake()
        self.blocked_ticks=0
        for point in self.path[1:4]:
            if np.linalg.norm(point)>36 or not fine.line_clear(np.zeros(2),point):break
            target=point.copy()
        distance=float(np.linalg.norm(target));direction=target/max(distance,1.)
        clearance=float(fine.sample(np.linspace(np.zeros(2),target,8)).min())
        speed=min(3.4,max(1.1,.32*(clearance-self.TRACK_RADIUS)),math.sqrt(max(0.,.8*(distance-4))))
        goal=direction*speed
        force=.085*goal+.65*(goal-self.velocity)
        force/=max(1.,float(np.linalg.norm(force)))
        predicted=.95*self.velocity+.6*force
        if not fine.line_clear(np.zeros(2),predicted*3):
            self.release('inertial_risk');return self.brake()
        self.stats['active_ticks']+=1
        return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)


class MyDroneRedExploration22(MyDroneRedRepair17):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rn7 = PassageMotion22(int(self.identifier or 0))
