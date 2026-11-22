"""LinkedPassage43 plus a short observed, coordinated launch; no later changes."""
from types import SimpleNamespace
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_linked_passage43 import MyDroneRedLinkedPassage43
from swarm_rescue.solutions.my_drone_red_stopturn import body_motion, turn_clear
from swarm_rescue.solutions.my_drone_red_advanced2 import _M11_safe_velocity
from swarm_rescue.solutions.red_startup44 import ANGLES, InitialMap44, assignment, valid_bid, valid_plan


class MyDroneRedStartup44(MyDroneRedLinkedPassage43):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._st44_scans=[];self._st44_bids={};self._st44_plan=None
        self._st44_done=False;self._st44_goal=None;self._st44_events=[]
        self._st44_best=float('inf');self._st44_advance=0
        self._st44_samples=[]
        self._st44_initial=InitialMap44();self._st44_aligned=False
        self._st44_stable=0;self._st44_started=False;self._st44_angle=None

    def _note44(self,kind,**data):
        self._st44_events.append(dict(step=self._steps,kind=kind,**data))

    def define_message_for_all(self):
        base=super().define_message_for_all()
        if self._steps>170:return base
        return ('startup44-v1',base,tuple(self._st44_bids[i] for i in sorted(self._st44_bids)),self._st44_plan)

    def _receive(self):
        stripped=[]
        for sender,message in self.communicator.received_messages:
            if isinstance(message,tuple) and len(message)==4 and message[0]=='startup44-v1':
                _,base,bids,plan=message;stripped.append((sender,base))
                if self._steps<=56:
                    if isinstance(bids,tuple) and len(bids)<=self._n:
                        for bid in bids:
                            if valid_bid(bid,self._n,int(self.elapsed_timestep)):
                                if bid[0] not in self._st44_bids or bid[1]>self._st44_bids[bid[0]][1]:
                                    self._st44_bids[bid[0]]=bid
                    if valid_plan(plan,self._n,self._steps) and self._st44_plan is None:
                        self._st44_plan=plan
                        self._note44('received_assignment',plan=plan)
            else:stripped.append((sender,message))
        proxy=SimpleNamespace(**self.__dict__)
        proxy.elapsed_timestep=self.elapsed_timestep
        proxy._valid_peer=self._valid_peer;proxy._valid_contract39=self._valid_contract39;proxy._note39=self._note39
        proxy.communicator=SimpleNamespace(received_messages=stripped)
        MyDroneRedLinkedPassage43._receive(proxy)
        for name in ('_f27_home','_d39_contract','_d39_rx_step','_lp43_receipt'):
            setattr(self,name,getattr(proxy,name))

    def _finish44(self,reason):
        if self._st44_done:return
        self._st44_done=True
        self._launch_open=False # Disable the legacy fixed-angle window only at launch.
        self._route=[];self._last_route_step=-100
        self._slow_ticks=0;self._motion_anchor=None;self._direction_anchor=None
        self._note44('released',reason=reason,position=None if self._position is None else self._position.tolist(),heading=float(self._heading))

    def _navigation(self):
        if self._st44_done:return super()._navigation()
        if self._steps>=168 or self._progress()>.04 or self._entry37.active:
            self._finish44('deadline_or_entry_guard');return super()._navigation()
        if self._position is None:return self._empty_command()
        measured=body_motion(self.odometer_values())
        if measured is not None:self._velocity=.4*self._velocity+.6*measured
        d=self.lidar_values();a=self.lidar_rays_angles();sem=self.semantic_values()
        command=self._empty_command()
        v=np.clip(-.7*self._velocity,-1,1)
        command.update(forward=float(v[0]),lateral=float(v[1]),rotation=0.)
        if d is None:return command
        if self._steps%2==0 and 6<=self._steps<=30:
            compass=self.measured_compass_angle()
            heading=self._heading if compass is None else compass
            self._st44_initial.add(d,a,self._position,heading,float(np.linalg.norm(self._velocity)),sem)
        self._explore.step=self._steps
        if self._steps%6==0:
            dynamic=np.zeros(len(d),bool)
            for obj in sem or ():
                if getattr(obj.entity_type,'name',None) in ('DRONE','BOMB'):
                    dynamic|=(np.abs((np.asarray(a)-obj.angle+np.pi)%(2*np.pi)-np.pi)<.19)&(np.abs(np.asarray(d)-obj.distance)<25)
            self._explore.observe(self._position,self._heading,d,a,dynamic)
        if self._steps%30==0:
            self._visited_snapshot=np.packbits(self._explore.shared_visited|(self._explore.visits>0)).tobytes()
            self._radio_maps.publish(int(self.identifier or 0),self._steps,self._map.mask())
        own=int(self.identifier or 0)
        if self._steps==32:
            profile=self._st44_initial.finish()
            if profile is not None:
                bid=(own,self._steps,tuple(map(float,self._st44_initial.origin)),tuple(map(float,profile)))
                self._st44_bids[own]=bid;self._note44('scan_completed',bid=bid,scans=len(self._st44_initial.frames),initial_map=self._st44_initial.audit())
        if own==0 and self._st44_plan is None and (self._steps==46 or (self._steps>=38 and len(self._st44_bids)==self._n)):
            self._st44_plan=(self._steps,assignment(self._st44_bids))
            self._note44('team_assignment',plan=self._st44_plan,received_ids=sorted(self._st44_bids))
        if self._steps<40:return command
        if self._st44_goal is None:
            if self._st44_plan is None and self._steps<56:return command
            plan=self._st44_plan[1] if self._st44_plan is not None else assignment(self._st44_bids)
            row=next((r for r in plan if r[0]==own),None)
            if row is None or own not in self._st44_bids:
                self._finish44('no_observed_exit');return super()._navigation()
            _,k,length=row;theta=float(ANGLES[k])
            self._st44_goal=np.asarray(self._st44_bids[own][2])+length*np.array([math.cos(theta),math.sin(theta)])
            self._launch_angle=theta;self._launch_open=False
            self._explore.forward_direction=np.array([math.cos(theta),math.sin(theta)])
            self._st44_angle=theta
            self._note44('direction_planned',assigned_sector=k,angle=theta,target=self._st44_goal.tolist(),
                coordinated=self._st44_plan is not None,received_ids=sorted(self._st44_bids))
        error=(self._st44_angle-self._heading+math.pi)%(2*math.pi)-math.pi
        speed=float(np.linalg.norm(self._velocity));odom=self.odometer_values()
        yaw_rate=None if odom is None else float(odom[2])
        if not self._st44_started:
            good=abs(error)<.07 and speed<.45 and yaw_rate is not None and abs(yaw_rate)<.045
            self._st44_stable=self._st44_stable+1 if good else 0
            if self._st44_stable>=3:self._st44_aligned=True
            if self._st44_aligned and abs(error)>.12:self._st44_aligned=False
            ready=self._st44_aligned and self._steps>=60+own*3
            timeout=self._steps>=112+own*3
            if ready or timeout:
                self._st44_started=True;self._st44_advance=self._steps
                self._note44('depart',assigned_sector=int(round(self._st44_angle/(2*math.pi/24)))%24,
                    angle=self._st44_angle,target=self._st44_goal.tolist(),heading=float(self._heading),
                    heading_error=float(error),speed=speed,aligned=bool(ready),fallback=bool(timeout and not ready),
                    coordinated=self._st44_plan is not None,received_ids=sorted(self._st44_bids))
            else:
                if speed<.45 and turn_clear(d,a) and abs(error)>=.045:
                    command['rotation']=float(np.clip(1.5*error,-.5,.5))
                if self._steps%3==0:self._st44_samples.append(dict(step=self._steps,phase='align',position=self._position.tolist(),heading=float(self._heading),heading_error=float(error),speed=speed,command=dict(command)))
                return command
        delta=self._st44_goal-self._position;distance=float(np.linalg.norm(delta))
        if distance<self._st44_best-3:self._st44_best=distance;self._st44_advance=self._steps
        if distance<24 or self._steps-self._st44_advance>=18:
            self._finish44('target_reached' if distance<24 else 'blocked_timeout');return super()._navigation()
        desired=(math.atan2(delta[1],delta[0])-self._heading+math.pi)%(2*math.pi)-math.pi
        move,self._last_motion_angle=_M11_safe_velocity(np.asarray(d).ravel(),np.asarray(a),desired,self._velocity,self._last_motion_angle,distance)
        command.update(forward=float(move[0]),lateral=float(move[1]),rotation=0.)
        if self._steps%6==0:self._st44_samples.append(dict(step=self._steps,phase='depart',position=self._position.tolist(),heading=float(self._heading),distance=distance,command=dict(command)))
        return command
