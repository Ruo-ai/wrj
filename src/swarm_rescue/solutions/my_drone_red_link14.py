"""RL14: independent delivery, opening-task and relative corridor experiments.

Only legal sensors/messages. Neither navigation pose nor RL11 mapping is
rewritten. Body-frame routes are transported by measured odometry, never by
commands. Each switch defaults off on the common class.
"""
from collections import Counter, deque
import math
from types import SimpleNamespace
import numpy as np
from swarm_rescue.solutions.my_drone_red_lines11 import MyDroneRedLines11, fit_lines
from swarm_rescue.solutions.my_drone_red_advanced2 import _M08_MyDroneRedV4
from swarm_rescue.solutions.my_drone_red_rn7 import LocalPassage, CoarseView, rotate
from swarm_rescue.solutions.my_drone_red_stopturn import body_motion, turn_clear
from swarm_rescue.solutions.my_drone_red_entry13 import corridor_direction
from swarm_rescue.solutions.semifinal_localization import wrap


def transport(points, odometry):
    if odometry is None or np.shape(odometry)!=(3,) or not np.isfinite(odometry).all():
        return None
    distance,alpha,theta=map(float,odometry)
    shift=distance*np.array([math.cos(alpha),math.sin(alpha)])
    return rotate(np.asarray(points,float)-shift,-theta)


def opening_candidates(d,a,grid,position,heading,objects=(),fine=None):
    """Observation points INSIDE today's measured free fan, not beyond it.

    Endpoints near coarse unknown are hypotheses, not proven physical doors.
    No unknown cell or shared visit becomes a traversable cell.
    """
    if d is None or a is None or position is None:return []
    d=np.asarray(d,float).ravel();a=np.asarray(a,float).ravel()
    if d.shape!=a.shape or not np.isfinite(position).all():return []
    fine=LocalPassage(d,a,objects) if fine is None else fine
    candidates=[]
    # 18 spaced sectors, bounded cost. The complete swept segment is checked.
    for angle in np.linspace(-math.pi,math.pi,18,endpoint=False):
        window=np.abs((a-angle+math.pi)%(2*math.pi)-math.pi)<.12
        lengths=d[window&np.isfinite(d)]
        if len(lengths)<3:continue
        length=min(145.,float(lengths.min())-32.)
        if length<60:continue
        point=length*np.array([math.cos(angle),math.sin(angle)])
        if not fine.line_clear(np.zeros(2),point):continue
        world=np.asarray(position)+rotate(point,heading);r,c=grid.cell(world)
        f=grid.free[max(0,r-2):r+3,max(0,c-2):c+3]
        wall=grid.obstacle[max(0,r-2):r+3,max(0,c-2):c+3]>=2
        gain=int((~f&~wall).sum())
        if gain<3:continue
        score=gain-2*min(10,int(grid.visits[r,c]))-3*bool(grid.shared_visited[r,c])
        candidates.append(dict(body=point,world=world,angle=float(angle),gain=gain,score=float(score)))
    return sorted(candidates,key=lambda x:x['score'],reverse=True)[:6]


class ReturnTrail:
    """Ordered measured retreat corridor; a broken link invalidates the trail."""
    MAX_POINTS=192
    def __init__(self):
        self.points=[];self.valid=False;self.reason='no_contact';self.distance=0.

    def reset(self):
        self.points=[np.zeros(2)];self.valid=True;self.reason='contact';self.distance=0.

    def update(self,odom):
        if not self.points:return
        moved=transport(self.points,odom)
        if moved is None:self.invalidate('odometry_missing');return
        self.points=list(moved);self.distance+=abs(float(odom[0]))

    def invalidate(self,reason):
        self.valid=False;self.reason=reason;self.points=[]

    def extend(self,fine):
        if not self.valid or not self.points:return
        if np.linalg.norm(self.points[-1])<28:return
        if not fine.line_clear(np.zeros(2),self.points[-1]):
            self.invalidate('unobserved_trail_link');return
        if len(self.points)>=self.MAX_POINTS:self.invalidate('trail_capacity');return
        self.points.append(np.zeros(2))

    def remaining(self):
        return float(sum(np.linalg.norm(v) for v in np.diff([*self.points,np.zeros(2)],axis=0))) if self.points else 0.


class ReceiptBook:
    """Only origin-0 reports of ACTUALLY accepted source versions count.

    A relayed receipt keeps origin sequence. Publishing/sending alone never
    sets an acknowledgement. This is team message content, not radio changes.
    """
    def __init__(self):self.step=-1;self.versions={}
    def accept(self,packet,now,number):
        if not isinstance(packet,tuple) or len(packet)!=3:return False
        origin,step,versions=packet
        if origin!=0 or type(step) is not int or not self.step<step<=now:return False
        if not isinstance(versions,tuple) or len(versions)>number:return False
        parsed={}
        for item in versions:
            if not isinstance(item,tuple) or len(item)!=2:return False
            source,seq=item
            if type(source)is not int or not 0<=source<number or type(seq)is not int or not 0<=seq<=step:return False
            if source in parsed:return False
            parsed[source]=seq
        self.step=step;self.versions=parsed;return True
    def packet(self):return (0,self.step,tuple(sorted(self.versions.items()))) if self.step>=0 else None
    def delivered(self,source,sequence):return self.versions.get(source,-1)>=sequence


class MyDroneRedLink14(MyDroneRedLines11):
    DELIVERY=False
    OPENING_TASKS=False
    RELATIVE_CORRIDOR=False

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._link_stats=Counter();self._link_events=deque(maxlen=1600)
        self._receipt=ReceiptBook();self._last_contact=-999;self._trail=ReturnTrail()
        self._return_state='IDLE';self._return_start=0;self._return_floor=-1
        self._return_cooldown=0;self._return_best=float('inf');self._return_progress=0
        self._opening=None;self._opening_serial=0;self._opening_after=180
        self._opening_history=deque(maxlen=48);self._corridor_history=deque(maxlen=3)
        self._corridor_start=0;self._corridor_until=0;self._corridor_still=0

    def _link_note(self,event,**data):
        self._link_stats[event]+=1
        self._link_events.append(dict(step=int(self._steps),event=event,**data))

    def define_message_for_all(self):
        original=super().define_message_for_all()
        if not self.DELIVERY:return original
        if int(self.identifier or 0)==0:
            self._receipt.accept((0,int(self._steps),tuple((int(k),int(v[1])) for k,v in sorted(self._radio_maps.snapshots.items()))),self._steps,self._n)
        # While returning, alternate own snapshot with the unmodified relay.
        if self._return_state!='IDLE' and self._steps%16<8:
            own=self._radio_maps.snapshots.get(int(self.identifier or 0))
            if own is not None:
                v3=original[1];v1=v3[1]
                original=(original[0],(v3[0],(v1[0],v1[1],own),v3[2],v3[3]),original[2],original[3])
        return ('red-link14',original,self._receipt.packet())

    def _receive(self):
        if not self.DELIVERY:return super()._receive()
        stripped=[]
        for sender,message in self.communicator.received_messages:
            if isinstance(message,tuple) and len(message)==3 and message[0]=='red-link14':
                message,receipt=message[1:]
                if self._receipt.accept(receipt,self.elapsed_timestep,self._n):
                    self._link_stats['received_leader_receipt']+=1
            stripped.append((sender,message))
            if isinstance(message,tuple) and len(message)==4 and message[0]=='red-v4':
                claim=message[2]
                if isinstance(claim,tuple) and len(claim)==3 and self._valid_peer((claim[0],claim[1],(0.,0.),claim[2])) and claim[0]!=int(self.identifier or 0):
                    self._last_contact=self._steps
        proxy=SimpleNamespace(_n=self._n,elapsed_timestep=self.elapsed_timestep,_status=self._status,
            _peers=self._peers,_radio_maps=self._radio_maps,_explore=self._explore,
            _valid_peer=self._valid_peer,_bomb_claims=self._bomb_claims,_bomb_records=self._bomb_records,
            communicator=SimpleNamespace(received_messages=stripped))
        _M08_MyDroneRedV4._receive(proxy)

    def _update_observation(self):
        super()._update_observation()
        if self.DELIVERY:
            self._trail.update(self.odometer_values())
            if self._return_state=='IDLE' and self._steps-self._last_contact<=2:
                self._trail.reset()
        if self._opening is not None:
            moved=transport([self._opening['body']],self.odometer_values())
            if moved is None:self._end_opening('odometry_missing')
            else:self._opening['body']=moved[0]

    def _fine(self):
        if self.lidar_values() is None or self.lidar_rays_angles() is None:return None
        return LocalPassage(self.lidar_values(),self.lidar_rays_angles(),self.semantic_values())

    def _available(self):
        return (self._position is not None and self.drone_health>10 and not self._exploration_submitted
            and self._attempt_step is None and self._steps>=self._yield_until
            and not (self._bomb_target is not None and np.linalg.norm(self._bomb_target-self._position)<55))

    def _brake14(self):
        v=body_motion(self.odometer_values())
        if v is None:return None
        f=np.clip(-.65*v,-1,1)
        return dict(forward=float(f[0]),lateral=float(f[1]),rotation=0.)

    def _move14(self,point,fine,speed=3.4):
        """Fresh swept free-space and measured inertia checks before each move."""
        v=body_motion(self.odometer_values())
        if v is None or fine is None:return None
        distance=float(np.linalg.norm(point))
        if not fine.line_clear(np.zeros(2),point):return None
        direction=np.asarray(point)/max(1.,distance)
        bearing=math.atan2(direction[1],direction[0])
        if any(getattr(o.entity_type,'name','')=='DRONE' and o.distance<60
               and abs(wrap(o.angle-bearing))<.8 for o in self.semantic_values() or ()):
            self._link_stats['dynamic_peer_brake']+=1
            return self._brake14()
        goal=direction*min(speed,math.sqrt(max(0.,.8*(distance-5))))
        force=.085*goal+.65*(goal-v);force/=max(1.,float(np.linalg.norm(force)))
        if not fine.line_clear(np.zeros(2),(.95*v+.6*force)*3):return self._brake14()
        return dict(forward=float(force[0]),lateral=float(force[1]),rotation=0.)

    def _stop_return(self,reason):
        self._link_note('return_end',reason=reason,floor=self._return_floor,
                        acknowledged=self._receipt.versions.get(int(self.identifier or 0),-1))
        self._return_state='IDLE';self._return_cooldown=self._steps+240
        self._trail.invalidate(reason)

    def _delivery_command(self):
        tick=self._steps;source=int(self.identifier or 0);trail=self._trail
        if tick%6==0 and self._return_state=='IDLE' and trail.valid and tick-self._last_contact>2:
            fine=self._fine()
            if fine is not None:
                previous=trail.valid;trail.extend(fine)
                if previous and not trail.valid:self._link_note('trail_invalid',reason=trail.reason)
        if not self._available():return None
        remaining=max(0.,(.82-self._progress())*float(getattr(self._misc_data,'max_timestep_limit',5400) or 5400))
        length=trail.remaining()
        if self._return_state=='IDLE':
            overdue=tick-self._last_contact>600 or self._progress()>.54
            budget=length/2.4+240+90*self.carried_bombs_count()
            if (source==0 or tick<self._return_cooldown or not overdue or tick-self._last_contact<120
                    or not trail.valid or len(trail.points)<2 or self._progress()>=.78
                    or self._bomb_target is not None):return None
            if remaining<budget:
                self._link_stats['return_insufficient_budget']+=1;return None
            self._radio_maps.publish(source,tick,self._map.mask())
            self._return_floor=tick;self._return_state='RETURN';self._return_start=tick
            self._return_progress=tick;self._return_best=length
            self._link_note('return_start',distance=length,budget=budget,remaining=remaining,
                            floor=tick,points=len(trail.points),communication_disabled=bool(self.communicator_is_disabled()))
        if self._receipt.delivered(source,self._return_floor):
            self._stop_return('leader_acknowledged');return None
        if self._progress()>=.79 or tick-self._return_start>650:
            self._stop_return('hard_budget');return None
        if not trail.valid:
            self._stop_return('invalid_trail');return None
        if self._return_state=='SYNC':
            if tick-self._return_progress>120:self._stop_return('sync_timeout');return None
            # Original safe navigation keeps moving: no permanent wait for peers.
            return None
        while trail.points and np.linalg.norm(trail.points[-1])<14:trail.points.pop()
        if not trail.points:
            self._return_state='SYNC';self._return_progress=tick
            self._link_note('returned_contact_site',communication_disabled=bool(self.communicator_is_disabled()))
            return None
        length=trail.remaining()
        if length<self._return_best-8:self._return_best=length;self._return_progress=tick
        if tick-self._return_progress>80:
            self._stop_return('no_progress');return None
        fine=self._fine();target=trail.points[-1]
        command=self._move14(target,fine)
        if command is None and fine is not None and tick%6==0:
            route=fine.route(np.zeros(2),target,CoarseView(self._explore,self._position,self._heading),target=target,limit=165)
            if route:
                command=self._move14(route[1],fine);self._link_stats['return_local_replan']+=1
        if command is None:
            self._link_stats['return_obstructed_ticks']+=1
            return self._brake14()
        self._link_stats['return_motion_ticks']+=1
        return command

    def _end_opening(self,reason):
        if self._opening is None:return
        task=self._opening
        self._link_note('opening_end',task=task['id'],reason=reason,entered=task['entered'])
        self._opening_history.append((task['world'].copy(),self._steps+200))
        self._opening=None;self._opening_after=self._steps+75

    def _opening_command(self):
        if not self._available() or self._gathering or self._progress()>=.64:
            self._end_opening('mission_budget');return None
        if self._bomb_target is not None:
            # Retain task during brief placement/local replans, bounded by age.
            if self._opening and self._steps-self._opening['start']>200:self._end_opening('placement_budget')
            return None
        fine=None
        if self._opening is None:
            if self._steps<self._opening_after or self._steps%18:return None
            fine=self._fine()
            options=opening_candidates(self.lidar_values(),self.lidar_rays_angles(),self._explore,
                self._position,self._heading,self.semantic_values(),fine)
            claims=[p[2] for i,p in self._peers.items() if i!=int(self.identifier or 0) and self._steps-p[0]<90 and p[2] is not None]
            options=[p for p in options if not any(t>self._steps and np.linalg.norm(p['world']-q)<90 for q,t in self._opening_history)
                     and not any(np.linalg.norm(p['world']-q)<100 for q in claims)]
            if not options:self._opening_after=self._steps+36;return None
            chosen=options[0];self._opening_serial+=1
            self._opening=dict(id=self._opening_serial,body=chosen['body'].copy(),world=chosen['world'].copy(),
                start=self._steps,progress=self._steps,best=float(np.linalg.norm(chosen['body'])),entered=False,stage=0)
            self._link_note('opening_selected',task=self._opening_serial,point=chosen['world'].tolist(),
                body=chosen['body'].tolist(),gain=chosen['gain'],navigation=self._position.tolist())
        task=self._opening;distance=float(np.linalg.norm(task['body']))
        if distance<task['best']-8:task['best']=distance;task['progress']=self._steps
        if self._steps-task['start']>200 or self._steps-task['progress']>65:
            self._end_opening('no_progress_or_budget');return None
        fine=self._fine() if fine is None else fine
        if distance<18:
            task['entered']=True;self._link_note('opening_waypoint_reached',task=task['id'],navigation=self._position.tolist())
            direction=task['world']-self._position
            angle=math.atan2(direction[1],direction[0])-self._heading
            options=opening_candidates(self.lidar_values(),self.lidar_rays_angles(),self._explore,
                self._position,self._heading,self.semantic_values(),fine)
            options=[p for p in options if abs(wrap(p['angle']-angle))<.9]
            if task['stage']<1 and options:
                choice=options[0];task.update(body=choice['body'],world=choice['world'],best=float(np.linalg.norm(choice['body'])),progress=self._steps,stage=1)
                self._link_note('opening_continue',task=task['id'],body=choice['body'].tolist(),point=choice['world'].tolist())
            else:self._end_opening('observation_complete');return None
        command=self._move14(task['body'],fine)
        if command is None:
            if self._steps%6==0 and fine is not None:
                route=fine.route(np.zeros(2),task['body'],CoarseView(self._explore,self._position,self._heading),target=task['body'],limit=165)
                if route:
                    command=self._move14(route[1],fine);self._link_stats['opening_local_replan']+=1
            if command is None:return self._brake14()
        self._link_stats['opening_motion_ticks']+=1;return command

    def _corridor_command(self):
        if (not self._available() or not self._l11_active or self._gathering or self._progress()>=.64
                or not self._route or self._steps<self._corridor_until):return None
        delta=self._explore.point(self._route[0])-self._position
        desired=wrap(math.atan2(delta[1],delta[0])-self._heading)
        if np.linalg.norm(delta)<40:return None
        d,a=self.lidar_values(),self.lidar_rays_angles()
        if self._steps%4==0:
            guide=corridor_direction(fit_lines(d,a,self.semantic_values()),desired)
            if guide is None or guide['width']>130:
                self._corridor_history.clear();return None
            self._corridor_history.append((self._steps,wrap(self._segment_motion.pose.heading+guide['angle']),guide))
        if len(self._corridor_history)<3 or self._steps-self._corridor_history[-1][0]>4:return None
        headings=[v[1] for v in self._corridor_history]
        if max(abs(wrap(h-headings[-1])) for h in headings)>.055:return None
        guide=self._corridor_history[-1][2];angle=wrap(headings[-1]-self._segment_motion.pose.heading)
        if abs(wrap(angle-desired))>.25:return None
        v=body_motion(self.odometer_values())
        if v is None:return None
        fine=self._fine();target=70*np.array([math.cos(angle),math.sin(angle)])
        if fine is None or not fine.line_clear(np.zeros(2),target):return None
        if not self._corridor_start:self._corridor_start=self._steps
        if self._steps-self._corridor_start>75:
            self._corridor_until=self._steps+100;self._corridor_start=0;self._corridor_history.clear();return None
        if abs(angle)>.22:
            if not turn_clear(d,a):return None
            self._corridor_still=self._corridor_still+1 if np.linalg.norm(v)<.45 and abs(float(self.odometer_values()[2]))<.04 else 0
            cmd=self._brake14()
            if self._corridor_still>=3:cmd['rotation']=float(np.clip(angle*.9,-.18,.18))
            self._link_stats['corridor_stop_turn_ticks']+=1;return cmd
        self._corridor_still=0;self._link_stats['corridor_motion_ticks']+=1
        return self._move14(target,fine,speed=3.4)

    def _navigation(self):
        original=super()._navigation()
        command=None
        if self.DELIVERY:command=self._delivery_command()
        elif self.OPENING_TASKS:command=self._opening_command()
        elif self.RELATIVE_CORRIDOR:command=self._corridor_command()
        if command is not None:
            original.update(command)
            # Original map/mission/deployment still update; pause only stall
            # detectors while the separately bounded local task is executing.
            self._slow_ticks=0;self._waypoint_progress.last_advance=self._steps
            if self._steps%60==0:self._motion_anchor=None
            if self._segment_motion.state not in ('NORMAL','COOLDOWN'):
                self._segment_motion.release('link14_local_task')
            if self._rn7.path:self._rn7.release('link14_local_task')
        return original


class MyDroneRedDelivery14(MyDroneRedLink14):DELIVERY=True
class MyDroneRedOpening14(MyDroneRedLink14):OPENING_TASKS=True
class MyDroneRedRelative14(MyDroneRedLink14):RELATIVE_CORRIDOR=True
class MyDroneRedLink14Off(MyDroneRedLink14):pass
