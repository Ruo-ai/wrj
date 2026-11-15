"""RX10 exploration-only experiments on the retained RN7-B.

A3 mapping/pose, RN7 passage executor, S6-B and placement remain inherited.
Remote observations affect utility only, never certify a navigation edge.
"""
from collections import Counter, deque
import math
from types import SimpleNamespace
import cv2
import numpy as np
from swarm_rescue.solutions.my_drone_red_rn7 import MyDroneRedRN7B
from swarm_rescue.solutions.my_drone_red_advanced2 import (
    _M19_SurveyGrid, _M08_MyDroneRedV4, _M16_path_to)


def observed_gain(cell, free, walls, remote_seen=None):
    """First unknown cell of each ray, stopping at known walls/occlusion.

    Unknown space beyond that first cell receives NO visibility credit.
    Remote observation discounts value, but cannot make an unsafe path valid.
    """
    angles=np.arange(24)*2*math.pi/24
    lengths=np.arange(1,12)
    rr=cell[0]+np.rint(np.sin(angles)[:,None]*lengths).astype(int)
    cc=cell[1]+np.rint(np.cos(angles)[:,None]*lengths).astype(int)
    valid=(rr>=0)&(cc>=0)&(rr<free.shape[0])&(cc<free.shape[1])
    r=np.clip(rr,0,free.shape[0]-1);c=np.clip(cc,0,free.shape[1]-1)
    stop=~valid|walls[r,c]|~free[r,c]
    has=stop.any(axis=1);first=stop.argmax(axis=1);ray=np.arange(len(angles))
    r=r[ray,first];c=c[ray,first]
    good=has&valid[ray,first]&~walls[r,c]&~free[r,c]
    ids=np.unique(r[good]*free.shape[1]+c[good])
    if not len(ids):return 0.
    if remote_seen is None:return float(len(ids))
    return float((1.-.55*remote_seen.ravel()[ids]).sum())


def safe_reachable(grid, position, limit=8192):
    walls=cv2.dilate((grid.obstacle>=2).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))).astype(bool)
    allowed=grid.free&~walls;start=grid.cell(position)
    # Same start-cell convention as the retained navigator. All later cells
    # must be locally observed and obey its original obstacle inflation.
    allowed[start]=True;distance={start:0};parents={start:None};queue=deque([start])
    while queue and len(distance)<limit:
        cell=queue.popleft();r,c=cell
        for nxt in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if (0<=nxt[0]<allowed.shape[0] and 0<=nxt[1]<allowed.shape[1]
                    and allowed[nxt] and nxt not in distance):
                distance[nxt]=distance[cell]+1;parents[nxt]=cell;queue.append(nxt)
    return allowed,walls,distance,parents


class BranchGrid(_M19_SurveyGrid):
    """Regional task memory is separate from the inherited temporary route."""
    def __init__(self,size,event,owner):
        super().__init__(size,event);self.owner=owner
        self.task=None;self.serial=0;self.discovered=set();self.cooling={}
        self.cache=None;self.cache_step=-999;self.remote_seen=np.zeros(self.shape,bool)
        self.remote_walls=np.zeros(self.shape,bool);self.branch_enabled=False
        self.continue_after_gather=False;self.max_gain=0.;self.return_steps=float('inf')
        self.stats=Counter();self.events=deque(maxlen=500)

    def note(self,kind,**data):
        self.stats[kind]+=1
        self.events.append(dict(step=int(self.step),event=kind,**data))

    def refresh(self,position,home,claims=(),blocked=(),force=False):
        if not force and self.cache is not None and self.step-self.cache_step<48:return self.cache
        allowed,safety,distances,parents=safe_reachable(self,position)
        wall=(self.obstacle>=2)|self.remote_walls
        unknown=~self.free&~safety&~self.remote_walls
        frontier=allowed&(cv2.dilate(unknown.astype(np.uint8),np.ones((3,3),np.uint8))>0)
        local_value=cv2.filter2D(unknown.astype(np.float32),-1,np.ones((5,5),np.float32))
        groups={};self.cooling={k:t for k,t in self.cooling.items() if t>self.step}
        avoid=[p for p,t in self.avoid_regions if t>self.step]
        for cell,steps in distances.items():
            if steps<3 or not frontier[cell] or cell in blocked:continue
            point=self.point(cell);key=(cell[0]//5,cell[1]//5)
            if key in self.cooling or any(np.linalg.norm(point-p)<100 for p in avoid):continue
            rough=float(local_value[cell])-steps*.10-min(12,int(self.visits[cell]))
            if key not in groups or rough>groups[key][0]:groups[key]=(rough,cell,steps,point)
        candidates=[]
        for _,cell,steps,point in sorted(groups.values(),key=lambda x:x[0],reverse=True)[:16]:
            gain=observed_gain(cell,self.free,wall,self.remote_seen)
            if gain<3.5:continue
            key=(cell[0]//5,cell[1]//5)
            if key not in self.discovered and len(self.discovered)<400:
                self.discovered.add(key);self.note('discovered',key=list(key),point=point.tolist(),gain=gain)
            peer_cost=sum(35*max(0.,1.-np.linalg.norm(point-p)/140) for p in claims)
            score=12*gain-.8*steps-5*min(10,int(self.visits[cell]))-peer_cost
            candidates.append(dict(cell=cell,steps=steps,point=point,gain=gain,score=score,key=key))
        self.max_gain=max((v['gain'] for v in candidates),default=0.)
        # Return to a communication-sized neighbourhood via observed paths.
        hr,hc=self.cell(home);close=[]
        for r in range(max(0,hr-8),min(self.shape[0],hr+9)):
            for c in range(max(0,hc-8),min(self.shape[1],hc+9)):
                distance=distances.get((r,c))
                if distance is None:continue
                dx=(c+.5)*self.CELL-self.width/2-home[0]
                dy=self.height/2-(r+.5)*self.CELL-home[1]
                if dx*dx+dy*dy<150**2:close.append(distance)
        self.return_steps=min(close)*self.CELL/2.5 if close else float('inf')
        self.cache=(candidates,parents);self.cache_step=self.step
        self.stats['frontier_refreshes']+=1
        return self.cache

    def finish_task(self,reason,cool=False):
        if self.task is None:return
        task=self.task
        if cool:self.cooling[task['key']]=self.step+180
        self.note('task_end',task=task['id'],reason=reason,entered=task['entered'])
        self.task=None

    def update_task(self,position):
        task=self.task
        if task is None:return
        count=int(self.free.sum())
        distance=float(np.linalg.norm(position-task['goal']))
        if count>=task['free']+3 or distance<task['best']-20:
            task['progress']=self.step;task['free']=count;task['best']=min(task['best'],distance)
        if distance<35 and not task['entered']:
            task['entered']=True;self.note('entered',task=task['id'],point=position.tolist())
        if self.step-task['start']>420:self.finish_task('task_budget',cool=True)
        elif self.step-task['progress']>120:self.finish_task('no_progress',cool=True)

    def retire_goal(self,cell):
        if self.task is not None and cell==self.task.get('cell'):
            # Reaching an observation waypoint need not retire the whole branch.
            self.note('waypoint_reached',task=self.task['id'])
            self.goal_visits[cell]+=0.15
            return
        super().retire_goal(cell)

    def plan(self,position,home,identifier,number,claims,blocked_goals=(),rendezvous=False):
        effective_gather=rendezvous and not self.continue_after_gather
        if not self.branch_enabled or effective_gather:
            return super().plan(position,home,identifier,number,claims,blocked_goals,effective_gather)
        candidates,parents=self.refresh(position,home,claims,blocked_goals,force=True)
        if not candidates:
            self.finish_task('no_reachable_value')
            return super().plan(position,home,identifier,number,claims,blocked_goals,False)
        if self.task is not None:
            continuing=[v for v in candidates if np.linalg.norm(v['point']-self.task['goal'])<260
                        and float((v['point']-self.task['anchor'])@self.task['axis'])>-40]
            if continuing:candidates=continuing
            else:self.finish_task('branch_resolved')
        for candidate in candidates:
            if self.task is not None:
                delta=candidate['point']-self.task['anchor'];length=np.linalg.norm(delta)
                candidate['score']+=28*float(delta@self.task['axis'])/max(60.,length)
                if np.linalg.norm(candidate['point']-self.task['goal'])<180:candidate['score']+=24
        chosen=max(candidates,key=lambda x:(x['score'],x['cell']))
        if self.task is None:
            self.serial+=1;axis=chosen['point']-position;axis/=max(1.,np.linalg.norm(axis))
            self.task=dict(id=self.serial,start=self.step,progress=self.step,anchor=position.copy(),
                axis=axis,key=chosen['key'],goal=chosen['point'],cell=chosen['cell'],
                free=int(self.free.sum()),best=float(np.linalg.norm(chosen['point']-position)),entered=False)
            self.note('selected',task=self.serial,point=chosen['point'].tolist(),gain=chosen['gain'])
        else:
            self.task.update(goal=chosen['point'],cell=chosen['cell'],best=float(np.linalg.norm(chosen['point']-position)))
            self.note('continued',task=self.task['id'],point=chosen['point'].tolist(),gain=chosen['gain'])
        return _M16_path_to(parents,chosen['cell'])


def finish_budget(progress,inventory,health,gain,return_steps,remaining_steps):
    return bool(.70<=progress<.80 and inventory==0 and health>10 and gain>=3.5
                and np.isfinite(return_steps) and remaining_steps>return_steps+180)


class MyDroneRedBranches10(MyDroneRedRN7B):
    BRANCH_TASKS=False
    ADAPTIVE_FINISH=False
    SHARE_OBSERVATIONS=True

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._rx_peers={};self._rx_continue=False;self._rx_map_floor=0
        self._rx_packet_step=-999;self._rx_packet_free=b'';self._rx_stats=Counter()
        self._rx_last_share=-999;self._rx_role='explore'
        if self.BRANCH_TASKS or self.ADAPTIVE_FINISH:
            self._explore=BranchGrid(self.size_area,self._event,self)

    @property
    def _rx_enabled(self):return self.BRANCH_TASKS or self.ADAPTIVE_FINISH

    def define_message_for_all(self):
        original=super().define_message_for_all()
        if not self._rx_enabled:return original
        grid=self._explore;goal=None
        if grid.task is not None and self._rx_role=='explore':goal=tuple(map(float,grid.task['goal']))
        trusted=self.measured_gps_position() is not None and self.measured_compass_angle() is not None
        packet=(int(self.identifier or 0),int(self._steps),self._rx_packet_free,goal,
                bool(self._rx_continue),int(self._rx_map_floor),bool(trusted))
        return ('red-rx10',original,packet)

    def _receive(self):
        if not self._rx_enabled:return super()._receive()
        stripped=[];grid=self._explore;size=(math.prod(grid.shape)+7)//8
        for sender,message in self.communicator.received_messages:
            if isinstance(message,tuple) and len(message)==3 and message[0]=='red-rx10':
                base,packet=message[1:];stripped.append((sender,base))
                if not isinstance(packet,tuple) or len(packet)!=7:continue
                source,seq,free,goal,wants,floor,trusted=packet
                if not (type(source) is int and 0<=source<self._n and source!=int(self.identifier or 0)
                        and type(seq) is int and 0<=seq<=self.elapsed_timestep
                        and isinstance(free,bytes) and len(free) in (0,size)
                        and type(wants) is bool and type(trusted) is bool
                        and type(floor) is int and 0<=floor<=seq
                        and self._valid_peer((source,seq,(0.,0.),goal))):continue
                if seq<=self._rx_peers.get(source,{}).get('seq',-1):continue
                self._rx_peers[source]=dict(seq=seq,free=free,goal=goal,wants=wants,floor=floor,trusted=trusted)
                self._rx_stats['received_packets']+=1
            else:stripped.append((sender,message))
        proxy=SimpleNamespace(_n=self._n,elapsed_timestep=self.elapsed_timestep,_status=self._status,
            _peers=self._peers,_radio_maps=self._radio_maps,_explore=self._explore,
            _valid_peer=self._valid_peer,_bomb_claims=self._bomb_claims,_bomb_records=self._bomb_records,
            communicator=SimpleNamespace(received_messages=stripped))
        _M08_MyDroneRedV4._receive(proxy)

    def _rx_remaining(self):
        limits=self._misc_data;step=float(self.elapsed_timestep);wall=float(self.elapsed_walltime)
        max_steps=getattr(limits,'max_timestep_limit',None);max_wall=getattr(limits,'max_walltime_limit',None)
        values=[]
        if max_steps:values.append(.84*max_steps-step)
        if max_wall and wall>0:values.append((.84*max_wall-wall)*step/wall)
        return max(0.,min(values)) if values else 0.

    def _rx_context(self):
        grid=self._explore;grid.step=self._steps
        if self._position is None or self._home is None:return
        if self._steps-self._rx_last_share>=48:
            self._rx_last_share=self._steps;grid.remote_seen.fill(False);grid.remote_walls.fill(False)
            if self.SHARE_OBSERVATIONS:
                for peer in self._rx_peers.values():
                    if self._steps-peer['seq']<=180 and peer['free'] and peer['trusted']:
                        grid.remote_seen|=np.unpackbits(np.frombuffer(peer['free'],np.uint8))[:math.prod(grid.shape)].reshape(grid.shape).astype(bool)
                # Only actually received, independently versioned snapshots.
                for source,mask in self._radio_maps.masks.items():
                    if source!=int(self.identifier or 0):
                        grid.remote_walls|=cv2.resize(mask.astype(np.float32),grid.shape[::-1],interpolation=cv2.INTER_AREA)>.05
            self._rx_packet_free=np.packbits(grid.free).tobytes() if self.SHARE_OBSERVATIONS else b''
            self._rx_packet_step=self._steps
        claims=[p[2] for i,p in self._peers.items() if i!=int(self.identifier or 0) and self._steps-p[0]<90 and p[2] is not None]
        if self.BRANCH_TASKS or self._progress()>=.69:
            grid.refresh(self._position,self._home,claims,self._blocked_goals)
        grid.update_task(self._position)
        old=self._rx_continue
        self._rx_continue=self.ADAPTIVE_FINISH and finish_budget(self._progress(),self.carried_bombs_count(),
            self.drone_health,grid.max_gain,grid.return_steps,self._rx_remaining())
        if old and not self._rx_continue:
            self._rx_map_floor=self._steps;self._rx_stats['finish_return']+=1
            if self._gathering and self._bomb_target is None:self._route=[]
        elif self._rx_continue and not old:self._rx_stats['finish_extension']+=1
        grid.continue_after_gather=self._rx_continue
        grid.branch_enabled=self.BRANCH_TASKS and self._bomb_target is None
        self._rx_role='placement' if self.carried_bombs_count()>0 and self._bomb_target is not None else 'gather' if self._gathering and not self._rx_continue else 'explore'
        if self.carried_bombs_count()>0 and grid.task is not None and self._bomb_target is None:
            initial=self._initial_inventory or self.carried_bombs_count();placed=initial-self.carried_bombs_count()
            due=.1+.36*(placed+1)/(initial+1)+.008*int(self.identifier or 0)
            if self._progress()>=due+.03:
                if self._route and self._route[-1]==grid.task['cell']:self._route=[]
                grid.finish_task('placement_due');grid.branch_enabled=False

    def _navigation(self):
        if self._rx_enabled:self._rx_context()
        return super()._navigation()

    def _try_hold(self,distances,dynamic,command):
        if self.ADAPTIVE_FINISH and self._rx_continue:
            self._holding=False;self._rx_stats['hold_deferred_ticks']+=1;return False
        return super()._try_hold(distances,dynamic,command)

    def _release_stale_route(self,reason):
        if self._rx_enabled and self._explore.task is not None:
            if reason=='new_wall':self._explore.note('local_replan_requested',task=self._explore.task['id'])
            elif reason=='waypoint_stall':self._explore.finish_task(reason,cool=True)
        return super()._release_stale_route(reason)

    def _submit(self):
        if self.ADAPTIVE_FINISH and self._progress()<.84:
            waiting=False
            for source,peer in self._rx_peers.items():
                if self._steps-peer['seq']>120:continue
                snapshot=self._radio_maps.snapshots.get(source)
                waiting|=peer['wants'] or (peer['floor']>0 and (snapshot is None or snapshot[1]<peer['floor']))
            if self._rx_continue or waiting:
                self._rx_stats['early_submit_deferred']+=1;return
        return super()._submit()


class MyDroneRedRX10Branch(MyDroneRedBranches10):BRANCH_TASKS=True
class MyDroneRedRX10Finish(MyDroneRedBranches10):ADAPTIVE_FINISH=True
class MyDroneRedRX10Combined(MyDroneRedRX10Branch):ADAPTIVE_FINISH=True
