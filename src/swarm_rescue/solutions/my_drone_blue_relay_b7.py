"""B7 task allocation experiment on read-only B0. No transport/localization rewrite."""
import math
from collections import Counter
from types import SimpleNamespace
import numpy as np
from swarm_rescue.solutions.my_drone_eval_blue import MyDroneBlueAdapted,wrap
from swarm_rescue.solutions.my_drone_blue_frontier import FrontierTaskGrid
from swarm_rescue.solutions.my_drone_blue_transport import TransportGrid
from swarm_rescue.solutions.blue_relay_b7_protocol import RelayPacket,RelayLease,near


def shortest(allowed,start):
    # Reuse B3's tested native graph routine, without replacing the B0 grid.
    view=SimpleNamespace(shape=allowed.shape,cell=lambda ignored:start)
    return TransportGrid._shortest(view,allowed,None)[1].reshape(allowed.shape)


class RelayView:
    """Use B5's bounded occlusion gain, without adopting its search controller."""
    visibility_gain=FrontierTaskGrid.visibility_gain

    def __init__(self,grid):
        self.grid=grid;self.shape=grid.shape;self.seen=np.zeros(self.shape,bool)
        a=np.linspace(-math.pi,math.pi,35,endpoint=False)
        self._ray_offsets=np.rint(np.stack((np.sin(a),np.cos(a)),axis=1)[:,None,:]*np.arange(1,8)[None,:,None]).astype(int)
        self.tick=-100;self.observed_samples=0;self.repeated_samples=0

    def observe(self,position,heading,distances,angles):
        if len(distances)!=len(angles) or not len(distances):return
        # Vectorized legal sensing opportunity; never call this detection rate.
        rays=np.linspace(-math.pi,math.pi,35)
        gaps=np.abs((np.asarray(angles)[None,:]-rays[:,None]+math.pi)%(2*math.pi)-math.pi)
        indices=np.argsort(gaps,axis=1)[:,:2];d=np.asarray(distances)[indices]
        limit=np.where(np.isfinite(d).all(axis=1),np.minimum(118,d.min(axis=1)-20),0)
        lengths=np.arange(0,119,16)
        directions=np.column_stack((np.cos(rays+heading),np.sin(rays+heading)))
        points=position+directions[:,None,:]*lengths[None,:,None]
        c=np.floor((points[:,:,0]+self.grid.width/2)/self.grid.CELL).astype(int)
        r=np.floor((self.grid.height/2-points[:,:,1])/self.grid.CELL).astype(int)
        valid=(r>=0)&(r<self.shape[0])&(c>=0)&(c<self.shape[1])&(lengths[None,:]<limit[:,None])
        rr=np.clip(r,0,self.shape[0]-1);cc=np.clip(c,0,self.shape[1]-1)
        stop=~valid|~self.grid.free[rr,cc]|(self.grid.evidence[rr,cc]>=2)
        valid &= ~np.maximum.accumulate(stop,axis=1)
        ids=np.unique(rr[valid]*self.shape[1]+cc[valid])
        self.observed_samples+=len(ids);self.repeated_samples+=int(self.seen.ravel()[ids].sum());self.seen.ravel()[ids]=True

    def refresh(self,position,tick):
        if tick-self.tick<24:return
        self.tick=tick;self.allowed,_=self.grid.layers();self.distance=shortest(self.allowed,self.grid.cell(position))
        self.free=self.grid.free.copy();self.walls=self.grid.evidence>=2
        for _,(free,walls,_) in self.grid.remote.values():
            self.free|=free;self.walls|=walls&~(self.grid.free&(self.grid.evidence<.5))

    def anchor(self,point,allowed,radius=40,object_endpoint=False):
        r,c=self.grid.cell(point);n=int(math.ceil(radius/self.grid.CELL));choices=[]
        if self.walls[r,c] or (not object_endpoint and not self.free[r,c]):return None
        if allowed[r,c]:return r,c
        for rr in range(max(0,r-n),min(self.shape[0],r+n+1)):
            for cc in range(max(0,c-n),min(self.shape[1],c+n+1)):
                if allowed[rr,cc]:
                    length=np.linalg.norm(self.grid.point((rr,cc))-point)
                    if length<=radius:choices.append((length,rr,cc))
        for _,rr,cc in sorted(choices):
            q=self.grid.point((rr,cc));points=np.linspace(point,q,max(2,int(np.linalg.norm(q-point)/6)+1))
            cols=np.clip(((points[:,0]+self.grid.width/2)/self.grid.CELL).astype(int),0,self.shape[1]-1)
            rows=np.clip(((self.grid.height/2-points[:,1])/self.grid.CELL).astype(int),0,self.shape[0]-1)
            known=self.free[rows,cols]
            if object_endpoint:known|=np.linalg.norm(points-point,axis=1)<=32
            if known.all() and not self.walls[rows,cols].any():return rr,cc
        return None

    def arrival(self,point):
        anchor=self.anchor(point,self.allowed&np.isfinite(self.distance))
        if anchor is None:return None
        return float(self.distance[anchor]*self.grid.CELL+np.linalg.norm(self.grid.point(anchor)-point))

    def return_hint(self,bomb,center):
        if center is None:return None
        # Separate conservative clearance proxy, never a proof of payload sweep.
        padded=np.pad(self.allowed,1,constant_values=False)
        wide=self.allowed.copy()
        for r in range(3):
            for c in range(3):wide&=padded[r:r+self.shape[0],c:c+self.shape[1]]
        start=self.anchor(bomb,wide,48);end=self.anchor(center,wide,64,object_endpoint=True)
        if start is None or end is None:return None
        length=shortest(wide,start)[end]*self.grid.CELL
        return float(length+112) if np.isfinite(length) else None

    def frontier(self,position,bomb,claims=()):
        unknown=~self.free&~self.walls;padded=np.pad(unknown,2)
        near_unknown=np.zeros(self.shape,bool)
        for dr,dc in ((0,2),(4,2),(2,0),(2,4)):
            near_unknown|=padded[dr:dr+self.shape[0],dc:dc+self.shape[1]]
        eligible=self.allowed&np.isfinite(self.distance)&(self.distance>=3)&(self.distance<=15)&near_unknown
        cells=np.argwhere(eligible)
        if not len(cells):return None
        direction=np.asarray(bomb)-position;norm=np.linalg.norm(direction)
        direction=direction/max(norm,1.)
        # At most 16 viewpoints; no full-map per-cell ray casting.
        order=np.argsort(self.distance[eligible],kind='stable')
        picks=order[np.linspace(0,len(order)-1,min(16,len(order))).astype(int)]
        choices=[]
        for k in picks:
            cell=tuple(cells[k]);q=self.grid.point(cell)
            if norm>8 and (q-np.asarray(bomb))@direction<24:continue
            if any(np.linalg.norm(q-claim)<96 for claim in claims):continue
            gain=self.visibility_gain(cell,self.free,self.walls,self.seen)
            length=self.distance[cell]*self.grid.CELL
            if gain>=6:choices.append((gain/(40+length),gain,length,q))
        if not choices:return None
        _,gain,length,q=max(choices,key=lambda v:v[0]);return dict(point=q,gain=gain,length=float(length))


class MyDroneBlueRelayDecision(MyDroneBlueAdapted):
    RELAY_ENABLED=True
    EXPLICIT_RELAY=False
    MAX_PEER_ETA=180.
    MAX_DEFERRAL=180

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._relay=RelayLease(self.identifier);self._relay_view=RelayView(self._grid)
        self._relay_packets=[];self._relay_meta={};self._relay_good=0;self._relay_trusted=False
        self._relay_hits=[];self._relay_explore=None;self._relay_wait=False;self._relay_passive=None
        self._relay_reasons=Counter();self._relay_last_mode='search';self._relay_queries={}
        self._relay_front_task=None;self._relay_front_cooldowns=[];self._relay_assessment={};self._relay_peer_eta=None

    def define_message_for_all(self):
        base=super().define_message_for_all()
        if not self.RELAY_ENABLED or base is None:return base
        return RelayPacket(base,self._relay.extension(self._tick,self._relay_trusted,self.drone_health))

    def _receive(self):
        super()._receive()
        self._relay_packets=[]
        if not self.RELAY_ENABLED:return
        for _,msg in self.communicator.received_messages:
            if not isinstance(msg,tuple) or len(msg)!=9 or msg[0]!='blue-v1':continue
            sender,stamp=msg[1:3];meta=getattr(msg,'relay',None)
            if type(sender) is not int or not 0<=sender<self._number or sender==self.identifier:continue
            if type(stamp) is not int or not 0<=self._tick-stamp<=12:continue
            if not isinstance(meta,dict) or meta.get('version')!=1:continue
            if type(meta.get('trusted')) is not bool or not isinstance(meta.get('health'),(int,float)) or not math.isfinite(meta['health']):continue
            good=True
            for job in (meta.get('offer'),meta.get('status')):
                if job is None:continue
                if not isinstance(job,dict):good=False;break
                token=job.get('token');point=job.get('point');receiver=job.get('receiver')
                if (not isinstance(token,(tuple,list)) or len(token)!=2 or any(type(i) is not int for i in token)
                    or not 0<=token[0]<self._number or token[1]<=0 or np.shape(point)!=(2,) or not np.isfinite(point).all()
                    or type(receiver) is not int or not 0<=receiver<self._number
                    or any(type(job.get(k)) is not int for k in ('created','deadline'))
                    or not 0<=job['created']<=stamp<job['deadline']+self._relay.LEASE
                    or not 0<job['deadline']-job['created']<=self._relay.MAX_AGE
                    or job.get('phase') not in ('offered','confirmed','accepted','held','cancelled','ended_unconfirmed')):good=False;break
                if job is meta.get('offer') and token[0]!=sender:good=False;break
                if job is meta.get('status') and receiver!=sender:good=False;break
            if good:
                self._relay_meta[sender]=(stamp,meta,msg)
                self._relay_packets.append((sender,stamp,meta))
        self._relay_meta={k:v for k,v in self._relay_meta.items() if self._tick-v[0]<=12}

    def _sense(self,distances,angles):
        hits=super()._sense(distances,angles);self._relay_hits=hits
        if not self.RELAY_ENABLED:return hits
        gps=self.measured_gps_position();compass=self.measured_compass_angle()
        good=(gps is not None and np.shape(gps)==(2,) and np.isfinite(gps).all() and compass is not None
              and np.isfinite(compass) and np.linalg.norm(gps-self._position)<15 and abs(wrap(compass-self._heading))<.12)
        self._relay_good=self._relay_good+1 if good else 0;self._relay_trusted=self._relay_good>=8
        if self._relay_trusted and self._tick%6==0 and not self.semantic_is_disabled():
            self._relay_view.observe(self._position,self._heading,distances,angles)
        self._relay_wait=False;self._relay_explore=None
        if self.EXPLICIT_RELAY:
            self._relay.update(self._relay_packets,self._tick,self._can_accept,self._arrival,
                bool(self.grasped_bombs()),self._mode in ('search','collect'),self._relay_trusted and self.drone_health>=10)
        return hits

    def _refresh(self):
        self._relay_view.refresh(self._position,self._tick)
        if self._relay_queries.get('tick')!=self._relay_view.tick:self._relay_queries={'tick':self._relay_view.tick}

    def _arrival(self,point):
        self._refresh();return self._relay_view.arrival(point)

    def _remaining(self):
        limit=getattr(self._misc_data,'max_timestep_limit',None) or 5400
        wall=getattr(self._misc_data,'max_walltime_limit',None) or 180
        return min(limit-self._tick,max(0.,wall-self.elapsed_walltime)*30.)

    def _return_hint(self,point):
        self._refresh();key=('return',self._grid.cell(point),None if self._center is None else self._grid.cell(self._center))
        if key not in self._relay_queries:self._relay_queries[key]=self._relay_view.return_hint(point,self._center)
        return self._relay_queries[key]

    def _can_accept(self,point):
        def reject(reason):self._relay_reasons[reason]+=1;return None
        if not self._relay_trusted:return reject('position_untrusted')
        if self.drone_health<10 or self.grasped_bombs():return reject('health_or_payload_busy')
        if self._mode=='collect' and self._target is not None and not near(self._target,point):return reject('different_collection_task')
        length=self._arrival(point);back=self._return_hint(point)
        if not self._relay_view.allowed[self._grid.cell(self._position)]:return reject('current_cell_uncertain')
        if length is None:return reject('no_known_approach')
        if back is None:return reject('return_unconfirmed')
        if length>self.MAX_PEER_ETA:return reject('arrival_too_late')
        if self._remaining()<length+back/.6+180:return reject('insufficient_remaining_time')
        return length

    def _frontier(self,point):
        if not self._relay_trusted or self.drone_health<10 or self._relay.blocked(point,self._tick):return None
        if self._can_accept(point) is None:return None
        claims=[p[3] for p in self._peers.values() if p[2]=='search' and p[3] is not None and self._tick-p[0]<=12]
        view=self._relay_view;task=self._relay_front_task
        if task and near(task['bomb'],point):
            cell=self._grid.cell(task['point']);length=view.distance[cell]*self._grid.CELL
            gain=view.visibility_gain(cell,view.free,view.walls,view.seen)
            if length<task['best']-8:task.update(best=length,progress=self._tick)
            if (view.allowed[cell] and np.isfinite(length) and length>=24 and gain>=2
                and self._tick<task['until'] and self._tick-task['progress']<=90
                and not any(near(task['point'],q,96) for q in claims)):
                return dict(point=task['point'],gain=gain,length=float(length))
            self._relay_front_cooldowns.append((task['point'],self._tick+120));self._relay_front_task=None
        self._relay_front_cooldowns=[(q,t) for q,t in self._relay_front_cooldowns[-8:] if t>self._tick]
        result=view.frontier(self._position,point,claims+[q for q,_ in self._relay_front_cooldowns])
        if result:self._relay_front_task=dict(result,bomb=point.copy(),until=self._tick+180,best=result['length'],progress=self._tick)
        else:self._relay_reasons['no_reachable_frontier_gain']+=1
        return result

    def _peer(self,point,claimed):
        self._refresh();best=[]
        anchor=self._relay_view.anchor(point,self._relay_view.allowed)
        if anchor is None:return None
        cache_key=('peer_distance',anchor)
        if cache_key not in self._relay_queries:self._relay_queries[cache_key]=shortest(self._relay_view.allowed,anchor)
        distances=self._relay_queries[cache_key]
        for ident,(stamp,position,mode,target) in self._peers.items():
            record=self._relay_meta.get(ident)
            if self._tick-stamp>12 or record is None:continue
            _,meta,msg=record
            if not meta['trusted'] or meta['health']<10:continue
            if claimed:
                if mode!='collect' or target is None or not near(target,point):continue
                if not any(len(b)==4 and b[3] and near(b[:2],point) for b in msg[6]):continue
            elif mode!='search' or meta.get('status') or meta.get('offer'):continue
            cell=self._relay_view.anchor(position,self._relay_view.allowed,24)
            if cell is None or not np.isfinite(distances[cell]):continue
            eta=float(distances[cell]*self._grid.CELL+40)
            own=self._relay_view.arrival(point)
            if eta<=self.MAX_PEER_ETA and own is not None and eta<=own+100:best.append((eta,ident))
        self._relay_peer_eta=min(best)[0] if best else None
        return min(best)[1] if best else None

    def _worthwhile(self,point,front):
        own=self._arrival(point);eta=self._relay_peer_eta
        if front is None or own is None or eta is None:return False
        extra=max(0.,eta-own)
        threshold=6+.05*extra+.01*front['length']
        self._relay_assessment=dict(frontier_gain=front['gain'],explore_distance=front['length'],
            own_arrival=own,peer_eta=eta,extra_pickup_delay=extra,required_gain=threshold,
            return_proxy=self._return_hint(point),return_status='coarse_clearance_proxy_not_payload_proof',remaining=self._remaining())
        return front['gain']>=threshold

    def _choose_bomb(self):
        ordinary=super()._choose_bomb()
        if not self.RELAY_ENABLED:return ordinary
        if self._relay.fallback and self._tick>=self._relay.fallback['until']:self._relay.fallback=None
        if self._relay.inc and self._relay.inc['phase']=='accepted':return np.array(self._relay.inc['point'])
        if self._relay.fallback:
            point=np.array(self._relay.fallback['point'])
            if any(b[3] and near(b[:2],point) for b in self._bombs):return point
        active=(self._relay.out if self._relay.out and self._relay.out['phase']=='confirmed' else self._relay_passive)
        if active:
            # Acknowledgment for one bomb never authorizes skipping a different
            # newly visible bomb. Preserve the first receiver's confirmed job.
            for obj,other in sorted(self._relay_hits,key=lambda hit:hit[0].distance):
                if obj.distance<85 and not near(other,active['point']) and self._peer(other,True) is None:
                    self._relay.stats['new_bomb_self_priority']+=1;return other
        point=None
        if self._relay.out and self._relay.out['phase']!='cancelled':point=np.array(self._relay.out['point'])
        elif self._relay_passive:point=self._relay_passive['point']
        elif self._relay_hits:
            point=min(self._relay_hits,key=lambda h:h[0].distance)[1]
            if ordinary is not None and not near(point,ordinary):return ordinary
        if point is None:return ordinary
        front=self._frontier(point)
        if self.EXPLICIT_RELAY:
            if self._relay.out and self._relay.out['phase']!='cancelled':
                if front is None:self._relay.fail_out(self._tick,'no_safe_frontier_or_budget');return point
                if self._relay.out['phase']=='confirmed':
                    self._relay_explore=front;return None
                self._relay_wait=True;return point
            if front is not None:
                receiver=self._peer(point,False)
                if receiver is not None and self._worthwhile(point,front) and self._relay.start(point,receiver,self._tick):
                    self._relay_wait=True;return point
            self._relay_reasons['no_confirmed_receiver_or_frontier']+=1
            return ordinary
        peer=self._peer(point,True) if front is not None else None
        if peer is not None and self._worthwhile(point,front):
            lease=self._relay_passive
            if lease is None:
                lease=self._relay_passive=dict(point=point.copy(),receiver=peer,start=self._tick,best=float('inf'),progress=self._tick)
                self._relay.event('passive_peer_claim_observed',self._tick,dict(token=(int(self.identifier),0),point=point,receiver=peer))
            distance=self._relay_peer_eta
            if distance<lease['best']-8:lease.update(best=distance,progress=self._tick)
            if self._tick-lease['start']<=self.MAX_DEFERRAL and self._tick-lease['progress']<=90:
                self._relay_explore=front;return None
        if self._relay_passive:
            self._relay.event('passive_fallback_self',self._tick,dict(token=(int(self.identifier),0),point=point),reason='claim_frontier_or_progress_expired')
            self._relay.cooldowns.append((tuple(point),self._tick+240));self._relay_passive=None
            self._relay.fallback=dict(point=tuple(point),until=self._tick+120)
            return point
        return ordinary

    def _navigate(self,goal,distances,angles,carrying=False):
        if self.RELAY_ENABLED and not carrying and goal is None and self._mode=='search' and self._relay_explore:
            goal=self._relay_explore['point'];self._target=goal.copy();self._command_source='relay_scout'
            if self._route and np.linalg.norm(self._route[-1]-goal)>32:self._route_step=-100
        return super()._navigate(goal,distances,angles,carrying)

    def control(self):
        self._command_source=None
        command=super().control()
        if self.RELAY_ENABLED and not self.grasped_bombs() and self._relay_wait and self._mode=='collect':
            command=dict(command,grasper=0);self._command_source='relay_wait_ack'
            self._relay.stats['wait_steps']+=1
        if self.RELAY_ENABLED:
            self._relay.stats['scout_steps']+=int(self._relay_explore is not None and self._mode=='search')
            if self._mode!=self._relay_last_mode:self._relay.stats['role_switches']+=1;self._relay_last_mode=self._mode
        return command


class MyDroneBlueRelayConfirmed(MyDroneBlueRelayDecision):
    EXPLICIT_RELAY=True


class MyDroneBlueRelayOff(MyDroneBlueRelayDecision):
    RELAY_ENABLED=False
