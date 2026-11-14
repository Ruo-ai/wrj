"""Bounded task leases using delivered messages only; no simulator references."""
import copy
import math
from collections import Counter,deque


class RelayPacket(tuple):
    """Keep B0's nine-field tuple compatible; engine Message is Any.

    Additional data is detached from live task state before transmission.
    Constructing this packet is not evidence that anyone received it.
    """
    def __new__(cls,base,relay):
        obj=super().__new__(cls,base);obj.relay=copy.deepcopy(relay);return obj

    def __getnewargs__(self):return tuple(self),self.relay


def near(a,b,r=32):return math.dist(a,b)<r


class RelayLease:
    WAIT=8
    LEASE=30
    MAX_AGE=240
    STALL=90
    COOLDOWN=240

    def __init__(self,identifier):
        self.identifier=int(identifier);self.sequence=0;self.out=None;self.inc=None
        self.fallback=None;self.cooldowns=[];self.events=deque(maxlen=4096)
        self.stats=Counter();self.last_packet={};self.terminal=None
        self.seen_offers=deque(maxlen=64);self.declined=deque(maxlen=64)

    def event(self,kind,tick,job=None,reason=None):
        self.stats[kind]+=1
        if len(self.events)==self.events.maxlen:self.stats['diagnostic_event_overflow']+=1
        self.events.append(dict(kind=kind,step=int(tick),token=list(job['token']) if job else None,
            point=list(job['point']) if job else None,receiver=job.get('receiver') if job else None,reason=reason))

    def blocked(self,point,tick):
        self.cooldowns=[(p,t) for p,t in self.cooldowns if t>tick]
        return any(near(point,p) for p,_ in self.cooldowns)

    def start(self,point,receiver,tick):
        if self.out or self.inc or self.blocked(point,tick):return False
        self.sequence+=1
        self.out=dict(token=(self.identifier,self.sequence),point=tuple(map(float,point)),receiver=int(receiver),
            created=tick,deadline=tick+self.MAX_AGE,phase='offered',ack=None)
        self.event('offer_created',tick,self.out);return True

    def fail_out(self,tick,reason):
        if not self.out or self.out['phase']=='cancelled':return
        job=self.out;self.event('fallback_self',tick,job,reason)
        self.fallback=dict(point=job['point'],until=tick+120,reason=reason)
        self.cooldowns.append((job['point'],tick+self.COOLDOWN));self.cooldowns=self.cooldowns[-8:]
        job.update(phase='cancelled',clear_at=tick+self.LEASE)

    def fail_in(self,tick,reason):
        if not self.inc:return
        self.event('receiver_cancelled',tick,self.inc,reason)
        self.terminal=dict(self.inc,phase='cancelled',clear_at=tick+self.LEASE)
        self.inc=None

    def update(self,packets,tick,can_accept,progress,carrying=False,available=True,healthy=True):
        """can_accept returns route length or None; progress is observed path length.

        These callbacks use the caller's own legal map, never a peer's objects.
        Lease deadlines are absolute; duplicate packets cannot extend them.
        """
        if self.out and self.out['phase']=='cancelled' and tick>=self.out['clear_at']:self.out=None
        if self.terminal and tick>=self.terminal['clear_at']:self.terminal=None
        if self.fallback and tick>=self.fallback['until']:self.fallback=None
        if self.inc:
            job=self.inc
            if carrying:
                if job['phase']!='held':self.event('pickup_observed',tick,job)
                job['phase']='held'
            elif job['phase']=='held':
                self.event('possession_ended_unconfirmed',tick,job)
                self.terminal=dict(job,phase='ended_unconfirmed',clear_at=tick+self.LEASE);self.inc=None
            elif not healthy:self.fail_in(tick,'unsafe_or_untrusted')
            elif tick>=job['deadline']:self.fail_in(tick,'deadline')
            else:
                length=progress(job['point'])
                if length is not None and length<job['best']-8:
                    job['best']=length;job['progress_step']=tick
                if tick-job['progress_step']>self.STALL:self.fail_in(tick,'no_route_progress')
        offers=[]
        for sender,stamp,data in packets:
            if not 0<=tick-stamp<=12 or stamp<=self.last_packet.get(sender,-1):continue
            self.last_packet[sender]=stamp
            offer=data.get('offer');status=data.get('status')
            if self.out and self.out['phase']!='cancelled' and status:
                job=self.out
                if sender==job['receiver'] and tuple(status['token'])==tuple(job['token']) and near(status['point'],job['point']):
                    if status['phase']=='accepted' and data.get('trusted') and data.get('health',0)>=10:
                        if job['phase']!='confirmed':self.event('takeover_confirmed',tick,job)
                        job.update(phase='confirmed',ack=stamp)
                    elif status['phase']=='held':
                        self.event('peer_pickup_confirmed',tick,job);self.out=None
                    elif status['phase'] in ('cancelled','ended_unconfirmed'):self.fail_out(tick,'receiver_'+status['phase'])
            if self.inc and self.inc['phase']!='held':
                if offer and tuple(offer['token'])==tuple(self.inc['token']) and offer['phase']=='cancelled':
                    self.fail_in(tick,'donor_cancelled')
                elif status and status['phase'] in ('accepted','held') and sender<self.identifier and near(status['point'],self.inc['point']):
                    self.fail_in(tick,'lower_id_conflicting_receiver')
            if offer and offer['phase'] in ('offered','confirmed') and offer['receiver']==self.identifier:
                if data.get('trusted') and offer['created']<=tick<offer['deadline'] and offer['deadline']-offer['created']<=self.MAX_AGE:
                    if tuple(offer['token']) not in self.seen_offers:
                        self.seen_offers.append(tuple(offer['token']));self.event('offer_received',tick,offer)
                    offers.append(offer)
        if self.out and self.out['phase']!='cancelled':
            if carrying and self.out['phase']!='confirmed':self.fail_out(tick,'donor_picked_up')
            elif not healthy:self.fail_out(tick,'unsafe_or_untrusted')
            elif tick>=self.out['deadline']:self.fail_out(tick,'deadline')
            elif self.out['ack'] is None and tick-self.out['created']>self.WAIT:self.fail_out(tick,'no_ack')
            elif self.out['ack'] is not None and tick-self.out['ack']>self.LEASE:self.fail_out(tick,'ack_expired')
        if not self.inc and not self.out and available and healthy and not carrying:
            for offer in sorted(offers,key=lambda o:tuple(o['token'])):
                if self.terminal and tuple(self.terminal['token'])==tuple(offer['token']):continue
                length=can_accept(offer['point'])
                if length is None:
                    if tuple(offer['token']) not in self.declined:
                        self.declined.append(tuple(offer['token']));self.event('receiver_rejected',tick,offer,'route_budget_or_task')
                    continue
                self.inc=copy.deepcopy(offer)
                self.inc.update(phase='accepted',best=length,progress_step=tick)
                self.event('receiver_accepted',tick,self.inc);break

    def extension(self,tick,trusted,health):
        def clean(job):
            return {k:job[k] for k in ('token','point','receiver','created','deadline','phase')} if job else None
        return dict(version=1,trusted=bool(trusted),health=float(health),
                    offer=clean(self.out),status=clean(self.inc or self.terminal))
