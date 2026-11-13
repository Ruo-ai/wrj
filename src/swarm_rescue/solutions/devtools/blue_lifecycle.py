"""Opt-in LOCAL evaluation observer. Truth stays here, never in a controller.

The mixin captures only legal inputs and the final returned command. The separate
observer joins exact renderer entity IDs and physics events OFFLINE. No commands,
sensor values, physics or scoring are modified. JSON is written once after play.
"""
import math
import time
from collections import Counter
import numpy as np


def point(value):
    return None if value is None else [float(x) for x in value]


def instrument(base):
    class Probe(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._probe_times=Counter()
            self._probe_calls=Counter()
            self._probe_nav=None
            self._probe_frame=None

        def _timed(self, name, fn, *args):
            start=time.perf_counter()
            result=fn(*args)
            self._probe_times[name]+=time.perf_counter()-start
            self._probe_calls[name]+=1
            return result

        def _receive(self):return self._timed('receive',super()._receive)
        def _pose(self):return self._timed('pose',super()._pose)
        def _sense(self,*args):return self._timed('sense',super()._sense,*args)
        def _choose_bomb(self):return self._timed('choose',super()._choose_bomb)
        def _navigate(self,*args,**kwargs):
            start=time.perf_counter()
            result=super()._navigate(*args,**kwargs)
            self._probe_times['navigate']+=time.perf_counter()-start
            self._probe_calls['navigate']+=1
            self._probe_nav=dict(result)
            return result

        def control(self):
            self._probe_nav=None
            start=time.perf_counter()
            command=super().control()
            self._probe_times['control']+=time.perf_counter()-start
            carrying=bool(self.grasped_bombs());hit=self._center_hit
            nav=self._probe_nav
            source=getattr(self,'_command_source',None)
            if source is None:
                source='navigation' if nav==command else 'post_navigation_override' if nav else 'early_return'
            route=self._route;end=route[-1] if route else None
            remaining=float(np.linalg.norm(end-self._target)) if end is not None and self._target is not None else None
            self._probe_frame=dict(step=int(self.elapsed_timestep),drone=int(self.identifier),
                mode=self._mode,carrying=carrying,target=point(self._target),position=point(self._position),
                command={k:float(v) for k,v in command.items()},navigation=nav,source=source,
                center_distance=float(hit.distance) if hit is not None else None,
                center_angle=float(hit.angle) if hit is not None else None,
                route_points=len(route),route_end=point(end),endpoint_remaining=remaining,
                delivery_ticks=int(self._delivery_ticks),gps_missing=self.measured_gps_position() is None,
                compass_missing=self.measured_compass_angle() is None,health=int(self.drone_health),
                peer_count=len(self._peers),terminal_state=getattr(self,'_contact_state',None))
            return command
    # Preserve base identity/source hashes; receipt explicitly records observer.
    Probe.__name__=base.__name__;Probe.__module__=base.__module__
    return Probe


class LifecycleObserver:
    def __init__(self):
        self.bombs=[];self.rows=[];self.before={};self.last={};self.frames=[]
        self.seconds=0.;self.ticks=0

    def begin(self,gui):
        start=time.perf_counter()
        if not self.bombs:
            self.bombs=list(gui._the_map.bombs)
            self.rows=[dict(bomb=i+1,initial=point(b.position),first_seen=None,seen_by=[],
                first_claim=None,first_near=None,pickups=[],losses=[],releases=[],
                first_center_near=None,disposed=None,phase_steps={},associations='claims matched offline within 60px; ambiguous excluded')
                for i,b in enumerate(self.bombs)]
            self.uid={b.uid:i for i,b in enumerate(self.bombs)}
        self.before={}
        for d in gui._drones:
            ident=int(d.identifier);held=[self.bombs.index(b) for b in d.grasped_bombs() if b in self.bombs]
            visible=[]
            # Only enabled public semantic BOMB rays authorize discovery. Raw
            # renderer IDs resolve identity offline, not visibility/occlusion.
            sem=(d.semantic_values() or ()) if d.drone_health>0 else ()
            angles={round(float(s.angle),8) for s in sem if getattr(s.entity_type,'name','')=='BOMB'}
            sensor=d.semantic();raw=getattr(sensor,'_hitpoints',None)
            if angles and raw is not None:
                for a,hit in zip(sensor.ray_angles,raw):
                    if round(float(a),8) in angles and int(hit[8]) in self.uid:
                        visible.append(self.uid[int(hit[8])])
            self.before[ident]=(held,set(visible))
        self.seconds+=time.perf_counter()-start

    def end(self,gui):
        start=time.perf_counter();self.ticks+=1
        step=int(gui._elapsed_timestep)
        for d in gui._drones:
            ident=int(d.identifier);old,visible=self.before.get(ident,([],set()))
            frame=getattr(d,'_probe_frame',None)
            current=[self.bombs.index(b) for b in d.grasped_bombs() if b in self.bombs]
            for i in visible:
                r=self.rows[i]
                if r['first_seen'] is None:r['first_seen']=step
                if ident not in r['seen_by']:r['seen_by'].append(ident)
            for i in set(current)-set(old):self.rows[i]['pickups'].append(dict(step=step,drone=ident))
            for i in set(old)-set(current):
                if not self.bombs[i].disposed:self.rows[i]['losses'].append(dict(step=step,drone=ident))
            # Dead/removed drones retain their final probe. Physics losses are
            # real, but old controls must not count as new decisions/phase time.
            if not frame or frame['step']!=step:continue
            associated=None
            if old:associated=old[0]
            elif frame['mode']=='collect' and frame['target'] is not None:
                # Correct only OFFLINE for the drone's estimated origin error.
                target=np.asarray(frame['target'])
                if frame['position'] is not None:target+=np.asarray(d.true_position())-frame['position']
                matches=[i for i,b in enumerate(self.bombs) if not b.disposed and math.dist(target,b.position)<60]
                if len(matches)==1:associated=matches[0]
            if associated is not None:
                r=self.rows[associated];distance=frame['center_distance']
                if old:
                    phase='center' if distance is not None and distance<105 else 'transport'
                    if phase=='center' and r['first_center_near'] is None:r['first_center_near']=step
                    if frame['command'].get('grasper')==0 and self.last.get(ident,{}).get('command',{}).get('grasper')!=0:
                        r['releases'].append(dict(step=step,drone=ident,distance=distance))
                else:
                    phase='approach'
                    if r['first_claim'] is None:r['first_claim']=step
                    if math.dist(d.true_position(),self.bombs[associated].position)<85 and r['first_near'] is None:r['first_near']=step
                r['phase_steps'][phase]=r['phase_steps'].get(phase,0)+1
            frame=dict(frame,bomb=None if associated is None else associated+1,
                       truth=point(d.true_position()))
            prev=self.last.get(ident,{})
            changed=any(frame.get(k)!=prev.get(k) for k in ('mode','carrying','source','terminal_state','bomb'))
            release=frame['command'].get('grasper')==0 and bool(old)
            if step%30==0 or release or (changed and step-prev.get('saved',-99)>=10):
                self.frames.append(frame);frame['saved']=step
            else:frame['saved']=prev.get('saved',-99)
            self.last[ident]=frame
        for r,b in zip(self.rows,self.bombs):
            if b.disposed and r['disposed'] is None:r['disposed']=step
        self.seconds+=time.perf_counter()-start

    def export(self,gui):
        return dict(scope='OFFLINE ONLY; no true state fed to controllers',bombs=self.rows,frames=self.frames,
            observer_seconds=self.seconds,observer_ticks=self.ticks,
            function_seconds={str(d.identifier):dict(getattr(d,'_probe_times',{})) for d in gui._drones},
            discovered=sum(r['first_seen'] is not None for r in self.rows),
            official_disposed=sum(r['disposed'] is not None for r in self.rows))
