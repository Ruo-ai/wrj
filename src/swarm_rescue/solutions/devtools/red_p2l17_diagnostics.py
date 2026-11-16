"""Passive RL17 trace: returned matching decisions and actual source versions.

All simulator truth stays here, after guarded control, never in a controller,
message, or mapper input. Optional instrumentation is identical for both arms.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def outgoing_snapshot(message):
    """Unwrap the retained red-v4 -> red-v3 -> red-v1 envelope."""
    for _ in range(4):
        if not isinstance(message,tuple) or not message:return None
        if message[0]=='red-v1':return message[2]
        if message[0] not in ('red-v3','red-v4'):raise ValueError('Unknown red envelope')
        message=message[1]
    raise ValueError('Too many envelopes')


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMatcher, WallLineMapper, MyDroneRedLines11
    from swarm_rescue.solutions.my_drone_red_repair17 import RepairMatcher17
    from swarm_rescue.solutions.my_drone_red_p2l17 import PointLineMatcher17
    old_control=gui_sr.run_control;rows={};truth={};events=[];radio=[];errors=[];seconds=0.;originals=[]

    def instrument(cls,name,kind):
        owned=name in cls.__dict__;original=getattr(cls,name);originals.append((cls,name,original,owned))
        def wrapped(self,*args,**kwargs):
            start=time.perf_counter();answer=original(self,*args,**kwargs);elapsed=time.perf_counter()-start
            if kind=='match':
                self._audit_decision17=dict(accepted=answer[0] is not None,info=answer[1],
                    offset=None if answer[0] is None else answer[0].tolist(),seconds=elapsed)
            elif kind=='update':
                self._audit_update17=getattr(self,'_audit_update17',0.)+elapsed
                self._audit_update_max17=max(getattr(self,'_audit_update_max17',0.),elapsed)
            elif kind=='send':
                # Message construction is a broadcast attempt, not a receipt.
                try:
                    snapshot=outgoing_snapshot(answer)
                    if snapshot is not None:
                        source,version,payload=snapshot
                        radio.append(dict(event='send_attempt',holder=int(self.identifier),step=int(self._steps),
                            source=int(source),version=int(version),sha256=hashlib.sha256(payload).hexdigest()))
                except (IndexError,TypeError,ValueError):
                    if len(errors)<12:errors.append('Unexpected outgoing message format')
            return answer
        setattr(cls,name,wrapped)

    # Wrap child first; its saved method still resolves the instrumented parent.
    instrument(RepairMatcher17,'match','match')
    instrument(PointLineMatcher17,'match','match')
    instrument(WallLineMatcher,'match','match')
    instrument(WallLineMapper,'update','update')
    instrument(MyDroneRedLines11,'define_message_for_all','send')

    def control(drone):
        nonlocal seconds
        answer=old_control(drone);start=time.perf_counter()
        try:
            ident=str(drone.identifier);tick=int(drone._steps);mapper=drone._submission_mapper
            row=rows.setdefault(ident,dict(versions={},max_pending=0,submitted=False,mapper_seconds=0.))
            missing=drone.measured_gps_position() is None or drone.measured_compass_angle() is None
            actual=np.asarray(drone.true_position());heading=float(drone.true_angle())
            truth.setdefault(ident,[]).append([tick,*actual,heading,*(mapper.position if mapper.position is not None else [np.nan,np.nan]),
                mapper.heading,missing,*drone._position,drone._heading])
            if tick%8==0 and missing:
                events.append(dict(drone=int(ident),step=tick,
                    decision=getattr(mapper.matcher,'_audit_decision17',{}),
                    p2l=getattr(mapper.matcher,'last_p2l',{})))
            row['max_pending']=max(row['max_pending'],len(mapper.pending))
            for source,snapshot in drone._radio_maps.snapshots.items():
                if row['versions'].get(str(source))!=snapshot[1]:
                    radio.append(dict(event='local_publish' if source==int(ident) else 'received',holder=int(ident),
                        step=tick,source=source,version=snapshot[1],sha256=hashlib.sha256(snapshot[2]).hexdigest()))
                    row['versions'][str(source)]=snapshot[1]
            if drone._exploration_submitted and not row['submitted']:
                radio.append(dict(event='submitted',holder=int(ident),step=tick,versions=dict(row['versions'])))
                row['submitted']=True
            row.update(mapper_seconds=getattr(mapper,'_audit_update17',0.),mapper_max_seconds=getattr(mapper,'_audit_update_max17',0.),
                p2l_seconds=getattr(mapper.matcher,'p2l_seconds',0.),p2l_stats=dict(getattr(mapper.matcher,'p2l_stats',{})))
        except Exception as exc:
            if len(errors)<12:errors.append(repr(exc))
        seconds+=time.perf_counter()-start
        return answer

    gui_sr.run_control=control
    try:yield
    finally:
        gui_sr.run_control=old_control
        for cls,name,original,owned in reversed(originals):
            if owned:setattr(cls,name,original)
            else:delattr(cls,name)
        path=Path(destination)
        np.savez_compressed(path.with_suffix('.npz'),**{'pose_'+k:np.asarray(v,float) for k,v in truth.items()})
        with path.open('x',encoding='utf-8') as f:
            json.dump(dict(schema='rl17-passive-v1',drones=rows,errors=errors,seconds=seconds,matches=events,radio=radio,
                columns=['step','true_x','true_y','true_heading','mapper_x','mapper_y','mapper_heading','missing','navigation_x','navigation_y','navigation_heading'],
                limitation='send_attempt != delivered. Received records are holder-local accepted versions; shared bias remains possible.'),f,indent=2)
