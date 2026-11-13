"""Offline diagnostics only: held payloads, event states, and planning timings."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from swarm_rescue.solutions.my_drone_eval_blue import RescueGrid
from swarm_rescue.solutions.my_drone_blue_transport import TransportGrid


def inspect(attempt):
    read=lambda name:json.loads((attempt/name).read_text(encoding='utf-8-sig'))
    r,t=read('result.json'),read('telemetry.json');pending={}
    for e in t.get('transport',{}).get('events',[]):
        key=(e['drone'],e['bomb'])
        if e['event']=='pickup':pending[key]=e['step']
        else:pending.pop(key,None)
    states={};events=[]
    for line in (attempt/'run.log').read_text(encoding='utf-8',errors='replace').splitlines():
        if line.startswith('B3_EVENT '):
            e=json.loads(line[9:]);events.append(e)
            if e['event']=='state':states[str(e['drone'])]=e
    pose=[p['error'] for p in t.get('pose_samples',[])]
    return dict(attempt=str(attempt.resolve()),result={k:r.get(k) for k in ('blue_score','disposed_count',
        'blue_health_score','elapsed_steps','elapsed_walltime','walltime_limited','controller_crashed','engine_callback_errors')},
        transport=t.get('transport'),pending=[dict(drone=d,bomb_event_id=b,started=s,duration=r['elapsed_steps']-s)
                                          for (d,b),s in pending.items()],
        pose=dict(mean=float(np.mean(pose)),p95=float(np.percentile(pose,95)),maximum=max(pose)) if pose else None,
        last_states=states,events=events)


def benchmark():
    records=[]
    for cls in (RescueGrid,TransportGrid):
        g=cls((1700,1100));g.free[:]=True
        for c in (20,40,60,80):
            g.evidence[:50 if c%40 else 35,c]=8
        position,target=g.point((8,8)),g.point((9,99))
        start=time.perf_counter()
        for _ in range(20):path=g.route(position,target)
        records.append(dict(class_name=cls.__name__,calls=20,seconds=time.perf_counter()-start,
                            route_points=len(path),endpoint=list(map(float,path[-1]))))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempts',type=Path,nargs='*');p.add_argument('--benchmark',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=dict(attempts=[inspect(v) for v in a.attempts],benchmark=benchmark() if a.benchmark else None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data if a.benchmark else [r['result'] for r in data['attempts']],ensure_ascii=False))
