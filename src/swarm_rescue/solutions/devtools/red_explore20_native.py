"""Small native RN7 component checks with an already-exhausted allowance.

Tick/allowance warm-start is a synthetic component fixture, NOT R19 replay.
Official hull, sensors and physics remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from swarm_rescue.solutions.devtools.red_pair19_batch import read,write

MODULE='swarm_rescue.solutions.devtools.red_explore20_native'


def child(case):
    from swarm_rescue.solutions import my_drone_red_rn7 as rn
    from swarm_rescue.solutions.my_drone_red_budget20 import RenewedPassage20
    from swarm_rescue.solutions.devtools.red_rn7_native import run_case
    original=rn.PassageMotion;base=RenewedPassage20 if case.pop('candidate') else original
    class Warm(base):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.tick=1199;self.episodes=8
    rn.PassageMotion=Warm
    try:return run_case(**case)
    finally:rn.PassageMotion=original


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--child-json')
    a=p.parse_args()
    if a.child_json:
        if sys.stdin.readline().strip()!='GO':raise RuntimeError('Missing resource handshake')
        write(a.output,child(json.loads(a.child_json)));return
    from swarm_rescue.solutions.devtools.optimization.evaluation.resources import run_limited
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for label,params in [('open55',dict(width=55)),('blocked25',dict(width=25)),('deadend55',dict(width=55,deadend=True))]:
        for candidate in (False,True):
            name=label+('_r20' if candidate else '_rl17');path=a.output/(name+'.json')
            argv=[MODULE,'--output',str(path),'--child-json',json.dumps(dict(candidate=candidate,steps=240,**params))]
            boot=(f'import runpy,sys;sys.path[:]={sys.path!r};sys.prefix={sys.prefix!r};sys.exec_prefix={sys.exec_prefix!r};'
                  f'sys.argv={argv!r};runpy.run_module({MODULE!r},run_name="__main__")')
            cmd=[sys.executable,'-B','-u','-S','-X','utf8','-c',boot]
            with path.with_suffix('.log').open('x',encoding='utf-8') as log:
                resources=run_limited(cmd,dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),log,timeout=45)
            text=path.with_suffix('.log').read_text('utf-8',errors='replace')
            row=read(path) if path.exists() else {}
            row={k:v for k,v in row.items() if k not in ('trace','events')}
            row.update(case=name,resources=resources,valid=resources['exit_code']==0 and not any(t in text for t in ('Traceback','Exception ignored','AssertionError')))
            rows.append(row);print(name,{k:row.get(k) for k in ('valid','passed','contacts','max_x','stats')},flush=True)
    write(a.output/'summary.json',dict(scope='synthetic pre-exhausted RN7 native component, no flight score',rows=rows))


if __name__=='__main__':main()
