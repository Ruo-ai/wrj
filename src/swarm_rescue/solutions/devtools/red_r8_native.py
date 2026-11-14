"""R8 native components and same-sensor mapper comparison. Offline only."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

MODULE='swarm_rescue.solutions.devtools.red_r8_native'


def case(params):
    if params.get('kind')=='mapping':
        from swarm_rescue.solutions import my_drone_red_mapping as original
        from swarm_rescue.solutions.red_pixel_faces import PixelFaceMapper
        from swarm_rescue.solutions.devtools.red_stopturn_native import motion_case
        from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer,breakdown_to_dict
        saved=original.SubmissionMapper;instances=[]
        class Shadow(saved):
            def __init__(self,size):
                super().__init__(size);self.pixel=PixelFaceMapper(size);instances.append(self)
            def update(self,*args,**kwargs):
                super().update(*args,**kwargs);self.pixel.update(*args,**kwargs)
        original.SubmissionMapper=Shadow
        try:row=motion_case(group='A',noise=params.get('noise',True),steps=420,scene='corner')
        finally:original.SubmissionMapper=saved
        pixel=instances[0].pixel;arrays=row.pop('arrays')
        arrays['pixel_prediction']=pixel.to_pixels(pixel.mask())
        row['pixel_map_formula']=breakdown_to_dict(MapExplorationScorer().score(arrays['truth'],arrays['pixel_prediction'],row['seconds']))
        row['pixel_stats']=dict(pixel.stats);row['same_sensor_inputs']=True
        row['arrays']=arrays;return row
    from swarm_rescue.solutions import my_drone_red_rn7 as old
    from swarm_rescue.solutions.red_footprint8 import FootprintMotion
    from swarm_rescue.solutions.devtools.red_rn7_native import run_case
    saved=old.PassageMotion
    if params['kind']=='n8':old.PassageMotion=FootprintMotion
    try:return run_case(**{k:v for k,v in params.items() if k!='kind'})
    finally:old.PassageMotion=saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case-json');p.add_argument('--suite',choices=['motion','mapping','verify'],default='motion');args=p.parse_args()
    if args.case_json:
        if sys.stdin.readline().strip()!='GO':raise RuntimeError('No handshake')
        row=case(json.loads(args.case_json));arrays=row.pop('arrays',None)
        if arrays is not None:np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
        with args.output.open('x',encoding='utf-8') as f:json.dump(row,f,indent=2,allow_nan=False)
        return
    from swarm_rescue.solutions.devtools.optimization.evaluation.resources import run_limited
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    if args.suite=='mapping':cases=[('mapping_'+str(noise),dict(kind='mapping',noise=noise)) for noise in (False,True)]
    elif args.suite=='verify':cases=[('n8_w46',dict(kind='n8',width=46)),('n8_w50',dict(kind='n8',width=50)),('n8_deadend',dict(kind='n8',width=55,deadend=True))]
    else:
        cases=[(f'{kind}_w{width}',dict(kind=kind,width=width)) for width in (46,50,80,25) for kind in ('rn7','n8')]
        cases+=[('n8_deadend',dict(kind='n8',width=55,deadend=True)),('n8_w50_ideal',dict(kind='n8',width=50,noise=False))]
    for name,params in cases:
        dest=args.output/(name+'.json');argv=[MODULE,'--output',str(dest),'--case-json',json.dumps(params)]
        boot=(f'import runpy,sys;sys.path[:]={sys.path!r};sys.prefix={sys.prefix!r};sys.exec_prefix={sys.exec_prefix!r};'
              f'sys.argv={argv!r};runpy.run_module({MODULE!r},run_name="__main__")')
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with dest.with_suffix('.log').open('x',encoding='utf-8') as log:
            resources=run_limited([sys.executable,'-B','-u','-S','-X','utf8','-c',boot],env,log,timeout=60)
        logtext=dest.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
        row=json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else dict(error=logtext)
        row.update(case=name,params=params,resources=resources,valid=resources['exit_code']==0 and
                   not any(s in logtext for s in ('Traceback','Exception ignored from cffi callback')))
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('trace','events')},ensure_ascii=True),flush=True)
    with (args.output/'results.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
