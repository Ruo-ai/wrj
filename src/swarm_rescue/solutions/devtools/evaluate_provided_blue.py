"""Evaluate the user-provided blue controller without replacing the red entry."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import traceback
from swarm_rescue.solutions.devtools import evaluate_red as red

SCOPE = 'local_provided_blue_only; not competition reference panel'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(source, score, disposed, timestep, crashed=False, reference='provided-my_drone_eval_blue'):
    from swarm_rescue.simulation.reporting.place_score_manager import (
        PlaceScoreManager, PlacePhase1Result, ReferenceBlueRun, difficulty_from_blue_score)
    manager=PlaceScoreManager()
    ref=ReferenceBlueRun(reference,score,difficulty_from_blue_score(score),
                         crashed,percent_disposed=disposed,full_disposal_timestep=timestep)
    bomb=manager.compute_bomb_score([ref],placed_count=source['placed_count'],expected_count=source['expected_count'])
    return asdict(manager.merge_final_score(PlacePhase1Result(**source['phase1']),bomb)) if bomb.sufficient_references else None


def reference_blue_class(variant='baseline'):
    from swarm_rescue.solutions.devtools.optimization.evaluation.identities import load_blue
    return load_blue(variant)


def worker(path):
    request=json.loads(Path(path).read_text(encoding='utf-8'));run=Path(request['run_dir'])
    result=dict(map=request['map'],round=request['round'],seed=request['seed'],status='error',phase1=None,score_scope=SCOPE)
    try:
        os.environ['SWARM_RESCUE_RESULT_PATH']=request['engine_dir']
        import numpy as np
        import yaml
        import swarm_rescue.launcher as launcher_module
        variant = request.get('blue_variant', 'baseline')
        MyDroneBlue = reference_blue_class(variant)
        if request.get('lifecycle_diagnostics'):
            from swarm_rescue.solutions.devtools.blue_lifecycle import instrument
            MyDroneBlue=instrument(MyDroneBlue)
        entry = getattr(launcher_module, 'drone_entry', launcher_module)
        entry.drone_class_for_mode=lambda mode: MyDroneBlue
        red.configure_ray_backend('cpu')
        random.seed(request['seed']);np.random.seed(request['seed'])
        config=red.evaluation_config(request['map'],drones=request['drones'],zones=request.get('zones'))
        config['evaluation_plan'][0]['team_mode']='rescue';red.write_json(run/'conditions.json',config)
        calibration = request.get('platform_layout_only', False)
        source = None if calibration else json.loads(Path(request['red_result']).read_text(encoding='utf-8'))
        layout=Path(request['layout']);payload=json.loads(layout.read_text(encoding='utf-8'))
        if not calibration:
            assert source['status']=='ok' and source['submitted']
        count = request['count'] if calibration else source['placed_count']
        assert payload['map_name']==request['map'] and len(payload['bombs'])==count
        if 'zones' in request and sorted(payload.get('zones_config',[]))!=sorted(request['zones']):
            raise ValueError('Bomb layout zones differ from blue scenario')
        with tempfile.TemporaryDirectory(prefix='provided-blue-') as tmp:
            config_path=Path(tmp)/'eval.yml';config_path.write_text(yaml.safe_dump(config),encoding='ascii')
            launcher=launcher_module.Launcher(config_path=str(config_path),default_bombs_file=str(layout))
            assert launcher.eval_plan_ok
            data=launcher.one_round(launcher.eval_plan.list_eval_config[0],request['round'],headless=True,competition_mode=True)
        assert data is not None and int(data[11])==count
        score,disposed,time_score=launcher.score_manager.compute_score(data[5],data[4],data[6],data[2],has_crashed=data[9])
        result.update(status='controller_crash' if data[9] else 'ok',blue_score=float(score),
            disposed_count=int(data[5]),disposed_percent=float(disposed),blue_time_score=float(time_score),
            blue_health_score=float(data[6]),bomb_count=int(data[11]),full_disposal_timestep=int(data[3]),
            elapsed_steps=int(data[2]),elapsed_walltime=float(data[7]),walltime_limited=bool(data[8]),
            controller_crashed=bool(data[9]),layout_sha256=digest(layout),red_result_sha256=None if calibration else digest(request['red_result']),
            blue_source_sha256=digest(Path(__file__).parent.parent / (MyDroneBlue.__module__.split('.')[-1]+'.py')),
            blue_variant=variant,blue_controller_class=MyDroneBlue.__name__,
            phase1=None if calibration else merge(source,float(score),float(disposed),int(data[3]),bool(data[9]),
                'provided-my_drone_eval_blue-'+variant))
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}';traceback.print_exc()
    red.write_json(run/'result.json',result)
    return 0 if result['status']=='ok' else 1


def command(path):
    script=str(Path(__file__).resolve())
    bootstrap=(f'import sys,runpy; sys.path[:]={sys.path!r}; '
        f'sys.prefix={sys.prefix!r}; sys.exec_prefix={sys.exec_prefix!r}; '
        f'sys.argv={[script,"--worker",str(path)]!r}; runpy.run_path({script!r},run_name="__main__")')
    return [sys.executable,'-u','-B','-S','-X','utf8','-c',bootstrap]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker');p.add_argument('--red-output',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--blue-variant',choices=('baseline','adapted','enhanced','sweep'),default='baseline')
    p.add_argument('--maps',nargs='+',choices=red.MAPS,default=list(red.MAPS))
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--drones',type=int,default=5)
    a=p.parse_args()
    if a.worker:return worker(a.worker)
    if not a.red_output or not a.output:p.error('--red-output and --output are required')
    source=a.red_output.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    manifest=red.environment_manifest(a);manifest['score_scope']=SCOPE
    manifest['arguments']={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    manifest['red_manifest_sha256']=digest(source/'manifest.json');red.write_json(a.output/'manifest.json',manifest)
    rows=json.loads((source/'results.json').read_text(encoding='utf-8'));results=[]
    red.worker_command=command
    environment=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',SWARM_BLUE_DEBUG='1')
    for r in rows:
        if r['map'] not in a.maps or r['round']>a.rounds:continue
        name=f"{r['map']}_r{r['round']:02}";run=a.output/name;run.mkdir()
        layout=next((source/name/'artifacts').glob('*_bombs.json'))
        request=dict(map=r['map'],round=r['round'],seed=r['seed'],drones=a.drones,run_dir=str(run),blue_variant=a.blue_variant,
                     layout=str(layout),red_result=str(source/name/'result.json'))
        print(f'Start {name}',flush=True)
        result=red.execute_round(request,600,environment,None);results.append(result)
        red.write_json(a.output/'results.json',results)
        print(f"Finish {name}: {result['status']}, disposed={result.get('disposed_count')}, blue={result.get('blue_score')}",flush=True)
    return 0 if results and all(r['status']=='ok' for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
