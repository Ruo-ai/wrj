"""Frozen plans, per-phase receipts, serial execution and interrupted-run recovery."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import zipfile

from ... import evaluate_verified_revision as ev
from ..dataset_manager.artifacts import latest_output,verify as verify_dataset
from ..dataset_manager.index import REPO,resolve_path,file_hash,path_name
from ..map_generator.validate import digest
from ..map_generator.append_maps import append_lock
from .resources import run_limited


ROOT = REPO/'output/optimization/evaluations'
read,write = ev.read,ev.atomic_json


def worker_command(request):
    module = __package__+'.worker'
    bootstrap = (f'import runpy,sys; sys.path[:]={sys.path!r}; sys.prefix={sys.prefix!r}; '
                 f'sys.exec_prefix={sys.exec_prefix!r}; sys.argv={[module,str(request)]!r}; '
                 f'runpy.run_module({module!r},run_name="__main__")')
    return [sys.executable,'-B','-u','-S','-X','utf8','-X','faulthandler','-c',bootstrap]


def source_receipt(map_rows):
    paths = [p for p in (REPO/'src').rglob('*') if p.is_file() and p.suffix in ('.py','.yml','.png')
             and '__pycache__' not in p.parts and 'generated_maps' not in p.parts]
    paths += [p for p in (REPO/'config').rglob('*') if p.is_file() and p.suffix in ('.yml','.yaml','.json')]
    paths += [resolve_path(name) for row in map_rows for name in row['files']]
    return {path_name(p):file_hash(p) for p in sorted(set(paths))}


def validate_sources(receipt):
    for name,expected in receipt.items():
        if file_hash(resolve_path(name))!=expected:
            raise ValueError(f'Frozen source/data changed: {name}; start a new batch')


def environment():
    packages = {}
    for name in ('numpy','opencv-contrib-python','arcade','pymunk','PyYAML'):
        packages[name] = importlib.metadata.version(name)
    return dict(python=sys.version,platform=platform.platform(),packages=packages)


def create_or_resume(args):
    from .scenarios import official_rows,scenario,engine_profile
    from .identities import blue_identity
    official=getattr(args,'official_maps',None)
    selected=getattr(args,'map_names',None)
    if selected:
        from .scenarios import training_rows
        dataset=(args.dataset or latest_output()).resolve()
        rows=training_rows(selected,dataset)
        envelope=read(dataset/'dataset.json')
        manifest_hash=file_hash(dataset/'dataset.json')
        if args.split!='train':raise ValueError('Mixed explicit development maps must use train')
    elif official:
        dataset=None;rows=official_rows(official)
        envelope={'dataset_id':'official-semifinal'}
        manifest_hash=digest(rows)
    else:
        dataset = (args.dataset or latest_output()).resolve()
        verify_dataset(dataset)
        envelope = read(dataset/'dataset.json')
        index = {r['id']:r for r in read(dataset/'index.json')['maps']}
        rows = [index[r['id']] for r in read(dataset/(args.split+'.json'))['maps']]
        manifest_hash=file_hash(dataset/'dataset.json')
    if args.limit is not None: rows=rows[:args.limit]
    if not rows: raise ValueError('Selected dataset split is empty')
    scenarios={r['id']:scenario(resolve_path(r['map_path']),r.get('official_map')) for r in rows}
    profile=engine_profile()
    for controller in args.controllers:
        ev.controller_identity(controller)
    config = dict(protocol='semifinal-r0-v1',engine_profile=profile,scenarios=scenarios,dataset=path_name(dataset) if dataset else None,dataset_id=envelope['dataset_id'],
        dataset_manifest_sha256=manifest_hash,split=args.split,
        controllers=args.controllers,rounds=args.rounds,seed=args.seed,timeout=args.timeout,
        memory_bytes=args.memory_bytes,simultaneous_simulations=1,cpu_affinity_count=1,
        recording=True,competition_mode=True,blue='provided_my_drone_eval_blue',
        **blue_identity(getattr(args,'blue_variant','adapted')),
        controller_identities={c:ev.controller_identity(c) for c in args.controllers},
        map_ids=[r['id'] for r in rows],smoke=args.smoke,
        diagnostic_steps=getattr(args,'diagnostic_steps',8) if args.smoke else None,environment=environment())
    sources = source_receipt(rows)
    if args.output:
        output = args.output.resolve()
    else:
        ROOT.mkdir(parents=True,exist_ok=True)
        output = None
        if not args.new:
            for candidate in sorted(ROOT.glob('batch_*'),reverse=True):
                if (candidate/'plan.json').is_file() and read(candidate/'plan.json')['configuration']==config:
                    output = candidate
                    break
        if output is None:
            number = 1
            while (ROOT/f'batch_{number:03d}').exists():
                number += 1
            output = ROOT/f'batch_{number:03d}'
    if output==REPO or any(output.is_relative_to(REPO/p) for p in ('src','map_data','config','.git','.venv','平台评测结果')):
        raise ValueError('Evaluation output must be a separate directory')
    if output.exists():
        plan = read(output/'plan.json')
        if plan['configuration']!=config or plan['source_sha256']!=sources:
            raise ValueError('Resume configuration/source mismatch; keep old data and use --new or a new --output')
        if file_hash(output/'source_snapshot.zip')!=plan['source_snapshot_sha256']:
            raise ValueError('Frozen source archive changed')
    else:
        output.mkdir(parents=True)
        schedule = []
        for row in rows:
            for number in range(1,args.rounds+1):
                # Same per-map/per-round seed for every candidate, independent of list order.
                run_seed = (args.seed+int(row['content_sha256'][:8],16)+(number-1)*1000003)%(2**32)
                order = args.controllers if number%2 else list(reversed(args.controllers))
                for controller in order:
                    schedule.append(dict(map_id=row['id'],name=row['name'],family=row['family'],
                        map_file=row['map_path'],map=row.get('official_map') or 'Generated'+row['geometry_sha256'],official_map=row.get('official_map'),scenario=scenarios[row['id']],round=number,
                        seed=run_seed,controller=controller))
        with zipfile.ZipFile(output/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(resolve_path(name),name if not Path(name).is_absolute() else 'external/'+digest(name))
        plan = dict(configuration=config,source_sha256=sources,schedule=schedule,map_rows=rows,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source_snapshot_sha256=file_hash(output/'source_snapshot.zip'))
        write(output/'plan.json',plan)
    return output,plan


def receipt(attempt,plan_sha):
    return dict(plan_sha256=plan_sha,files={p.relative_to(attempt).as_posix():file_hash(p)
        for p in sorted(attempt.rglob('*')) if p.is_file() and p.name!='integrity.json'})


def load_done(phase_dir,plan_sha):
    marker = phase_dir/'done.json'
    if not marker.is_file():
        return None,None
    done = read(marker)
    attempt = phase_dir/done['attempt']
    if attempt.parent.resolve()!=phase_dir.resolve():
        raise ValueError('Invalid attempt path')
    stored = read(attempt/'integrity.json')
    if stored!=receipt(attempt,plan_sha) or file_hash(attempt/'integrity.json')!=done['receipt_sha256']:
        raise ValueError(f'Completed artifacts changed: {attempt}')
    return read(attempt/'result.json'),attempt


def phase_valid(row,phase):
    if row.get('status')!='ok' or row.get('exit_code')!=0 or row.get('walltime_limited') or row.get('controller_crashed') or row.get('engine_callback_errors'):
        return False
    if phase=='red':
        return bool(row.get('submitted') and row.get('valid_shape') and row.get('phase1'))
    return bool(row.get('phase1') and row['phase1'].get('score_status')=='final')


def execute_phase(output,plan,item,phase,red_attempt=None,retry_failed=False):
    plan_sha = file_hash(output/'plan.json')
    phase_dir = output/'runs'/f"{item['name']}_r{item['round']:02d}"/item['controller']/phase
    phase_dir.mkdir(parents=True,exist_ok=True)
    row,old_attempt = load_done(phase_dir,plan_sha)
    if row is not None and (not retry_failed or phase_valid(row,phase)):
        if phase=='blue' and red_attempt and not plan['configuration']['smoke'] and row.get('red_result_sha256')!=file_hash(red_attempt/'result.json'):
            raise ValueError('Blue result belongs to a different red attempt')
        return row,old_attempt
    n = 1
    while (phase_dir/f'attempt_{n:03d}').exists():
        n += 1
    attempt = phase_dir/f'attempt_{n:03d}'
    attempt.mkdir()
    conditions=item.get('scenario',{'stage':'preliminary','zones':[]})
    count=10 if phase=='blue' and conditions['stage']=='semifinal' else 5
    request = dict(official_map=item.get('official_map'),stage=conditions['stage'],zones=conditions['zones'],map=item['map'],map_file=str(resolve_path(item['map_file'])),round=item['round'],
        seed=item['seed'],phase=phase,run_dir=str(attempt),drones=count,bombs=2,recording=True,
        ray_backend='cpu',visual=False,smoke=plan['configuration']['smoke'],
        diagnostic_steps=plan['configuration'].get('diagnostic_steps',8),
        plan_sha256=plan_sha,**ev.controller_identity(item['controller']))
    if phase=='blue':
        from .identities import blue_identity
        request.update(blue_identity(item.get('blue_variant', plan['configuration'].get('blue_variant', 'baseline'))))
        if item.get('oracle_layout'):
            request.update(layout=str(resolve_path(item['oracle_layout'])),platform_layout_only=True,count=10,
                           red_result=None)
        else:
            request.update(layout=str(ev.one_layout(red_attempt)),red_result=str(red_attempt/'result.json'))
        if request['smoke']:
            # A diagnostic-only one-bomb layout exercises rescue initialization.
            from .scenarios import scenario
            spec=scenario(resolve_path(item['map_file']),item.get('official_map'))
            x,y = spec['spawn'];w,h = spec['size']
            layout = attempt/'smoke_layout.json'
            write(layout,dict(map_name=item['map'],zones_config=request['zones'],bombs=[dict(x=x-w/2,y=h/2-y,theta=0)]))
            request.update(layout=str(layout),platform_layout_only=True,count=1)
    settings = dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1',
        G3_MEMORY_BYTES=str(plan['configuration']['memory_bytes']))
    with tempfile.TemporaryDirectory(prefix='g3-engine-') as scratch:
        if not scratch.isascii():
            raise ValueError('The system TEMP path must be ASCII for native image export')
        request['engine_dir'] = str(Path(scratch)/'artifacts')
        write(attempt/'request.json',request)
        try:
            with (attempt/'run.log').open('w',encoding='utf-8') as log:
                try:
                    limits = run_limited(worker_command(attempt/'request.json'),settings,log,
                                        plan['configuration']['timeout'],plan['configuration']['memory_bytes'])
                except (OSError,RuntimeError) as error:
                    limits = dict(exit_code=None,status='resource_setup_error',error=str(error))
            write(attempt/'resources.json',limits)
        finally:
            artifacts = Path(request['engine_dir'])
            if artifacts.exists():
                shutil.copytree(artifacts,attempt/'artifacts')
    row = read(attempt/'result.json') if (attempt/'result.json').is_file() else dict(
        map=item['map'],round=item['round'],seed=item['seed'],status=limits['status'],phase1=None)
    row.update(exit_code=limits['exit_code'],phase=phase,smoke=request['smoke'])
    # cffi may print an exception and continue with exit 0. Such a run is not a
    # clean reference, even when the controller and score manager finish normally.
    row['engine_callback_errors'] = (attempt/'run.log').read_text(encoding='utf-8',errors='replace').count('Exception ignored from cffi callback')
    if phase=='blue' and row.get('status')=='ok' and any(
            row.get(k)!=request[k] for k in ('blue_variant','blue_controller_class')):
        row.update(status='invalid_blue_identity',error='Loaded blue differs from frozen request')
    if limits['status']!='exited':
        row['status'] = limits['status']
        if limits.get('error'):
            row['error'] = limits['error']
    if any(row.get(key)!=request[key] for key in ('map','round','seed')):
        row.update(status='invalid_result_identity',error='Worker identity differs from request')
    if phase=='blue' and not request['smoke'] and row.get('status')=='ok':
        if (row.get('layout_sha256')!=file_hash(request['layout']) or
                (not request.get('platform_layout_only') and row.get('red_result_sha256')!=file_hash(request['red_result']))):
            row.update(status='invalid_layout_binding',error='Blue result is not bound to the exact red result/layout')
    if request['smoke']:
        row['score_scope'] = f"diagnostic_{request['diagnostic_steps']}_steps_not_a_score"
    elif item.get('oracle_layout'):
        row['score_scope']='offline_oracle_pressure_blue_only; no red score'
    write(attempt/'result.json',row)
    write(attempt/'integrity.json',receipt(attempt,plan_sha))
    write(phase_dir/'done.json',dict(attempt=attempt.name,receipt_sha256=file_hash(attempt/'integrity.json')))
    return row,attempt


def run(args):
    output,plan = create_or_resume(args)
    from .report import save_report
    with append_lock(output):
        pairs = []
        for number,item in enumerate(plan['schedule'],1):
            validate_sources(plan['source_sha256'])
            print(f"[{number}/{len(plan['schedule'])}] {item['name']} r{item['round']} {item['controller']}",flush=True)
            write(output/'progress.json',dict(active=item,completed_pairs=len(pairs),expected_pairs=len(plan['schedule'])))
            red,red_run = execute_phase(output,plan,item,'red',retry_failed=args.retry_failed)
            blue,blue_run = None,None
            if plan['configuration']['smoke'] or (phase_valid(red,'red') and red.get('placed_count',0)>0):
                blue,blue_run = execute_phase(output,plan,item,'blue',red_run,args.retry_failed)
            pairs.append(dict(**item,red=red,blue=blue,red_run=str(red_run),
                              blue_run=str(blue_run) if blue_run else None))
            save_report(output,plan,pairs)
        write(output/'progress.json',dict(active=None,completed_pairs=len(pairs),expected_pairs=len(plan['schedule']),complete=True))
    print('Completed:',output,flush=True)
    return output
