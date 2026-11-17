"""Frozen 6 x 2 x 2 red-stage comparison. No controller/engine files changed.

Prepare is the default; full execution requires --run --prepared. Source bytes
are preserved in a JSON receipt, not a submission package or source ZIP.
"""
import argparse
import base64
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

from swarm_rescue.solutions.devtools.optimization.evaluation import runner
from swarm_rescue.solutions.devtools.optimization.evaluation.scenarios import training_rows, scenario, engine_profile
from swarm_rescue.solutions.devtools.optimization.dataset_manager.artifacts import verify
from swarm_rescue.solutions.devtools.optimization.map_generator.official_format import read_spec, check_sources

REPO = runner.REPO
DATASET = REPO / 'output/optimization/datasets/dataset_001'
MODULE = 'swarm_rescue.solutions.devtools.red_pair19_batch'
CONTROLLERS = {'rl11_lines': ('my_drone_red_lines11', 'MyDroneRedLines11'),
               'rl17_repair': ('my_drone_red_repair17', 'MyDroneRedRepair17')}
SEEDS = (2026111701, 2026111702)
SELECTION = {
    'yh_map_fs_24': '训练集中墙占比最低；88px名义通道，相对开阔混合分区。',
    'yh_map_fs_37': '37段/块墙，72px名义通道；少分区但通道较窄。',
    'yh_map_fs_27': '56px名义通道、50段/块墙；较密集窄口。',
    'yh_map_fs_72': '56px名义通道、较低墙占比；窄口与较开阔房间共存。',
    'yh_map_fs_55': '训练集最多65段/块墙、56px名义通道；密集曲折分区。',
    'yh_map_fs_90': '63段/块墙、64px名义通道；第二个密集分区样本。',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    records = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata['Name']
        record = dist.read_text('RECORD')
        records[name] = dict(version=dist.version, record_sha256=
            hashlib.sha256(record.encode()).hexdigest() if record else None)
    return dict(python=sys.version, executable=sys.executable,
                executable_sha256=digest(sys.executable), distributions=records)


def register():
    runner.ev.CONTROLLERS = {**runner.ev.CONTROLLERS, **CONTROLLERS}


def source_files(rows):
    # Broad protection deliberately fails closed if another window edits a
    # potentially imported shared/blue module. No old batch is ever re-frozen.
    hashes = runner.source_receipt(rows)
    for p in [REPO/'pyproject.toml', REPO/'doc/复赛调整说明.md', *DATASET.glob('*.json')]:
        hashes[p.relative_to(REPO).as_posix()] = digest(p)
    return hashes


def prepare(output, smoke=False):
    if output.exists():
        raise FileExistsError('New output required: '+str(output))
    info = verify(DATASET)
    if info['split_counts']['train'] != 70 or info['split_counts']['validation'] != 15 or info['split_counts']['test'] != 15:
        raise ValueError('Dataset split changed')
    names = list(SELECTION)[:1] if smoke else list(SELECTION)
    rows = training_rows(names, DATASET)
    train = {r['id'] for r in read(DATASET/'train.json')['maps']}
    groups={member:g['id'] for g in read(DATASET/'groups.json')['groups'] for member in g['members']}
    if any(r['id'] not in train for r in rows):
        raise ValueError('Only original training maps are permitted')
    schedule, selections = [], []
    for row in rows:
        path = REPO/row['map_path']
        spec = read_spec(path)
        check_sources(spec)  # Read/validate existing generated Python only.
        if (spec['rules']['timestep_limit'], spec['rules']['walltime_limit']) != (5400, 180):
            raise ValueError('Unexpected native limits')
        conditions = scenario(path)
        selections.append(dict(name=row['name'], reason=SELECTION[row['name']],
            dataset_group=groups[row['id']],
            map_sha256=digest(path), content_sha256=row['content_sha256'],
            passage_width=spec['passage_width'], wall_count=len(spec['walls']),
            wall_fraction=row['wall_fraction'], zones=spec['zones'], rules=spec['rules']))
        for number, seed in enumerate(SEEDS[:1] if smoke else SEEDS, 1):
            order = list(CONTROLLERS) if number == 1 else list(reversed(CONTROLLERS))
            for controller in order:
                schedule.append(dict(map_id=row['id'], name=row['name'], family=row['family'],
                    map_file=row['map_path'], map='R19'+row['geometry_sha256'][:12], official_map=None,
                    scenario=conditions, round=number, seed=seed, controller=controller))
    hashes = source_files(rows)
    plan = dict(schema='paired-red19-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=REPO).decode('utf-8'),
        configuration=dict(smoke=smoke, diagnostic_steps=16 if smoke else None,
            timeout=240, memory_bytes=4_000_000_000, cpu_affinity_count=1,
            simultaneous_simulations=1, engine_profile=engine_profile(),
            controllers=list(CONTROLLERS),controller_identities={c:runner.ev.controller_identity(c) for c in CONTROLLERS},
            environment=environment(), blue_evaluated=False, full_score=None),
        selection=selections, schedule=schedule, map_rows=rows, source_sha256=hashes,
        scope='training development screen, red only; no independent holdout inference',
        limitations=['All 100 maps have both static zone types; no zone-free stratum exists.',
                    'One mixed-partition family; nominal passages are not measured minimum navigable width.',
                    'Runtime seed changes noise/initial headings, not frozen wall or zone geometry.',
                    'Same seeds do not guarantee equal trajectories under wall-clock budgets.',
                    'Local CPU/commit constraints do not establish equivalence to remote platform hardware.'])
    output.mkdir(parents=True)
    # JSON holds exact source bytes without creating Python files outside solutions.
    payload={}
    for name,sha in hashes.items():
        content=(REPO/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=sha:raise ValueError('Source changed while freezing: '+name)
        payload[name]=base64.b64encode(content).decode('ascii')
    write(output/'source_bytes.json',payload)
    plan['source_bytes_sha256'] = digest(output/'source_bytes.json')
    write(output/'plan.json', plan)
    write(output/'prepared.json', dict(plan_sha256=digest(output/'plan.json'), runs=len(schedule), executed=0))
    return plan


def validate_plan(folder, plan, check_live=True):
    if digest(folder/'plan.json') != read(folder/'prepared.json')['plan_sha256']:
        raise ValueError('Prepared plan changed')
    if digest(folder/'source_bytes.json') != plan['source_bytes_sha256']:
        raise ValueError('Frozen source bytes changed')
    if check_live:
        runner.validate_sources(plan['source_sha256'])
        if environment() != plan['configuration']['environment']:
            raise ValueError('Python/dependency environment changed')


def worker(request):
    if sys.stdin.readline().strip() != 'GO':
        raise RuntimeError('Missing resource-wrapper startup authorization')
    if os.name != 'nt':
        from swarm_rescue.solutions.devtools.optimization.evaluation.resources import linux_limit
        linux_limit(4_000_000_000)
    register()
    request_doc = read(request)
    plan_path = Path(request_doc['run_dir']).parents[4]/'plan.json'
    runner.validate_sources(read(plan_path)['source_sha256'])
    from swarm_rescue.solutions.devtools.optimization.evaluation.worker import run
    from swarm_rescue.solutions.devtools.optimization.evaluation import scenarios
    from swarm_rescue.solutions.devtools.optimization.map_generator.validate import geometry_hash
    original_resolve=scenarios.resolve_map
    def short_recording_identity(settings):
        # Cosmetic evaluation/recording identity only. Generated geometry,
        # native map implementation, random seed and online controller unchanged.
        spec=read_spec(settings['map_file'])
        expected='R19'+geometry_hash(spec)[:12]
        if settings['map']!=expected:raise ValueError('Short map identity mismatch')
        cls=original_resolve(settings)
        cls.__name__=expected
        from swarm_rescue import launcher
        setattr(launcher,expected,cls)
        return cls
    scenarios.resolve_map=short_recording_identity
    from swarm_rescue.solutions.devtools.red_stopturn_diagnostics import recording as motion
    from swarm_rescue.solutions.devtools.red_rn7_batch import narrow_recording
    from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording as exploration
    from swarm_rescue.solutions.devtools.red_lines11_diagnostics import recording as inputs
    from swarm_rescue.solutions.devtools.red_local13_diagnostics import recording as coverage
    from swarm_rescue.solutions.devtools.red_p2l17_diagnostics import recording as matches
    root = Path(request_doc['run_dir'])
    instances={}
    with ExitStack() as contexts:
        for context, filename in [(motion,'motion_diagnostics.json'),(narrow_recording,'narrow_diagnostics.json'),
            (exploration,'exploration_diagnostics.json'),(inputs,'line_diagnostics.json'),
            (coverage,'local13_diagnostics.json'),(matches,'p2l17_diagnostics.json')]:
            contexts.enter_context(context(root/filename))
        from swarm_rescue.simulation.gui_map import gui_sr
        original_control=gui_sr.run_control
        def identified_control(drone):
            instances[str(drone.identifier)]=dict(controller=type(drone).__module__+'.'+type(drone).__name__,
                mapper=type(drone._submission_mapper).__module__+'.'+type(drone._submission_mapper).__name__)
            return original_control(drone)
        gui_sr.run_control=identified_control
        try:code = run(request)
        finally:gui_sr.run_control=original_control
    # Record actual loaded controller/mapper classes, not only request labels.
    cls = runner.ev.load_controller(request_doc)
    loaded = {}
    for name, mod in list(sys.modules.items()):
        file = getattr(mod, '__file__', None)
        if file:
            p = Path(file).resolve()
            if p.is_file() and p.is_relative_to(REPO/'src'):
                loaded[p.relative_to(REPO).as_posix()] = digest(p)
    write(root/'loaded_identity.json', dict(controller=cls.__module__+'.'+cls.__name__,
        mro=[c.__module__+'.'+c.__name__ for c in cls.__mro__], observed_instances=instances,loaded_sources=loaded))
    return code


def command(request):
    argv = [MODULE, '--worker', str(request)]
    code = (f'import runpy,sys; sys.path[:]={sys.path!r}; sys.prefix={sys.prefix!r}; '
            f'sys.exec_prefix={sys.exec_prefix!r}; sys.argv={argv!r}; runpy.run_module({MODULE!r},run_name="__main__")')
    return [sys.executable,'-B','-u','-S','-X','utf8','-X','faulthandler','-c',code]


def execute(folder, plan):
    runner.worker_command = command
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        buffer=ctypes.create_unicode_buffer(32768)
        if kernel.GetShortPathNameW(tempfile.gettempdir(),buffer,len(buffer)) and buffer.value.isascii():
            tempfile.tempdir=buffer.value
    if not tempfile.gettempdir().isascii() or len(tempfile.gettempdir())>115:
        raise ValueError('Need an existing ASCII TEMP path <=115 characters for native export')
    # O_EXCL marker remains as evidence. No automatic retry, deletion or reuse.
    write(folder/'started.json', dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat()))
    completed = []
    for n, item in enumerate(plan['schedule'], 1):
        validate_plan(folder, plan)
        print(f"[{n}/{len(plan['schedule'])}] {item['name']} r{item['round']} {item['controller']} seed={item['seed']}", flush=True)
        runner.write(folder/'progress.json', dict(completed=len(completed), expected=len(plan['schedule']), active=item))
        try:
            result, attempt = runner.execute_phase(folder, plan, item, 'red')
            changed = [p for p,h in plan['source_sha256'].items() if not (REPO/p).is_file() or digest(REPO/p) != h]
            loaded = read(attempt/'loaded_identity.json') if (attempt/'loaded_identity.json').exists() else {}
            row = dict(item=item, attempt=str(attempt), result=result, changed_during_run=changed,
                       actual_identity=loaded.get('controller'))
            completed.append(row)
            write(folder/f'run_{n:02d}.json', row)
            if changed:
                raise RuntimeError('Source changed during flight; row invalid, stopping batch')
        except Exception:
            write(folder/f'failure_{n:02d}.json', dict(item=item,traceback=traceback.format_exc()))
            raise
    from swarm_rescue.solutions.devtools.red_pair19_report import summarize
    report = summarize(folder)
    runner.write(folder/'progress.json', dict(completed=len(completed), expected=len(plan['schedule']), active=None,
        complete=True, valid=report['valid_count'], invalid=report['invalid_count'], smoke=plan['configuration']['smoke']))
    print('Completed raw runs:', folder, 'full-stage valid=',report['valid_count'],
          'execution checks passed=',sum(r['execution_ok'] for r in report['rows']),flush=True)
    return 0 if report['all_execution_checks_pass'] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path); p.add_argument('--prepared',type=Path)
    p.add_argument('--worker',type=Path); p.add_argument('--run',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); register()
    if args.worker:
        return worker(args.worker)
    if not args.output:
        p.error('--output required')
    folder=args.output.resolve()
    if folder.exists():
        raise FileExistsError('Output exists; all failures retained, choose a new directory')
    if args.run and not args.prepared and not args.smoke:
        p.error('Full flights require --prepared pointing at the reviewed frozen preparation')
    if args.prepared:
        source=args.prepared.resolve(); plan=read(source/'plan.json'); validate_plan(source,plan)
        if args.smoke != plan['configuration']['smoke']:
            p.error('Smoke flag differs from prepared plan')
        folder.mkdir(parents=True)
        for name in ('plan.json','prepared.json','source_bytes.json'):
            with (folder/name).open('xb') as f: f.write((source/name).read_bytes())
    else:
        plan=prepare(folder,args.smoke)
    if args.run:
        return execute(folder,plan)
    print('Prepared only; 0 flights:',folder,'planned=',len(plan['schedule']),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
