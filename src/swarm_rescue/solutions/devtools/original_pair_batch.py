"""Fresh full competition matches for the two original submission archives."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

os.environ['PYGLET_SHADOW_WINDOW'] = 'false'

from .optimization.evaluation import runner
from .optimization.evaluation.identities import blue_identity
from .optimization.evaluation.report import flatten
from .optimization.evaluation.scenarios import official_rows, scenario, engine_profile
from .red_exploration22_archive import load_archived_controller, archive_runtime_identity, audit_archive
from .red_exploration22_batch import digest, write_new, use_ascii_temp

MODULE = 'swarm_rescue.solutions.devtools.original_pair_batch'
ARCHIVES = {
    'RL11': ('team742_preliminary（RL11）.zip', '7794b4a316a6b3bfedf22a944f597252286f47f4c0577ce21d773e644ebde99c',
             'my_drone_red_lines11', 'MyDroneRedLines11'),
    'RL17': ('team742_preliminary（RL17）.zip', 'c07d210f0f92c57aaa8b318b8887cc5bc72c1d51a3cfdf7575fc959b55642f4c',
             'my_drone_red_repair17', 'MyDroneRedRepair17'),
}


def register():
    runner.ev.CONTROLLERS = {**runner.ev.CONTROLLERS,
                             **{key: values[2:] for key, values in ARCHIVES.items()}}
    original = runner.ev.load_controller

    def load(request):
        key = request['controller_selector']
        if key not in ARCHIVES:
            return original(request)
        runner.ev.verify_controller(request)
        filename, expected, _, name = ARCHIVES[key]
        archive = runner.REPO / '复赛版本' / filename
        if digest(archive) != expected:
            raise ValueError('Original archive changed')
        cls = load_archived_controller(archive)
        if cls.__name__ != name:
            raise ValueError('Unexpected archive factory')
        return cls

    runner.ev.load_controller = load


def prepare(output, smoke):
    rows = official_rows(('Map06',) if smoke else ('Map06', 'Map08', 'Map09'))
    audits = {key: audit_archive(runner.REPO / '复赛版本' / entry[0]) for key, entry in ARCHIVES.items()}
    schedule = []
    for row in rows:
        for number, seed in enumerate((2026112001,) if smoke else (2026112001, 2026112002, 2026112003), 1):
            for key in (list(ARCHIVES) if number % 2 else list(reversed(ARCHIVES))):
                schedule.append(dict(map_id=row['id'], name=row['name'], family='official',
                    map_file=row['map_path'], map=row['name'], official_map=row['name'],
                    scenario=scenario(runner.REPO / row['map_path'], row['name']), round=number,
                    seed=seed, controller=key, blue_variant='adapted'))
    sources = runner.source_receipt(rows)
    for filename, expected, _, _ in ARCHIVES.values():
        path = runner.REPO / '复赛版本' / filename
        if digest(path) != expected:
            raise ValueError('Original ZIP hash mismatch')
        sources[path.relative_to(runner.REPO).as_posix()] = expected
    config = dict(smoke=smoke, diagnostic_steps=16 if smoke else None, timeout=300,
        memory_bytes=4_000_000_000, cpu_affinity_count=1, simultaneous_simulations=1,
        controllers=list(ARCHIVES), controller_identities={k: runner.ev.controller_identity(k) for k in ARCHIVES},
        engine_profile=engine_profile(), environment=runner.environment(), recording=True,
        competition_mode=True, red_drones=5, blue_drones=10, initial_bombs_per_red_drone=2,
        max_steps=5400, max_walltime=180, ray_backend='cpu', **blue_identity('adapted'))
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for filename in sources:
            archive.write(runner.REPO / filename, filename)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        configuration=config, archives=audits, schedule=schedule, map_rows=rows,
        source_sha256=sources, source_snapshot_sha256=digest(output / 'source_snapshot.zip'),
        scope='Local full two-phase competition, original archive controllers, fresh paired seeds',
        blue_selection='B0 adopted stable reference; no candidate has demonstrated overall superiority',
        retry_policy='Retain every attempt. Automatic resume reuses completed receipts. No score-based reruns.')
    write_new(output / 'plan.json', plan)
    write_new(output / 'frozen.json', dict(plan_sha256=digest(output / 'plan.json')))
    return plan


def worker(request_path):
    if sys.stdin.readline().strip() != 'GO':
        raise RuntimeError('Missing resource wrapper')
    request = runner.read(request_path)
    attempt = Path(request['run_dir'])
    output = next(p for p in attempt.parents if (p / 'plan.json').is_file())
    plan = runner.read(output / 'plan.json')
    if digest(output / 'plan.json') != request['plan_sha256']:
        raise ValueError('Plan hash mismatch')
    runner.validate_sources(plan['source_sha256'])
    register()
    from .optimization.evaluation.worker import run
    from swarm_rescue.simulation.gui_map import gui_sr
    original = gui_sr.run_control
    identities = {}

    def observed(drone):
        identities[str(drone.identifier)] = type(drone).__module__ + '.' + type(drone).__name__
        return original(drone)

    with ExitStack() as stack:
        stack.enter_context(patch.object(gui_sr, 'run_control', observed))
        if request['phase'] == 'red':
            from .red_lines11_diagnostics import recording as legal_inputs
            from .red_rx10_diagnostics import recording as received_maps
            stack.enter_context(legal_inputs(attempt / 'line_diagnostics.json'))
            stack.enter_context(received_maps(attempt / 'exploration_diagnostics.json'))
        code = run(request_path)
    archive = archive_runtime_identity(runner.ev.load_controller(request)) if request['phase'] == 'red' else None
    write_new(attempt / 'loaded_identity.json', dict(instances=identities, archive=archive))
    expected_count = 5 if request['phase'] == 'red' else 10
    expected_class = (archive['isolated_module'] + '.' + archive['controller_class'] if archive else
                      request['blue_controller_module'] + '.' + request['blue_controller_class'])
    if len(identities) != expected_count or set(identities.values()) != {expected_class}:
        raise ValueError('Actual controller identity mismatch')
    if archive and archive['zip_sha256'] != ARCHIVES[request['controller_selector']][1]:
        raise ValueError('Loaded archive hash mismatch')
    for filename in ('line_diagnostics.json', 'exploration_diagnostics.json') if archive else ():
        if runner.read(attempt / filename)['errors']:
            raise ValueError('Passive recorder failed: ' + filename)
    runner.validate_sources(plan['source_sha256'])
    return code


def command(request):
    bootstrap = (f'import runpy,sys; sys.path[:]={sys.path!r}; sys.prefix={sys.prefix!r}; '
        f'sys.exec_prefix={sys.exec_prefix!r}; sys.argv={[MODULE, "--worker", str(request)]!r}; '
        f'runpy.run_module({MODULE!r},run_name="__main__")')
    return [sys.executable, '-B', '-u', '-S', '-X', 'utf8', '-X', 'faulthandler', '-c', bootstrap]


def execute(output, smoke):
    register()
    use_ascii_temp()
    plan = runner.read(output / 'plan.json') if output.exists() else prepare(output, smoke)
    if digest(output / 'plan.json') != runner.read(output / 'frozen.json')['plan_sha256']:
        raise ValueError('Frozen plan changed')
    if plan['configuration']['smoke'] != smoke:
        raise ValueError('Scope mismatch')
    runner.worker_command = command
    pairs = []
    for number, item in enumerate(plan['schedule'], 1):
        runner.validate_sources(plan['source_sha256'])
        print(f"[{number}/{len(plan['schedule'])}] {item['map']} r{item['round']} {item['controller']} RED", flush=True)
        runner.write(output / 'progress.json', dict(active=item, phase='red', completed=len(pairs), expected=len(plan['schedule'])))
        red, red_run = runner.execute_phase(output, plan, item, 'red')
        blue, blue_run = None, None
        if smoke or (red.get('status') == 'ok' and red.get('placed_count', 0) > 0):
            print(f"[{number}/{len(plan['schedule'])}] BLUE map_score={(red.get('phase1') or {}).get('score_map')}", flush=True)
            runner.write(output / 'progress.json', dict(active=item, phase='blue', completed=len(pairs), expected=len(plan['schedule'])))
            blue, blue_run = runner.execute_phase(output, plan, item, 'blue', red_run)
        pair = dict(item, red=red, blue=blue, red_run=str(red_run), blue_run=str(blue_run) if blue_run else None)
        pairs.append(pair)
        runner.write(output / 'pairs.json', pairs)
        rows = [flatten(p) for p in pairs]
        runner.write(output / 'results.json', rows)
        row = rows[-1]
        print(f"DONE valid={row['valid_score']} total={row['total_score']} blue={row['blue_score']} reason={row['failure_reason']}", flush=True)
        if red.get('status') != 'ok' and number == 1:
            raise RuntimeError('First worker failed; inspect its log before continuing')
    runner.write(output / 'progress.json', dict(active=None, completed=len(pairs), expected=len(plan['schedule']), complete=True))
    print('COMPLETE ' + str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--worker', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.worker))
    execute(args.output.resolve(), args.smoke)
