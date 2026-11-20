"""One archived-seed Map09 simulator run using the actual extracted submission."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

from .optimization.evaluation import runner


REPO = runner.REPO
HISTORY = REPO / 'output/sf13/rl17_m89_once_20261116'
OLD_PHASE = HISTORY / 'runs/Map09_r01/rl17_repair/red'
EXTRACTED = REPO / 'team742_preliminary'
ARCHIVE = REPO / 'team742_preliminary.zip'
MODULE = 'swarm_rescue.solutions.devtools.map09_original_conditions_reproduction'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def command(request):
    argv = [MODULE, '--worker', str(request)]
    bootstrap = (f'import runpy,sys; sys.path[:]={sys.path!r}; sys.prefix={sys.prefix!r}; '
                 f'sys.exec_prefix={sys.exec_prefix!r}; sys.argv={argv!r}; '
                 f'runpy.run_module({MODULE!r},run_name="__main__")')
    return [sys.executable, '-B', '-u', '-S', '-X', 'utf8', '-X', 'faulthandler', '-c', bootstrap]


def worker(request_path):
    if sys.stdin.readline().strip() != 'GO':
        raise ValueError('Resource wrapper authorization missing')
    from .optimization.evaluation.worker import run
    from .map09_extracted_mapping_worker import load_solution, loaded_identity
    from .red_stopturn_diagnostics import recording as motion
    from .red_rn7_batch import narrow_recording
    from .red_rx10_diagnostics import recording as exploration
    from .red_lines11_diagnostics import recording as inputs
    from .red_local13_diagnostics import recording as coverage
    from .red_p2l17_diagnostics import recording as matching

    request = runner.read(request_path)
    attempt = Path(request['run_dir'])
    root = next(p for p in attempt.parents if (p / 'plan.json').is_file())
    plan = runner.read(root / 'plan.json')
    runner.validate_sources(plan['source_sha256'])
    if digest(root / 'plan.json') != request['plan_sha256']:
        raise ValueError('Request plan identity mismatch')
    controller, folder, members = load_solution(dict(
        version='RL17-R', extracted_dir=str(EXTRACTED),
        expected_members=plan['extracted_members']))
    from swarm_rescue.simulation.gui_map import gui_sr
    from swarm_rescue.solutions import my_drone_eval
    if my_drone_eval.drone_class_for_mode('place') is not controller:
        raise ValueError('Extracted entry returned a different controller')
    original = gui_sr.run_control
    observed = {}

    def identified(drone):
        if type(drone) is not controller:
            raise ValueError('Unexpected instantiated controller')
        observed[str(drone.identifier)] = {
            'controller': type(drone).__module__ + '.' + type(drone).__name__,
            'mapper': type(getattr(drone, '_submission_mapper', None)).__name__,
        }
        return original(drone)

    gui_sr.run_control = identified
    try:
        with ExitStack() as stack:
            for recorder, filename in (
                    (motion, 'motion_diagnostics.json'),
                    (narrow_recording, 'narrow_diagnostics.json'),
                    (exploration, 'exploration_diagnostics.json'),
                    (inputs, 'line_diagnostics.json'),
                    (coverage, 'local13_diagnostics.json'),
                    (matching, 'p2l17_diagnostics.json')):
                stack.enter_context(recorder(attempt / filename))
            code = run(request_path)
        if set(observed) != {str(n) for n in range(5)}:
            raise ValueError('Five controller instances were not observed')
        return code
    finally:
        gui_sr.run_control = original
        write_new(attempt / 'loaded_identity.json', dict(
            observed_instances=observed, extracted_dir=str(folder),
            loaded_members=loaded_identity(folder, members),
            extracted_entry=str(folder / 'my_drone_eval.py')))
        runner.validate_sources(plan['source_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    runner.ev.CONTROLLERS = {
        **runner.ev.CONTROLLERS,
        'rl17_repair': ('my_drone_red_repair17', 'MyDroneRedRepair17'),
    }
    if args.worker:
        return worker(args.worker)
    old_plan = runner.read(HISTORY / 'plan.json')
    historical, old_attempt = runner.load_done(OLD_PHASE, digest(HISTORY / 'plan.json'))
    if not runner.phase_valid(historical, 'red'):
        raise ValueError('Invalid historical result')
    if digest(HISTORY / 'source_snapshot.zip') != old_plan['source_snapshot_sha256']:
        raise ValueError('Historical source archive changed')
    if runner.environment() != old_plan['configuration']['environment']:
        raise ValueError('Dependency environment differs from historical conditions')
    allowed_drift = {
        'src/swarm_rescue/solutions/devtools/red_p2l17_batch.py',
        'src/swarm_rescue/solutions/devtools/red_p2l17_review.py',
    }
    drift = [name for name, expected in old_plan['source_sha256'].items()
             if digest(REPO / name) != expected]
    if set(drift) - allowed_drift:
        raise ValueError(f'Historical runtime files changed: {drift}')
    with ZipFile(ARCHIVE) as archive, ZipFile(HISTORY / 'source_snapshot.zip') as frozen:
        if archive.testzip() is not None:
            raise ValueError('Submission CRC error')
        members = {name: hashlib.sha256(archive.read(name)).hexdigest()
                   for name in archive.namelist()}
        actual = {p.relative_to(EXTRACTED).as_posix(): digest(p)
                  for p in EXTRACTED.rglob('*') if p.is_file()}
        if actual != members:
            raise ValueError('Extracted folder differs from ZIP')
        for name in members:
            if name.endswith('.py') and name not in ('__init__.py', 'my_drone_eval.py'):
                source = 'src/swarm_rescue/solutions/' + name
                if hashlib.sha256(frozen.read(source)).hexdigest() != members[name]:
                    raise ValueError('Extracted controller differs from historical: ' + name)
    item, = [i for i in old_plan['schedule'] if i['map'] == 'Map09' and i['controller'] == 'rl17_repair']
    if (item['seed'], item['round']) != (2477227837, 1):
        raise ValueError('Historical round changed')
    map_rows = [row for row in old_plan['map_rows'] if row['name'] == 'Map09']
    sources = runner.source_receipt(map_rows)
    for path in (ARCHIVE, *EXTRACTED.iterdir()):
        if path.is_file():
            sources[path.relative_to(REPO).as_posix()] = digest(path)
    previous = {
        'src/swarm_rescue/solutions/my_drone_eval.py': digest(REPO / 'src/swarm_rescue/solutions/my_drone_eval.py'),
        'output/sf13/rl22_official_20261119_run/runs/Map09_r01/rl11_lines/red/attempt_001/run.log':
            digest(REPO / 'output/sf13/rl22_official_20261119_run/runs/Map09_r01/rl11_lines/red/attempt_001/run.log'),
    }
    output = Path(tempfile.mkdtemp(prefix='map09_rl17_original_', dir=REPO / 'output/sf13'))
    plan = dict(configuration=old_plan['configuration'], schedule=[item], map_rows=map_rows,
                source_sha256=sources, historical_attempt=str(old_attempt),
                historical_plan_sha256=digest(HISTORY / 'plan.json'), historical_source_drift=drift,
                extracted_members=members, zip_sha256=digest(ARCHIVE), protected_sha256=previous,
                execution='One full red-stage simulator run, extracted entry and original payload count',
                graphics='PYGLET_SHADOW_WINDOW=false to allow the current desktop OpenGL context')
    write_new(output / 'plan.json', plan)
    print(output, flush=True)
    os.environ['PYGLET_SHADOW_WINDOW'] = 'false'
    from .red_exploration22_batch import use_ascii_temp
    use_ascii_temp()
    runner.worker_command = command
    result, attempt = runner.execute_phase(output, plan, item, 'red', retry_failed=False)
    runner.validate_sources(sources)
    runner.validate_sources(previous)
    runner.load_done(OLD_PHASE, digest(HISTORY / 'plan.json'))
    issues = []
    if not runner.phase_valid(result, 'red'):
        issues.append('invalid_red_phase')
    if (result.get('placed_count'), result.get('expected_count')) != (10, 10):
        issues.append('incomplete_payload_count')
    for marker in ('Traceback (most recent call last)', 'Fatal Python error', 'AssertionError'):
        if marker in (attempt / 'run.log').read_text(encoding='utf-8', errors='replace'):
            issues.append(marker)
    write_new(output / 'completion.json', dict(attempt=str(attempt), valid=not issues,
                                               issues=issues, old_records_verified=True))
    print('COMPLETE', result.get('status'), result.get('phase1'), issues, flush=True)
    return 0 if not issues else 2


if __name__ == '__main__':
    raise SystemExit(main())
