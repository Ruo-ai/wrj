"""Frozen official Map06/08/09 comparison; default is preparation, never flight.

18 paired red/blue matches = 18 red + at most 18 blue phases. Blue always uses
the actual layout from its paired red phase, except explicitly labelled smoke.
The archived arm executes the verified ZIP through an isolated import loader.
"""
import argparse
import base64
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

from swarm_rescue.solutions.devtools.optimization.evaluation import runner
from swarm_rescue.solutions.devtools.optimization.evaluation.scenarios import (
    official_rows, scenario, engine_profile,
)
from swarm_rescue.solutions.devtools.optimization.evaluation.identities import blue_identity
from swarm_rescue.solutions.devtools.red_pair19_batch import environment

REPO = runner.REPO
MODULE = 'swarm_rescue.solutions.devtools.red_exploration22_batch'
MAPS = ('Map06', 'Map08', 'Map09')
SEEDS = (2026111901, 2026111902)
CONTROLLERS = {
    'candidate_rl22': ('my_drone_red_exploration22', 'MyDroneRedExploration22'),
    'rl11_lines': ('my_drone_red_lines11', 'MyDroneRedLines11'),
    'archived_rl17r': ('my_drone_red_repair17', 'MyDroneRedRepair17'),
}
ARCHIVE = REPO / 'team742_preliminary.zip'
EXPECTED_ARCHIVE_SHA256 = 'c07d210f0f92c57aaa8b318b8887cc5bc72c1d51a3cfdf7575fc959b55642f4c'
read = runner.read


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def register(archive=ARCHIVE):
    """Process-local registry; never changes the shared evaluator or entry file."""
    from swarm_rescue.solutions.devtools.red_exploration22_archive import (
        load_archived_controller,
    )
    archive = Path(archive).resolve()
    runner.ev.CONTROLLERS = {**runner.ev.CONTROLLERS, **CONTROLLERS}
    original = getattr(runner.ev.load_controller, '_rl22_original', runner.ev.load_controller)

    def load(request):
        runner.ev.verify_controller(request)
        if request['controller_selector'] == 'archived_rl17r':
            if digest(archive) != EXPECTED_ARCHIVE_SHA256:
                raise ValueError('Previously packaged ZIP identity changed')
            return load_archived_controller(archive)
        return original(request)

    load._rl22_original = original
    runner.ev.load_controller = load


def schedule_for(rows, smoke=False):
    result = []
    for row in rows:
        if row['official_map'] not in MAPS or row['name'] != row['official_map']:
            raise ValueError('Only the three exact official map identities are permitted')
        conditions = scenario(REPO / row['map_path'], row['official_map'])
        for number, seed in enumerate(SEEDS[:1] if smoke else SEEDS, 1):
            order = list(CONTROLLERS)
            if number == 2:
                order.reverse()
            for controller in order:
                result.append(dict(map_id=row['id'], name=row['name'], family='official',
                    map_file=row['map_path'], map=row['name'], official_map=row['name'],
                    scenario=conditions, round=number, seed=seed, controller=controller,
                    blue_variant='adapted'))
    return result


def validate_schedule(plan):
    smoke = plan['configuration']['smoke']
    wanted_maps = MAPS[:1] if smoke else MAPS
    if tuple(r['name'] for r in plan['map_rows']) != wanted_maps:
        raise ValueError('Frozen official map set changed')
    if plan['schedule'] != schedule_for(plan['map_rows'], smoke):
        raise ValueError('Frozen paired schedule changed')
    if len(plan['schedule']) != (3 if smoke else 18):
        raise ValueError('Unexpected match count')


def prepare(output, smoke=False):
    from swarm_rescue.solutions.devtools.red_exploration22_archive import audit_archive
    if output.exists():
        raise FileExistsError('A new output directory is required')
    if digest(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError('Old package no longer matches the audited ZIP; do not replace it')
    register()
    archive_audit = audit_archive(ARCHIVE)
    rows = official_rows(MAPS[:1] if smoke else MAPS)
    sources = runner.source_receipt(rows)
    for path in (ARCHIVE, REPO / 'pyproject.toml', REPO / 'req.txt'):
        sources[path.relative_to(REPO).as_posix()] = digest(path)
    identities = {key: runner.ev.controller_identity(key) for key in CONTROLLERS}
    # Assert the archived arm really differs from RL11's controller bytes.
    archived_members = archive_audit['red_dependency_sha256']
    current_rl11 = digest(REPO / 'src/swarm_rescue/solutions/my_drone_red_lines11.py')
    if archive_audit['controller_class'] == 'MyDroneRedLines11':
        raise ValueError('Archive now resolves RL11: review dependency/config equivalence and deduplicate before preparing')
    plan = dict(schema='official-red-exploration22-v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=REPO).decode('utf-8'),
        configuration=dict(smoke=smoke, diagnostic_steps=16 if smoke else None,
            timeout=240, memory_bytes=4_000_000_000, cpu_affinity_count=1,
            simultaneous_simulations=1, controllers=list(CONTROLLERS),
            controller_identities=identities, engine_profile=engine_profile(),
            environment=environment(), recording=True, competition_mode=True,
            **blue_identity('adapted'), blue_seed_policy='same explicit seed as paired red',
            blue_layout_policy='exact paired red exported layout; smoke uses labelled one-bomb fixture',
            red_drones=5, blue_drones=10, initial_bombs_per_red_drone=2,
            max_steps=5400, max_walltime=180, ray_backend='cpu',
            archive_path=ARCHIVE.relative_to(REPO).as_posix(), archive_sha256=digest(ARCHIVE)),
        archive_audit=archive_audit,
        provenance=dict(candidate='independent minimal bug fix; not yet adopted',
            rl11='workspace retained RL11', archived='actual previous ZIP, private namespace',
            archive_same_as_rl11=False, rl11_controller_sha256=current_rl11,
            archived_dependency_count=len(archived_members)),
        map_rows=rows, schedule=schedule_for(rows, smoke), source_sha256=sources,
        expected_matches=3 if smoke else 18, expected_red_phases=3 if smoke else 18,
        maximum_blue_phases=3 if smoke else 18,
        diagnostic_policy='Identical passive motion, local13 and RX10 GUI observers for all three red arms; no planner patch',
        limitations=['Same seeds do not remove wall-clock variation.',
            'Passive observer cost is inside the official walltime budget for each arm.',
            'local13 physical visibility is 8-tick / 4px sampled approximation.',
            'RN7 task events retain at most 160 entries; completed means local waypoint exhaustion, not verified crossing.',
            'Local adapted-blue official formula scores are not platform blind-test scores.',
            'No frontier strategy change is combined with the bug fix.'])
    validate_schedule(plan)
    output.mkdir(parents=True, exist_ok=False)
    payload = {}
    for name, sha in sources.items():
        content = (REPO / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError('Source changed during preparation: ' + name)
        payload[name] = base64.b64encode(content).decode('ascii')
    write_new(output / 'source_bytes.json', payload)
    plan['source_bytes_sha256'] = digest(output / 'source_bytes.json')
    write_new(output / 'plan.json', plan)
    write_new(output / 'prepared.json', dict(plan_sha256=digest(output / 'plan.json'),
        matches=len(plan['schedule']), red_phases=len(plan['schedule']),
        maximum_blue_phases=len(plan['schedule']), executed=0))
    return plan


def validate_plan(folder, plan, live=True):
    if digest(folder / 'plan.json') != read(folder / 'prepared.json')['plan_sha256']:
        raise ValueError('Prepared plan hash changed')
    if digest(folder / 'source_bytes.json') != plan['source_bytes_sha256']:
        raise ValueError('Frozen source byte archive changed')
    validate_schedule(plan)
    if live:
        runner.validate_sources(plan['source_sha256'])
        if environment() != plan['configuration']['environment']:
            raise ValueError('Python or dependency environment changed')


def worker(request_path):
    if sys.stdin.readline().strip() != 'GO':
        raise RuntimeError('Resource wrapper did not authorize worker startup')
    if os.name != 'nt':
        from swarm_rescue.solutions.devtools.optimization.evaluation.resources import linux_limit
        linux_limit(4_000_000_000)
    request = read(request_path)
    root = Path(request['run_dir'])
    folder = next(p for p in root.parents if (p / 'plan.json').is_file())
    plan = read(folder / 'plan.json')
    validate_plan(folder, plan)
    register(REPO / plan['configuration']['archive_path'])
    from swarm_rescue.solutions.devtools.optimization.evaluation.worker import run
    from swarm_rescue.solutions.devtools.red_exploration22_archive import archive_runtime_identity
    from swarm_rescue.simulation.gui_map import gui_sr
    instances = {}
    references = {}
    old_control = gui_sr.run_control

    def identified_control(drone):
        key = str(drone.identifier)
        references[key] = drone
        mapper = getattr(drone, '_submission_mapper', None)
        instances[key] = dict(controller=type(drone).__module__ + '.' + type(drone).__name__,
            mapper=type(mapper).__module__ + '.' + type(mapper).__name__ if mapper is not None else None)
        return old_control(drone)

    gui_sr.run_control = identified_control
    try:
        with ExitStack() as contexts:
            if request['phase'] == 'red':
                from swarm_rescue.solutions.devtools.red_stopturn_diagnostics import recording as motion
                from swarm_rescue.solutions.devtools.red_local13_diagnostics import recording as local
                from swarm_rescue.solutions.devtools.red_rx10_diagnostics import recording as exploration
                contexts.enter_context(motion(root / 'motion_diagnostics.json'))
                contexts.enter_context(local(root / 'local13_diagnostics.json'))
                contexts.enter_context(exploration(root / 'exploration_diagnostics.json'))
            return run(request_path)
    finally:
        gui_sr.run_control = old_control
        loaded = {}
        for mod in list(sys.modules.values()):
            filename = getattr(mod, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                if path.is_file() and path.is_relative_to(REPO / 'src'):
                    loaded[path.relative_to(REPO).as_posix()] = digest(path)
        archive_runtime = None
        if request['phase'] == 'red' and request['controller_selector'] == 'archived_rl17r':
            archive_runtime = archive_runtime_identity(runner.ev.load_controller(request))
        write_new(root / 'loaded_identity.json', dict(phase=request['phase'],
            observed_instances=instances, loaded_sources=loaded, archive_runtime=archive_runtime))
        if request['phase'] == 'red':
            tasks = {}
            for key, drone in references.items():
                rn = getattr(drone, '_rn7', None)
                events = list(getattr(rn, 'events', []))
                tasks[key] = dict(stats=dict(getattr(rn, 'stats', {})), events=events,
                    event_retention_limit=160, possibly_truncated=len(events) >= 160,
                    episodes=getattr(rn, 'episodes', None),
                    lifetime_episodes=getattr(rn, 'lifetime_episodes', None))
            write_new(root / 'task_events.json', dict(drones=tasks,
                interpretation='Local RN7 events; completed does not establish physical crossing. No truth is fed into policy.'))
        runner.validate_sources(plan['source_sha256'])


def command(request):
    argv = [MODULE, '--worker', str(request)]
    bootstrap = (f'import runpy,sys; sys.path[:]={sys.path!r}; sys.prefix={sys.prefix!r}; '
        f'sys.exec_prefix={sys.exec_prefix!r}; sys.argv={argv!r}; '
        f'runpy.run_module({MODULE!r},run_name="__main__")')
    return [sys.executable, '-B', '-u', '-S', '-X', 'utf8', '-X', 'faulthandler', '-c', bootstrap]


def use_ascii_temp():
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        buffer = ctypes.create_unicode_buffer(32768)
        if kernel.GetShortPathNameW(tempfile.gettempdir(), buffer, len(buffer)) and buffer.value.isascii():
            tempfile.tempdir = buffer.value
    if not tempfile.gettempdir().isascii() or len(tempfile.gettempdir()) > 115:
        raise ValueError('An existing ASCII TEMP path <=115 characters is required by native export')


def phase_issues(row, attempt, phase, smoke):
    issues = []
    if row.get('status') != 'ok': issues.append(str(row.get('status', 'missing_status')))
    if row.get('exit_code') != 0: issues.append('exit_code')
    if row.get('walltime_limited'): issues.append('walltime_limit')
    if row.get('controller_crashed'): issues.append('controller_crash')
    if row.get('engine_callback_errors'): issues.append('engine_callback_error')
    log_file = attempt / 'run.log'
    if log_file.is_file():
        log = log_file.read_text(encoding='utf-8', errors='replace')
        for marker in ('Traceback (most recent call last)', 'AssertionError', 'Fatal Python error',
                       'Exception ignored from cffi callback'):
            if marker in log: issues.append('log_exception:' + marker)
    else:
        issues.append('missing_run_log')
    identity_file = attempt / 'loaded_identity.json'
    if not identity_file.is_file() or not read(identity_file).get('observed_instances'):
        issues.append('missing_actual_loaded_identity')
    else:
        issues.extend(loaded_issues(row, attempt, phase, smoke))
    if phase == 'red':
        for name in ('local13_diagnostics', 'exploration_diagnostics'):
            diag_file = attempt / (name + '.json')
            if not diag_file.is_file(): issues.append('missing_' + name)
            elif read(diag_file).get('errors'): issues.append('diagnostic_invalid:' + name)
        if not (attempt / 'motion_diagnostics.json').is_file(): issues.append('missing_motion_diagnostics')
        if not smoke and (row.get('placed_count') != 10 or row.get('expected_count') != 10):
            issues.append('incomplete_simulated_payload_placement')
    if not smoke and not runner.phase_valid(row, phase): issues.append('phase_invalid')
    return list(dict.fromkeys(issues))


def loaded_issues(row, attempt, phase, smoke):
    """Check actual instances, byte origins and resource receipts, not labels alone."""
    issues = []
    try:
        request = read(attempt / 'request.json')
        folder = next(p for p in attempt.parents if (p / 'plan.json').is_file())
        plan = read(folder / 'plan.json')
        identity = read(attempt / 'loaded_identity.json')
        config = plan['configuration']
        resources = read(attempt / 'resources.json')
        if resources.get('cpu_count') != 1 or resources.get('memory_limit_bytes') != 4_000_000_000:
            issues.append('resource_envelope_mismatch')
        if resources.get('status') != 'exited': issues.append('resource_wrapper_not_exited')
        if request['phase'] != phase or bool(request['smoke']) != smoke:
            issues.append('request_phase_or_scope_mismatch')
        for key in ('map', 'round', 'seed', 'controller_selector', 'controller_module', 'controller_class'):
            if row.get(key) != request.get(key): issues.append('result_identity_mismatch:' + key)
        expected_count = 5 if phase == 'red' else 10
        instances = identity.get('observed_instances', {})
        if set(instances) != {str(i) for i in range(expected_count)}:
            issues.append('observed_drone_set_mismatch')
        if phase == 'blue':
            expected_controller = config['blue_controller_module'] + '.' + config['blue_controller_class']
            expected_mapper = None
        elif request['controller_selector'] == 'archived_rl17r':
            namespace = '_swarm_red_archive_' + config['archive_sha256']
            expected_controller = namespace + '.my_drone_red_repair17.MyDroneRedRepair17'
            expected_mapper = namespace + '.my_drone_red_repair17.RepairMapper17'
            archive = identity.get('archive_runtime') or {}
            if archive.get('zip_sha256') != config['archive_sha256'] or archive.get('workspace_solution_fallback_allowed') is not False:
                issues.append('archive_runtime_identity_mismatch')
            if archive.get('controller_class') != 'MyDroneRedRepair17' or archive.get('isolated_module') != namespace + '.my_drone_red_repair17':
                issues.append('archive_loaded_class_mismatch')
            loaded_members = archive.get('loaded_member_sha256', {})
            for member, sha in plan['archive_audit']['red_dependency_sha256'].items():
                if loaded_members.get(member) != sha: issues.append('archive_dependency_mismatch:' + member)
            for member, sha in loaded_members.items():
                if plan['archive_audit']['entries'].get(member, {}).get('sha256') != sha:
                    issues.append('unfrozen_archive_member:' + member)
        else:
            expected_controller = request['controller_module'] + '.' + request['controller_class']
            expected_mapper = ('swarm_rescue.solutions.my_drone_red_lines11.WallLineMapper'
                if request['controller_selector'] == 'rl11_lines'
                else 'swarm_rescue.solutions.my_drone_red_repair17.RepairMapper17')
        for ident, actual in instances.items():
            if actual.get('controller') != expected_controller:
                issues.append('actual_controller_mismatch:' + ident)
            if actual.get('mapper') != expected_mapper:
                issues.append('actual_mapper_mismatch:' + ident)
        loaded_sources = identity.get('loaded_sources', {})
        if not loaded_sources: issues.append('missing_loaded_source_hashes')
        for name, sha in loaded_sources.items():
            if plan['source_sha256'].get(name) != sha:
                issues.append('loaded_source_not_frozen:' + name)
        conditions = row.get('actual_conditions') or {}
        if (conditions.get('drones') != expected_count or conditions.get('zones') != request['zones']
                or conditions.get('instantiated_zone_types') != request['zones']
                or conditions.get('max_steps') != (config['diagnostic_steps'] if smoke else 5400)
                or conditions.get('max_walltime') != (20 if smoke else 180)):
            issues.append('actual_map_conditions_mismatch')
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        issues.append('actual_identity_audit_failed:' + repr(error))
    return issues


def summarize_pairs(pairs, smoke):
    from swarm_rescue.solutions.devtools.optimization.evaluation.report import flatten
    rows = []
    for pair in pairs:
        row = flatten(pair)
        row['red_issues'] = pair['red_issues']
        row['blue_issues'] = pair['blue_issues']
        row['execution_ok'] = not (pair['red_issues'] or pair['blue_issues']) and pair['blue'] is not None
        row['valid_score'] = bool(row['valid_score'] and row['execution_ok'] and not smoke)
        row['red_attempt'] = pair['red_run']
        row['blue_attempt'] = pair['blue_run']
        row['red_valid'] = bool(not smoke and runner.phase_valid(pair['red'], 'red') and not pair['red_issues'])
        row['blue_valid'] = bool(not smoke and pair['blue'] is not None and runner.phase_valid(pair['blue'], 'blue') and not pair['blue_issues'])
        row['full_valid'] = row['valid_score']
        row['reasons'] = ['red:' + r for r in pair['red_issues']] + ['blue:' + r for r in pair['blue_issues']]
        if not row['valid_score']: row['total_score'] = None
        rows.append(row)
    return dict(rows=rows, completed_matches=len(rows), smoke=smoke,
        valid_complete_matches=sum(r['valid_score'] for r in rows),
        execution_ok_matches=sum(r['execution_ok'] for r in rows),
        failures=[r for r in rows if not r['execution_ok']],
        interpretation='Every map/seed/round is retained; no aggregate average is used for adoption.')


def execute(folder, plan):
    validate_plan(folder, plan)
    register(REPO / plan['configuration']['archive_path'])
    use_ascii_temp()
    runner.worker_command = command
    write_new(folder / 'started.json', dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat()))
    pairs = []
    smoke = plan['configuration']['smoke']
    for number, item in enumerate(plan['schedule'], 1):
        validate_plan(folder, plan)
        print(f"[{number}/{len(plan['schedule'])} matches] {item['name']} r{item['round']} "
              f"{item['controller']} red/blue seed={item['seed']}", flush=True)
        runner.write(folder / 'progress.json', dict(active=item, completed_matches=len(pairs),
            expected_matches=len(plan['schedule']), smoke=smoke))
        try:
            red, red_run = runner.execute_phase(folder, plan, item, 'red', retry_failed=False)
            validate_plan(folder, plan)
            red_issues = phase_issues(red, red_run, 'red', smoke)
            blue, blue_run = None, None
            blue_issues = ['skipped_invalid_red_or_no_layout']
            if smoke or (runner.phase_valid(red, 'red') and red.get('placed_count', 0) > 0):
                blue, blue_run = runner.execute_phase(folder, plan, item, 'blue', red_run, retry_failed=False)
                validate_plan(folder, plan)
                blue_issues = phase_issues(blue, blue_run, 'blue', smoke)
            pair = dict(**item, red=red, blue=blue, red_run=str(red_run),
                blue_run=str(blue_run) if blue_run else None, red_issues=red_issues, blue_issues=blue_issues)
            pairs.append(pair)
            write_new(folder / f'match_{number:02d}.json', pair)
            runner.write(folder / 'paired_summary.json', summarize_pairs(pairs, smoke))
        except Exception:
            write_new(folder / f'failure_{number:02d}.json', dict(item=item, traceback=traceback.format_exc()))
            raise
    validate_plan(folder, plan)
    summary = summarize_pairs(pairs, smoke)
    runner.write(folder / 'paired_summary.json', summary)
    runner.write(folder / 'progress.json', dict(active=None, completed_matches=len(pairs),
        expected_matches=len(plan['schedule']), complete=True, smoke=smoke,
        red_phases=len(pairs), blue_phases=sum(p['blue'] is not None for p in pairs)))
    print('Completed:', folder, 'matches=', len(pairs), 'red=', len(pairs),
          'blue=', sum(p['blue'] is not None for p in pairs),
          'valid formal matches=', summary['valid_complete_matches'], flush=True)
    return 0 if summary['execution_ok_matches'] == len(pairs) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if not args.output:
        parser.error('--output is required')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('New output required; previous failures are retained, never retried automatically')
    if args.run and not args.prepared and not args.smoke:
        parser.error('Formal execution requires --prepared pointing to the reviewed frozen plan')
    if args.prepared:
        source = args.prepared.resolve()
        plan = read(source / 'plan.json')
        validate_plan(source, plan)
        if args.smoke != plan['configuration']['smoke']:
            parser.error('--smoke must match the prepared plan')
        output.mkdir(parents=True, exist_ok=False)
        for name in ('plan.json', 'prepared.json', 'source_bytes.json'):
            with (output / name).open('xb') as stream:
                stream.write((source / name).read_bytes())
    else:
        plan = prepare(output, args.smoke)
    if args.run:
        return execute(output, plan)
    print('PREPARED ONLY:', output, 'executed=0;', len(plan['schedule']),
          'matches =', len(plan['schedule']), 'red + at most', len(plan['schedule']), 'blue phases', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
