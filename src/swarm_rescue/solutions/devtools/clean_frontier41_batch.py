"""One predeclared Map08 flight against the user's historical Clean screenshot."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).resolve()
OLD = ROOT/'output/sf13/clean_boundary_official_run_20261122'
OUT = ROOT/'output/sf13/clean_frontier41_run_20261122'
SELECTOR = 'candidate_clean_frontier41'
spec = importlib.util.spec_from_file_location('cf41_native', ROOT/'output/sf13/clean_boundary_official_run_20261122.py')
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
read, sha, write = driver.read, driver.digest, driver.runner.write
driver.SCRIPT = SCRIPT
driver.SELECTORS = (SELECTOR,)
driver.CONTROLLERS = {SELECTOR: ('my_drone_red_clean_frontier41', 'MyDroneRedCleanFrontier41')}


def validate(output, worker_role=False):
    plan = read(output/'plan.json')
    assert sha(output/'plan.json') == read(output/'frozen.json')['plan_sha256']
    driver.runner.validate_sources(plan['source_sha256'])
    key = 'worker_environment' if worker_role else 'environment'
    assert driver.runner.environment() == plan['configuration'][key]
    assert all(sha(ROOT/f) == v for f, v in plan['historical_sha256'].items())
    assert [(p['map'], p['round'], p['seed'], p['controller']) for p in plan['schedule']] == [('Map08', 2, 1393965797, SELECTOR)]
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return plan


driver.validate = validate


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    old = read(OLD/'plan.json')
    base = next(p for p in read(OLD/'pairs.json') if p['map'] == 'Map08'
                and p['seed'] == 1393965797 and p['controller'] == 'candidate_clean_boundary34')
    for f, value in read(Path(base['red_run'])/'loaded_identity.json')['loaded_sources'].items():
        assert sha(ROOT/f) == value, ('Historical source changed', f)
    OUT.mkdir(); driver.register(); driver.use_ascii_temp()
    environment = driver.probe_worker_environment(OUT)
    assert environment == old['configuration']['worker_environment']
    item = dict(next(p for p in old['schedule'] if p['map'] == 'Map08'
                     and p['seed'] == 1393965797 and p['controller'] == 'candidate_clean_boundary34'))
    item.update(controller=SELECTOR, arm='candidate', version='CleanFrontier41 / Clean + boundary + frontier queue', execution_index=1)
    config = dict(old['configuration'])
    config.update(controllers=[SELECTOR], rounds=1,
        controller_identities={SELECTOR: driver.runner.ev.controller_identity(SELECTOR)},
        environment=driver.runner.environment(), worker_environment=environment)
    sources = driver.runner.source_receipt(old['map_rows'])
    for path in (SCRIPT, ROOT/'output/sf13/clean_boundary_official_run_20261122.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    history = {f.relative_to(ROOT).as_posix(): sha(f)
               for f in Path(base['red_run']).rglob('*') if f.is_file()}
    for f in (OLD/'plan.json', OLD/'pairs.json'):
        history[f.relative_to(ROOT).as_posix()] = sha(f)
    with zipfile.ZipFile(OUT/'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(sources):
            archive.write(ROOT/f, f)
    plan = dict(schema='clean-frontier41-native-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        configuration=config, schedule=[item], map_rows=old['map_rows'], source_sha256=sources,
        source_snapshot_sha256=sha(OUT/'source_snapshot.zip'), historical_sha256=history,
        baseline_batch=str(OLD), baseline_runs=[base['red_run']], component_checks_passed=5,
        delta='Clean EvidenceNavigation32 replaced by FrontierQueue39; enable only with empty inventory, before original gathering; no 60% gate',
        preserved='Clean localization, mapper, boundary, navigation clearance, deployment, radio protocol, return/gathering, freeze and submission deadlines',
        scope='One new full native red flight; historical screenshot baseline reused; no blue/full total; does not isolate all differences from Continuous38/Delivery39',
        retry_policy='One attempt, no retries, no score-selected tuning')
    write(OUT/'plan.json', plan)
    write(OUT/'frozen.json', dict(plan_sha256=sha(OUT/'plan.json')))
    validate(OUT)
    print('PREPARED', sha(OUT/'plan.json'), flush=True)


def worker(request):
    from swarm_rescue.simulation.gui_map import gui_sr
    native = gui_sr.run_control; refs = {}; samples = []; errors = []
    def observed(drone):
        command = native(drone)
        try:
            key = int(drone.identifier); refs[key] = drone
            if drone._steps % 30 == 0:
                samples.append(dict(drone=key, step=drone._steps, gathering=bool(drone._gathering),
                    enabled=drone._explore.extension_enabled, inventory=drone.carried_bombs_count(),
                    route_length=len(drone._route), stats=dict(drone._cf41_stats)))
        except Exception as exc:
            if len(errors) < 12:
                errors.append(repr(exc))
        return command
    try:
        with patch.object(gui_sr, 'run_control', observed):
            code = driver.worker(request)
    finally:
        run = Path(read(request)['run_dir'])
        write(run/'clean_frontier41_diagnostics.json', dict(passive_only=True, errors=errors, samples=samples,
            drones={str(k): dict(stats=dict(d._cf41_stats), frontier_stats=dict(d._explore.stats39),
                frontier_choices=d._explore.audit39, mro=[c.__name__ for c in type(d).__mro__]) for k, d in refs.items()}))
    return code


def run():
    plan = validate(OUT); driver.register(); driver.use_ascii_temp()
    driver.runner.worker_command = driver.command
    with driver.append_lock(OUT):
        item = plan['schedule'][0]
        driver.no_retry_guard(OUT/'runs'/'Map08_r02'/SELECTOR/'red')
        write(OUT/'progress.json', dict(complete=False, current_map='Map08', expected=1))
        result, attempt = driver.runner.execute_phase(OUT, plan, item, 'red', retry_failed=False)
        issues = driver.phase_issues(result, attempt, plan, 'red')
        observer = read(attempt/'clean_frontier41_diagnostics.json')
        if observer['errors'] or len(observer['drones']) != 5:
            issues.append('frontier_observer')
        pair = dict(item, red=result, red_run=str(attempt), blue=None, blue_run=None,
                    red_issues=issues, blue_issues=['not_requested'])
        write(OUT/'pairs.json', [pair])
    validate(OUT)
    done = dict(complete=True, new_red_runs=1, new_blue_runs=0, retries=0,
                valid_red=int(not issues), total_score=None, issues=issues)
    write(OUT/'completion.json', done); write(OUT/'progress.json', done)
    print('COMPLETE', done, result.get('phase1'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', type=Path); parser.add_argument('--environment-worker', type=Path)
    args = parser.parse_args()
    if args.environment_worker:
        sys.exit(driver.environment_worker(args.environment_worker))
    if args.worker:
        sys.exit(worker(args.worker))
    if args.prepare:
        prepare()
    if args.run:
        run()
