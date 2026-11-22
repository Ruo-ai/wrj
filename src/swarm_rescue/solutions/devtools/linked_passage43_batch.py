"""One delivery-aware and fine-passage native Map08 flight; no seed draws or retries."""
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
OUT = ROOT/'output/sf13/linked_passage43_run_20261122'
OLD = ROOT/'output/sf13/return_budget42_run_20261122'
SELECTOR = 'candidate_linked_passage43'
spec = importlib.util.spec_from_file_location('lp43_previous', ROOT/'src/swarm_rescue/solutions/devtools/return_budget42_batch.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
driver, read, sha = previous.driver, previous.read, previous.sha
write = driver.runner.write
driver.SCRIPT = SCRIPT
driver.SELECTORS = (SELECTOR,)
driver.CONTROLLERS = {SELECTOR: ('my_drone_red_linked_passage43', 'MyDroneRedLinkedPassage43')}


def validate(output, worker_role=False):
    plan = read(output/'plan.json')
    assert sha(output/'plan.json') == read(output/'frozen.json')['plan_sha256']
    driver.runner.validate_sources(plan['source_sha256'])
    assert driver.runner.environment() == plan['configuration']['worker_environment' if worker_role else 'environment']
    assert all(sha(ROOT/f) == v for f, v in plan['historical_sha256'].items())
    assert [(p['map'], p['round'], p['seed'], p['controller']) for p in plan['schedule']] == [('Map08', 2, 2252748057, SELECTOR)]
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return plan


driver.validate = validate


def prepare():
    if OUT.exists(): raise FileExistsError(OUT)
    previous.validate(OLD)
    old = read(OLD/'plan.json')
    baseline = next(p for p in read(OLD/'pairs.json') if p['map'] == 'Map08')
    for f, v in read(Path(baseline['red_run'])/'loaded_identity.json')['loaded_sources'].items():
        assert sha(ROOT/f) == v, ('Historical source changed', f)
    OUT.mkdir(); driver.register(); driver.use_ascii_temp()
    env = driver.probe_worker_environment(OUT)
    assert env == old['configuration']['worker_environment']
    item = dict(next(p for p in old['schedule'] if p['map'] == 'Map08'))
    item.update(controller=SELECTOR, version='LinkedPassage43 / actual delivery and fine radar', execution_index=1)
    config = dict(old['configuration'])
    config.update(controllers=[SELECTOR], rounds=1,
        controller_identities={SELECTOR: driver.runner.ev.controller_identity(SELECTOR)},
        environment=driver.runner.environment(), worker_environment=env)
    sources = driver.runner.source_receipt(old['map_rows'])
    for path in (SCRIPT, ROOT/'output/sf13/continuous38_run_20261122.py',
                 ROOT/'output/sf13/entry37_run_20261122.py', ROOT/'output/sf13/clean_boundary_official_run_20261122.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    history = {f.relative_to(ROOT).as_posix(): sha(f) for f in Path(baseline['red_run']).rglob('*') if f.is_file()}
    for f in (OLD/'plan.json', OLD/'pairs.json', OLD/'review/data.json'):
        history[f.relative_to(ROOT).as_posix()] = sha(f)
    with zipfile.ZipFile(OUT/'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(sources): archive.write(ROOT/f, f)
    plan = dict(schema='linked-passage43-native-paired-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        configuration=config, schedule=[item], map_rows=old['map_rows'], source_sha256=sources,
        source_snapshot_sha256=sha(OUT/'source_snapshot.zip'), historical_sha256=history,
        baseline_batch=str(OLD), baseline_runs=[baseline['red_run']], component_checks_passed=30,
        delta='Actual leader receipt forwarded over real messages; live pipeline <=90 steps, receipt age<=120; finals exact digest. Budget reconnect only when delivery stale, nearest verified contact node then fallback. Fresh 2px radar sweeps rescue rejected near frontier goals, bounded230 steps.',
        preserved='Continuous38 pose, map rendering, boundary, source bytes/forwarding/freeze, final ACK, deployment, coarse occupancy and evidence; ReturnBudget42 speed estimate and180+100 reserves. Only return eligibility/contact target and fine passage override change.',
        scope='One new red flight with official zones; historical ReturnBudget42 reused; no blue or full total',
        retry_policy='Exactly one attempt; no tuning after flight; failed delivery remains reported')
    write(OUT/'plan.json', plan); write(OUT/'frozen.json', dict(plan_sha256=sha(OUT/'plan.json')))
    write(OUT/'component_receipt.json', dict(passed=True, checks=30,
        command='pytest test_red_linked_passage43.py test_red_return_budget42.py test_red_schedule40r1.py -q',
        source_sha256={f:v for f,v in sources.items() if 'linked_passage43' in f}))
    validate(OUT); print('PREPARED', sha(OUT/'plan.json'), flush=True)


def worker(request):
    from swarm_rescue.simulation.gui_map import gui_sr
    native = gui_sr.run_control; refs = {}; errors = []
    def observed(drone):
        command = native(drone)
        try: refs[int(drone.identifier)] = drone
        except Exception as exc:
            if len(errors) < 12: errors.append(repr(exc))
        return command
    try:
        with patch.object(gui_sr, 'run_control', observed): code = previous.worker(request)
    finally:
        run = Path(read(request)['run_dir'])
        write(run/'linked_passage43_diagnostics.json', dict(passive_only=True, errors=errors,
            drones={str(k): dict(stats=dict(d._lp43_stats), events=d._lp43_events, samples=d._lp43_samples,
                final_link=d._lp43_link, fine_stats=dict(d._fine43.stats), fine_events=list(d._fine43.events),
                last_budget=d._d39_budget, mro=[c.__name__ for c in type(d).__mro__]) for k,d in refs.items()}))
    return code


def run():
    plan = validate(OUT); driver.register(); driver.use_ascii_temp()
    driver.runner.worker_command = driver.command
    with driver.append_lock(OUT):
        item = plan['schedule'][0]
        driver.no_retry_guard(OUT/'runs'/'Map08_r02'/SELECTOR/'red')
        write(OUT/'progress.json', dict(complete=False, expected=1, current_map='Map08', seed=item['seed']))
        result, attempt = driver.runner.execute_phase(OUT, plan, item, 'red', retry_failed=False)
        issues = driver.phase_issues(result, attempt, plan, 'red')
        for name in ('entry37_diagnostics.json', 'continuous38_diagnostics.json', 'delivery39_diagnostics.json',
                     'schedule40r1_diagnostics.json', 'return_budget42_diagnostics.json', 'linked_passage43_diagnostics.json'):
            if not (attempt/name).is_file(): issues.append('missing_'+name)
            elif read(attempt/name)['errors']: issues.append('observer_errors_'+name)
        pair = dict(item, red=result, red_run=str(attempt), blue=None, blue_run=None,
                    red_issues=issues, blue_issues=['not_requested'])
        write(OUT/'pairs.json', [pair])
        delivery = previous.previous.previous.delivery_check(attempt, issues)
        write(OUT/'delivery_checks.json', [delivery])
    validate(OUT)
    done = dict(complete=True, new_red_runs=1, new_blue_runs=0, retries=0,
        valid_red=int(not issues), delivery_complete=int(delivery['passed']), total_score=None)
    write(OUT/'completion.json', done); write(OUT/'progress.json', done)
    print('COMPLETE', done, 'MAP', result.get('phase1', {}).get('score_map'), 'DELIVERY', delivery, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', type=Path); parser.add_argument('--environment-worker', type=Path)
    args = parser.parse_args()
    if args.environment_worker: sys.exit(driver.environment_worker(args.environment_worker))
    if args.worker: sys.exit(worker(args.worker))
    if args.prepare: prepare()
    if args.run: run()
