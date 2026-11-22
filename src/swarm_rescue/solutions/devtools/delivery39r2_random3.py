"""Unchanged Delivery39R2: three pre-drawn seeds on native Map08 and Map09."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import secrets
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / 'output/sf13/delivery39r2_random3_20261122'
OLD = ROOT / 'output/sf13/delivery39r2_run_20261122'
CATALOG = ROOT / 'output/sf13/clean_boundary_official_run_20261122/plan.json'
spec = importlib.util.spec_from_file_location('delivery39r2_frozen_worker', ROOT / 'src/swarm_rescue/solutions/devtools/delivery39r2_batch.py')
single = importlib.util.module_from_spec(spec); spec.loader.exec_module(single)
driver = single.driver
read, write, sha = single.read, single.write, single.sha
SELECTOR = single.SELECTOR
driver.SCRIPT = SCRIPT


def schedule_for(rows, seeds):
    items = []
    for number, seed in enumerate(seeds, 1):
        for row in rows:
            name = row['name']
            items.append(dict(map_id=row['id'], name=name, family=row['family'],
                map_file=row['map_path'], map=name, official_map=name,
                scenario=driver.NATIVE_SCENARIO(ROOT / row['map_path'], name),
                round=number, seed=seed, controller=SELECTOR, arm='candidate',
                version='Delivery39R2 unchanged / random-three validation', blue_variant='adapted',
                pair_id=f'{name}_r{number:02d}_s{seed}', execution_index=len(items)+1))
    return items


def validate(output, worker_role=False):
    plan = read(output / 'plan.json')
    assert sha(output / 'plan.json') == read(output / 'frozen.json')['plan_sha256']
    driver.runner.validate_sources(plan['source_sha256'])
    expected = plan['configuration']['worker_environment' if worker_role else 'environment']
    assert driver.runner.environment() == expected
    assert all(sha(ROOT / f) == value for f, value in plan['historical_sha256'].items())
    assert sha(output / 'seed_draw.json') == plan['seed_draw_sha256']
    draw = read(output / 'seed_draw.json'); seeds = draw['accepted_seeds']
    assert len(seeds) == len(set(seeds)) == 3
    assert not set(seeds).intersection(draw['excluded_previous_seeds'])
    assert [row['name'] for row in plan['map_rows']] == ['Map08', 'Map09']
    assert plan['schedule'] == schedule_for(plan['map_rows'], seeds)
    assert len(plan['schedule']) == 6 and driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return plan


driver.validate = validate


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    single.validate(OLD)
    old = read(OLD / 'plan.json'); pairs = read(OLD / 'pairs.json')
    for pair in pairs:
        for name, value in read(Path(pair['red_run']) / 'loaded_identity.json')['loaded_sources'].items():
            assert sha(ROOT / name) == value, ('Candidate or engine changed', name)
    previous = set(); previous_plans = {}
    for path in (ROOT / 'output/sf13').rglob('plan.json'):
        value = read(path)
        previous.update(x['seed'] for x in value.get('schedule', []) if isinstance(x.get('seed'), int))
        previous.update(x for x in value.get('seeds', []) if isinstance(x, int))
        previous_plans[path.relative_to(ROOT).as_posix()] = sha(path)
    OUT.mkdir()
    seeds = []; draws = []
    while len(seeds) < 3:
        value = secrets.randbits(32)
        accepted = value not in previous and value not in seeds
        draws.append(dict(value=value, accepted=accepted))
        if accepted:
            seeds.append(value)
    write(OUT / 'seed_draw.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        method='secrets.randbits(32), operating-system randomness; independent of map/hash/date',
        accepted_seeds=seeds, draws=draws, excluded_previous_seeds=sorted(previous),
        inspected_previous_plans=previous_plans,
        policy='Reject only previously scheduled seeds and duplicate draws; no performance filtering; same three seeds on both maps'))
    rows = sorted((r for r in read(CATALOG)['map_rows'] if r['name'] in ('Map08', 'Map09')), key=lambda r:r['name'])
    driver.register(); driver.use_ascii_temp()
    env = driver.probe_worker_environment(OUT)
    assert env == old['configuration']['worker_environment']
    config = dict(old['configuration']); config.update(rounds=3,
        environment=driver.runner.environment(), worker_environment=env)
    sources = driver.runner.source_receipt(rows)
    for path in (SCRIPT, ROOT / 'src/swarm_rescue/solutions/devtools/delivery39r2_batch.py',
                 ROOT / 'output/sf13/continuous38_run_20261122.py',
                 ROOT / 'output/sf13/entry37_run_20261122.py',
                 ROOT / 'output/sf13/clean_boundary_official_run_20261122.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    history = {p.relative_to(ROOT).as_posix(): sha(p)
               for pair in pairs for p in Path(pair['red_run']).rglob('*') if p.is_file()}
    for path in (OLD / 'plan.json', OLD / 'pairs.json', OLD / 'completion.json', OLD / 'source_snapshot.zip', CATALOG):
        history[path.relative_to(ROOT).as_posix()] = sha(path)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(ROOT / name, name)
    plan = dict(schema='delivery39r2-two-maps-three-random-seeds-native-v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration=config, schedule=schedule_for(rows,seeds), map_rows=rows,
        source_sha256=sources, historical_sha256=history,
        source_snapshot_sha256=sha(OUT / 'source_snapshot.zip'), seed_draw_sha256=sha(OUT / 'seed_draw.json'),
        candidate_reference_plan_sha256=sha(OLD / 'plan.json'),
        scope='Six fresh complete red phases; Map08/09 native special zones; unchanged Delivery39R2; no blue/full total',
        delta='Only evaluation map, new independently pre-drawn seed and round labels; contestant, engine, driver and passive observers unchanged',
        seed_policy='Three seeds shared across maps, fixed before all flights; no retries, replacement seeds or mid-batch tuning',
        historical_score_policy='Prior diagnostic successes and failures excluded from new random sample statistics')
    write(OUT / 'plan.json', plan)
    write(OUT / 'frozen.json', dict(plan_sha256=sha(OUT / 'plan.json')))
    validate(OUT)
    print('PREPARED', seeds, sha(OUT / 'plan.json'), flush=True)


def delivery_check(attempt, issues):
    import numpy as np
    diag_path = attempt / 'dualmap32_diagnostics.json'
    if not diag_path.is_file():
        return dict(passed=False, verified_final_sources=[], acknowledged_sources=[], issues=issues)
    diag = read(diag_path); receipt = diag['drones'].get('0',{}).get('_s28_submission') or {}
    verified = []
    with np.load(attempt / 'dualmap32_snapshots.npz', allow_pickle=False) as data:
        for i in range(5):
            key = str(i)
            try:
                own = diag['snapshot_payloads'][key][key]; got = diag['snapshot_payloads']['0'][key]
                ok = (own['own_sealed'] and receipt['received_versions'][key] == own['sequence']
                      and receipt['received_digests'][key] == own['payload_sha256']
                      and got['payload_sha256'] == own['payload_sha256']
                      and data[got['array']].tobytes() == data[own['array']].tobytes())
                if ok: verified.append(i)
            except (KeyError, TypeError):
                pass
    acknowledgements = receipt.get('acknowledged_sources', [])
    return dict(passed=not issues and len(verified)==5 and bool(receipt.get('final_complete')) and len(acknowledgements)==5,
        verified_final_sources=verified, acknowledged_sources=acknowledgements,
        actual_received_versions=receipt.get('received_versions', {}), issues=issues)


def run():
    plan = validate(OUT); driver.register(); driver.use_ascii_temp()
    driver.runner.worker_command = driver.command
    pairs = []; delivery = []
    with driver.append_lock(OUT):
        for item in plan['schedule']:
            validate(OUT)
            phase = OUT / 'runs' / f"{item['map']}_r{item['round']:02d}" / SELECTOR / 'red'
            driver.no_retry_guard(phase)
            write(OUT / 'progress.json', dict(complete=False, completed=len(pairs), expected=6,
                current_map=item['map'], current_seed=item['seed']))
            print('START',item['execution_index'],'/6',item['map'],item['seed'],flush=True)
            result, attempt = driver.runner.execute_phase(OUT,plan,item,'red',retry_failed=False)
            issues = driver.phase_issues(result,attempt,plan,'red')
            for name in ('entry37_diagnostics.json','continuous38_diagnostics.json','delivery39_diagnostics.json'):
                if not (attempt/name).is_file(): issues.append('missing_'+name)
                elif read(attempt/name)['errors']: issues.append('observer_errors_'+name)
            pair = dict(item,red=result,red_run=str(attempt),blue=None,blue_run=None,
                red_issues=issues,blue_issues=['not_requested'])
            pairs.append(pair); write(OUT/'pairs.json',pairs)
            check = delivery_check(attempt,issues)
            delivery.append(dict(map=item['map'],round=item['round'],seed=item['seed'],**check))
            write(OUT/'delivery_checks.json',delivery)
            print('FINISHED',item['execution_index'],'/6',item['map'],item['seed'],
                'map',result.get('phase1',{}).get('score_map'),'delivery',check,flush=True)
    validate(OUT)
    done = dict(complete=True,new_red_runs=len(pairs),new_blue_runs=0,retries=0,
        valid_red=sum(not p['red_issues'] for p in pairs),delivery_complete=sum(x['passed'] for x in delivery),
        total_score=None,completed=len(pairs),expected=6)
    write(OUT/'completion.json',done); write(OUT/'progress.json',done)
    print('COMPLETE',done,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--worker',type=Path);parser.add_argument('--environment-worker',type=Path)
    args=parser.parse_args()
    if args.environment_worker:sys.exit(driver.environment_worker(args.environment_worker))
    if args.worker:sys.exit(single.worker(args.worker))
    if args.prepare:prepare()
    if args.run:run()
