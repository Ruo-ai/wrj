"""Six fresh full red/blue evaluations of the released Startup44 Delivery R2."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import secrets
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / 'output/sf13/startup44_delivery_r2_m89_random3_20261123_152910_169'
OLD = ROOT / 'output/sf13/startup44_deep_bomb_random3_20261123'
RELEASE = ROOT / 'output/releases/startup44_delivery_r2_20261123'
PACKAGE = ROOT / 'team742_preliminary.zip'
PACKAGE_SHA = '2c0314cccf30095c7f6f640276b1f704f2c4b3d37b89bf2f1c9c926441446427'
MAPS = ('Map08', 'Map09')
SELECTOR = 'released_startup44_delivery_r2'
spec = importlib.util.spec_from_file_location('released_r2_observers',
    ROOT / 'src/swarm_rescue/solutions/devtools/startup44_delivery_r2_batch.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
driver, read, sha, write = previous.driver, previous.read, previous.sha, previous.write
driver.SCRIPT = SCRIPT
driver.SELECTORS = (SELECTOR,)
driver.CONTROLLERS = {SELECTOR: ('my_drone_red_startup44_delivery', 'MyDroneRedStartup44Delivery')}
driver.ZONE_CLASSES = {**driver.ZONE_CLASSES, 'Map06': ['NoComZone']}


def schedule_for(rows, seeds):
    result = []
    for number in range(1, 4):
        for row in rows:
            name = row['name']; seed = seeds[name][number - 1]
            result.append(dict(map_id=row['id'], name=name, family=row['family'],
                map_file=row['map_path'], map=name, official_map=name,
                scenario=driver.NATIVE_SCENARIO(ROOT / row['map_path'], name),
                round=number, seed=seed, controller=SELECTOR, arm='released',
                version='Startup44 Delivery R2 released ZIP', blue_variant='adapted',
                pair_id=f'{name}_r{number:02d}_s{seed}', execution_index=len(result) + 1))
    return result


def validate(output, worker_role=False):
    p = read(output / 'plan.json')
    assert sha(output / 'plan.json') == read(output / 'frozen.json')['plan_sha256']
    driver.runner.validate_sources(p['source_sha256'])
    assert driver.runner.environment() == p['configuration']['worker_environment' if worker_role else 'environment']
    assert sha(PACKAGE) == PACKAGE_SHA == p['submission_sha256']
    assert all(sha(ROOT / f) == v for f, v in p['historical_sha256'].items())
    assert sha(output / 'seed_draw.json') == p['seed_draw_sha256']
    draw = read(output / 'seed_draw.json'); seeds = draw['accepted_seeds_by_map']
    flat = [v for name in MAPS for v in seeds[name]]
    assert len(flat) == len(set(flat)) == 6
    assert not set(flat).intersection(draw['excluded_previous_seeds'])
    assert tuple(r['name'] for r in p['map_rows']) == MAPS
    assert p['schedule'] == schedule_for(p['map_rows'], seeds)
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return p


driver.validate = validate


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha(PACKAGE) == PACKAGE_SHA
    release = read(RELEASE / 'package_manifest.json')
    with zipfile.ZipFile(PACKAGE) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(release['files'])
        for name, v in release['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == v['sha256']
        for name, value in release['runtime_source_sha256'].items():
            assert sha(ROOT / 'src/swarm_rescue/solutions' / name) == value
        assert b'MyDroneRedStartup44Delivery' in archive.read('my_drone_eval.py')
    excluded = set(); histories = {}
    def collect(value, key=''):
        if isinstance(value, dict):
            for k, v in value.items(): collect(v, k)
        elif isinstance(value, list):
            for v in value: collect(v, key)
        elif 'seed' in key and type(value) is int and 0 <= value < 2**32:
            excluded.add(value)
    for path in (ROOT / 'output/sf13').rglob('plan.json'):
        collect(read(path)); histories[path.relative_to(ROOT).as_posix()] = sha(path)
    OUT.mkdir()
    seeds = {}; draws = []; accepted = set()
    for name in MAPS:
        seeds[name] = []
        while len(seeds[name]) < 3:
            value = secrets.randbits(32)
            ok = value not in excluded and value not in accepted
            draws.append(dict(map=name, value=value, accepted=ok))
            if ok:
                accepted.add(value); seeds[name].append(value)
    write(OUT / 'seed_draw.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        accepted_seeds_by_map=seeds, draws=draws, excluded_previous_seeds=sorted(excluded),
        method='Independent secrets.randbits(32) OS draws; no map/hash/date-derived seeds; no score selection.',
        inspected_previous_plans=histories))
    old = read(OLD / 'plan.json')
    rows = driver.scenarios.official_rows(MAPS)
    driver.register(); driver.use_ascii_temp()
    env = driver.probe_worker_environment(OUT)
    assert env == old['configuration']['worker_environment']
    config = dict(old['configuration'])
    config.update(controllers=[SELECTOR], rounds=3,
        controller_identities={SELECTOR: driver.runner.ev.controller_identity(SELECTOR)},
        environment=driver.runner.environment(), worker_environment=env)
    sources = driver.runner.source_receipt(rows)
    for f in old['source_sha256']:
        if not f.startswith(('src/', 'config/')):
            sources[f] = sha(ROOT / f)
    history = {p.relative_to(ROOT).as_posix(): sha(p) for p in
        (RELEASE / 'package_manifest.json', RELEASE / 'release_receipt.json',
         RELEASE / 'import_verification.json', OLD / 'plan.json')}
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(sources): archive.write(ROOT / f, f)
    (OUT / 'team742_preliminary.zip').write_bytes(PACKAGE.read_bytes())
    plan = dict(schema='released-startup44-delivery-r2-two-maps-six-full-pairs-v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        configuration=config, schedule=schedule_for(rows, seeds), map_rows=rows,
        source_sha256=sources, source_snapshot_sha256=sha(OUT / 'source_snapshot.zip'),
        historical_sha256=history, seed_draw_sha256=sha(OUT / 'seed_draw.json'),
        submission_sha256=PACKAGE_SHA, submission_runtime_sha256=release['runtime_source_sha256'],
        preserved='Released Startup44 Delivery R2 algorithm, official maps/zones/spawn/engine/scorer/resources unchanged.',
        loading='Native driver registers exact class selected by ZIP entry; every loaded runtime module must equal ZIP bytes.',
        observer_adaptation='Map06 has one NoComZone; extend read-only zone audit from Map08/09 two-zone contract. No map or physics changes.',
        scope='6 new red phases, each followed by fixed B0 Adapted blue on its actual recorded bomb layout.',
        score_scope='Native locally measured full score against one fixed B0 blue, not online reference-panel score.',
        retry_policy='One attempt per phase, all outcomes retained; no seed replacement, best-of selection, retuning or score gates.')
    write(OUT / 'plan.json', plan)
    write(OUT / 'frozen.json', dict(plan_sha256=sha(OUT / 'plan.json')))
    validate(OUT)
    print('PREPARED', seeds, sha(OUT / 'plan.json'), flush=True)


def worker(request):
    if read(request)['phase'] == 'blue':
        return driver.worker(request)
    return previous.worker(request)


def phase_issues(result, attempt, plan, phase, red_attempt=None):
    issues = driver.phase_issues(result, attempt, plan, phase, red_attempt)
    req = read(attempt / 'request.json')
    if req['map'] == 'Map06' and (attempt / 'zones_observation.json').exists():
        z = read(attempt / 'zones_observation.json')
        expected = dict(configured_zones=['NO_COM_ZONE'], disabler_zone_count=1,
            disabler_zone_classes=['NoComZone'], drone_count=5 if phase == 'red' else 10,
            phase=phase, observation_only=True)
        if z == expected:
            issues = [x for x in issues if x != 'actual_official_zone_entities']
        elif 'actual_official_zone_entities' not in issues:
            issues.append('actual_official_zone_entities')
    if phase == 'red':
        for name in ('startup44_diagnostics.json', 'startup44_delivery_diagnostics.json'):
            if not (attempt / name).is_file(): issues.append('missing_' + name)
            elif read(attempt / name)['errors']: issues.append('observer_' + name)
        if (attempt / 'loaded_identity.json').is_file():
            loaded = read(attempt / 'loaded_identity.json')['loaded_sources']
            mismatched = [n for n, h in plan['submission_runtime_sha256'].items()
                          if loaded.get('src/swarm_rescue/solutions/' + n) != h]
            if mismatched: issues.append('loaded_package_source_mismatch:' + ','.join(mismatched))
    return issues


def run():
    plan = validate(OUT)
    driver.register(); driver.use_ascii_temp(); driver.runner.worker_command = driver.command
    pairs = []
    with driver.append_lock(OUT):
        for item in plan['schedule']:
            validate(OUT)
            folder = OUT / 'runs' / f"{item['map']}_r{item['round']:02d}" / SELECTOR
            print('START', item['execution_index'], '/6', item['map'], item['seed'], flush=True)
            write(OUT / 'progress.json', dict(complete=False, completed=len(pairs), expected=6, phase='red', current=item))
            driver.no_retry_guard(folder / 'red')
            red, red_run = driver.runner.execute_phase(OUT, plan, item, 'red', retry_failed=False)
            red_issues = phase_issues(red, red_run, plan, 'red')
            pair = dict(item, red=red, red_run=str(red_run), blue=None, blue_run=None,
                        red_issues=red_issues, blue_issues=['blue_pending'])
            write(OUT / 'pairs.json', pairs + [pair])
            print('RED_DONE', item['execution_index'], red.get('phase1', {}).get('score_map'), red_issues, flush=True)
            if driver.runner.phase_valid(red, 'red') and red.get('placed_count', 0) > 0:
                validate(OUT); driver.no_retry_guard(folder / 'blue')
                write(OUT / 'progress.json', dict(complete=False, completed=len(pairs), expected=6, phase='blue', current=item))
                blue, blue_run = driver.runner.execute_phase(OUT, plan, item, 'blue', red_run, retry_failed=False)
                pair.update(blue=blue, blue_run=str(blue_run),
                    blue_issues=phase_issues(blue, blue_run, plan, 'blue', red_run))
            else:
                pair['blue_issues'] = ['no_valid_actual_red_layout']
            pairs.append(pair); write(OUT / 'pairs.json', pairs)
            rows = [driver.strict_flatten(p) for p in pairs]; write(OUT / 'results.json', rows)
            print('PAIR_DONE', item['execution_index'],
                  {k: rows[-1].get(k) for k in ('map', 'seed', 'map_score', 'wall_score', 'blue_score', 'total_score', 'valid_score')}, flush=True)
        validate(OUT)
        done = dict(complete=True, expected=6, completed=len(pairs), new_red_runs=len(pairs),
            new_blue_runs=sum(p['blue'] is not None for p in pairs),
            valid_red=sum(not p['red_issues'] for p in pairs),
            valid_blue=sum(p['blue'] is not None and not p['blue_issues'] for p in pairs),
            valid_full=sum(driver.strict_flatten(p)['valid_score'] for p in pairs), retries=0)
        write(OUT / 'completion.json', done); write(OUT / 'progress.json', done)
        print('COMPLETE', done, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', type=Path); parser.add_argument('--environment-worker', type=Path)
    args = parser.parse_args()
    if args.environment_worker: sys.exit(driver.environment_worker(args.environment_worker))
    if args.worker: sys.exit(worker(args.worker))
    if args.prepare: prepare()
    if args.run: run()
