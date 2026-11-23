"""Frozen 2-map x 3-new-seed x 2-controller full red/blue experiment."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import secrets
import subprocess
import sys
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).resolve()
OUT = ROOT/'output/sf13/startup44_deep_bomb_random3_20261123'
OLD = ROOT/'output/sf13/startup44_delivery_r2_20261123'
spec = importlib.util.spec_from_file_location('deep_bomb_previous',
    ROOT/'src/swarm_rescue/solutions/devtools/startup44_delivery_r2_batch.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
driver, read, sha, write = previous.driver, previous.read, previous.sha, previous.write
SELECTORS = ('baseline_startup44_delivery', 'candidate_startup44_deep_bomb')
driver.SCRIPT = SCRIPT
driver.SELECTORS = SELECTORS
driver.CONTROLLERS = {
    SELECTORS[0]: ('my_drone_red_startup44_delivery', 'MyDroneRedStartup44Delivery'),
    SELECTORS[1]: ('my_drone_red_startup44_deep_bomb', 'MyDroneRedStartup44DeepBomb'),
}


def schedule_for(rows, seeds):
    items = []
    for number, seed in enumerate(seeds, 1):
        for mi, row in enumerate(rows):
            for selector in (SELECTORS if (mi+number)%2 else SELECTORS[::-1]):
                name = row['name']
                items.append(dict(map_id=row['id'], name=name, family=row['family'],
                    map_file=row['map_path'], map=name, official_map=name,
                    scenario=driver.NATIVE_SCENARIO(ROOT/row['map_path'],name),
                    round=number, seed=seed, controller=selector,
                    arm='baseline' if selector == SELECTORS[0] else 'candidate',
                    version='Startup44 Delivery R2' if selector == SELECTORS[0] else 'Startup44 DeepBomb45',
                    blue_variant='adapted', pair_id=f'{name}_r{number:02d}_s{seed}',
                    execution_index=len(items)+1))
    return items


def validate(output, worker_role=False):
    p = read(output/'plan.json')
    assert sha(output/'plan.json') == read(output/'frozen.json')['plan_sha256']
    driver.runner.validate_sources(p['source_sha256'])
    assert driver.runner.environment() == p['configuration']['worker_environment' if worker_role else 'environment']
    assert all(sha(ROOT/f) == v for f,v in p['historical_sha256'].items())
    assert sha(output/'seed_draw.json') == p['seed_draw_sha256']
    draw = read(output/'seed_draw.json'); seeds = draw['accepted_seeds']
    assert len(seeds) == len(set(seeds)) == 3
    assert not set(seeds).intersection(draw['excluded_previous_seeds'])
    assert [r['name'] for r in p['map_rows']] == ['Map08','Map09']
    assert p['schedule'] == schedule_for(p['map_rows'], seeds)
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return p


driver.validate = validate


def prepare():
    if OUT.exists(): raise FileExistsError(OUT)
    old = read(OLD/'plan.json')
    driver.runner.validate_sources(old['source_sha256'])
    excluded = set(); oldplans = {}
    for path in (ROOT/'output/sf13').rglob('plan.json'):
        value = read(path)
        excluded.update(x['seed'] for x in value.get('schedule',[]) if isinstance(x.get('seed'), int))
        excluded.update(x for x in value.get('seeds',[]) if isinstance(x,int))
        oldplans[path.relative_to(ROOT).as_posix()] = sha(path)
    OUT.mkdir(); seeds = []; draws = []
    while len(seeds) < 3:
        n = secrets.randbits(32); ok = n not in excluded and n not in seeds
        draws.append(dict(value=n,accepted=ok))
        if ok: seeds.append(n)
    write(OUT/'seed_draw.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        accepted_seeds=seeds, draws=draws, excluded_previous_seeds=sorted(excluded),
        method='secrets.randbits(32); OS randomness independent of map/hash/date; no score selection',
        inspected_previous_plans=oldplans))
    rows = sorted(old['map_rows'],key=lambda r:r['name'])
    driver.register(); driver.use_ascii_temp(); env = driver.probe_worker_environment(OUT)
    config = dict(old['configuration']); config.update(controllers=list(SELECTORS),rounds=3,
        controller_identities={s:driver.runner.ev.controller_identity(s) for s in SELECTORS},
        environment=driver.runner.environment(), worker_environment=env)
    sources = driver.runner.source_receipt(rows)
    for f in old['source_sha256']:
        if not f.startswith(('src/','config/')): sources[f] = sha(ROOT/f)
    history = {p.relative_to(ROOT).as_posix():sha(p) for p in
               (OLD/'plan.json',OLD/'frozen.json',OLD/'source_snapshot.zip',OLD/'pairs.json')}
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(sources): z.write(ROOT/f,f)
    plan = dict(schema='startup44-deep-bomb-full-paired-v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration=config, schedule=schedule_for(rows,seeds), map_rows=rows,
        source_sha256=sources, source_snapshot_sha256=sha(OUT/'source_snapshot.zip'),
        historical_sha256=history, seed_draw_sha256=sha(OUT/'seed_draw.json'),
        baseline_batch=str(OLD), component_checks_passed=54,
        delta='Reserve late second charge; observed route depth/detour and first-visit ranking with 65% marginal-wall floor; preserve frontier/fine passage eligibility while retaining inventory.',
        preserved='Startup44 optimized launch, Continuous38 pose/map, observed boundary, Delivery R2 protocol, freeze and original release safety/fallback. Official special zones and native resources unchanged.',
        scope='12 new red phases plus 12 fixed B0 Adapted blue phases on their actual layouts.',
        score_scope='Locally measured full score against one fixed B0 blue, not the online official reference-panel score.',
        retry_policy='One attempt per scheduled phase; no score-driven retuning, replacement seeds or best-score reruns.')
    write(OUT/'plan.json',plan); write(OUT/'frozen.json',dict(plan_sha256=sha(OUT/'plan.json')))
    write(OUT/'component_receipt.json',dict(passed=True,checks=54))
    validate(OUT); print('PREPARED',seeds,sha(OUT/'plan.json'),flush=True)


def worker(request):
    if read(request)['phase'] == 'blue': return driver.worker(request)
    from swarm_rescue.simulation.gui_map import gui_sr
    native = gui_sr.run_control; refs = {}; errors = []
    def observe(d):
        result = native(d)
        try: refs[int(d.identifier)] = d
        except Exception as exc:
            if len(errors) < 12: errors.append(repr(exc))
        return result
    try:
        with patch.object(gui_sr,'run_control',observe): code = previous.worker(request)
    finally:
        run = Path(read(request)['run_dir'])
        write(run/'deep_bomb45_diagnostics.json',dict(passive_only=True,errors=errors,
            drones={str(i):previous.clean_json(dict(events=getattr(d,'_db45_events',[]),
                first_visits=getattr(d,'_db45_first',{}), reserve_explore_ticks=getattr(d,'_db45_reserve_ticks',0),
                placements=d._placements)) for i,d in refs.items()}))
    return code


def run():
    plan = validate(OUT); driver.register(); driver.use_ascii_temp(); driver.runner.worker_command = driver.command
    pairs = []
    with driver.append_lock(OUT):
        for item in plan['schedule']:
            validate(OUT)
            folder = OUT/'runs'/f"{item['map']}_r{item['round']:02d}"/item['controller']
            print('START',item['execution_index'],'/12',item['map'],item['seed'],item['version'],flush=True)
            write(OUT/'progress.json',dict(complete=False,completed=len(pairs),expected=12,phase='red',current=item))
            driver.no_retry_guard(folder/'red')
            red, red_run = driver.runner.execute_phase(OUT,plan,item,'red',retry_failed=False)
            red_issues = driver.phase_issues(red,red_run,plan,'red')
            for name in ('startup44_diagnostics.json','startup44_delivery_diagnostics.json','deep_bomb45_diagnostics.json'):
                if not (red_run/name).is_file(): red_issues.append('missing_'+name)
                elif read(red_run/name)['errors']: red_issues.append('observer_'+name)
            pair = dict(item,red=red,red_run=str(red_run),blue=None,blue_run=None,
                        red_issues=red_issues,blue_issues=['blue_pending'])
            write(OUT/'pairs.json',pairs+[pair])
            print('RED_DONE',item['execution_index'],red.get('phase1',{}).get('score_map'),red_issues,flush=True)
            if driver.runner.phase_valid(red,'red') and red.get('placed_count',0)>0:
                validate(OUT); driver.no_retry_guard(folder/'blue')
                write(OUT/'progress.json',dict(complete=False,completed=len(pairs),expected=12,phase='blue',current=item))
                blue, blue_run = driver.runner.execute_phase(OUT,plan,item,'blue',red_run,retry_failed=False)
                pair.update(blue=blue,blue_run=str(blue_run),blue_issues=driver.phase_issues(blue,blue_run,plan,'blue',red_run))
            else: pair['blue_issues'] = ['no_valid_actual_red_layout']
            pairs.append(pair); write(OUT/'pairs.json',pairs)
            rows = [driver.strict_flatten(p) for p in pairs]; write(OUT/'results.json',rows)
            print('PAIR_DONE',item['execution_index'],rows[-1],flush=True)
        validate(OUT)
        done = dict(complete=True,expected=12,completed=len(pairs),new_red_runs=len(pairs),
            new_blue_runs=sum(p['blue'] is not None for p in pairs),
            valid_red=sum(not p['red_issues'] for p in pairs),
            valid_blue=sum(p['blue'] is not None and not p['blue_issues'] for p in pairs),
            valid_full=sum(driver.strict_flatten(p)['valid_score'] for p in pairs),retries=0)
        write(OUT/'completion.json',done); write(OUT/'progress.json',done)
        print('COMPLETE',done,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',action='store_true')
    p.add_argument('--worker',type=Path); p.add_argument('--environment-worker',type=Path); a = p.parse_args()
    if a.environment_worker: sys.exit(driver.environment_worker(a.environment_worker))
    if a.worker: sys.exit(worker(a.worker))
    if a.prepare: prepare()
    if a.run: run()
