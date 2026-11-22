"""Three pre-drawn random seeds; unchanged frozen Continuous38/native worker."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import secrets
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[4]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'output/sf13/continuous38_random3_20261122'
OLD=ROOT/'output/sf13/continuous38_run_20261122'
BASE=OLD/'runs/Map09_r01/candidate_continuous38/red/attempt_001'
spec=importlib.util.spec_from_file_location('continuous38_frozen_worker',ROOT/'output/sf13/continuous38_run_20261122.py')
single=importlib.util.module_from_spec(spec);spec.loader.exec_module(single)
driver=single.driver;read,write,sha=single.read,single.write,single.sha
SELECTOR='candidate_continuous38'
driver.SCRIPT=SCRIPT

def validate(output,worker_role=False):
    plan=read(output/'plan.json')
    assert sha(output/'plan.json')==read(output/'frozen.json')['plan_sha256']
    driver.runner.validate_sources(plan['source_sha256'])
    expected=plan['configuration']['worker_environment' if worker_role else 'environment']
    assert driver.runner.environment()==expected,'Execution environment changed'
    assert all(sha(ROOT/name)==value for name,value in plan['historical_sha256'].items()),'Frozen reference changed'
    draw=read(output/'seed_draw.json')
    assert sha(output/'seed_draw.json')==plan['seed_draw_sha256']
    assert [s['seed'] for s in plan['schedule']]==draw['accepted_seeds']
    assert len(set(draw['accepted_seeds']))==3 and 1492628140 not in draw['accepted_seeds']
    assert len(plan['schedule'])==3 and all(s['map']=='Map09' and s['controller']==SELECTOR for s in plan['schedule'])
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return plan

driver.validate=validate

def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    single.base.validate(OLD)
    old=read(OLD/'plan.json')
    for name,value in read(BASE/'loaded_identity.json')['loaded_sources'].items():
        assert sha(ROOT/name)==value,('Algorithm/engine changed',name)
    OUT.mkdir();drawn=[];seeds=[]
    while len(seeds)<3:
        value=secrets.randbits(32)
        accepted=value not in seeds and value!=1492628140
        drawn.append(dict(value=value,accepted=accepted))
        if accepted:seeds.append(value)
    write(OUT/'seed_draw.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        method='secrets.randbits(32), operating-system randomness; no map/hash/time-derived seeds',
        draws=drawn,accepted_seeds=seeds,rejection_rule='duplicates or preceding diagnostic seed only; no performance filtering'))
    driver.register();driver.use_ascii_temp();env=driver.probe_worker_environment(OUT)
    row=old['map_rows'][0];schedule=[]
    for number,seed in enumerate(seeds,1):
        item=dict(old['schedule'][0]);item.update(round=number,seed=seed,execution_index=number,
            pair_id=f'Map09_r{number:02d}_s{seed}')
        schedule.append(item)
    config=dict(old['configuration']);config.update(rounds=3,environment=driver.runner.environment(),worker_environment=env)
    sources=driver.runner.source_receipt([row])
    for path in (SCRIPT,ROOT/'output/sf13/continuous38_run_20261122.py',ROOT/'output/sf13/entry37_run_20261122.py',ROOT/'output/sf13/clean_boundary_official_run_20261122.py'):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    history={p.relative_to(ROOT).as_posix():sha(p) for p in BASE.rglob('*') if p.is_file()}
    for path in (OLD/'plan.json',OLD/'source_snapshot.zip',OLD/'completion.json'):
        history[path.relative_to(ROOT).as_posix()]=sha(path)
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(sources):archive.write(ROOT/name,name)
    plan=dict(schema='continuous38-three-random-seeds-native-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration=config,schedule=schedule,map_rows=[row],source_sha256=sources,historical_sha256=history,
        source_snapshot_sha256=sha(OUT/'source_snapshot.zip'),seed_draw_sha256=sha(OUT/'seed_draw.json'),
        algorithm_identity_verified_against=str(BASE/'loaded_identity.json'),
        frozen_candidate_plan_sha256=sha(OLD/'plan.json'),scope='Map09 only, three new complete red phases with official zones, no blue/full total',
        delta='Only new independently drawn seeds and round labels; contestant algorithms and native execution/observers identical',
        seed_policy='All three seeds fixed before any flight; no retries or replacement seeds',
        historical_score_policy='Previous diagnostic seed 1492628140 is excluded from random-three statistics')
    write(OUT/'plan.json',plan);write(OUT/'frozen.json',dict(plan_sha256=sha(OUT/'plan.json')))
    validate(OUT);print('PREPARED',seeds,sha(OUT/'plan.json'),flush=True)

def run():
    plan=validate(OUT);driver.register();driver.use_ascii_temp();driver.runner.worker_command=driver.command
    pairs=[]
    with driver.append_lock(OUT):
        for item in plan['schedule']:
            validate(OUT)
            phase=OUT/'runs'/f"Map09_r{item['round']:02d}"/SELECTOR/'red'
            driver.no_retry_guard(phase)
            driver.runner.write(OUT/'progress.json',dict(complete=False,completed=len(pairs),expected=3,current_seed=item['seed']))
            print(f"START {item['round']}/3 seed={item['seed']}",flush=True)
            result,attempt=driver.runner.execute_phase(OUT,plan,item,'red',retry_failed=False)
            issues=driver.phase_issues(result,attempt,plan,'red')
            for name in ('entry37_diagnostics.json','continuous38_diagnostics.json'):
                if not (attempt/name).is_file():issues.append('missing_'+name)
                elif read(attempt/name)['errors']:issues.append('observer_errors_'+name)
            pair=dict(item,red=result,red_run=str(attempt),blue=None,blue_run=None,red_issues=issues,blue_issues=['not_requested'])
            pairs.append(pair);driver.runner.write(OUT/'pairs.json',pairs)
            print(f"FINISHED {item['round']}/3 valid={not issues} map={(result.get('phase1') or {}).get('score_map')} issues={issues}",flush=True)
    validate(OUT)
    done=dict(complete=True,new_red_runs=3,new_blue_runs=0,retries=0,valid_red=sum(not p['red_issues'] for p in pairs),
        map_scores=[None if p['red_issues'] else p['red']['phase1']['score_map'] for p in pairs],total_score=None)
    write(OUT/'completion.json',done);driver.runner.write(OUT/'progress.json',dict(done,completed=3,expected=3))
    print('COMPLETE',done,flush=True)
    return int(any(p['red_issues'] for p in pairs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--worker',type=Path);p.add_argument('--environment-worker',type=Path);a=p.parse_args()
    if a.environment_worker:sys.exit(driver.environment_worker(a.environment_worker))
    if a.worker:sys.exit(single.worker(a.worker))
    if a.prepare:prepare()
    if a.run:sys.exit(run())
