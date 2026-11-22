"""Two frozen native red flights; same bad seed/round as historical R2."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[4];SCRIPT=Path(__file__).resolve()
OUT=ROOT/'output/sf13/schedule40_run_20261122'
OLD=ROOT/'output/sf13/delivery39r2_random3_20261122'
SELECTOR='candidate_schedule40'
spec=importlib.util.spec_from_file_location('schedule40_previous',ROOT/'src/swarm_rescue/solutions/devtools/delivery39r2_random3.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
driver=previous.driver;read,sha=previous.read,previous.sha
write=driver.runner.write
driver.SCRIPT=SCRIPT;driver.SELECTORS=(SELECTOR,)
driver.CONTROLLERS={SELECTOR:('my_drone_red_schedule40','MyDroneRedSchedule40')}


def validate(output,worker_role=False):
    plan=read(output/'plan.json')
    assert sha(output/'plan.json')==read(output/'frozen.json')['plan_sha256']
    driver.runner.validate_sources(plan['source_sha256'])
    expected=plan['configuration']['worker_environment' if worker_role else 'environment']
    assert driver.runner.environment()==expected
    assert all(sha(ROOT/f)==v for f,v in plan['historical_sha256'].items())
    assert [(p['map'],p['round'],p['seed']) for p in plan['schedule']]==[('Map08',2,2252748057),('Map09',2,2252748057)]
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return plan


driver.validate=validate


def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    previous.validate(OLD)
    old=read(OLD/'plan.json')
    baseline=[p for p in read(OLD/'pairs.json') if p['seed']==2252748057]
    for p in baseline:
        for f,v in read(Path(p['red_run'])/'loaded_identity.json')['loaded_sources'].items():
            assert sha(ROOT/f)==v,('Historical source changed',f)
    OUT.mkdir();driver.register();driver.use_ascii_temp()
    env=driver.probe_worker_environment(OUT)
    assert env==old['configuration']['worker_environment']
    schedule=[]
    for index,p in enumerate(baseline,1):
        item=dict(next(x for x in old['schedule'] if x['map']==p['map'] and x['seed']==p['seed']))
        item.update(controller=SELECTOR,version='Schedule40 / unified exploration scheduling',execution_index=index)
        schedule.append(item)
    config=dict(old['configuration']);config.update(controllers=[SELECTOR],rounds=1,
        controller_identities={SELECTOR:driver.runner.ev.controller_identity(SELECTOR)},
        environment=driver.runner.environment(),worker_environment=env)
    sources=driver.runner.source_receipt(old['map_rows'])
    for f in (SCRIPT,ROOT/'src/swarm_rescue/solutions/devtools/delivery39r2_batch.py',
              ROOT/'src/swarm_rescue/solutions/devtools/delivery39r2_random3.py',
              ROOT/'output/sf13/continuous38_run_20261122.py',ROOT/'output/sf13/entry37_run_20261122.py',
              ROOT/'output/sf13/clean_boundary_official_run_20261122.py'):
        sources[f.relative_to(ROOT).as_posix()]=sha(f)
    history={f.relative_to(ROOT).as_posix():sha(f) for p in baseline for f in Path(p['red_run']).rglob('*') if f.is_file()}
    for f in (OLD/'plan.json',OLD/'pairs.json',OLD/'review/data.json',OLD/'discussion_audit_20261122.json'):
        history[f.relative_to(ROOT).as_posix()]=sha(f)
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(sources):z.write(ROOT/f,f)
    plan=dict(schema='schedule40-native-paired-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration=config,schedule=schedule,map_rows=old['map_rows'],source_sha256=sources,
        source_snapshot_sha256=sha(OUT/'source_snapshot.zip'),historical_sha256=history,
        baseline_batch=str(OLD),baseline_runs=[p['red_run'] for p in baseline],component_checks_passed=18,
        delta=['Remove inherited .72 route clear during exploration','Empty-inventory frontier queue enabled without .60 gate',
               'RN7 narrow executor uses exploration mode and actual delivery preemption instead of .68 cutoff'],
        preserved='Continuous38 localization, mapping, boundary, source freeze, deployment, protocol, return budget/executor, initial direction strategy, native driver/resources/zones',
        scope='Two new complete red flights; prior matching R2 flights reused; no blue or full total',
        retry_policy='One attempt each, no retries, no tuning between maps')
    write(OUT/'plan.json',plan);write(OUT/'frozen.json',dict(plan_sha256=sha(OUT/'plan.json')))
    validate(OUT);print('PREPARED',sha(OUT/'plan.json'),flush=True)


def worker(request):
    from swarm_rescue.simulation.gui_map import gui_sr
    native=gui_sr.run_control;refs={};samples=[];errors=[]
    def observed(drone):
        command=native(drone)
        try:
            key=int(drone.identifier);refs[key]=drone
            if drone._steps%60==0:
                samples.append(dict(drone=key,step=drone._steps,mode=drone._d39_mode,
                    progress=drone._progress(),route_length=len(drone._route),
                    goal=list(drone._route[-1]) if drone._route else None,
                    frontiers_enabled=drone._explore.extension_enabled,
                    stats=dict(drone._schedule40_stats)))
        except Exception as exc:
            if len(errors)<12:errors.append(repr(exc))
        return command
    try:
        with patch.object(gui_sr,'run_control',observed):code=previous.single.worker(request)
    finally:
        run=Path(read(request)['run_dir'])
        write(run/'schedule40_diagnostics.json',dict(passive_only=True,errors=errors,samples=samples,
            drones={str(k):dict(stats=dict(d._schedule40_stats),mro=[c.__name__ for c in type(d).__mro__]) for k,d in refs.items()}))
    return code


def run():
    plan=validate(OUT);driver.register();driver.use_ascii_temp();driver.runner.worker_command=driver.command
    pairs=[];delivery=[]
    with driver.append_lock(OUT):
        for item in plan['schedule']:
            validate(OUT)
            driver.no_retry_guard(OUT/'runs'/f"{item['map']}_r{item['round']:02d}"/SELECTOR/'red')
            write(OUT/'progress.json',dict(complete=False,completed=len(pairs),expected=2,current_map=item['map']))
            print('START',item['map'],item['seed'],flush=True)
            result,attempt=driver.runner.execute_phase(OUT,plan,item,'red',retry_failed=False)
            issues=driver.phase_issues(result,attempt,plan,'red')
            for name in ('entry37_diagnostics.json','continuous38_diagnostics.json','delivery39_diagnostics.json','schedule40_diagnostics.json'):
                if not (attempt/name).is_file():issues.append('missing_'+name)
                elif read(attempt/name)['errors']:issues.append('observer_errors_'+name)
            pairs.append(dict(item,red=result,red_run=str(attempt),blue=None,blue_run=None,red_issues=issues,blue_issues=['not_requested']))
            write(OUT/'pairs.json',pairs)
            check=previous.delivery_check(attempt,issues);delivery.append(dict(map=item['map'],**check))
            write(OUT/'delivery_checks.json',delivery)
            print('FINISHED',item['map'],'map',result.get('phase1',{}).get('score_map'),'delivery',check,flush=True)
    validate(OUT)
    done=dict(complete=True,new_red_runs=len(pairs),new_blue_runs=0,retries=0,
        valid_red=sum(not p['red_issues'] for p in pairs),delivery_complete=sum(p['passed'] for p in delivery),total_score=None)
    write(OUT/'completion.json',done);write(OUT/'progress.json',done);print('COMPLETE',done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--worker',type=Path);p.add_argument('--environment-worker',type=Path);a=p.parse_args()
    if a.environment_worker:sys.exit(driver.environment_worker(a.environment_worker))
    if a.worker:sys.exit(worker(a.worker))
    if a.prepare:prepare()
    if a.run:run()
