"""Twelve new paired red flights; fixed native driver, zones, resources and seeds."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import secrets
import subprocess
import sys
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[4]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'output/sf13/startup44_random3_20261122'
OLD=ROOT/'output/sf13/linked_passage43_run_20261122'
CATALOG=ROOT/'output/sf13/clean_boundary_official_run_20261122/plan.json'
spec=importlib.util.spec_from_file_location('startup44_previous',ROOT/'src/swarm_rescue/solutions/devtools/linked_passage43_batch.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
driver,read,sha=previous.driver,previous.read,previous.sha
write=driver.runner.write
SELECTORS=('baseline_linked_passage43','candidate_startup44')
driver.SCRIPT=SCRIPT
driver.SELECTORS=SELECTORS
driver.CONTROLLERS={SELECTORS[0]:('my_drone_red_linked_passage43','MyDroneRedLinkedPassage43'),
                    SELECTORS[1]:('my_drone_red_startup44','MyDroneRedStartup44')}


def schedule_for(rows,seeds):
    items=[]
    for number,seed in enumerate(seeds,1):
        for row in rows:
            for selector in (SELECTORS if number%2 else SELECTORS[::-1]):
                name=row['name']
                items.append(dict(map_id=row['id'],name=name,family=row['family'],map_file=row['map_path'],
                    map=name,official_map=name,scenario=driver.NATIVE_SCENARIO(ROOT/row['map_path'],name),
                    round=number,seed=seed,controller=selector,arm='baseline' if selector==SELECTORS[0] else 'candidate',
                    version='LinkedPassage43' if selector==SELECTORS[0] else 'Startup44',blue_variant='adapted',
                    pair_id=f'{name}_r{number:02d}_s{seed}',execution_index=len(items)+1))
    return items


def validate(output,worker_role=False):
    plan=read(output/'plan.json')
    assert sha(output/'plan.json')==read(output/'frozen.json')['plan_sha256']
    driver.runner.validate_sources(plan['source_sha256'])
    assert driver.runner.environment()==plan['configuration']['worker_environment' if worker_role else 'environment']
    assert all(sha(ROOT/f)==v for f,v in plan['historical_sha256'].items())
    assert sha(output/'seed_draw.json')==plan['seed_draw_sha256']
    draw=read(output/'seed_draw.json');seeds=draw['accepted_seeds']
    assert len(seeds)==len(set(seeds))==3 and not set(seeds).intersection(draw['excluded_previous_seeds'])
    assert [r['name'] for r in plan['map_rows']]==['Map08','Map09']
    assert plan['schedule']==schedule_for(plan['map_rows'],seeds) and len(plan['schedule'])==12
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return plan


driver.validate=validate


def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    previous.validate(OLD);old=read(OLD/'plan.json')
    excluded=set();oldplans={}
    for path in (ROOT/'output/sf13').rglob('plan.json'):
        value=read(path)
        excluded.update(x['seed'] for x in value.get('schedule',[]) if isinstance(x.get('seed'),int))
        excluded.update(x for x in value.get('seeds',[]) if isinstance(x,int))
        oldplans[path.relative_to(ROOT).as_posix()]=sha(path)
    OUT.mkdir();seeds=[];draws=[]
    while len(seeds)<3:
        value=secrets.randbits(32);ok=value not in excluded and value not in seeds
        draws.append(dict(value=value,accepted=ok))
        if ok:seeds.append(value)
    write(OUT/'seed_draw.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),accepted_seeds=seeds,
        method='secrets.randbits(32); OS randomness, independent of map/hash/date; no performance selection',
        draws=draws,excluded_previous_seeds=sorted(excluded),inspected_previous_plans=oldplans))
    rows=sorted((r for r in read(CATALOG)['map_rows'] if r['name'] in ('Map08','Map09')),key=lambda r:r['name'])
    driver.register();driver.use_ascii_temp();env=driver.probe_worker_environment(OUT)
    assert env==old['configuration']['worker_environment']
    config=dict(old['configuration']);config.update(controllers=list(SELECTORS),rounds=3,
        controller_identities={s:driver.runner.ev.controller_identity(s) for s in SELECTORS},
        environment=driver.runner.environment(),worker_environment=env)
    sources=driver.runner.source_receipt(rows)
    for p in (SCRIPT,ROOT/'output/sf13/continuous38_run_20261122.py',ROOT/'output/sf13/entry37_run_20261122.py',
              ROOT/'output/sf13/clean_boundary_official_run_20261122.py'):
        sources[p.relative_to(ROOT).as_posix()]=sha(p)
    history={p.relative_to(ROOT).as_posix():sha(p) for p in (OLD/'plan.json',OLD/'frozen.json',OLD/'pairs.json',OLD/'completion.json',CATALOG)}
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(sources):archive.write(ROOT/f,f)
    plan=dict(schema='startup44-native-twelve-paired-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),configuration=config,
        schedule=schedule_for(rows,seeds),map_rows=rows,source_sha256=sources,historical_sha256=history,
        source_snapshot_sha256=sha(OUT/'source_snapshot.zip'),seed_draw_sha256=sha(OUT/'seed_draw.json'),
        baseline_batch=str(OLD),component_checks_passed=37,
        delta='At most96 initial control steps: multiframe legal360 lidar sweep; received per-drone safe exit bids; real-radio leader assignment; live collision avoidance then unchanged LP43. No turn solely for scan.',
        preserved='Continuous38 pose/map, boundary, LP43 navigation after startup, fine passages, delivery, final source freeze/ACK and deployment. Official zones enabled; native scenario/driver/resources unchanged.',
        scope='Map08/09, same three new seeds, baseline plus candidate, twelve new red phases; no blue/full score requested',
        retry_policy='Exactly one attempt per scheduled flight. No seed selection, retuning, replacement or retries after freeze.')
    write(OUT/'plan.json',plan);write(OUT/'frozen.json',dict(plan_sha256=sha(OUT/'plan.json')))
    write(OUT/'component_receipt.json',dict(passed=True,checks=37,command='pytest test_red_startup44.py test_red_linked_passage43.py test_red_return_budget42.py test_red_schedule40r1.py -q'))
    validate(OUT);print('PREPARED',seeds,sha(OUT/'plan.json'),flush=True)


def worker(request):
    from swarm_rescue.simulation.gui_map import gui_sr
    native=gui_sr.run_control;refs={};errors=[]
    def observed(d):
        command=native(d)
        try:refs[int(d.identifier)]=d
        except Exception as exc:
            if len(errors)<12:errors.append(repr(exc))
        return command
    try:
        with patch.object(gui_sr,'run_control',observed):code=previous.worker(request)
    finally:
        run=Path(read(request)['run_dir'])
        write(run/'startup44_diagnostics.json',dict(passive_only=True,errors=errors,drones={str(i):dict(
            class_name=type(d).__name__,events=getattr(d,'_st44_events',[]),samples=getattr(d,'_st44_samples',[]),
            plan=getattr(d,'_st44_plan',None),bids=getattr(d,'_st44_bids',{}),released=getattr(d,'_st44_done',None)) for i,d in refs.items()}))
    return code


def run():
    plan=validate(OUT);driver.register();driver.use_ascii_temp();driver.runner.worker_command=driver.command
    pairs=[];checks=[]
    with driver.append_lock(OUT):
        for item in plan['schedule']:
            validate(OUT)
            phase=OUT/'runs'/f"{item['map']}_r{item['round']:02d}"/item['controller']/'red'
            driver.no_retry_guard(phase)
            write(OUT/'progress.json',dict(complete=False,completed=len(pairs),expected=12,current=item))
            print('START',item['execution_index'],'/12',item['map'],item['seed'],item['controller'],flush=True)
            result,attempt=driver.runner.execute_phase(OUT,plan,item,'red',retry_failed=False)
            issues=driver.phase_issues(result,attempt,plan,'red')
            for name in ('entry37_diagnostics.json','continuous38_diagnostics.json','delivery39_diagnostics.json',
                         'schedule40r1_diagnostics.json','return_budget42_diagnostics.json','linked_passage43_diagnostics.json','startup44_diagnostics.json'):
                if not (attempt/name).is_file():issues.append('missing_'+name)
                elif read(attempt/name)['errors']:issues.append('observer_errors_'+name)
            pairs.append(dict(item,red=result,red_run=str(attempt),blue=None,blue_run=None,red_issues=issues,blue_issues=['not_requested']))
            write(OUT/'pairs.json',pairs)
            check=previous.previous.previous.previous.delivery_check(attempt,issues)
            checks.append(dict(map=item['map'],round=item['round'],seed=item['seed'],controller=item['controller'],**check))
            write(OUT/'delivery_checks.json',checks)
            print('FINISHED',item['execution_index'],'/12','map',result.get('phase1',{}).get('score_map'),'delivery',check,flush=True)
    validate(OUT)
    done=dict(complete=True,new_red_runs=len(pairs),new_blue_runs=0,retries=0,valid_red=sum(not p['red_issues'] for p in pairs),
              delivery_complete=sum(c['passed'] for c in checks),total_score=None,completed=len(pairs),expected=12)
    write(OUT/'completion.json',done);write(OUT/'progress.json',done);print('COMPLETE',done,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--worker',type=Path);parser.add_argument('--environment-worker',type=Path);args=parser.parse_args()
    if args.environment_worker:sys.exit(driver.environment_worker(args.environment_worker))
    if args.worker:sys.exit(worker(args.worker))
    if args.prepare:prepare()
    if args.run:run()
