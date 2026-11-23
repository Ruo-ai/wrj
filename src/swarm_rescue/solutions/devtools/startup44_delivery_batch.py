"""Conditional two-pair native flights; fixed seeds, official zones, no retries."""
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
OUT = ROOT/'output/sf13/startup44_delivery_20261123'
OLD = ROOT/'output/sf13/startup44_refined_random3_20261123'
spec = importlib.util.spec_from_file_location('delivery44_previous', ROOT/'src/swarm_rescue/solutions/devtools/startup44_refined_batch.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
driver, read, sha = previous.driver, previous.read, previous.sha
write = driver.runner.write
SELECTORS = ('baseline_startup44', 'candidate_startup44_delivery')
driver.SCRIPT = SCRIPT
driver.SELECTORS = SELECTORS
driver.CONTROLLERS = {SELECTORS[0]:('my_drone_red_startup44','MyDroneRedStartup44'),
                     SELECTORS[1]:('my_drone_red_startup44_delivery','MyDroneRedStartup44Delivery')}


def validate(output, worker_role=False):
    plan = read(output/'plan.json')
    assert sha(output/'plan.json') == read(output/'frozen.json')['plan_sha256']
    driver.runner.validate_sources(plan['source_sha256'])
    assert driver.runner.environment() == plan['configuration']['worker_environment' if worker_role else 'environment']
    assert all(sha(ROOT/f) == v for f,v in plan['historical_sha256'].items())
    assert [(p['map'],p['round'],p['seed'],p['controller']) for p in plan['schedule']] == [
        (name, number, seed, selector) for name,number,seed in [('Map09',3,1872360317),('Map08',2,4237853630)] for selector in SELECTORS]
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return plan


driver.validate = validate


def prepare():
    if (OUT/'plan.json').exists(): raise FileExistsError(OUT/'plan.json')
    old = read(OLD/'plan.json')
    driver.runner.validate_sources(old['source_sha256'])
    assert sha(ROOT/'src/swarm_rescue/solutions/my_drone_red_startup44.py') == read(OUT/'before_receipt.json')['sha256']
    driver.register(); driver.use_ascii_temp()
    env = driver.probe_worker_environment(OUT)
    assert env == old['configuration']['worker_environment']
    schedule = []
    for name,number,seed in [('Map09',3,1872360317),('Map08',2,4237853630)]:
        template = next(p for p in old['schedule'] if p['map']==name and p['seed']==seed)
        for selector in SELECTORS:
            item = dict(template)
            item.update(controller=selector, arm='baseline' if selector==SELECTORS[0] else 'candidate',
                version='Startup44 原版' if selector==SELECTORS[0] else 'Startup44 交付修复',
                execution_index=len(schedule)+1)
            schedule.append(item)
    config = dict(old['configuration'])
    config.update(controllers=list(SELECTORS), rounds=1,
        controller_identities={s:driver.runner.ev.controller_identity(s) for s in SELECTORS},
        environment=driver.runner.environment(), worker_environment=env)
    sources = driver.runner.source_receipt(old['map_rows'])
    for f,v in old['source_sha256'].items():
        if not f.startswith('src/') and not f.startswith('config/'): sources[f] = sha(ROOT/f)
    history = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
               [OLD/'plan.json',OLD/'pairs.json',OLD/'review/data.json',OUT/'before_receipt.json',OUT/'startup44_before_delivery.py']}
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(sources): archive.write(ROOT/f,f)
    plan = dict(schema='startup44-delivery-gated-pairs-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration=config,schedule=schedule,map_rows=old['map_rows'],source_sha256=sources,
        source_snapshot_sha256=sha(OUT/'source_snapshot.zip'),historical_sha256=history,
        component_checks_passed=38,baseline_batch=str(OLD),
        gate='Map09 valid and 5/5 exact final source bytes at actual submission; common-cutoff all, north_deep and northwest_deep visit cells >= fresh baseline. Otherwise Map08 not run.',
        delta='Persistent exact final receipts; missing-source relay priority in existing slot; mutual contact tickets from actual radio; bounded late carrier/receiver and waiting; same initial return gate.',
        preserved='Original optimized Startup44 untouched. Inherited startup, pose, mapping, rendering, boundary, exploration, freeze and native driver/resources/zones unchanged.',
        retry_policy='One attempt per arm. Freeze before running. No flight repeats or score-driven retuning.',
        scoring_scope='Native red phase including ten bombs. No blue/full total score in this task.')
    write(OUT/'plan.json',plan);write(OUT/'frozen.json',dict(plan_sha256=sha(OUT/'plan.json')))
    write(OUT/'component_receipt.json',dict(passed=True,checks=38,
        command='pytest test_red_startup44_delivery.py test_red_startup44.py test_red_linked_passage43.py -q'))
    validate(OUT);print('PREPARED',sha(OUT/'plan.json'),flush=True)


def worker(request):
    from swarm_rescue.simulation.gui_map import gui_sr
    native = gui_sr.run_control; refs={}; errors=[]
    def observe(d):
        command = native(d)
        try: refs[int(d.identifier)] = d
        except Exception as exc:
            if len(errors)<12: errors.append(repr(exc))
        return command
    try:
        with patch.object(gui_sr,'run_control',observe): code=previous.worker(request)
    finally:
        run=Path(read(request)['run_dir'])
        write(run/'startup44_delivery_diagnostics.json',dict(passive_only=True,errors=errors,
            drones={str(i):driver.json_state(getattr(d,'_dl44',None)) for i,d in refs.items()}))
    return code


def gate(pairs):
    import numpy as np
    from swarm_rescue.solutions.devtools import schedule40r1_report as h
    diags = [read(Path(p['red_run'])/'dualmap32_diagnostics.json') for p in pairs]
    common = min(v['_f27_stats']['cutoff_elapsed'] for d in diags for v in d['drones'].values())
    rows=[]
    for pair,diag in zip(pairs,diags):
        metric,data=h.atlas.inspect(pair,OUT);nav=h.navplot.load(pair)
        with np.load(Path(pair['red_run'])/'dualmap32_snapshots.npz',allow_pickle=False) as z:
            sources,acks=h.h.delivery(diag,nav['receipt'],z)
            exact=[]
            for i in range(5):
                own=diag['snapshot_payloads'][str(i)][str(i)]
                received=diag['snapshot_payloads']['0'][str(i)]
                exact.append(z[own['array']].tobytes()==z[received['array']].tobytes())
        coverage,_=h.h.coverage(data,common,pair['map'])
        rows.append(dict(arm=pair['arm'],valid=metric['valid_red'],score=metric['map_score'],
            final_sources=sum(s['latest'] for s in sources),exact_source_bytes=sum(exact),
            acknowledgements=len(acks),coverage=coverage,common_cutoff=common))
    a,b=rows
    checks=dict(valid=all(r['valid'] for r in rows),finals=b['final_sources']==5,
                exact=b['exact_source_bytes']==5)
    for key in ('all','north_deep','northwest_deep'):
        checks[key+'_not_reduced']=b['coverage'][key]['cells40']>=a['coverage'][key]['cells40']
    return dict(passed=all(checks.values()),checks=checks,records=rows)


def run():
    plan=validate(OUT);driver.register();driver.use_ascii_temp();driver.runner.worker_command=driver.command
    pairs=[];gating=None
    with driver.append_lock(OUT):
        for item in plan['schedule']:
            if item['map']=='Map08' and not gating['passed']:break
            validate(OUT)
            phase=OUT/'runs'/f"{item['map']}_r{item['round']:02d}"/item['controller']/'red'
            driver.no_retry_guard(phase)
            write(OUT/'progress.json',dict(complete=False,completed=len(pairs),current=item))
            print('START',item['execution_index'],item['map'],item['seed'],item['controller'],flush=True)
            result,attempt=driver.runner.execute_phase(OUT,plan,item,'red',retry_failed=False)
            issues=driver.phase_issues(result,attempt,plan,'red')
            for name in ('startup44_diagnostics.json','startup44_delivery_diagnostics.json','linked_passage43_diagnostics.json'):
                if not (attempt/name).is_file():issues.append('missing_'+name)
                elif read(attempt/name)['errors']:issues.append('observer_'+name)
            pairs.append(dict(item,red=result,red_run=str(attempt),blue=None,blue_run=None,
                              red_issues=issues,blue_issues=['not_requested']))
            write(OUT/'pairs.json',pairs)
            print('FINISHED',item['execution_index'],'map',result.get('phase1',{}).get('score_map'),'issues',issues,flush=True)
            if len(pairs)==2:
                gating=gate(pairs);write(OUT/'map09_gate.json',gating)
                print('MAP09_GATE',gating,flush=True)
    validate(OUT)
    done=dict(complete=True,new_red_runs=len(pairs),new_blue_runs=0,retries=0,
        valid_red=sum(not p['red_issues'] for p in pairs),total_score=None,map09_passed=gating['passed'],
        map08_executed=len(pairs)==4,map08_skip_reason=None if len(pairs)==4 else 'Map09 predeclared delivery/depth gate failed')
    write(OUT/'completion.json',done);write(OUT/'progress.json',done);print('COMPLETE',done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--worker',type=Path);p.add_argument('--environment-worker',type=Path);a=p.parse_args()
    if a.environment_worker:sys.exit(driver.environment_worker(a.environment_worker))
    if a.worker:sys.exit(worker(a.worker))
    if a.prepare:prepare()
    if a.run:run()
