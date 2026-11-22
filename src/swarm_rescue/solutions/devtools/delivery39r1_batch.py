"""Frozen, gated two-seed native evaluation; historical Continuous38 reused."""
import argparse
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[4];SCRIPT=Path(__file__).resolve()
OUT=ROOT/'output/sf13/delivery39r1_run_20261122'
OLD=ROOT/'output/sf13/continuous38_random3_20261122'
SELECTOR='candidate_delivery39r1'
spec=importlib.util.spec_from_file_location('delivery39_native_base',ROOT/'output/sf13/continuous38_run_20261122.py')
single=importlib.util.module_from_spec(spec);spec.loader.exec_module(single)
driver=single.driver;read,write,sha=single.read,single.write,single.sha
driver.SCRIPT=SCRIPT;driver.SELECTORS=(SELECTOR,)
driver.CONTROLLERS={SELECTOR:('my_drone_red_delivery39r1','MyDroneRedDelivery39R1')}

def validate(output,worker_role=False):
    plan=read(output/'plan.json')
    assert sha(output/'plan.json')==read(output/'frozen.json')['plan_sha256']
    driver.runner.validate_sources(plan['source_sha256'])
    expected=plan['configuration']['worker_environment' if worker_role else 'environment']
    assert driver.runner.environment()==expected
    assert all(sha(ROOT/f)==v for f,v in plan['historical_sha256'].items())
    assert [i['seed'] for i in plan['schedule']]==[4233377800,2315063155]
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    return plan
driver.validate=validate

def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    old=read(OLD/'plan.json');pairs=read(OLD/'pairs.json')
    for p in pairs:
        for name,value in read(Path(p['red_run'])/'loaded_identity.json')['loaded_sources'].items():
            assert sha(ROOT/name)==value,('Historical module changed',name)
    OUT.mkdir();driver.register();driver.use_ascii_temp();env=driver.probe_worker_environment(OUT)
    schedule=[]
    for index,seed in enumerate((4233377800,2315063155),1):
        item=dict(next(x for x in old['schedule'] if x['seed']==seed))
        item.update(controller=SELECTOR,version='Delivery39R1 / Continuous38 + observed boundary',execution_index=index)
        schedule.append(item)
    config=dict(old['configuration']);config.update(controllers=[SELECTOR],rounds=2,
        controller_identities={SELECTOR:driver.runner.ev.controller_identity(SELECTOR)},
        environment=driver.runner.environment(),worker_environment=env)
    sources=driver.runner.source_receipt(old['map_rows'])
    extra=[SCRIPT,ROOT/'src/swarm_rescue/solutions/devtools/test_red_delivery39.py',
        ROOT/'output/sf13/continuous38_run_20261122.py',ROOT/'output/sf13/entry37_run_20261122.py',
        ROOT/'output/sf13/clean_boundary_official_run_20261122.py']
    for path in extra:sources[path.relative_to(ROOT).as_posix()]=sha(path)
    history={path.relative_to(ROOT).as_posix():sha(path) for p in pairs if p['seed'] in (4233377800,2315063155)
        for path in Path(p['red_run']).rglob('*') if path.is_file()}
    for path in (OLD/'plan.json',OLD/'pairs.json',OLD/'review/data.json'):history[path.relative_to(ROOT).as_posix()]=sha(path)
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(sources):z.write(ROOT/name,name)
    plan=dict(schema='delivery39r1-gated-native-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration=config,schedule=schedule,map_rows=old['map_rows'],source_sha256=sources,historical_sha256=history,
        source_snapshot_sha256=sha(OUT/'source_snapshot.zip'),baseline_batch=str(OLD),
        component_checks=17,scope='R1 fixes reached-node executor defect; initial Delivery39 failed trial retained. Two gated red flights; official zones; no blue or total; no retries',
        delta='Observed walk graph return, actual-radio final meeting agreement, measured time/route budget, persistent unfinished frontiers after .60; inherited Continuous38 pose/mapper/renderer/sealing/deployment unchanged',
        gate='First run valid red, all five actual final source payloads/digests received at submission and five matching source-set ACKs; otherwise do not run seed2315063155',
        preserved='Native driver/seed/round/scenario/resources/observation cadence; baseline native receipts reused')
    write(OUT/'plan.json',plan);write(OUT/'frozen.json',dict(plan_sha256=sha(OUT/'plan.json')))
    validate(OUT);print('PREPARED',sha(OUT/'plan.json'),flush=True)

def worker(request):
    from swarm_rescue.simulation.gui_map import gui_sr
    native=gui_sr.run_control;refs={};errors=[]
    def observed(drone):
        command=native(drone)
        try:refs[int(drone.identifier)]=drone
        except Exception as exc:
            if len(errors)<12:errors.append(repr(exc))
        return command
    try:
        with patch.object(gui_sr,'run_control',observed):code=single.worker(request)
    finally:
        attempt=Path(read(request)['run_dir'])
        write(attempt/'delivery39_diagnostics.json',dict(passive_only=True,errors=errors,
            drones={str(k):dict(stats=dict(d._d39_stats),events=d._d39_events,samples=d._d39_samples,
                walk_stats=dict(d._walk39.stats),walk_points=[p.tolist() for p in d._walk39.points],
                walk_edges=[list(e.items()) for e in d._walk39.edges],final_mode=d._d39_mode,
                contract=d._d39_contract,contract_accepts=sorted(d._d39_accepts),
                frontier_stats=dict(d._explore.stats39),frontier_choices=d._explore.audit39,
                last_direct_contacts={str(i):(t,p.tolist()) for i,(t,p) in d._d39_contacts.items()}) for k,d in refs.items()}))
    return code

def gate(attempt,issues):
    import numpy as np
    d=read(attempt/'dualmap32_diagnostics.json')
    receipt=d['drones']['0']['_s28_submission'] or {}
    data=np.load(attempt/'dualmap32_snapshots.npz',allow_pickle=False)
    verified=[]
    for i in range(5):
        try:
            actual=d['snapshot_payloads']['0'][str(i)];own=d['snapshot_payloads'][str(i)][str(i)]
            # Receipt digests checked separately: end-of-run equality alone is insufficient.
            ok=(receipt['received_versions'][str(i)]==own['sequence'] and
                receipt['received_digests'][str(i)]==own['payload_sha256'] and
                actual['payload_sha256']==own['payload_sha256'])
            if ok:verified.append(i)
        except (KeyError,TypeError):pass
    result=dict(passed=not issues and len(verified)==5 and bool(receipt.get('final_complete')),
        verified_final_sources=verified,acknowledged_sources=receipt.get('acknowledged_sources',[]),
        actual_received_versions=receipt.get('received_versions',{}),issues=issues)
    data.close();return result

def run():
    plan=validate(OUT);driver.register();driver.use_ascii_temp();driver.runner.worker_command=driver.command
    pairs=[];gates=[]
    with driver.append_lock(OUT):
        for item in plan['schedule']:
            phase=OUT/'runs'/f"Map09_r{item['round']:02d}"/SELECTOR/'red'
            driver.no_retry_guard(phase)
            driver.runner.write(OUT/'progress.json',dict(complete=False,completed=len(pairs),current_seed=item['seed']))
            print('START',item['seed'],flush=True)
            result,attempt=driver.runner.execute_phase(OUT,plan,item,'red',retry_failed=False)
            issues=driver.phase_issues(result,attempt,plan,'red')
            for name in ('entry37_diagnostics.json','continuous38_diagnostics.json','delivery39_diagnostics.json'):
                if not (attempt/name).exists():issues.append('missing_'+name)
                elif read(attempt/name)['errors']:issues.append('observer_errors_'+name)
            pair=dict(item,red=result,red_run=str(attempt),blue=None,blue_run=None,red_issues=issues,blue_issues=['not_requested'])
            pairs.append(pair);driver.runner.write(OUT/'pairs.json',pairs)
            g=gate(attempt,issues);gates.append(dict(seed=item['seed'],**g));driver.runner.write(OUT/'gates.json',gates)
            print('FINISHED',item['seed'],'map',result.get('phase1',{}).get('score_map'),'gate',g,flush=True)
            if item['execution_index']==1 and not g['passed']:break
    validate(OUT)
    done=dict(complete=True,new_red_runs=len(pairs),new_blue_runs=0,retries=0,
        valid_red=sum(not p['red_issues'] for p in pairs),total_score=None,
        gate_first_passed=gates[0]['passed'],skipped_seeds=[] if len(pairs)==2 else [2315063155])
    write(OUT/'completion.json',done);driver.runner.write(OUT/'progress.json',done)
    print('COMPLETE',done,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--worker',type=Path);p.add_argument('--environment-worker',type=Path);a=p.parse_args()
    if a.environment_worker:sys.exit(driver.environment_worker(a.environment_worker))
    if a.worker:sys.exit(worker(a.worker))
    if a.prepare:prepare()
    if a.run:run()
