"""Runtime repair continuation; preserve failed attempt and reuse its valid baseline.

The first candidate crashed at step 41, before a submission. R1 fixes the
startup wrapper callback/plan handoff and diagnostic JSON nonfinite handling.
No performance-driven tuning or seed replacement is done.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[4]
SCRIPT=Path(__file__).resolve()
FAILED=ROOT/'output/sf13/startup44_delivery_20261123'
OUT=ROOT/'output/sf13/startup44_delivery_r1_20261123'
spec=importlib.util.spec_from_file_location('delivery44_base',ROOT/'src/swarm_rescue/solutions/devtools/startup44_delivery_batch.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
driver,read,sha,write=base.driver,base.read,base.sha,base.write
driver.SCRIPT=SCRIPT
base.OUT=OUT


def validate(output,worker_role=False):
    plan=read(output/'plan.json')
    assert sha(output/'plan.json')==read(output/'frozen.json')['plan_sha256']
    driver.runner.validate_sources(plan['source_sha256'])
    assert driver.runner.environment()==plan['configuration']['worker_environment' if worker_role else 'environment']
    assert all(sha(ROOT/f)==v for f,v in plan['historical_sha256'].items())
    assert driver.scenarios.scenario is driver.NATIVE_SCENARIO
    assert [(p['map'],p['seed']) for p in plan['schedule']]==[('Map09',1872360317)]*2+[('Map08',4237853630)]*2
    return plan


driver.validate=validate


def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    old=read(FAILED/'plan.json');pairs=read(FAILED/'pairs.json')
    assert len(pairs)==2 and not pairs[0]['red_issues'] and pairs[1]['red_issues']
    assert pairs[1]['red']['elapsed_steps']==41
    baseline=pairs[0]
    for f,v in read(Path(baseline['red_run'])/'loaded_identity.json')['loaded_sources'].items():
        assert sha(ROOT/f)==v,('valid baseline loaded source changed',f)
    OUT.mkdir();driver.register();driver.use_ascii_temp();env=driver.probe_worker_environment(OUT)
    assert env==old['configuration']['worker_environment']
    config=dict(old['configuration']);config.update(environment=driver.runner.environment(),worker_environment=env)
    sources=driver.runner.source_receipt(old['map_rows'])
    for f in old['source_sha256']:
        if not f.startswith(('src/','config/')):sources[f]=sha(ROOT/f)
    history={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
             [FAILED/'plan.json',FAILED/'pairs.json',FAILED/'frozen.json',FAILED/'source_snapshot.zip',FAILED/'before_receipt.json']}
    for p in Path(baseline['red_run']).rglob('*'):
        if p.is_file():history[p.relative_to(ROOT).as_posix()]=sha(p)
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(sources):z.write(ROOT/f,f)
    plan=dict(old)
    plan.update(schema='startup44-delivery-runtime-repair-r1',created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),configuration=config,
        source_sha256=sources,historical_sha256=history,source_snapshot_sha256=sha(OUT/'source_snapshot.zip'),
        component_checks_passed=39,reused_baseline=baseline,
        prior_failed_run=pairs[1]['red_run'],prior_failure='step41 missing _note44 callback in startup receive proxy; then nonfinite diagnostic serialization',
        revision='Bind unchanged startup callback and propagate its plan; serialize nonfinite diagnostic state as null; increase event log capacity. No strategy parameter changes.',
        retry_policy='Preserve one invalid runtime attempt. Reuse the valid same-session baseline; run repaired candidate once. Map08 remains gated.')
    write(OUT/'plan.json',plan);write(OUT/'frozen.json',dict(plan_sha256=sha(OUT/'plan.json')))
    write(OUT/'component_receipt.json',dict(passed=True,checks=39))
    write(FAILED/'aborted.json',dict(complete=True,valid_red=1,invalid_red=1,map08_executed=False,
          reason=plan['prior_failure'],continuation=str(OUT),scores_from_invalid_run_not_reported=True))
    validate(OUT);print('PREPARED_R1',sha(OUT/'plan.json'),flush=True)


def clean_json(value):
    value=driver.json_state(value)
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,list):return [clean_json(v) for v in value]
    return value


def worker(request):
    from swarm_rescue.simulation.gui_map import gui_sr
    native=gui_sr.run_control;refs={};errors=[]
    def observed(d):
        result=native(d)
        try:refs[int(d.identifier)]=d
        except Exception as exc:
            if len(errors)<12:errors.append(repr(exc))
        return result
    try:
        with patch.object(gui_sr,'run_control',observed):code=base.previous.worker(request)
    finally:
        run=Path(read(request)['run_dir'])
        write(run/'startup44_delivery_diagnostics.json',dict(passive_only=True,errors=errors,
            drones={str(i):clean_json(getattr(d,'_dl44',None)) for i,d in refs.items()}))
    return code


def run():
    plan=validate(OUT);driver.register();driver.use_ascii_temp();driver.runner.worker_command=driver.command
    pairs=[dict(plan['reused_baseline'],reused_same_session=True)];write(OUT/'pairs.json',pairs);gating=None
    with driver.append_lock(OUT):
        for item in plan['schedule'][1:]:
            if item['map']=='Map08' and not gating['passed']:break
            validate(OUT)
            phase=OUT/'runs'/f"{item['map']}_r{item['round']:02d}"/item['controller']/'red'
            driver.no_retry_guard(phase)
            write(OUT/'progress.json',dict(complete=False,completed=len(pairs),current=item))
            print('START_R1',item['execution_index'],item['map'],item['seed'],item['controller'],flush=True)
            result,attempt=driver.runner.execute_phase(OUT,plan,item,'red',retry_failed=False)
            issues=driver.phase_issues(result,attempt,plan,'red')
            for f in ('startup44_diagnostics.json','startup44_delivery_diagnostics.json','linked_passage43_diagnostics.json'):
                if not (attempt/f).is_file():issues.append('missing_'+f)
                elif read(attempt/f)['errors']:issues.append('observer_'+f)
            pairs.append(dict(item,red=result,red_run=str(attempt),blue=None,blue_run=None,red_issues=issues,blue_issues=['not_requested']))
            write(OUT/'pairs.json',pairs)
            print('FINISHED_R1',item['execution_index'],'map',result.get('phase1',{}).get('score_map'),'issues',issues,flush=True)
            if len(pairs)==2:
                gating=base.gate(pairs) if not issues else dict(passed=False,checks={'valid':False},records=[])
                write(OUT/'map09_gate.json',gating);print('MAP09_GATE',gating,flush=True)
    validate(OUT)
    done=dict(complete=True,new_red_runs=len(pairs),new_blue_runs=0,valid_red=sum(not p['red_issues'] for p in pairs),
        map09_passed=gating['passed'],map08_executed=len(pairs)==4,total_score=None,
        map08_skip_reason=None if len(pairs)==4 else 'Map09 delivery/depth gate failed',
        runtime_repair_attempts=1,invalid_prior_attempts=1,performance_retries=0,
        new_attempts_this_batch=len(pairs)-1,baseline_reused_from_same_session=plan['reused_baseline']['red_run'])
    write(OUT/'completion.json',done);write(OUT/'progress.json',done);print('COMPLETE_R1',done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--worker',type=Path);p.add_argument('--environment-worker',type=Path);a=p.parse_args()
    if a.environment_worker:sys.exit(driver.environment_worker(a.environment_worker))
    if a.worker:sys.exit(worker(a.worker))
    if a.prepare:prepare()
    if a.run:run()
