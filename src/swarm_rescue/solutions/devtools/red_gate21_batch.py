"""R21 diagnostic preparation; unchanged RL17-R, six new instrumented flights.

This is explicitly NOT a candidate comparison. Existing R20 results remain the
reference; full execution requires the user's --run with a frozen preparation.
"""
import argparse
import base64
from datetime import datetime,timezone
from pathlib import Path
import subprocess

from swarm_rescue.solutions.devtools import red_pair19_batch as b
from swarm_rescue.solutions.devtools import red_pair19_report as old_report
from swarm_rescue.solutions.devtools.red_explore20_report import summarize as base_summary
from swarm_rescue.solutions.devtools.red_explore20_diagnostics import recording as global_recording
from swarm_rescue.solutions.devtools.red_gate21_diagnostics import recording,verify

MODULE='swarm_rescue.solutions.devtools.red_gate21_batch'
CONTROLLERS={'rl17_repair':('my_drone_red_repair17','MyDroneRedRepair17')}
SELECTION={'yh_map_fs_72':'入口主诊断，保留两个既有种子；补录实际 RN7 探测与排序快照。',
           'yh_map_fs_37':'R20 两轮明显退步的诊断对照，仍运行原版 RL17-R。',
           'yh_map_fs_24':'较正常探索的训练地图保护性对照；不修改原地图或划分。'}


def summarize(batch):
    target=batch/'base_summary';target.mkdir(exist_ok=False)
    result=base_summary(batch,target)
    for row in result['rows']:
        try:
            check=verify(Path(row['attempt'])/'gate21_diagnostics.json');row['gate21_check']=check
            if check['errors'] or check['rank_mismatches'] or check['unattributed_contacts'] or not check['calls'] or not check['truth_rows']:
                row['reasons'].append('gate21_diagnostic_invalid')
        except Exception as e:row['reasons'].append('gate21_missing:'+repr(e))
        row['execution_ok']=not [r for r in row['reasons'] if r.startswith('diagnostic_') is False]
        row['valid']=row['execution_ok'] and not b.read(batch/'plan.json')['configuration']['smoke']
    result.update(valid_count=sum(r['valid'] for r in result['rows']),invalid_count=sum(not r['valid'] for r in result['rows']),
        all_execution_checks_pass=all(r['execution_ok'] for r in result['rows']),
        interpretation='Unchanged RL17-R diagnostic flights, no new policy or paired candidate effect.')
    b.write(batch/'summary.json',result)
    return result


def register():
    b.CONTROLLERS=CONTROLLERS;b.MODULE=MODULE;b.register();old_report.summarize=summarize


def prepare(out,smoke=False):
    b.verify(b.DATASET);names=list(SELECTION)[:2] if smoke else list(SELECTION)
    rows=b.training_rows(names,b.DATASET);selection=[];schedule=[]
    for row in rows:
        path=b.REPO/row['map_path'];spec=b.read_spec(path);b.check_sources(spec);conditions=b.scenario(path)
        if (spec['rules']['timestep_limit'],spec['rules']['walltime_limit'])!=(5400,180):raise ValueError('Native engine limits changed')
        selection.append(dict(name=row['name'],reason=SELECTION[row['name']],map_sha256=b.digest(path),zones=conditions['zones']))
        for number,seed in enumerate(b.SEEDS,1):
            if smoke and number!=(1 if row['name']=='yh_map_fs_72' else 2):continue
            schedule.append(dict(map_id=row['id'],name=row['name'],family=row['family'],map_file=row['map_path'],
                map='R19'+row['geometry_sha256'][:12],official_map=None,scenario=conditions,round=number,seed=seed,controller='rl17_repair'))
    sources=b.source_files(rows)
    plan=dict(schema='red-gate21-diagnostic-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.REPO,text=True).strip(),
        configuration=dict(smoke=smoke,diagnostic_steps=128 if smoke else None,timeout=240,memory_bytes=4_000_000_000,
            cpu_affinity_count=1,simultaneous_simulations=1,controllers=list(CONTROLLERS),
            controller_identities={c:b.runner.ev.controller_identity(c) for c in CONTROLLERS},
            engine_profile=b.engine_profile(),environment=b.environment(),blue_evaluated=False,full_score=None),
        selection=selection,map_rows=rows,schedule=schedule,source_sha256=sources,
        intervention='NONE: passive actual-probe snapshots, local scores, execution, coverage timeline, contact attribution',
        reference_batch='output/sf13/r20a_20261117_150615',candidate_prepared=False,
        limitations=['Six diagnostic flights, not a new-policy paired comparison. RL11 and failed R20-B remain historical references.',
            'Same seed does not guarantee exact full-flight replay: observer overhead is inside the 180-second budget.',
            'No global inflation changes, no new navigation target, no mapper or communication changes.',
            'Physical visibility is the established 8-tick/4px estimate, not exact coverage.'])
    out.mkdir(parents=True,exist_ok=False)
    b.write(out/'source_bytes.json',{k:base64.b64encode((b.REPO/k).read_bytes()).decode('ascii') for k in sources})
    b.runner.validate_sources(sources);plan['source_bytes_sha256']=b.digest(out/'source_bytes.json')
    b.write(out/'plan.json',plan);b.write(out/'prepared.json',dict(plan_sha256=b.digest(out/'plan.json'),runs=len(schedule),executed=0))
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--prepared',type=Path)
    p.add_argument('--worker',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--run',action='store_true')
    a=p.parse_args();register()
    if a.worker:
        root=Path(b.read(a.worker)['run_dir'])
        with recording(root/'gate21_diagnostics.json'),global_recording(root/'explore20_diagnostics.json'):
            return b.worker(a.worker)
    if not a.output:p.error('--output required')
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Fresh output required; failures retained: '+str(out))
    if a.prepared:
        source=a.prepared.resolve();plan=b.read(source/'plan.json');b.validate_plan(source,plan)
        if a.smoke!=plan['configuration']['smoke']:p.error('Smoke flag differs from frozen plan')
        out.mkdir(parents=True)
        for name in ('plan.json','prepared.json','source_bytes.json'):
            with (out/name).open('xb') as f:f.write((source/name).read_bytes())
    else:
        if a.run and not a.smoke:p.error('Full diagnostics require --prepared')
        plan=prepare(out,a.smoke)
    if a.run:return b.execute(out,plan)
    print('PREPARED ONLY',out,'runs',len(plan['schedule']),'executed 0; no candidate policy',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
