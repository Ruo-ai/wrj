"""Independent frozen R20 batch: four training maps x two seeds x three arms."""
import argparse
import base64
from datetime import datetime,timezone
from pathlib import Path
import subprocess

from swarm_rescue.solutions.devtools import red_pair19_batch as b
from swarm_rescue.solutions.devtools import red_pair19_report as old_report
from swarm_rescue.solutions.devtools.red_explore20_report import summarize
from swarm_rescue.solutions.devtools.red_explore20_diagnostics import recording

MODULE='swarm_rescue.solutions.devtools.red_explore20_batch'
CONTROLLERS={**b.CONTROLLERS,'r20_budget':('my_drone_red_budget20','MyDroneRedBudget20')}
SELECTION={'yh_map_fs_72':'R19最低地图分，主诊断样本；入口与局部执行预算。',
           'yh_map_fs_55':'R19低图分且有长期缺测，主诊断样本；保留反例。',
           'yh_map_fs_37':'同训练集较开阔对照；R19出现地图分退步。',
           'yh_map_fs_24':'同训练集开阔对照，检查局部任务增加是否损害原有优势。'}


def register():
    b.CONTROLLERS=CONTROLLERS;b.MODULE=MODULE;b.register();old_report.summarize=summarize


def prepare(output,smoke=False):
    if output.exists():raise FileExistsError(output)
    b.verify(b.DATASET)
    names=list(SELECTION)[:1] if smoke else list(SELECTION)
    rows=b.training_rows(names,b.DATASET);schedule=[];selection=[]
    for row in rows:
        path=b.REPO/row['map_path'];spec=b.read_spec(path);b.check_sources(spec);conditions=b.scenario(path)
        if (spec['rules']['timestep_limit'],spec['rules']['walltime_limit'])!=(5400,180):raise ValueError('Engine limits changed')
        selection.append(dict(name=row['name'],reason=SELECTION[row['name']],map_sha256=b.digest(path),zones=conditions['zones']))
        for number,seed in enumerate(b.SEEDS[:1] if smoke else b.SEEDS,1):
            order=['rl17_repair','r20_budget'] if smoke else list(CONTROLLERS)
            if number==2:order.reverse()
            for c in order:
                schedule.append(dict(map_id=row['id'],name=row['name'],family=row['family'],map_file=row['map_path'],
                    map='R19'+row['geometry_sha256'][:12],official_map=None,scenario=conditions,round=number,seed=seed,controller=c))
    sources=b.source_files(rows)
    plan=dict(schema='red-explore20-v1',created_utc=datetime.now(timezone.utc).isoformat(),
              head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.REPO,text=True).strip(),
              configuration=dict(smoke=smoke,diagnostic_steps=128 if smoke else None,timeout=240,memory_bytes=4_000_000_000,
                cpu_affinity_count=1,simultaneous_simulations=1,controllers=list(CONTROLLERS),
                controller_identities={c:b.runner.ev.controller_identity(c) for c in CONTROLLERS},
                engine_profile=b.engine_profile(),environment=b.environment(),blue_evaluated=False,full_score=None),
              selection=selection,map_rows=rows,schedule=schedule,source_sha256=sources,
              intervention='RN7 eight-episode allowance can renew twice, after ticks1200/2400; no other policy change',
              primary_contrast='r20_budget minus rl17_repair',reference_contrast='rl17_repair minus rl11_lines',
              limitations=['Training maps only, all contain both static special zones. No independent test-set inference.',
                           'Same seeds do not imply same candidate trajectories or sensors.',
                           'Visibility sampled every8 steps on4px cells is an estimate, not exact coverage.'])
    output.mkdir(parents=True)
    b.write(output/'source_bytes.json',{k:base64.b64encode((b.REPO/k).read_bytes()).decode('ascii') for k in sources})
    b.runner.validate_sources(sources)
    plan['source_bytes_sha256']=b.digest(output/'source_bytes.json');b.write(output/'plan.json',plan)
    b.write(output/'prepared.json',dict(plan_sha256=b.digest(output/'plan.json'),runs=len(schedule),executed=0))
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--prepared',type=Path)
    p.add_argument('--worker',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--run',action='store_true')
    a=p.parse_args();register()
    if a.worker:
        req=b.read(a.worker)
        with recording(Path(req['run_dir'])/'explore20_diagnostics.json'):return b.worker(a.worker)
    if not a.output:p.error('--output required')
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a fresh output; failed attempts are retained: '+str(out))
    if a.prepared:
        source=a.prepared.resolve();plan=b.read(source/'plan.json');b.validate_plan(source,plan)
        if a.smoke!=plan['configuration']['smoke']:p.error('Smoke flag differs from frozen plan')
        out.mkdir(parents=True)
        for name in ('plan.json','prepared.json','source_bytes.json'):
            with (out/name).open('xb') as f:f.write((source/name).read_bytes())
    else:
        if a.run and not a.smoke:p.error('Full batch requires --prepared')
        plan=prepare(out,a.smoke)
    if a.run:return b.execute(out,plan)
    print('PREPARED ONLY',out,'runs',len(plan['schedule']),'executed 0',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
