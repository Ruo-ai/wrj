"""B8 independent adapter. Default: blue loader check only, no freeze or games."""
import argparse
import json
import os
from pathlib import Path
import sys

MODULE='swarm_rescue.solutions.devtools.blue_nav_b8_run'
EXTRA={key:('my_drone_blue_nav_b8',name) for key,name in (
    ('b8_empty','MyDroneBlueB8Empty'),('b8_payload','MyDroneBlueB8Payload'),('b8_off','MyDroneBlueB8Off'))}


def register(extended=False):
    from .optimization.evaluation import identities
    identities.BLUE_CONTROLLERS={**identities.BLUE_CONTROLLERS,**EXTRA}
    if extended:
        from . import blue_lifecycle
        from .blue_nav_b8_diagnostics import instrument,Observer
        blue_lifecycle.instrument=instrument;blue_lifecycle.LifecycleObserver=Observer


def worker_command(request):
    argv=[MODULE,'--worker',str(request)]
    bootstrap=(f'import runpy,sys; sys.path[:]={sys.path!r}; sys.prefix={sys.prefix!r}; '
               f'sys.exec_prefix={sys.exec_prefix!r}; sys.argv={argv!r}; runpy.run_module({MODULE!r},run_name="__main__")')
    return [sys.executable,'-B','-u','-S','-X','utf8','-X','faulthandler','-c',bootstrap]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--red-idle-confirmed',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--maps',nargs='+',default=['Map08']);p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--variants',nargs='+',default=['adapted','b8_empty','b8_payload'])
    a=p.parse_args()
    if a.worker:
        if sys.stdin.readline().strip()!='GO':raise SystemExit('No resource startup handshake')
        if os.name!='nt':
            from .optimization.evaluation.resources import linux_limit
            linux_limit(int(os.environ['G3_MEMORY_BYTES']))
        register(True)
        from .optimization.evaluation.worker import run
        raise SystemExit(run(a.worker))
    if (a.prepare or a.run) and not a.red_idle_confirmed:
        p.error('Freeze/run only after red development and evaluation finish: --red-idle-confirmed required')
    if a.rounds<1 or len(set(a.variants))!=len(a.variants):p.error('Positive rounds and unique variants required')
    if not set(a.variants)<={'adapted',*EXTRA}:p.error('Use the B0/B8 independent selectors only')
    register()
    from .optimization.evaluation.identities import load_blue
    classes={v:load_blue(v) for v in a.variants}
    if not(a.prepare or a.run):
        print(json.dumps({v:c.__module__+'.'+c.__name__ for v,c in classes.items()}))
        print('No simulation, output creation or red source freeze.');return
    if a.output is None:p.error('--output required')
    from .optimization.evaluation import runner,blue_transport_batch as bt
    from .optimization.map_generator.append_maps import append_lock
    from .blue_b6_run_20261113 import valid_result,write_atlas
    from .blue_nav_b8_report import summarize,write_report
    root=a.output.resolve()
    if root==runner.REPO/'output' or not root.is_relative_to(runner.REPO/'output'):p.error('Use a new output subdirectory')
    # No resume into an existing result folder, even this module's own old batch.
    root.mkdir(parents=True,exist_ok=False)
    runner.write(root/'b8_scope.json',dict(maps=a.maps,rounds=a.rounds,variants=a.variants,baseline='adapted',relay=False))
    plans={};rows=[]
    with append_lock(root):
        for v in a.variants:plans[v]=bt.prepare(root/v,bt.LAYOUTS,a.maps,v,a.rounds,False,'b8_navigation')
        receipt=plans[a.variants[0]]['source_sha256']
        if any(x['source_sha256']!=receipt for x in plans.values()):raise ValueError('Sources changed during freeze; use a new batch')
        runner.validate_sources(receipt)
        if not a.run:
            print('Prepared',root,'blue jobs',len(a.maps)*a.rounds*len(a.variants),'red jobs 0');return
        runner.worker_command=worker_command
        for index,first in enumerate(plans[a.variants[0]]['schedule']):
            order=a.variants if first['round']%2 else list(reversed(a.variants))
            for v in order:
                plan=plans[v];item=plan['schedule'][index];runner.validate_sources(plan['source_sha256'])
                runner.write(root/'progress.json',dict(active=item,completed=len(rows)))
                print(item['name'],item['round'],v,'seed',item['seed'],flush=True)
                result,attempt=runner.execute_phase(root/v,plan,item,'blue')
                rows.append(dict(item=item,result=result,run=str(attempt),valid_blue=valid_result(result)))
                write_atlas(root,rows,bt.LAYOUTS)
                report=root/'report.md';report.write_text(report.read_text(encoding='utf-8').replace('# B6 蓝队','# B8 蓝队',1),encoding='utf-8')
                write_report([summarize(Path(row['run'])) for row in rows],root)
                print('valid',rows[-1]['valid_blue'],'score',result.get('blue_score'),'disposed',result.get('disposed_count'),flush=True)
        runner.write(root/'progress.json',dict(active=None,completed=len(rows),complete=True))
    print('Complete',root)


if __name__=='__main__':main()
