"""B7 isolated adapter. Default only checks blue loaders, without freezing red.

--prepare / --run require --red-idle-confirmed from the person starting the job.
Never use that flag while the other red development/evaluation is still active.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

MODULE='swarm_rescue.solutions.devtools.blue_relay_b7_run'
EXTRA={k:('my_drone_blue_relay_b7',v) for k,v in (
    ('relay_decision','MyDroneBlueRelayDecision'),('relay_confirmed','MyDroneBlueRelayConfirmed'),('relay_off','MyDroneBlueRelayOff'))}


def register(extended=False):
    from .optimization.evaluation import identities
    identities.BLUE_CONTROLLERS={**identities.BLUE_CONTROLLERS,**EXTRA}
    if extended:
        from . import blue_lifecycle
        from .blue_relay_b7_diagnostics import instrument,Observer
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
    p.add_argument('--maps',nargs='+',default=['Map08']);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--variants',nargs='+',default=['adapted','relay_decision','relay_confirmed'])
    a=p.parse_args()
    if a.worker:
        if sys.stdin.readline().strip()!='GO':raise SystemExit('No parent startup authorization')
        if os.name!='nt':
            from .optimization.evaluation.resources import linux_limit
            linux_limit(int(os.environ['G3_MEMORY_BYTES']))
        register(True)
        from .optimization.evaluation.worker import run
        raise SystemExit(run(a.worker))
    if (a.prepare or a.run) and not a.red_idle_confirmed:
        p.error('Red is still active or unconfirmed: no source freeze or game. Use --red-idle-confirmed only after red finishes.')
    register()
    from .optimization.evaluation.identities import load_blue
    if a.rounds<1 or len(set(a.variants))!=len(a.variants):p.error('Positive rounds and unique variants required')
    classes={v:load_blue(v) for v in a.variants}
    if not(a.prepare or a.run):
        print(json.dumps({v:dict(module=c.__module__,name=c.__name__) for v,c in classes.items()}))
        print('Blue loaders only. No red source snapshot, output creation or simulation.');return
    if a.output is None:p.error('--output must be a new blue-owned directory')
    from .optimization.evaluation import runner,blue_transport_batch as bt
    from .optimization.map_generator.append_maps import append_lock
    from .blue_b6_run_20261113 import valid_result,write_atlas
    from .blue_relay_b7_report import summarize,write_report
    root=a.output.resolve()
    if root==runner.REPO/'output' or not root.is_relative_to(runner.REPO/'output'):p.error('Output must be a new subdirectory of project output')
    marker=root/'b7_scope.json';scope=dict(owner='blue_relay_b7',maps=a.maps,rounds=a.rounds,variants=a.variants)
    if root.exists():
        if not marker.is_file() or runner.read(marker)!=scope:raise ValueError('Existing output is not this B7 scope; use a new directory')
    else:
        root.mkdir(parents=True,exist_ok=False);runner.write(marker,scope)
    plans={};rows=[]
    with append_lock(root):
        for v in a.variants:plans[v]=bt.prepare(root/v,bt.LAYOUTS,a.maps,v,a.rounds,False,'b7_relay')
        receipt=plans[a.variants[0]]['source_sha256']
        if any(plan['source_sha256']!=receipt for plan in plans.values()):raise ValueError('Sources changed during freezing; use a new batch')
        runner.validate_sources(receipt)
        if not a.run:
            print('Prepared:',root,'blue jobs:',len(a.maps)*a.rounds*len(a.variants),'red jobs: 0');return
        runner.worker_command=worker_command
        for index,first in enumerate(plans[a.variants[0]]['schedule']):
            order=a.variants if first['round']%2 else list(reversed(a.variants))
            for v in order:
                plan=plans[v];item=plan['schedule'][index];runner.validate_sources(plan['source_sha256'])
                runner.write(root/'progress.json',dict(active=item,completed=len(rows),expected=len(a.maps)*a.rounds*len(a.variants)))
                print(item['name'],item['round'],v,'seed',item['seed'],flush=True)
                result,attempt=runner.execute_phase(root/v,plan,item,'blue')
                rows.append(dict(item=item,result=result,run=str(attempt),valid_blue=valid_result(result)))
                write_atlas(root,rows,bt.LAYOUTS)
                # The old renderer writes only this new B7-owned output.
                report=root/'report.md';text=report.read_text(encoding='utf-8')
                report.write_text(text.replace('# B6 蓝队独立对照','# B7 蓝队独立对照',1),encoding='utf-8')
                write_report([summarize(Path(row['run'])) for row in rows],root)
                print('valid',rows[-1]['valid_blue'],'score',result.get('blue_score'),'disposed',result.get('disposed_count'),flush=True)
        runner.write(root/'progress.json',dict(active=None,completed=len(rows),complete=True))
    print('Complete:',root,'core atlas: report.md; discovery/relay detail: relay_report.md')


if __name__=='__main__':main()
