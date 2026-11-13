"""B6 blue-only adapter. Default: freeze plans, never start a simulation.

Registry and diagnostic replacement exist only in this dedicated process.
Existing entry/registry/runner/engine files are never edited. Use --run explicitly
after other evaluations finish; a fresh short output path is recommended.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

MODULE='swarm_rescue.solutions.devtools.blue_b6_run_20261113'
EXTRA={name:('my_drone_blue_b6_local_20261113',cls) for name,cls in (
    ('b6_local','MyDroneBlueB6Local'),('b6_private','MyDroneBlueB6Private'),
    ('b6_off','MyDroneBlueB6Off'),('b6_portal','MyDroneBlueB6Portal'))}


def register_local(extended=False):
    from .optimization.evaluation import identities
    identities.BLUE_CONTROLLERS={**identities.BLUE_CONTROLLERS,**EXTRA}
    if extended:
        from . import blue_lifecycle
        from .blue_b6_diagnostics_20261113 import instrument,Observer
        blue_lifecycle.instrument=instrument;blue_lifecycle.LifecycleObserver=Observer


def worker_command(request):
    argv=[MODULE,'--worker',str(request)]
    bootstrap=(f'import runpy,sys; sys.path[:]={sys.path!r}; sys.prefix={sys.prefix!r}; '
               f'sys.exec_prefix={sys.exec_prefix!r}; sys.argv={argv!r}; '
               f'runpy.run_module({MODULE!r},run_name="__main__")')
    return [sys.executable,'-B','-u','-S','-X','utf8','-X','faulthandler','-c',bootstrap]


def valid_result(result):
    from .optimization.evaluation.joint_batch import oracle_valid
    return bool(oracle_valid(result) and not result.get('smoke'))


def write_atlas(output,rows,layouts,historical=False):
    """Reuse existing geometry/coverage rendering; include failed rows safely."""
    from .optimization.evaluation import runner,blue_transport_batch as bt
    from .blue_narrow_report import coverage_figure
    read=runner.read
    lines=['# B6 蓝队独立对照（所有已结束局）','',
        '本图册是历史B5结果的重新分析，没有运行B6完整局。' if historical else '本图册只统计此新批次实际结束的对局。',
        '固定离线强布局；在线控制器未知地图与弹位。0 红阶段：红地图分/墙分/时间分/难度分/原始与有效总分及红种子均不适用。',
        '所有失败、超时、异常保留；原始蓝分不等于有效分。单CPU/4GB/5400步/180秒；诊断开销计入墙钟。',
        'Python及旧NumPy全局随机源固定；Playground独立default_rng(None)仍未被覆盖，种子不保证逐位复现。',
        '碰撞回调计数不是独立接触次数；稀疏位姿与局部地图不能证明每步可达性。','',
        '|地图/轮/版本|蓝种子|原始/有效蓝分|处置/发现|抓取/脱手事件|健康/蓝时间分|步数/墙钟秒|有效|',
        '|---|---|---|---|---|---|---|---|']
    for row in rows:
        item,r=row['item'],row['result'];a=Path(row['run'])
        t=read(a/'telemetry.json') if (a/'telemetry.json').exists() else {}
        l=t.get('lifecycle',{});transport=t.get('transport',{})
        lines.append(f"|{item['name']}/r{item['round']}/{item['blue_variant']}|{item['seed']}|"
            f"{r.get('blue_score')}/{r.get('blue_score') if row['valid_blue'] else '无有效分'}|"
            f"{r.get('disposed_count')}/{l.get('discovered')}|{transport.get('pickups')}/{transport.get('unconfirmed_losses')}|"
            f"{r.get('blue_health_score')}/{r.get('blue_time_score')}|{r.get('elapsed_steps')}/{r.get('elapsed_walltime')}|{row['valid_blue']}|")
    for row in rows:
        item,r=row['item'],row['result'];a=Path(row['run']);folder=output/item['blue_variant']
        t=read(a/'telemetry.json') if (a/'telemetry.json').exists() else {};l=t.get('lifecycle',{})
        extra=l.get('b6_diagnostics',{});gates={};poses={}
        for f in l.get('frames',[]):
            if f.get('narrow'):gates[str(f['drone'])]=f['narrow']
            if f.get('b6_pose'):poses[str(f['drone'])]=f['b6_pose']
        aggregate=Counter()
        for g in gates.values():aggregate.update(g['stats'])
        resources=read(a/'resources.json') if (a/'resources.json').exists() else {}
        lines+=['',f"## {item['name']} / r{item['round']} / {item['blue_variant']}",'',
            f"实际特殊区：{r.get('actual_conditions')}；种子：{item['seed']}；状态：{r.get('status')}。",
            f"蓝源码SHA256：{r.get('blue_source_sha256')}；布局SHA256：{r.get('layout_sha256')}。",
            f"触限：{r.get('walltime_limited')}；控制器异常：{r.get('controller_crashed')}；回调异常：{r.get('engine_callback_errors')}；退出码：{r.get('exit_code')}。",
            f"资源（进程时间含启动）：{resources}。",
            f"窄路末次采样累计：{dict(aggregate)}（控制器自判，需回放确认）；局部定位末次状态：{poses}。",
            f"新增诊断秒：{sum(v.get('extra_seconds',0) for v in extra.get('drones',{}).values()):.6f}；原始快照数：{sum(len(v.get('snapshots',[])) for v in extra.get('drones',{}).values())}。",
            '逐弹时刻来自既有离线生命周期观察器；无观察记录填空，不推测。','',
            '|弹|发现/认领/接近|抓取机@步|中心附近|释放/脱手次数|官方处置|',
            '|---|---|---|---|---|---|---|']
        for b in l.get('bombs',[]):
            lines.append(f"|{b['bomb']}|{b['first_seen']}/{b['first_claim']}/{b['first_near']}|"
                f"{','.join(str(e['drone'])+'@'+str(e['step']) for e in b['pickups']) or '无'}|"
                f"{b['first_center_near']}|{len(b['releases'])}/{len(b['losses'])}|{b['disposed']}|")
        if t and not r.get('smoke'):
            stem=f"{item['name']}_{item['blue_variant']}_r{item['round']:02d}"
            trajectory=folder/'figures'/(stem+'_trajectory.png')
            if not trajectory.exists():bt.figures(folder,item,a,layouts)
            coverage=folder/'figures'/(stem+'_coverage.png')
            if not coverage.exists():coverage_figure(t,coverage)
            for title,path in (('探索轨迹/估计对比',trajectory),('机身经过/雷达自由/语义机会',coverage),
                ('墙覆盖：历史固定输入，非新红队成绩',layouts/(item['name']+'.png')),
                ('实际初始弹位',folder/'figures'/(stem+'_positions.png'))):
                if path.exists():lines+=['',title,'',f'![{title}](<{path.resolve().as_posix()}>)','']
        else:lines+=['','本局无可用图件，保留失败/诊断记录，不补造图片。','']
        lines+=['','红回放：不适用，未运行红队。蓝回放（有录像时）：','```powershell',
                f"Set-Location -LiteralPath '{runner.REPO}'",f".\\replay_map.cmd '{a}'",'```','']
    runner.write(output/'results.json',rows)
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--report-existing',type=Path,help='Read-only historical reanalysis to a NEW output; never runs games')
    p.add_argument('--maps',nargs='+',default=['Map08'])
    p.add_argument('--variants',nargs='+',default=['adapted','b6_private','b6_local'])
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--diagnostics',choices=['off','lifecycle','extended'],default='extended')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.worker:
        if sys.stdin.readline().strip()!='GO':raise SystemExit('Parent did not authorize startup')
        if os.name!='nt':
            from .optimization.evaluation.resources import linux_limit
            linux_limit(int(os.environ['G3_MEMORY_BYTES']))
        import json
        request=json.loads(a.worker.read_text(encoding='utf-8'))
        register_local(request.get('lifecycle_diagnostics')=='b6_extended')
        from .optimization.evaluation.worker import run
        raise SystemExit(run(a.worker))
    if a.output is None:p.error('--output required; old batches never selected automatically')
    if a.rounds<1 or len(set(a.variants))!=len(a.variants):p.error('Positive rounds and unique variants required')
    register_local()
    from .optimization.evaluation import runner,blue_transport_batch as bt,identities
    from .optimization.map_generator.append_maps import append_lock
    if any(v not in identities.BLUE_CONTROLLERS for v in a.variants):p.error('Unknown blue variant')
    root=a.output.resolve()
    if root==runner.REPO/'output' or not root.is_relative_to(runner.REPO/'output'):
        p.error('Use a separate new directory below project output')
    if a.report_existing:
        if a.run:p.error('--report-existing and --run are mutually exclusive')
        root.mkdir(parents=True,exist_ok=False)
        sources=sorted(a.report_existing.glob('*/results.json'))
        if not sources:raise ValueError('No variant/results.json inputs; no games started')
        rows=[];receipt={}
        for source in sources:
            receipt[str(source.resolve())]=runner.file_hash(source)
            for row in runner.read(source):
                row['valid_blue']=valid_result(row['result']);rows.append(row)
                (root/row['item']['blue_variant']).mkdir(exist_ok=True)
                for name in ('result.json','telemetry.json','resources.json','run.log'):
                    path=Path(row['run'])/name
                    if path.exists():receipt[str(path.resolve())]=runner.file_hash(path)
        runner.write(root/'historical_inputs.json',receipt)
        write_atlas(root,rows,bt.LAYOUTS,historical=True)
        print('Historical reanalysis only; no simulation:',root,flush=True)
        return
    scope=dict(owner='blue_b6_20261113',maps=a.maps,variants=a.variants,rounds=a.rounds,diagnostics=a.diagnostics)
    marker=root/'b6_scope.json'
    if root.exists():
        if not marker.is_file() or runner.read(marker)!=scope:
            raise ValueError('Existing directory is not this exact B6 scope; choose a new output')
    else:
        root.mkdir(parents=True,exist_ok=False);runner.write(marker,scope)
    diagnostic={'off':False,'lifecycle':True,'extended':'b6_extended'}[a.diagnostics]
    plans={}
    with append_lock(root):
        for variant in a.variants:
            folder=root/variant
            if folder.exists():
                plan=runner.read(folder/'plan.json')
                if runner.file_hash(folder/'source_snapshot.zip')!=plan['source_snapshot_sha256']:
                    raise ValueError('Source snapshot changed')
                runner.validate_sources(plan['source_sha256'])
            else:plan=bt.prepare(folder,bt.LAYOUTS,a.maps,variant,a.rounds,False,diagnostic)
            plans[variant]=plan
        receipt=plans[a.variants[0]]['source_sha256']
        if any(plan['source_sha256']!=receipt for plan in plans.values()):
            raise ValueError('Sources changed while freezing versions; preserve this batch and choose a new output')
        runner.validate_sources(receipt)
        if not a.run:
            print('Prepared only; no simulation started:',root,flush=True)
            print('Blue tasks:',len(a.maps)*len(a.variants)*a.rounds,'; red tasks: 0',flush=True)
            print('Seeds:',[(i['name'],i['round'],i['seed']) for i in plans[a.variants[0]]['schedule']],flush=True)
            return
        runner.worker_command=worker_command
        rows=[];schedule=plans[a.variants[0]]['schedule']
        for index,first in enumerate(schedule):
            # Interleave versions and reverse on alternate rounds, rather than
            # running all B0 first while machine load changes over time.
            order=a.variants if first['round']%2 else list(reversed(a.variants))
            for variant in order:
                plan=plans[variant];item=plan['schedule'][index]
                runner.validate_sources(plan['source_sha256'])
                runner.write(root/'progress.json',dict(active=item,completed=len(rows),expected=len(schedule)*len(a.variants)))
                print(item['name'],item['round'],variant,'seed',item['seed'],flush=True)
                result,attempt=runner.execute_phase(root/variant,plan,item,'blue')
                rows.append(dict(item=item,result=result,run=str(attempt),valid_blue=valid_result(result)))
                write_atlas(root,rows,bt.LAYOUTS)
                print('raw',result.get('blue_score'),'disposed',result.get('disposed_count'),'valid',rows[-1]['valid_blue'],flush=True)
        runner.write(root/'progress.json',dict(active=None,completed=len(rows),expected=len(schedule)*len(a.variants),complete=True))
    print('Complete:',root,flush=True)


if __name__=='__main__':main()
