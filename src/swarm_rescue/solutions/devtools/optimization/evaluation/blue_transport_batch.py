"""Explicit fixed strong-layout B0/B3 tests; no red rerun, no online truth."""
import argparse
import importlib.metadata
from pathlib import Path
import zipfile
from . import runner
from .runner import REPO,read,write,file_hash
from .joint_batch import oracle_valid
from .oracle_layouts import MAPS,DATASET
from .scenarios import training_rows,engine_profile
from .identities import BLUE_CONTROLLERS,blue_identity
from ..dataset_manager.index import path_name
from ..map_generator.append_maps import append_lock

LAYOUTS=REPO/'output/semifinal_joint_20261113/oracle_layouts_v2'


def prepare(output,layouts,maps,variant,rounds,diagnostic=False,lifecycle=False):
    manifest=read(layouts/'manifest.json')
    records={r['name']:r for r in manifest['maps']}
    rows=training_rows(maps,DATASET);sources=runner.source_receipt(rows)
    sources[path_name(layouts/'manifest.json')]=file_hash(layouts/'manifest.json')
    schedule=[]
    for row in rows:
        record=records[row['name']]
        if file_hash(REPO/record['layout'])!=record['layout_sha256']:raise ValueError('Frozen layout changed')
        sources[record['layout']]=record['layout_sha256']
        for number in range(1,rounds+1):
            schedule.append(dict(name=row['name'],map_id=row['id'],family=row['family'],
                map_file=row['map_path'],map=record['map'],official_map=row.get('official_map'),
                scenario=record['scenario'],round=number,
                seed=(742+int(row['content_sha256'][:8],16)+(number-1)*1000003)%(2**32),
                controller='advanced2',blue_variant=variant,oracle_layout=record['layout']))
    config=dict(**blue_identity(variant),memory_bytes=4_000_000_000,timeout=240,
        smoke=diagnostic,diagnostic_steps=128 if diagnostic else None,diagnostic_profile=diagnostic,
        cpu_affinity_count=1,lifecycle_diagnostics=lifecycle,scope='128-step profiling, NOT a score' if diagnostic else 'offline fixed strong layout, blue only; red scores N/A',
        environment={**runner.environment(),'scipy':importlib.metadata.version('scipy')},
        rounds=rounds,seed=742,engine_profile=engine_profile())
    draft=dict(configuration=config,source_sha256=sources,schedule=schedule)
    if output.exists():
        plan=read(output/'plan.json')
        if any(plan[k]!=v for k,v in draft.items()):raise ValueError('Frozen batch changed; use a NEW output')
        if file_hash(output/'source_snapshot.zip')!=plan['source_snapshot_sha256']:raise ValueError('Snapshot changed')
    else:
        output.mkdir(parents=True)
        with zipfile.ZipFile(output/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in sources:z.write(REPO/name,name)
        plan=dict(draft,source_snapshot_sha256=file_hash(output/'source_snapshot.zip'))
        write(output/'plan.json',plan)
    return plan


def figures(folder,item,attempt,layouts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    doc=read(REPO/item['map_file']);w,h=doc['canvas']['width'],doc['canvas']['height']
    background=plt.imread((REPO/item['map_file']).with_suffix('.png'))
    out=folder/'figures';out.mkdir(exist_ok=True)
    stem=f"{item['name']}_{item['blue_variant']}_r{item['round']:02d}"
    layout=read(REPO/item['oracle_layout'])
    positions=out/(stem+'_positions.png');trajectory=out/(stem+'_trajectory.png')
    fig,ax=plt.subplots(figsize=(11,7));ax.imshow(background,extent=(0,w,h,0))
    for i,b in enumerate(layout['bombs'],1):
        x,y=b['x']+w/2,h/2-b['y'];ax.plot(x,y,'bo',ms=4);ax.text(x,y,str(i),fontsize=8)
        ax.add_patch(Circle((x,y),round(.12*min(w,h)),fill=False,color='royalblue',alpha=.35))
    ax.set(title=item['name']+' / fixed OFFLINE initial layout; NOT a red run',xlim=(0,w),ylim=(h,0))
    fig.tight_layout();fig.savefig(positions,dpi=110);plt.close(fig)
    samples=read(attempt/'telemetry.json').get('pose_samples',[])
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax in axes:ax.imshow(background,extent=(0,w,h,0))
    for i in sorted({p['drone'] for p in samples}):
        rows=[p for p in samples if p['drone']==i]
        for ax,key in zip(axes,('truth','estimate')):
            ax.plot([p[key][0]+w/2 for p in rows],[h/2-p[key][1] for p in rows],lw=.7,label=str(i))
    axes[0].set_title('OFFLINE sampled true trajectories')
    axes[1].set_title('Controller estimated trajectories')
    for ax in axes:ax.set(xlim=(0,w),ylim=(h,0));ax.legend(fontsize=6,ncol=5)
    fig.tight_layout();fig.savefig(trajectory,dpi=110);plt.close(fig)
    return trajectory,layouts/(item['name']+'.png'),positions


def report(folder,results,layouts):
    lines=['# 固定强布局蓝队运输专项','',
        '离线生成并冻结的10弹布局；在线蓝队未知弹位及地图。未运行红队，红地图/时间/难度/总分均不适用。墙覆盖仅为离线几何指标。',
        '所有失败保留，超时/控制器或回调异常的有效蓝分为空；退出码0不代表有效。','',
        '|地图|版本|轮|原始蓝分|有效蓝分|处置|健康|蓝时间分|步数/秒|有效|',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for row in results:
        i,r=row['item'],row['result']
        lines.append(f"|{i['name']}|{i['blue_variant']}|{i['round']}|{r.get('blue_score')}|"
            f"{r.get('blue_score') if row['valid_blue'] else '空'}|{r.get('disposed_count')}|{r.get('blue_health_score')}|"
            f"{r.get('blue_time_score')}|{r.get('elapsed_steps')}/{r.get('elapsed_walltime')}|{row['valid_blue']}|")
    for row in results:
        item,r,attempt=row['item'],row['result'],Path(row['run'])
        t=read(attempt/'telemetry.json').get('transport',{}) if (attempt/'telemetry.json').exists() else {}
        resource=read(attempt/'resources.json')
        lines += ['',f"## {item['name']} / {item['blue_variant']} / r{item['round']}",'',
            f"红种子：不适用；蓝种子：{item['seed']}；实际区域：{r.get('actual_conditions')}。",
            f"布局SHA256：{r.get('layout_sha256')}；蓝源码SHA256：{r.get('blue_source_sha256')}。",
            f"触限：{r.get('walltime_limited')}；控制器异常：{r.get('controller_crashed')}；回调异常：{r.get('engine_callback_errors')}。",
            f"抓取事件：{t.get('pickups')}；确认处置：{t.get('verified_disposals')}；未确认脱手：{t.get('unconfirmed_losses')}；成功运输步数：{t.get('disposed_transport_steps')}。发现数未记录，不推测。",
            f"峰值Job字节：{resource.get('peak_job_commit_bytes')}；进程秒：{resource.get('process_seconds')}；CPU：{resource.get('cpu_count')}。",'']
        if r.get('smoke'):
            lines+=['128步剖析诊断，出生点一弹。不是强布局正式结果，所有有效成绩与三类正式图不适用。','']
        elif (attempt/'telemetry.json').exists():
            for title,p in zip(('蓝队稀疏轨迹对比（不是红提交地图）','离线墙覆盖（历史固定布局）','实际初始弹位（未运行红队）'),figures(folder,item,attempt,layouts)):
                lines += [title,'',f'![{title}](<{p.resolve().as_posix()}>)','']
        lines += ['红回放：不适用，没有红队运行；蓝回放：','```powershell',
            f"Set-Location -LiteralPath '{REPO}'",f".\\replay_map.cmd '{attempt}'",'```','']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPO/'output/sf13/bt_p1')
    p.add_argument('--layouts',type=Path,default=LAYOUTS)
    p.add_argument('--maps',nargs='+',choices=MAPS,default=MAPS)
    p.add_argument('--variants',nargs='+',choices=BLUE_CONTROLLERS,default=['adapted','transport_compute'])
    p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--lifecycle',action='store_true',help='Passive per-bomb and final-command diagnostics; no per-step printing')
    p.add_argument('--profile-smoke',action='store_true',help='128-step unscored profiler; one map/variant only')
    args=p.parse_args()
    if args.rounds<1:p.error('rounds must be positive')
    if args.profile_smoke and (len(args.maps)!=1 or len(args.variants)!=1 or args.rounds!=1):p.error('Profiler must select one map and one variant')
    prepared=[(v,args.output.resolve()/v,prepare(args.output.resolve()/v,args.layouts.resolve(),args.maps,v,args.rounds,args.profile_smoke,args.lifecycle)) for v in args.variants]
    for variant,folder,plan in prepared:
        if args.prepare_only:continue
        with append_lock(folder):
            results=[]
            for item in plan['schedule']:
                runner.validate_sources(plan['source_sha256'])
                print(f"Blue only: {item['name']} / {variant} / r{item['round']}",flush=True)
                write(folder/'progress.json',dict(completed=len(results),expected=len(plan['schedule']),active=item))
                r,attempt=runner.execute_phase(folder,plan,item,'blue')
                valid=oracle_valid(r) and not r.get('smoke')
                results.append(dict(item=item,result=r,run=str(attempt),valid_blue=valid))
                write(folder/'results.json',results)
                print(f"score={r.get('blue_score')} disposed={r.get('disposed_count')} valid={valid}",flush=True)
            write(folder/'progress.json',dict(completed=len(results),expected=len(plan['schedule']),active=None,complete=True))
        report(folder,results,args.layouts.resolve())
    print('Prepared' if args.prepare_only else 'Complete',args.output,flush=True)


if __name__=='__main__':main()
