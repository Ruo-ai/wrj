"""One-command four-map A0/A2 x B0/B2 matrix, with frozen source checks."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import zipfile
from . import runner
from .runner import REPO,read,write,file_hash,validate_sources,phase_valid
from .oracle_layouts import MAPS,DATASET
from .identities import blue_identity
from .report import save_report
from .scenarios import training_rows,engine_profile
from ..dataset_manager.index import path_name
from ..map_generator.append_maps import append_lock

ROOT=REPO/'output/sf13'  # keep original engine artifact names below Windows MAX_PATH


def arguments(output,blue='adapted',smoke=False):
    return SimpleNamespace(dataset=DATASET,split='train',map_names=MAPS,
        controllers=['advanced2_semifinal'] if smoke else ['advanced2','advanced2_semifinal'],
        rounds=1,seed=742,timeout=240,memory_bytes=4_000_000_000,limit=None,
        smoke=smoke,diagnostic_steps=64,new=True,output=output,blue_variant=blue,retry_failed=False)


def prepare(output,smoke=False):
    if smoke:
        return runner.create_or_resume(arguments(output,'semifinal',True))
    _,reference=runner.create_or_resume(arguments(output/'reference'))
    _,candidate=runner.create_or_resume(arguments(output/'candidate_blue','semifinal'))
    if reference['source_sha256']!=candidate['source_sha256']:
        raise ValueError('Both halves must freeze the same sources')
    manifest=dict(maps=MAPS,rounds=1,seed=742,red=['advanced2','advanced2_semifinal'],
        blue=['adapted','semifinal'],red_phases=8,blue_phases=16,
        source_sha256=reference['source_sha256'],
        reference_plan_sha256=file_hash(output/'reference/plan.json'),
        candidate_plan_sha256=file_hash(output/'candidate_blue/plan.json'),
        design='same exact red layout for both blues; no red rerun in candidate_blue; failed source rows retained',
        adoption='not adopted; single-seed development screen only')
    path=output/'matrix.json'
    if path.exists() and read(path)!=manifest:raise ValueError('Frozen matrix changed; use a new output')
    if not path.exists():write(path,manifest)
    return output,manifest


def run_matrix(output):
    prepare(output)
    reference=output/'reference';candidate=output/'candidate_blue'
    runner.run(arguments(reference))
    plan=read(candidate/'plan.json')
    source={(r['map'],r['controller'],r['round']):r for r in read(reference/'results.json')}
    with append_lock(candidate):
        pairs=[]
        for n,item in enumerate(plan['schedule'],1):
            validate_sources(plan['source_sha256'])
            src=source[(item['name'],item['controller'],item['round'])]
            original=Path(src['red_run'])
            red,red_run=runner.load_done(original.parent,file_hash(reference/'plan.json'))
            if red_run!=original:raise ValueError('Source red attempt changed')
            blue,blue_run=None,None
            print(f"[fixed {n}/8] {item['name']} / {item['controller']} / semifinal",flush=True)
            write(candidate/'progress.json',dict(active=item,completed_pairs=len(pairs),expected_pairs=8))
            if phase_valid(red,'red') and red.get('placed_count',0)>0:
                blue,blue_run=runner.execute_phase(candidate,plan,item,'blue',red_run)
            pairs.append(dict(**item,red=red,blue=blue,red_run=str(red_run),
                              blue_run=str(blue_run) if blue_run else None))
            save_report(candidate,plan,pairs)
        write(candidate/'progress.json',dict(active=None,completed_pairs=len(pairs),expected_pairs=8,complete=True))
    summarize(output)
    from .placement_audit import audit
    audit(reference)


def summarize(output):
    refs=read(output/'reference/results.json');news=read(output/'candidate_blue/results.json')
    old={(r['map'],r['controller'],r['round']):r for r in refs}
    lines=['# 四图固定布局红蓝交叉矩阵','',
           '每份红布局只运行一次，两蓝复用同一布局及种子。红图在candidate_blue图册中复用reference；失败不剔除。',
           '一轮开发筛查不能证明全面最优，也不能替代多种子/留出/平台验收。','',
           '|地图|红布局|B0/B2有效|B0/B2处置|B0/B2蓝分|B2−B0蓝分|','|---|---|---|---|---|---|']
    comparisons=[]
    for new in news:
        prev=old[(new['map'],new['controller'],new['round'])]
        a=read(Path(prev['blue_run'])/'result.json') if prev['blue_run'] else {}
        b=read(Path(new['blue_run'])/'result.json') if new['blue_run'] else {}
        valid=phase_valid(a,'blue') and phase_valid(b,'blue')
        delta=b['blue_score']-a['blue_score'] if valid else None
        comparisons.append(dict(map=new['map'],red=new['controller'],round=new['round'],
            both_blue_valid=valid,delta_blue=delta,disposed_B0=a.get('disposed_count'),disposed_B2=b.get('disposed_count')))
        lines.append(f"|{new['map']}|{new['controller']}|{phase_valid(a,'blue')}/{phase_valid(b,'blue')}|"
            f"{a.get('disposed_count')}/{b.get('disposed_count')}|{a.get('blue_score')}/{b.get('blue_score')}|{delta}|")
    lines += ['',f"[B0固定对手：逐轮分项、资源、三类图、完整回放](<{(output/'reference/report.md').as_posix()}>)",'',
              f"[B2固定对手：逐轮分项、资源、三类图、完整回放](<{(output/'candidate_blue/report.md').as_posix()}>)",'',
              f"[红队逐弹真实新增墙面积与实际落点审计](<{(output/'reference/placement_audit.md').as_posix()}>)",'',
              '采用状态：待分析。正式入口、基线和既有提交包保留。']
    write(output/'blue_comparison.json',comparisons)
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def oracle_plan(output,layouts,variant):
    records=read(layouts/'manifest.json')['maps']
    if [r['name'] for r in records]!=MAPS:raise ValueError('Oracle maps differ from four-map contract')
    rows=training_rows(MAPS,DATASET)
    sources=runner.source_receipt(rows)
    for record in records:
        if file_hash(REPO/record['layout'])!=record['layout_sha256']:raise ValueError('Oracle layout changed')
        sources[record['layout']]=record['layout_sha256']
    sources[path_name(layouts/'manifest.json')]=file_hash(layouts/'manifest.json')
    config=dict(blue_variant=variant,**{k:v for k,v in blue_identity(variant).items() if k!='blue_variant'},
        memory_bytes=4_000_000_000,timeout=240,smoke=False,engine_profile=engine_profile(),
        cpu_affinity_count=1,scope='blue-only oracle pressure; no red score',rounds=1,seed=742)
    schedule=[]
    for record,row in zip(records,rows):
        schedule.append(dict(name=row['name'],map_id=row['id'],family=row['family'],map_file=row['map_path'],
            map=record['map'],official_map=row.get('official_map'),scenario=record['scenario'],round=1,
            seed=(742+int(row['content_sha256'][:8],16))%(2**32),controller='advanced2',
            blue_variant=variant,oracle_layout=record['layout']))
    draft=dict(configuration=config,source_sha256=sources,schedule=schedule)
    if output.exists():
        plan=read(output/'plan.json')
        if any(plan[k]!=v for k,v in draft.items()):raise ValueError('Frozen oracle batch changed')
        if file_hash(output/'source_snapshot.zip')!=plan['source_snapshot_sha256']:raise ValueError('Oracle archive changed')
    else:
        output.mkdir(parents=True)
        with zipfile.ZipFile(output/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in sources:archive.write(REPO/name,name)
        plan=dict(draft,source_snapshot_sha256=file_hash(output/'source_snapshot.zip'))
        write(output/'plan.json',plan)
    return plan


def oracle_valid(r):
    return bool(r.get('status')=='ok' and r.get('exit_code')==0 and not any(
        r.get(k) for k in ('walltime_limited','controller_crashed','engine_callback_errors')) and r.get('blue_score') is not None)


def oracle_run(output,layouts,prepare_only):
    for variant in ('adapted','semifinal'):
        folder=output/('oracle_'+variant);plan=oracle_plan(folder,layouts,variant)
        if prepare_only:continue
        with append_lock(folder):
            results=[]
            for item in plan['schedule']:
                validate_sources(plan['source_sha256'])
                print('Oracle pressure:',item['name'],variant,flush=True)
                result,attempt=runner.execute_phase(folder,plan,item,'blue')
                results.append(dict(item=item,result=result,run=str(attempt),valid_blue=oracle_valid(result)))
                write(folder/'results.json',results)
                oracle_report(folder,results,layouts)


def oracle_report(folder,results,layouts):
    lines=['# 离线强布局蓝队压力测试','',
           '没有红队在线运行，红地图/墙体/时间/难度/总分均不适用，不产生红队有效总分。离线弹图不是最优证明。','']
    for row in results:
        item,r,run=row['item'],row['result'],Path(row['run'])
        t=read(run/'telemetry.json').get('transport',{}) if (run/'telemetry.json').exists() else {}
        resource=read(run/'resources.json')
        lines += [f"## {item['name']} / {item['blue_variant']} / r1",'',
            f"蓝种子{item['seed']}；区域{item['scenario']['zones']}；有效蓝结果{row['valid_blue']}；状态{r.get('status')}；回调异常{r.get('engine_callback_errors')}。",
            f"蓝分{r.get('blue_score')}，处置{r.get('disposed_count')}/10，健康{r.get('blue_health_score')}，步数{r.get('elapsed_steps')}，墙钟秒{r.get('elapsed_walltime')}，峰值Job字节{resource.get('peak_job_commit_bytes')}。",
            f"抓取事件{t.get('pickups')}；持有期间确认处置{t.get('verified_disposals')}；未确认脱手{t.get('unconfirmed_losses')}；成功运输步数{t.get('disposed_transport_steps')}。",
            '红探索/建图图：不适用，未运行红队。下图仅显示离线墙覆盖与人工生成的初始弹位。','',
            f"![离线布局 {item['name']}](<{(layouts/(item['name']+'.png')).as_posix()}>)",'',
            '红回放：不适用。蓝回放：','```powershell',f"Set-Location -LiteralPath '{REPO}'",f".\\replay_map.cmd '{run}'",'```','']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'p2')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--oracle',action='store_true',help='Only 8 extra blue pressure phases; no red')
    parser.add_argument('--layouts',type=Path,default=REPO/'output/semifinal_joint_20261113/oracle_layouts_v2')
    args=parser.parse_args();output=args.output.resolve()
    if args.smoke and args.oracle:parser.error('Smoke and oracle modes are separate')
    if args.oracle:oracle_run(output,args.layouts.resolve(),args.prepare_only)
    elif args.prepare_only:prepare(output,args.smoke)
    elif args.smoke:runner.run(arguments(output,'semifinal',True))
    else:run_matrix(output)
    print('Prepared' if args.prepare_only else 'Completed',output,flush=True)


if __name__=='__main__':main()
