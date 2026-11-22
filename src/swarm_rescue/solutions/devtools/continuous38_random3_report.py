"""Read-only atlas/audit for all three predeclared Continuous38 random seeds."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
os.environ['PYGLET_SHADOW_WINDOW']='false';os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from swarm_rescue.solutions.red_boundary33 import FinalBoundary33

ROOT=Path(__file__).resolve().parents[4]
BATCH=ROOT/'output/sf13/continuous38_random3_20261122';OUT=BATCH/'review'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

helpers=module('continuous38_random_helpers','output/sf13/continuous38_report_20261122.py')
atlas=helpers.atlas;navplot=helpers.navplot;read=atlas.read
REPORT=ROOT/'红队_Continuous38_Map09随机三种子稳定性评测_20261122.md'

def save(fig,path):
    fig.savefig(path,dpi=155,bbox_inches='tight',facecolor='white');plt.close(fig)

def fmt(value,n=5):return '—' if value is None else f'{value:.{n}f}'

def main():
    driver=module('continuous38_random_audit','src/swarm_rescue/solutions/devtools/continuous38_random3.py')
    plan=driver.validate(BATCH);assert read(BATCH/'completion.json')['complete']
    pairs=read(BATCH/'pairs.json');assert len(pairs)==3
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();checks=[];records=[];loaded=[];samples_all=[];curves=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    check('plan, algorithm sources and previous diagnostic attempt unchanged',True)
    check('three seeds equal the preflight random draw',[p['seed'] for p in pairs]==read(BATCH/'seed_draw.json')['accepted_seeds'])
    check('one attempt per seed, no retries',len(list((BATCH/'runs').glob('**/attempt_*')))==3)
    cutoffs=[]
    for p in pairs:
        diag=read(Path(p['red_run'])/'dualmap32_diagnostics.json')
        cutoffs.append(min(d['_f27_stats']['cutoff_elapsed'] for d in diag['drones'].values()))
    common=min(cutoffs)
    for number,p in enumerate(pairs,1):
        run=Path(p['red_run']);m,d=atlas.inspect(p,BATCH);loaded.append((m,d))
        m['version']=f"Continuous38 · 第{number}组 · seed {p['seed']}"
        nav=navplot.load(p);rows,alignment=helpers.observations(p,d,common);samples_all.append(rows)
        covered,curve=helpers.h.coverage(d,common);curves.append(curve)
        outage=np.concatenate([a[a[:,1]>0] for a in rows.values()])
        identity=read(run/'loaded_identity.json');req=read(run/'request.json');native=read(run/'result.json')
        integrity=read(run/'integrity.json');resource=read(run/'resources.json')
        check(f'{number}: saved artifact hashes',all(atlas.sha(run/f)==v for f,v in integrity['files'].items()))
        check(f'{number}: actual loaded sources frozen',all(plan['source_sha256'].get(f)==v for f,v in identity['loaded_sources'].items()))
        check(f'{number}: five Continuous38 instances',len(identity['instances'])==5 and all(v.endswith('.MyDroneRedContinuous38') for v in identity['instances'].values()))
        check(f'{number}: native zones and fixed resources',req['zones']==['NO_COM_ZONE','NO_GPS_ZONE'] and resource['cpu_count']==1 and resource['memory_limit_bytes']==4000000000)
        check(f'{number}: actual submitted pixels/native scoring audit',not m['audit_issues'])
        cd=read(run/'continuous38_diagnostics.json')
        check(f'{number}: passive observer errors absent',not cd['errors'])
        agreement={k:max(s[k] for s in cd['samples']) for k in ('shared_position_difference','shared_heading_difference','stopturn_heading_difference')}
        check(f'{number}: shared pose agreement',max(agreement.values())==0.)
        receipt=nav['receipt'];versions=receipt['received_versions'];expected=receipt['epoch_elapsed']-1
        final_versions={k:v['expected_final_sequence'] for k,v in nav['source_metadata'].items()}
        fresh=[k for k,v in versions.items() if v==final_versions[k]]
        record=dict(round=number,seed=p['seed'],valid_red=m['valid_red'],map_score=m['map_score'],
            wall_score=m['wall_score'],time_score=m['time_score'],blue_score=None,total_score=None,
            wall_hits=m['error_metric']['wall_hits'],wall_misses=m['error_metric']['wall_misses'],penalty=m['error_metric']['penalty'],
            charged_pixels=native['phase1']['map_breakdown']['charged_pixels'],bombs=m['placed_count'],steps=m['red_steps'],seconds=m['red_seconds'],
            freeze_elapsed=cutoffs[number-1],common_cutoff_elapsed=common,coverage=covered,outage=helpers.stats(outage),
            per_drone={str(i):helpers.stats(a[a[:,1]>0]) for i,a in rows.items()},
            received_versions=versions,expected_source_final_versions=final_versions,latest_sources=len(fresh),
            missing_final_sources=[k for k in versions if k not in fresh],acknowledged_sources=receipt['acknowledged_sources'],
            acknowledgements=len(receipt['acknowledged_sources']),boundary_added_pixels=m['boundary']['added_pixels'],
            boundary_inferred=m['boundary']['inferred'],alignment=alignment,shared_pose_agreement=agreement,
            red_issues=m['red_issues'],audit_issues=m['audit_issues'],red_run=run.as_posix(),submission_sha256=m['actual_submission_sha256'])
        # Diagnose delivery loss without inventing observations or another flight.
        # Each replacement is that source's actually sealed pre-deadline payload.
        raw=read(run/'dualmap32_diagnostics.json');own_masks={};own_receipts={}
        coarse=((d['shape'][0]+2)//3,(d['shape'][1]+2)//3);count=int(np.prod(coarse));length=(count+7)//8
        with np.load(run/'dualmap32_snapshots.npz',allow_pickle=False) as tape:
            for i in range(5):
                k=str(i);meta=raw['snapshot_payloads'][k][k];blob=tape[meta['array']].tobytes()
                check(f'{number}: source {i} own frozen payload valid',meta['own_sealed'] and meta['sequence']==final_versions[k] and hashlib.sha256(blob).hexdigest()==meta['payload_sha256'])
                own_masks[i]=np.unpackbits(np.frombuffer(blob[length:],np.uint8))[:count].reshape(coarse).astype(bool)
                own_receipts[k]=dict(sequence=meta['sequence'],sha256=meta['payload_sha256'],epoch=meta['final_epoch'])
        renderer=FinalBoundary33(d['shape'][::-1])
        rerender=renderer.fuse({int(k):v[::3,::3] for k,v in nav['sources'].items()})
        check(f'{number}: same received inputs reproduce submitted pixels',np.array_equal(rerender,d['pred']))
        hypothetical=renderer.fuse(own_masks)
        score=atlas.MapExplorationScorer().score(d['truth'],hypothetical,m['red_seconds'])
        record['delivery_counterfactual']=dict(scope='Offline replacement with each source actual sealed final payload, same renderer; not a new flight or valid measured score',
            hypothetical_map_score=100*float(score.s_map),actual_received_map_score=m['map_score'],
            delta=None if m['map_score'] is None else 100*float(score.s_map)-m['map_score'],own_source_receipts=own_receipts)
        records.append(record)
        folder=OUT/f"r{number:02d}_s{p['seed']}";folder.mkdir()
        atlas.write(folder/'data.json',record)
        np.savez_compressed(folder/'drift_samples.npz',**{f'drone_{i}':a for i,a in rows.items()})
        fig,axs=plt.subplots(1,3,figsize=(21,6.8))
        atlas.draw_submission(axs[0],m,d);atlas.draw_bombs(axs[1],m,d);atlas.draw_tracks(axs[2],m,d,'red')
        fig.suptitle(f"Map09 · 第{number}组 · 随机种子 {p['seed']} · 官方特殊区域开启\n实收最新源图 {len(fresh)}/5 · 源集合ACK {record['acknowledgements']}/5",fontsize=16)
        fig.subplots_adjust(top=.75,bottom=.08,left=.01,right=.99,wspace=.05)
        fig.legend([Line2D([0],[0],color=atlas.COLORS[i]) for i in range(5)],[f'无人机{i}' for i in range(5)],loc='lower center',ncol=5,frameon=False)
        save(fig,folder/'submission_bombs_trajectory.png')
        fig,axs=plt.subplots(1,2,figsize=(16,7))
        atlas.draw_errors(axs[0],m,d);axs[1].imshow(Image.open(d['native_wall']));axs[1].set_axis_off()
        axs[1].set_title(f"原生墙体分 {fmt(m['wall_score'])}\n原始覆盖率 {native['phase1']['wall_breakdown']['coverage_fraction']:.2%}",fontsize=12)
        fig.suptitle(f"Map09 · seed {p['seed']} · 地图误差与实际炸弹墙体覆盖\n误差：绿命中／蓝漏画／红受罚／黄容差；右图红色为爆炸覆盖范围",fontsize=14)
        fig.tight_layout(rect=(0,0,1,.90));save(fig,folder/'errors_wall.png')
        fig,axs=plt.subplots(1,5,figsize=(23,5.4))
        for i in range(5):navplot.draw(axs[i],navplot.nav_colors(nav,str(i)),f'无人机{i}',d['shape'])
        fig.suptitle(f"Map09 · seed {p['seed']} · 各机赛末原导航图\n灰未知／白自由／黑占据；20px规划格；保留Clean清噪，无最终提交渲染和补边",fontsize=15)
        fig.subplots_adjust(top=.76,bottom=.04,left=.01,right=.99,wspace=.04);save(fig,folder/'navigation_all5.png')
        fig,axs=plt.subplots(1,5,figsize=(23,5.4))
        for i in range(5):
            k=str(i);navplot.draw(axs[i],nav['sources'][k],f'{i}号实际收到源图\n版本{versions[k]}／终版{final_versions[k]}',d['shape'],True)
        fig.suptitle(f"Map09 · seed {p['seed']} · 实际提交所用五源地图（融合前）\n只显示收到的字节；不补入未交付终图，白色不证明已观测自由",fontsize=15)
        fig.subplots_adjust(top=.76,bottom=.04,left=.01,right=.99,wspace=.04);save(fig,folder/'received_sources.png')
        Image.fromarray(np.uint8(~d['pred'])*255).save(folder/'actual_submitted_pixels.png')
        Image.fromarray(np.uint8(~hypothetical)*255).save(folder/'offline_all_final_sources_NOT_SUBMITTED.png')

    scores=[r['map_score'] for r in records if r['valid_red']]
    statistics=dict(valid_count=len(scores),mean=float(np.mean(scores)) if scores else None,
        minimum=float(np.min(scores)) if scores else None,maximum=float(np.max(scores)) if scores else None,
        population_std=float(np.std(scores)) if scores else None,range=float(np.ptp(scores)) if scores else None,
        excludes_prior_diagnostic_seed=True)
    summary=dict(map='Map09',controller='Continuous38',new_red_flights=3,new_blue_flights=0,retries=0,
        seeds=read(BATCH/'seed_draw.json')['accepted_seeds'],statistics=statistics,metrics=records,
        common_cutoff_elapsed=common,scope='Three newly drawn seeds, one unchanged controller, no new baseline arms or blue phases',
        caveat='Three samples describe this batch; not proof of general stability. Previous 75.05072 diagnostic score excluded.')
    atlas.write(OUT/'data.json',summary);atlas.write(BATCH/'audit.json',dict(passed=True,count=len(checks),checks=checks))
    fig,axs=plt.subplots(2,2,figsize=(15,9))
    for ax,key,title in zip(axs.flat,('all','north','north_deep','northwest_deep'),('全图','北侧','北侧深处','西北深处')):
        for i,c in enumerate(curves):ax.plot(np.arange(len(c[key]))+2,c[key],label=f"第{i+1}组／{records[i]['seed']}")
        ax.set(title=title,xlabel='原生步数',ylabel='中心到访40px格数');ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.suptitle(f'三随机种子到访曲线 · 统一到原生第{common}步 · 非雷达视野覆盖率',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,OUT/'coverage.png')
    fig,axs=plt.subplots(3,2,figsize=(15,12),sharex=True,sharey='col')
    for row,samples in enumerate(samples_all):
        for i,a in samples.items():
            part=a[a[:,1]>0]
            for col,index in enumerate((2,3)):axs[row,col].scatter(part[:,0],abs(part[:,index]),s=3,alpha=.45,color=atlas.COLORS[i],label=f'{i}号')
        for col in range(2):
            axs[row,col].set(title=f"第{row+1}组 · seed {records[row]['seed']}",xlabel='控制步数',ylabel='位置误差 / px' if col==0 else '航向误差 / 度')
            axs[row,col].legend(ncol=5,fontsize=8,markerscale=3);axs[row,col].grid(alpha=.2)
    fig.suptitle('三组真实飞行漂移 · 封存前每10步采样，仅GPS/罗盘失效期间\n不同种子的路径/失定位时长不同；导航与建图共享位姿',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,OUT/'drift.png')
    write_report(summary,checks)
    print(json.dumps(dict(statistics=statistics,metrics=[{k:r[k] for k in ('round','seed','valid_red','map_score','bombs','latest_sources','acknowledgements','seconds')} for r in records],audit_checks=len(checks),report=str(REPORT)),ensure_ascii=False,indent=2))

def write_report(summary,checks):
    rows=summary['metrics'];s=summary['statistics']
    text=['# Continuous38：Map09随机三种子稳定性评测','',
        '按用户要求，使用上一轮冻结的Continuous38原代码，在Map09新抽取3个随机种子，各新执行一次完整红阶段。恢复并保持官方无GPS/无罗盘、无通信区域；沿用原生CPU雷达、1CPU、4GB、5400步/180秒上限。没有修改算法、没有重试或换种子。',
        '',f"种子为 **{', '.join(str(v) for v in summary['seeds'])}**，由系统随机源 `secrets.randbits(32)` 产生，在任何飞行前写入并冻结；与地图哈希和日期无关。抽样只排除重复及上一诊断种子1492628140。",'',
        f"新红3场、新蓝0场。有效红阶段 **{s['valid_count']}/3**；有效地图分均值 **{fmt(s['mean'])}**，最低 **{fmt(s['minimum'])}**，最高 **{fmt(s['maximum'])}**，极差 **{fmt(s['range'])}**，总体标准差 **{fmt(s['population_std'])}**。上一轮75.05072是单独诊断样本，没有混入这些统计。三个样本不能证明所有种子稳定。",'',
        '**红阶段实际分项；未评测蓝队和完整总分**','',
        '|组|随机种子|有效地图分|墙体分|时间分|实际布弹|步数|耗时秒|最新源图|集合ACK|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"|{r['round']}|{r['seed']}|{fmt(r['map_score'])}|{fmt(r['wall_score'])}|{fmt(r['time_score'])}|{r['bombs']}/10|{r['steps']}|{r['seconds']:.2f}|{r['latest_sources']}/5|{r['acknowledgements']}/5|")
    text+=['','红阶段原生partial总分不作为完整总分，蓝分和难度终分均未测。每组提交图来自实际提交文件；布弹为实际落点；轨迹来自原生录像。ACK是算法源图集合确认，不是另一个官方分数条件。','',
        '**探索、误画与漂移**','',f"统一统计到原生第{summary['common_cutoff_elapsed']}步。到访是40px格内机身中心到达，不是雷达覆盖率；北侧深处取图像y<320，西北深处另要求x<622，仅为赛后分区。漂移使用封存前每10步、GPS/罗盘失效时刻，与控制后/物理前的真值对齐。",'',
        '|组|真墙命中|误画扣分|全图到访格|北侧深处格|西北深处格|航向P95|位置P95|补边新增像素|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"|{r['round']}|{r['wall_hits']:,}|{r['penalty']:.5f}|{r['coverage']['all']['cells40']}|{r['coverage']['north_deep']['cells40']}|{r['coverage']['northwest_deep']['cells40']}|{fmt(r['outage']['navigation_heading']['p95'],2)}°|{fmt(r['outage']['navigation_position']['p95'],2)}px|{r['boundary_added_pixels']:,}|")
    text+=['','**交付损失的离线检查（不计入实测成绩）**','',
        '使用每架机实际在截止前封存的源图，离线替换未交付旧版，再用相同最终渲染和补边生成诊断图。先验证实收源图重放能逐像素复现实际提交；没有增加观测，没有新飞行。下表只帮助判断交付缺口的影响，不能把假设分数当实际成绩或理论上界。','',
        '|组|实际地图分|假设全部封存源图到达的离线地图分|离线变化|','|---|---:|---:|---:|']
    for r in rows:
        c=r['delivery_counterfactual'];text.append(f"|{r['round']}|{fmt(r['map_score'])}|{fmt(c['hypothetical_map_score'])}|{fmt(c['delta'])}|")
    text+=['',f"![三组到访曲线]({(OUT/'coverage.png').as_posix()})",'',f"![三组漂移]({(OUT/'drift.png').as_posix()})",'',
        '**逐组图示与实际交付版本**','',
        '导航图是赛末20px局部规划图，保留Clean日常清噪、没有最终提交规整和补边；粗格不是最终墙体厚度。源图只显示提交者收到的版本，不补入失联飞机尚未交付的数据。']
    for r in rows:
        folder=OUT/f"r{r['round']:02d}_s{r['seed']}"
        versions='，'.join(f"{i}号={r['received_versions'][str(i)]}（终版{r['expected_source_final_versions'][str(i)]}）" for i in range(5))
        text+=['',f"**第{r['round']}组：种子{r['seed']}**",'',
            f"有效红阶段：{r['valid_red']}；实际实收：{versions}。未收到终图的来源：{r['missing_final_sources']}。",'',
            f"![提交图、炸弹图、轨迹图]({(folder/'submission_bombs_trajectory.png').as_posix()})",'',
            f"![地图误差与原生墙体覆盖]({(folder/'errors_wall.png').as_posix()})",'',
            f"![五机导航图]({(folder/'navigation_all5.png').as_posix()})",'',
            f"![实际收到的五源地图]({(folder/'received_sources.png').as_posix()})",'',
            f"[本组原始数据]({(folder/'data.json').as_posix()})"]
        if r['red_issues']:text+=['',f"本组无效原因：{r['red_issues']}。不把原始诊断分作为有效成绩。"]
    text+=['','**冻结与核验**','',
        f"算法及旧实测源码身份保持不变。{len(checks)}项产物检查通过，覆盖随机抽样冻结、全部已存文件哈希、真实类身份、资源/区域、实际提交像素、共享位姿及日志。没有额外飞行，也没有用这三场结果调代码。",'',
        f"[冻结计划]({(BATCH/'plan.json').as_posix()}) · [随机抽样原记录]({(BATCH/'seed_draw.json').as_posix()}) · [完成回执]({(BATCH/'completion.json').as_posix()}) · [核验]({(BATCH/'audit.json').as_posix()}) · [完整数据]({(OUT/'data.json').as_posix()})",'',
        '以下仅回放已保存的红队录像，不新增仿真；本轮无蓝阶段录像。','','```powershell',
        "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'","$env:PYTHONPATH='src'","$env:PYGLET_SHADOW_WINDOW='false'","$env:PYTHONIOENCODING='utf-8'"]
    for r in rows:text.append(f".venv/Scripts/python.exe -m swarm_rescue.solutions.devtools.optimization.evaluation.replay '{r['red_run']}'")
    text+=['```','']
    REPORT.write_text('\n'.join(text),encoding='utf-8')

if __name__=='__main__':main()
