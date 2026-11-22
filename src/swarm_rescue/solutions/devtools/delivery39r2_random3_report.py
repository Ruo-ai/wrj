"""Post-run-only figures and byte-level delivery audit for all six fixed flights."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
os.environ['PYGLET_SHADOW_WINDOW']='false'; os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
BATCH=ROOT/'output/sf13/delivery39r2_random3_20261122'; OUT=BATCH/'review'
REPORT=ROOT/'红队_Delivery39R2_Map08_Map09新随机三种子稳定性评测_20261122.md'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


helpers=module('d39r2_random_figures','output/sf13/continuous38_report_20261122.py')
atlas=helpers.atlas;navplot=helpers.navplot;read=atlas.read


def save(fig,path):
    fig.savefig(path,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig)


def fmt(value,n=5):
    return '—' if value is None else f'{value:.{n}f}'


def coverage(data,cutoff,name):
    state=data['red'][:max(0,cutoff-1)]
    h,w=data['shape'];x=state['x']+w/2;y=h/2-state['y']
    live=state['active']&(x>=0)&(x<w)&(y>=0)&(y<h)
    rois=dict(all=live)
    if name=='Map09':
        rois.update(north=live&(y<563),north_deep=live&(y<320),northwest_deep=live&(x<622)&(y<320))
    else:
        rois.update(south_strip=live&(y>=h-100),east_third=live&(x>=2*w/3),southeast=live&(x>=2*w/3)&(y>=h-100))
    counts={};curves={}
    for key,mask in rois.items():
        seen=set();curve=[]
        for t in range(len(state)):
            for i in range(5):
                if mask[t,i]:seen.add((int(x[t,i]//40),int(y[t,i]//40)))
            curve.append(len(seen))
        counts[key]=dict(cells40=len(seen),drone_frames=int(mask.sum()));curves[key]=curve
    return counts,curves


def delivery(diag,receipt,tape):
    sources=[]
    for i in range(5):
        key=str(i);own=diag['snapshot_payloads'][key][key];got=diag['snapshot_payloads']['0'][key]
        actual=tape[got['array']].tobytes();sealed=tape[own['array']].tobytes()
        assert hashlib.sha256(actual).hexdigest()==got['payload_sha256']
        assert hashlib.sha256(sealed).hexdigest()==own['payload_sha256']
        fresh=bool(own['own_sealed'] and receipt['received_versions'][key]==own['sequence']
            and receipt['received_digests'][key]==own['payload_sha256'] and actual==sealed)
        sources.append(dict(drone=i,received=receipt['received_versions'][key],final=own['sequence'],
            epoch=own['final_epoch'],latest=fresh,sealed=own['own_sealed'],
            received_sha256=receipt['received_digests'][key],final_sha256=own['payload_sha256']))
    verified=[]
    for i in receipt['acknowledged_sources']:
        capture=diag['acknowledgement_capture'].get(str(i),{})
        good=True
        for j in range(5):
            own=diag['snapshot_payloads'][str(j)][str(j)]
            got=capture.get('snapshots',{}).get(str(j))
            if (not got or got['sequence']!=own['sequence'] or got['payload_sha256']!=own['payload_sha256']
                or tape[got['array']].tobytes()!=tape[own['array']].tobytes()):
                good=False;break
        if good:verified.append(i)
    return sources,verified


def figures(pair,metric,data,nav,diag,record,folder):
    title=f"{pair['map']} · 第{pair['round']}组 · 新随机种子 {pair['seed']} · 官方特殊区域开启"
    metric['version']='Delivery39R2 · 实际提交'
    fig,axes=plt.subplots(1,3,figsize=(21,7.1))
    atlas.draw_submission(axes[0],metric,data);atlas.draw_bombs(axes[1],metric,data);atlas.draw_tracks(axes[2],metric,data,'red')
    fig.suptitle(title+f"\n最新终图 {record['latest_sources']}/5 · 实收字节核验ACK {record['acknowledgements']}/5",fontsize=16)
    fig.legend([Line2D([0],[0],color=atlas.COLORS[i]) for i in range(5)],[f'无人机{i}' for i in range(5)],loc='lower center',ncol=5,frameon=False)
    fig.subplots_adjust(top=.73,bottom=.06,left=.01,right=.99,wspace=.06);save(fig,folder/'submission_bombs_trajectory.png')
    for kind,draw in [('submission',lambda ax:atlas.draw_submission(ax,metric,data)),('bombs',lambda ax:atlas.draw_bombs(ax,metric,data)),('trajectory',lambda ax:atlas.draw_tracks(ax,metric,data,'red'))]:
        fig,ax=plt.subplots(figsize=(10,7));draw(ax);fig.tight_layout();save(fig,folder/f'{kind}.png')
    Image.fromarray(np.uint8(~data['pred'])*255).save(folder/'actual_submitted_pixels.png')
    fig,axes=plt.subplots(1,5,figsize=(24,6))
    for i in range(5):
        pixels=navplot.nav_colors(nav,str(i));navplot.draw(axes[i],pixels,f'{i}号机 · 赛末导航网格',data['shape'])
        Image.fromarray(pixels).save(folder/f'drone{i}_navigation.png')
    fig.suptitle(title+'\n灰未知、白观测自由、黑占据；20px规划格，未经过最终提交规整和补边',fontsize=15)
    fig.subplots_adjust(top=.70,bottom=.04,left=.01,right=.99,wspace=.04);save(fig,folder/'navigation_all5.png')
    fig,axes=plt.subplots(1,5,figsize=(24,6))
    for i,src in enumerate(record['sources']):
        navplot.draw(axes[i],nav['sources'][str(i)],f"{i}号实收源图\n版本{src['received']} / 终版{src['final']}",data['shape'],True)
    fig.suptitle(title+'\n提交者实际收到的五源占据图；白色不证明已观测自由',fontsize=15)
    fig.subplots_adjust(top=.68,bottom=.04,left=.01,right=.99,wspace=.04);save(fig,folder/'received_sources.png')
    fig,axes=plt.subplots(1,2,figsize=(16,7))
    atlas.draw_errors(axes[0],metric,data)
    axes[1].imshow(Image.open(data['native_wall']));axes[1].set_axis_off()
    axes[1].set_title(f"原生炸弹墙体覆盖 · 墙体分 {fmt(record['wall_score'])}")
    fig.suptitle(title+'\n左：绿命中／蓝漏画／红受罚／黄容差；右：红色爆炸覆盖',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.88));save(fig,folder/'errors_wall.png')
    fig,ax=plt.subplots(figsize=(12,5))
    rows=[r for r in diag['return_samples'] if r['drone']==0]
    for i in range(5):
        ax.step([r['control_step'] for r in rows],[r['received_versions'].get(str(i),0) for r in rows],where='post',color=atlas.COLORS[i],label=f'来源{i}')
    ax.axvline(record['freeze_elapsed']-1,c='#555',ls='--',label='源图冻结')
    ax.set(title=title+'\n提交者实收版本进度（30步采样；精确终版见表）',xlabel='控制步数',ylabel='实收源版本')
    ax.grid(alpha=.2);ax.legend(ncol=6);fig.tight_layout();save(fig,folder/'delivery_timeline.png')


def main():
    driver=module('d39r2_random_audit','src/swarm_rescue/solutions/devtools/delivery39r2_random3.py')
    plan=driver.validate(BATCH);done=read(BATCH/'completion.json');assert done['complete']
    pairs=read(BATCH/'pairs.json');assert len(pairs)==6
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();records=[];checks=[];curves={};samples={};loaded={}
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    check('exact predeclared schedule',[(p['map'],p['round'],p['seed']) for p in pairs]==[(p['map'],p['round'],p['seed']) for p in plan['schedule']])
    check('six attempts and no extra flights',len(list((BATCH/'runs').glob('**/attempt_*')))==6)
    resumed=read(BATCH/'coordinator_continuation.json')
    check('coordinator-only resume frozen',atlas.sha(ROOT/resumed['script'])==resumed['sha256'] and atlas.sha(BATCH/'plan.json')==resumed['plan_sha256'])
    cutoffs={}
    for p in pairs:
        diag=read(Path(p['red_run'])/'dualmap32_diagnostics.json')
        cutoffs[p['pair_id']]=min(int(x['_f27_stats']['cutoff_elapsed']) for x in diag['drones'].values())
    common={name:min(cutoffs[p['pair_id']] for p in pairs if p['map']==name) for name in ('Map08','Map09')}
    for p in pairs:
        key=p['pair_id'];run=Path(p['red_run']);folder=OUT/key;folder.mkdir()
        metric,data=atlas.inspect(p,BATCH);loaded[key]=(metric,data);nav=navplot.load(p)
        diag=read(run/'dualmap32_diagnostics.json');receipt=nav['receipt']
        identity=read(run/'loaded_identity.json');request=read(run/'request.json');resource=read(run/'resources.json')
        check(key+' evidence hashes',all(atlas.sha(run/f)==v for f,v in read(run/'integrity.json')['files'].items()))
        check(key+' loaded frozen sources',all(plan['source_sha256'][f]==v for f,v in identity['loaded_sources'].items()))
        check(key+' actual five R2 instances',len(identity['instances'])==5 and all(v.endswith('.MyDroneRedDelivery39R2') for v in identity['instances'].values()))
        check(key+' native zones and resource limits',request['zones']==p['scenario']['zones'] and resource['cpu_count']==1 and resource['memory_limit_bytes']==4000000000)
        check(key+' native score and submission audit',not metric['audit_issues'])
        cd=read(run/'continuous38_diagnostics.json');dd=read(run/'delivery39_diagnostics.json')
        check(key+' passive observer errors',not cd['errors'] and not dd['errors'] and not diag['return_sample_errors'])
        check(key+' shared continuous pose',max(s[k] for s in cd['samples'] for k in ('shared_position_difference','shared_heading_difference','stopturn_heading_difference'))==0)
        with np.load(run/'dualmap32_snapshots.npz',allow_pickle=False) as tape:
            sources,acks=delivery(diag,receipt,tape)
        check(key+' claimed ACKs supported by actual bytes',sorted(acks)==sorted(receipt['acknowledged_sources']))
        covered,curve=coverage(data,common[p['map']],p['map']);curves[key]=curve
        rows,alignment=helpers.observations(p,data,common[p['map']]);samples[key]=rows
        outage=np.concatenate([a[a[:,1]>0] for a in rows.values()])
        latest=sum(s['latest'] for s in sources)
        record=dict(map=p['map'],round=p['round'],seed=p['seed'],pair_id=key,valid_red=metric['valid_red'],
            map_score=metric['map_score'],wall_score=metric['wall_score'],time_score=metric['time_score'],blue_score=None,total_score=None,
            bombs=metric['placed_count'],steps=metric['red_steps'],seconds=metric['red_seconds'],
            wall_hits=metric['error_metric']['wall_hits'],wall_misses=metric['error_metric']['wall_misses'],penalty=metric['error_metric']['penalty'],
            sources=sources,latest_sources=latest,missing_final_sources=[s['drone'] for s in sources if not s['latest']],
            acknowledged_sources=acks,acknowledgements=len(acks),delivery_complete=bool(latest==5 and len(acks)==5 and receipt['final_complete']),
            submit_control=receipt['submit_control_step'],submit_reason=receipt['reason'],freeze_elapsed=cutoffs[key],
            common_cutoff_elapsed=common[p['map']],coverage=covered,outage=helpers.stats(outage),alignment=alignment,
            per_drone_outage={str(i):helpers.stats(a[a[:,1]>0]) for i,a in rows.items()},
            boundary=metric['boundary'],actual_zones=metric['actual_zones'],
            red_issues=metric['red_issues'],audit_issues=metric['audit_issues'],red_run=run.as_posix(),
            submission_sha256=metric['actual_submission_sha256'],
            final_delivery_samples={i:d['samples'][-1] if d['samples'] else None for i,d in dd['drones'].items()},
            delivery_drones={i:{k:v for k,v in d.items() if k not in ('walk_points','walk_edges','samples')} for i,d in dd['drones'].items()})
        records.append(record);atlas.write(folder/'data.json',record)
        figures(p,metric,data,nav,diag,record,folder)
    statistics={}
    for name in ('Map08','Map09'):
        rows=[r for r in records if r['map']==name];scores=[r['map_score'] for r in rows if r['valid_red']]
        statistics[name]=dict(valid_count=len(scores),mean=float(np.mean(scores)) if scores else None,
            minimum=min(scores) if scores else None,maximum=max(scores) if scores else None,
            population_std=float(np.std(scores)) if scores else None,range=float(np.ptp(scores)) if scores else None,
            delivery_complete=sum(r['delivery_complete'] for r in rows),bombs=sum(r['bombs'] for r in rows))
        fig,axes=plt.subplots(3,3,figsize=(21,17.5))
        for index,r in enumerate(rows):
            metric,data=loaded[r['pair_id']]
            atlas.draw_submission(axes[index,0],metric,data)
            axes[index,0].set_title(f"第{r['round']}组 / seed {r['seed']}\n地图分 {fmt(r['map_score'])} · 终图 {r['latest_sources']}/5 · ACK {r['acknowledgements']}/5",fontsize=12)
            atlas.draw_bombs(axes[index,1],metric,data);atlas.draw_tracks(axes[index,2],metric,data,'red')
        fig.suptitle(f'{name} / Delivery39R2 原代码 / 3个新随机种子 / 原生特殊区域开启\n左：实际提交；中：实际炸弹；右：真实轨迹',fontsize=17)
        fig.legend([Line2D([0],[0],color=atlas.COLORS[i]) for i in range(5)],[f'无人机{i}' for i in range(5)],loc='lower center',ncol=5,frameon=False)
        fig.subplots_adjust(top=.90,bottom=.035,left=.01,right=.99,wspace=.055,hspace=.24);save(fig,OUT/f'{name}_all3_submission_bombs_trajectory.png')
        fig,axes=plt.subplots(2,2,figsize=(15,9))
        for ax,key in zip(axes.flat,curves[rows[0]['pair_id']]):
            for r in rows:
                curve=curves[r['pair_id']][key];ax.plot(np.arange(len(curve))+2,curve,label=f"组{r['round']} / {r['seed']}")
            ax.set(title=key,xlabel='原生步数',ylabel='中心到访40px格');ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.suptitle(f'{name} 三种子到访 · 共同截止{common[name]}步 · 不是雷达视野覆盖率',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.95));save(fig,OUT/f'{name}_coverage.png')
        fig,axes=plt.subplots(3,2,figsize=(15,12),sharex=True,sharey='col')
        for row,r in enumerate(rows):
            for i,a in samples[r['pair_id']].items():
                part=a[a[:,1]>0]
                for col,index in enumerate((2,3)):axes[row,col].scatter(part[:,0],abs(part[:,index]),s=3,alpha=.45,color=atlas.COLORS[i],label=f'{i}号')
            for col in range(2):
                axes[row,col].set(title=f"第{r['round']}组 / seed {r['seed']}",xlabel='控制步数',ylabel='位置误差 / px' if col==0 else '航向误差 / 度')
                axes[row,col].grid(alpha=.2);axes[row,col].legend(ncol=5,fontsize=8,markerscale=3)
        fig.suptitle(f'{name} 实际失定位期间漂移 · 封存前每10步采样 · 各组路径不相同',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.95));save(fig,OUT/f'{name}_drift.png')
    summary=dict(completion=done,controller='Delivery39R2',seeds=read(BATCH/'seed_draw.json')['accepted_seeds'],
        statistics=statistics,metrics=records,common_cutoff_elapsed=common,
        scope='Six newly run red flights; unchanged candidate, no new baseline/blue; prior diagnostic seeds excluded')
    atlas.write(OUT/'data.json',summary);atlas.write(BATCH/'audit.json',dict(passed=True,count=len(checks),checks=checks))
    write_report(summary,checks)
    print(json.dumps(dict(statistics=statistics,results=[{k:r[k] for k in ('map','round','seed','valid_red','map_score','bombs','latest_sources','acknowledgements','seconds')} for r in records],checks=len(checks),report=str(REPORT)),ensure_ascii=False,indent=2))


def write_report(summary,checks):
    records=summary['metrics']
    text=['# Delivery39R2：Map08、Map09新随机三种子稳定性评测','',
        '日期：2026-11-22。按用户要求继续测试上一轮版本，参赛算法逐文件保持一致。Map08保留原生无GPS区，Map09保留原生无GPS区和无通信区；原生CPU雷达、1CPU／4GB／5400步／180秒，每机两弹。新红6场，新蓝0场，不报告完整总分。',
        '',f"三枚种子 **{', '.join(map(str,summary['seeds']))}** 由 `secrets.randbits(32)` 系统随机源产生，在全部飞行前记录并冻结；两张地图共用这三枚种子。只排除已有计划中的种子和重复抽样，不按成绩筛选，不重试、不更换种子、不按结果改算法。",'',
        '执行记录：第一场完成后，批次协调器向已存在的进度文件执行仅新建写入而停止，第二场尚未启动。另存协调器续跑脚本，只修正累计进度／回执写入并从第二场续跑；原计划、种子、参赛算法、评测worker及第一场证据保持原样。实际仍为每图每种子一次，共六场，没有重复第一场。', '',
        '**本轮汇总；不混入之前两个问题种子的成绩**','','|地图|有效红阶段|地图均值|最低|最高|极差|总体标准差|五源交付＋确认|实际弹|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summary['statistics'].items():text.append(f"|{name}|{s['valid_count']}/3|{fmt(s['mean'])}|{fmt(s['minimum'])}|{fmt(s['maximum'])}|{fmt(s['range'])}|{fmt(s['population_std'])}|{s['delivery_complete']}/3|{s['bombs']}/30|")
    text+=['','**逐场实测分项**','','|地图／组|种子|有效地图分|墙体分|时间分|实际弹|步数／秒|最新终图|源集合ACK|蓝分／完整总分|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in records:text.append(f"|{r['map']}／{r['round']}|{r['seed']}|{fmt(r['map_score'])}|{fmt(r['wall_score'])}|{fmt(r['time_score'])}|{r['bombs']}/10|{r['steps']}／{r['seconds']:.2f}|{r['latest_sources']}/5|{r['acknowledgements']}/5|未测／不报告|")
    text+=['','红阶段有效性与通信成功分别检查。ACK是算法内部对五份源图集合的确认，不是官方新增的提交要求；本报告以实际捕获的源图版本、摘要和字节核实，不能以靠近或发出消息代替交付。导航图为各机赛末20px规划网格，保留日常清噪，不经过最终规整或补边，也不等同于最终融合地图。','',
        '**探索与误差**','','Map09诊断区沿用北侧y<563、深处y<320、西北深处x<622且y<320；Map08最底部为图像底部100px带，右侧为最右1/3。这些分区只用于赛后统计，底部带到访不单独证明窄路全程通过。到访格只计地图图像范围内存活飞机的机身中心，不是雷达视野覆盖；漂移只统计封存前每10步的GPS／罗盘失效时刻。','',
        '|地图／组|真墙命中|误画扣分|全图到访格|重点区域到访格|航向P95|位置P95|补边新增像素|','|---|---:|---:|---:|---|---:|---:|---:|']
    for r in records:
        region='西北深处 '+str(r['coverage']['northwest_deep']['cells40']) if r['map']=='Map09' else '最底100px带 '+str(r['coverage']['south_strip']['cells40'])
        text.append(f"|{r['map']}／{r['round']}|{r['wall_hits']:,}|{r['penalty']:.5f}|{r['coverage']['all']['cells40']}|{region}|{fmt(r['outage']['navigation_heading']['p95'],2)}°|{fmt(r['outage']['navigation_position']['p95'],2)}px|{r['boundary']['added_pixels']:,}|")
    for name in ('Map08','Map09'):
        text+=['',f"**{name}三组曲线：统一截止原生{summary['common_cutoff_elapsed'][name]}步**",'',f"![到访曲线]({(OUT/f'{name}_coverage.png').as_posix()})",'',f"![漂移曲线]({(OUT/f'{name}_drift.png').as_posix()})"]
    for r in sorted(records,key=lambda x:(x['map'],x['round'])):
        folder=OUT/r['pair_id']
        text+=['',f"**{r['map']} 第{r['round']}组／seed {r['seed']}**",'',
            f"有效红阶段：{r['valid_red']}；地图分{fmt(r['map_score'])}；五源交付＋确认：{r['delivery_complete']}。源图冻结原生步{r['freeze_elapsed']}，提交控制步{r['submit_control']}，原因 `{r['submit_reason']}`。",'',
            '|来源|提交时实收版|自身封存终版|字节一致|','|---|---:|---:|---|']
        for s in r['sources']:text.append(f"|{s['drone']}号|{s['received']}|{s['final']}|{s['latest']}|")
        for source in r['missing_final_sources']:
            d=r['delivery_drones'][str(source)];last=r['final_delivery_samples'][str(source)]
            started=next((e['step'] for e in d['events'] if e['kind']=='return_started'),None)
            if last:
                text+=['',f"缺图来源{source}号：返程开始控制步{started}，结束状态 `{d['final_mode']}`；最后状态采样步{last['step']}，最近实际收信控制步{last['last_rx']}。此时返程估计路径仍有{last['budget'].get('route_pixels')}px。日志反映未完成返送，不能仅凭飞机靠近或曾经过某区域认定已交付。"]
        if r['red_issues']:text+=['',f"无效原因：{r['red_issues']}；原始分不列为有效成绩。"]
        for filename,title in [('submission_bombs_trajectory.png','实际提交图、炸弹图、轨迹图'),('errors_wall.png','提交误差及原生墙体覆盖'),('navigation_all5.png','五机导航图'),('received_sources.png','实际实收五源图'),('delivery_timeline.png','通信交付进度')]:
            text+=['',f"![{title}]({(folder/filename).as_posix()})"]
        text+=['',f"[本组完整数据]({(folder/'data.json').as_posix()})"]
    text+=['','**冻结与核验**','',f"{len(checks)}项产物检查通过，包含冻结代码、真实加载身份、原生资源和特殊区域、实际提交计分、共享位姿、全部源图及ACK字节。三种子只是本批样本，不能证明未知地图全面稳定。没有把之前的开发或诊断飞行混入均值，也没有替换正式比赛ZIP。",'',
        f"[冻结计划]({(BATCH/'plan.json').as_posix()}) · [种子抽样记录]({(BATCH/'seed_draw.json').as_posix()}) · [完成回执]({(BATCH/'completion.json').as_posix()}) · [核验]({(BATCH/'audit.json').as_posix()}) · [完整数据]({(OUT/'data.json').as_posix()})",'',
        '**录像回放：不新增仿真，本轮未运行蓝阶段**','','```powershell',
        "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'","$env:PYTHONPATH='src'","$env:PYGLET_SHADOW_WINDOW='false'","$env:PYTHONIOENCODING='utf-8'"]
    for r in records:text.append(f".venv/Scripts/python.exe -m swarm_rescue.solutions.devtools.optimization.evaluation.replay '{r['red_run']}'")
    text+=['```','']
    text[2:2]=['**结论：六场有效、60/60弹，五源终图与全队确认成功5/6场；地图分仍有明显随机种子波动，不能判为稳定高分版本。** 保留Delivery39R2作为后续优化基线，本轮不修改参赛算法，不替换正式ZIP。Map08均值58.56600、极差18.63337；Map09均值64.33992、极差28.96080。\n\n本批把三类问题区分开了。Map08第一组只收到2号机3810旧版而非4643终版；2号第3889控制步开始返程，第5250步仍处于return、最近实际收信停在3822步。这是交付未完成，且Map08没有禁通信区，不能归咎于禁通信区屏蔽。\n\n其余五场全部收齐终图，低分不能继续归为漏交付。Map09第二组西北深处实际到访0格，五机导航图中这一块也保持未知，轨迹没有进入；三组该区到访依次53、0、10格。Map08三场右下最底100px带（x≥地图宽度2/3、y≥高度−100）均0到访，右侧内部仍有大片未进入区域。以上证据支持探索覆盖不足是关键问题；本轮没有隔离初始朝向、窄路判断与任务调度各自的因果贡献。\n\n另发现补边误判：Map08第三组把x=1499附近的竖向证据选成右外框，真值右外墙在约1660处，提交误差图出现错误竖线。已确认的是外框位置错误；究竟来自内墙误认还是定位漂移后的外墙证据，尚未拆分，不把猜测当成原因。该组补边新增21,808px并不代表全部正确，也不能仅凭本批认定补边整体有害。\n\n两张地图第二组的真墙命中率分别约55.93%和54.04%，误画扣分3.60384和5.14811，主要损失为漏画。最终图的空白必须结合导航与真实轨迹解释，不能仅从提交图判断通信失败。当前建议下一步优先讨论未知区域覆盖、右下窄路与外框置信度，同时保留Map08第一组作返程交付回归样本；本轮没有开始这些新修改。','']
    REPORT.write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':main()
