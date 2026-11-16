"""Offline complete RL17 atlas, official score/receipt audit and input comparison."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_rx10_review as evidence
from swarm_rescue.solutions.devtools import red_lines11_review as lines
from swarm_rescue.solutions.devtools import red_consistent12_review as panels
from swarm_rescue.solutions.devtools.red_wall_staged16_review import compare_inputs
from swarm_rescue.solutions.devtools.red_p2l17_replay import write, summary


NAMES={'rl11_lines':'RL11 基线','rl17_repair':'RL17-R 拒绝后补充','rl17_p2l':'RL17-A 点到线','rl17_off':'RL17 关闭对照'}
KINDS={'exploration':'雷达观察范围','mapping':'实际提交建图差异','trajectories':'完整实际轨迹','bombs':'实际炸弹位置','walls':'官方墙体覆盖'}
NOTES={'exploration':'绿色：8步/4px采样的雷达可见区估计，含射线间插值；黑：真墙。',
       'mapping':'绿：真墙命中；蓝：漏墙；深红：受罚误画；黄：免罚误画。',
       'trajectories':'颜色为录像槽位；灰色为无GPS/罗盘区，黄色为无通信区。轨迹不等于雷达覆盖。',
       'bombs':'编号为实际落点产物顺序，非已核实机号；圆为爆炸半径，重叠不等于墙体收益重复。',
       'walls':'绿：覆盖内墙；蓝：覆盖边界墙；灰：未覆盖墙；红：爆炸范围。'}


def panel(ax,kind,row,data):
    panels.NAMES=NAMES
    panels.panel(ax,'mapping' if kind=='exploration' else kind,row,data)
    if kind=='exploration':
        img=np.ones((*data['truth'].shape,3))*.97;img[data['visible']]=[.55,.84,.69]
        img[data['truth'].astype(bool)]=[.12,.12,.12];ax.imshow(img);base.areas(ax,data['spec'])


def audit(row,data):
    run=Path(row['attempt']);doc=base.read(run/'p2l17_diagnostics.json')
    row['p2l_diagnostic_errors']=doc['errors'];row['p2l_diagnostic_seconds']=doc['seconds']
    counts=Counter();decisions=Counter()
    for d in doc['drones'].values():counts.update(d['p2l_stats'])
    for m in doc['matches']:
        decision=m['decision'];decisions[('accept_' if decision.get('accepted') else 'reject_')+decision.get('info',{}).get('reason','missing')]+=1
    row.update(p2l_stats=dict(counts),match_decisions=dict(decisions),
        mapper_seconds=sum(d['mapper_seconds'] for d in doc['drones'].values()),
        mapper_max_seconds=max((d['mapper_max_seconds'] for d in doc['drones'].values()),default=0.),
        incremental_p2l_seconds=sum(d['p2l_seconds'] for d in doc['drones'].values()),
        maximum_pending=max((d['max_pending'] for d in doc['drones'].values()),default=0),
        submission_receipts=[r for r in doc['radio'] if r['event']=='submitted'],
        radio_counts=dict(Counter(r['event'] for r in doc['radio'])))
    with np.load(run/'p2l17_diagnostics.npz') as f:
        poses=np.concatenate([f[k] for k in f.files]);outage=poses[poses[:,7].astype(bool)]
        row['full_outage_position']=summary(np.linalg.norm(outage[:,4:6]-outage[:,1:3],axis=1))
        row['full_outage_heading_degrees']=summary(np.abs((outage[:,6]-outage[:,3]+np.pi)%(2*np.pi)-np.pi)*180/np.pi)
    with np.load(run/'local13_diagnostics.npz') as f:
        visible=np.logical_or.reduce([f[k].astype(bool) for k in f.files])
    data['visible']=np.repeat(np.repeat(visible,4,0),4,1)[:data['h'],:data['w']]
    with np.load(run/'exploration_stages.npz') as f:
        data['local_maps']={k:f[k].copy() for k in f.files if k.startswith('local_final_pixels_')}
        data['local_union']=np.logical_or.reduce(list(data['local_maps'].values()))
    coverage=base.read(run/'local13_diagnostics.json')
    row['coverage_diagnostic_errors']=coverage['errors']
    if doc['errors'] or coverage['errors']:
        row['diagnostic_valid']=False
    else:row['diagnostic_valid']=True


def figures(rows,assets,out):
    maps=list(dict.fromkeys(r['map'] for r in rows));controllers=list(dict.fromkeys(r['controller'] for r in rows))
    for kind,title in KINDS.items():
        fig,axes=base.plt.subplots(len(maps),len(controllers),figsize=(9*len(controllers),7*len(maps)),squeeze=False)
        fig.subplots_adjust(left=.015,right=.99,top=.87,bottom=.06,hspace=.42,wspace=.04)
        for y,m in enumerate(maps):
            for x,c in enumerate(controllers):
                r=next((r for r in rows if r['map']==m and r['controller']==c),None)
                if r:panel(axes[y,x],kind,r,assets[m,c,r['round']])
                else:axes[y,x].axis('off')
        fig.suptitle('RL11 / RL17-R · '+title,fontproperties=lines.FONT,fontsize=22,y=.98)
        fig.text(.5,.94,'完整红阶段；主对比展示各图第1轮，其余轮次见逐局图册。蓝队难度与完整总分未评测。',ha='center',fontproperties=lines.FONT,fontsize=12)
        fig.text(.5,.025,NOTES[kind],ha='center',fontproperties=lines.FONT,fontsize=11)
        base.save(fig,out/f'comparison_{kind}.png')
    for r in rows:
        data=assets[r['map'],r['controller'],r['round']];prefix=f"{r['map']}_{r['controller']}_r{r['round']:02d}"
        for kind in KINDS:
            fig,ax=base.plt.subplots(figsize=(12,8));panel(ax,kind,r,data)
            fig.text(.5,.015,NOTES[kind],ha='center',fontproperties=lines.FONT,fontsize=10)
            fig.tight_layout(rect=(0,.04,1,1));base.save(fig,out/f'{prefix}_{kind}.png')
        fig,axes=base.plt.subplots(2,3,figsize=(19,12))
        images=[data['truth'],data['observed'],data['local_union'],data['pred'],data['merged'],data['visible']]
        titles=['真值墙体（仅离线）','实际墙面端点（40步采样下界）','各机本机终图并集（非送达）','实际提交图','官方漏画 / 误画差异','雷达观察范围（8步/4px近似）']
        for ax,img,title in zip(axes.ravel(),images,titles):
            ax.imshow(img,cmap='Greys',vmin=0,vmax=1) if img.ndim==2 else ax.imshow(img)
            ax.axis('off');ax.set_title(title,fontproperties=lines.FONT,fontsize=12)
        fig.suptitle(f"{r['map']} / {NAMES[r['controller']]} / r{r['round']} / seed {r['red_seed']} / 有效={r['red_valid']}",fontproperties=lines.FONT,fontsize=19)
        fig.tight_layout(rect=(0,0,1,.95));base.save(fig,out/f'{prefix}_layers.png')
        fig,axes=base.plt.subplots(2,3,figsize=(19,12))
        for ax,(key,img) in zip(axes.ravel(),data['local_maps'].items()):
            ax.imshow(img,cmap='Greys',vmin=0,vmax=1);ax.axis('off')
            ident=int(key.rsplit('_',1)[1]);s=next(s for s in r['source_snapshots'] if s['source']==ident)
            ax.set_title(f"本机 {ident} 终图；提交端版本={s['sequence']}，落后={s['age']}步",fontproperties=lines.FONT,fontsize=12)
        axes.ravel()[-1].axis('off')
        fig.suptitle(f"{r['map']} / {NAMES[r['controller']]} / r{r['round']} · 单机本地地图与送达版本",fontproperties=lines.FONT,fontsize=19)
        fig.tight_layout(rect=(0,0,1,.95));base.save(fig,out/f'{prefix}_local_maps.png')


def review(batch,out=None):
    batch=Path(batch).resolve();out=Path(out).resolve() if out else batch.with_name(batch.name+'_figures')
    out.mkdir(parents=True,exist_ok=False);plan=base.read(batch/'plan.json')
    # Check all saved hashes; no exceptions for changes to a frozen candidate.
    planhash=lines.frozen(batch)
    rows=[];assets={};failures=[]
    for source in base.read(batch/'results.json'):
        try:
            row,data=base.review_row(source,planhash)
            evidence.audit(row,data);lines.accuracy(row,data);audit(row,data)
            rows.append(row);assets[row['map'],row['controller'],row['round']]=data
        except Exception as exc:
            failures.append(dict(source=source,error=repr(exc)))
    pairs=[]
    for r in rows:
        if r['controller']=='rl11_lines':continue
        old=next((a for a in rows if (a['map'],a['round'],a['controller'])==(r['map'],r['round'],'rl11_lines')),None)
        if old is None:continue
        a,b=assets[old['map'],old['controller'],old['round']],assets[r['map'],r['controller'],r['round']]
        pairs.append(dict(map=r['map'],round=r['round'],controller=r['controller'],inputs=compare_inputs(old,r),
            trajectories_exact=bool(np.array_equal(a['trajectories'],b['trajectories'])),bombs_exact=a['bombs']==b['bombs'],
            delta={k:r[k]-old[k] for k in ('map_score','wall_score','time_score','wall_hits','charged_pixels','red_steps','red_walltime')}))
    figures(rows,assets,out)
    write(out/'summary.json',dict(rows=rows,pairs=pairs,failures=failures,source=str(batch),plan_sha256=planhash,
        blue_evaluated=False,full_score=None,complete_atlas=len(rows)==len(base.read(batch/'results.json'))))
    doc=['# RL17 完整红阶段图册','',f'冻结批次：{batch}；计划 SHA256 `{planhash}`。',
         '蓝队难度与完整比赛总分未评测。图中真值和实体数据只用于离线审计。',
         '', '|地图/版本/轮|有效|M|W|T|弹|步|墙钟s|健康|墙/机接触|真墙命中|受罚误画|',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in rows:
        doc.append(f"|{r['map']}/{NAMES[r['controller']]}/r{r['round']}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['red_steps']}|{r['red_walltime']:.3f}|{r['health']:.2f}|{r['contacts_wall']}/{r['contacts_drone']}|{r['wall_hits']}|{r['charged_pixels']}|")
    doc+=['','所有回调异常、超时和诊断异常保留于原始结果及summary.json；退出码0不是有效性判断依据。']
    for kind in KINDS:doc+=['',f'![{KINDS[kind]}对照]({(out/f"comparison_{kind}.png").as_posix()})']
    for r in rows:
        prefix=f"{r['map']}_{r['controller']}_r{r['round']:02d}"
        doc+=['',f"## {r['map']} / {NAMES[r['controller']]} / r{r['round']} / seed {r['red_seed']}",'',
            f"红有效={r['red_valid']}；诊断有效={r['diagnostic_valid']}；峰值Job内存={r['resources'].get('peak_job_commit_bytes')}字节；单CPU/4GB。",
            f"地图更新累计 {r['mapper_seconds']:.4f}s；新增点线匹配 {r['incremental_p2l_seconds']:.4f}s；匹配决策 {r['match_decisions']}。",
            f"缺测位置均值 {r['full_outage_position']['mean']:.4f}px；角误差均值 {r['full_outage_heading_degrees']['mean']:.4f}°。",
            f"实际提交回执 {r['submission_receipts']}。广播尝试不能当成已送达。"]
        for kind in list(KINDS)+['layers','local_maps']:
            doc+=['',f'![{kind}]({(out/f"{prefix}_{kind}.png").as_posix()})']
        doc+=['','```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{r['attempt']}'",'```']
    if failures:doc+=['',f'未完成图册条目：{failures}。不能视为完整交付。']
    (out/'report.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
    print(dict(rows=len(rows),valid=sum(r['red_valid'] and r['diagnostic_valid'] for r in rows),failures=failures,pairs=pairs),flush=True)
    if failures:raise RuntimeError('Incomplete atlas; preserve outputs and repair in a new directory')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();review(a.batch,a.output)
