"""RX10 full red comparisons and offline observation-to-submission audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
import numpy as np
from PIL import Image
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Circle
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools.red_rn7_review import regions, REGIONS

NAMES={'advanced2':'进阶2原版','rn7_b':'RN7-B原版','rx10_branch':'路口深入版',
       'rx10_finish':'自适应收尾版','rx10_combined':'组合（实验）','rx10_off':'RN7-B关闭对照'}
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')


def write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False)


def verify_frozen_batch(batch,plan):
    """Verify saved sources; allow only the concurrently edited, unused B8 file.

    This is offline review, never permission to resume an old simulation plan.
    All red strategies, physics, scorer and review dependencies must still match.
    """
    archive=batch/'source_snapshot.zip'
    assert base.digest(archive)==plan['source_snapshot_sha256']
    with zipfile.ZipFile(archive) as frozen:
        for name,expected in plan['source_sha256'].items():
            assert hashlib.sha256(frozen.read(name)).hexdigest()==expected,name
    changed=[name for name,expected in plan['source_sha256'].items()
             if not Path(name).is_file() or base.digest(name)!=expected]
    allowed={'src/swarm_rescue/solutions/blue_nav_b8.py'}
    unexpected=set(changed)-allowed
    if unexpected:raise ValueError(f'Used or unreviewed source changed: {sorted(unexpected)}')
    return dict(batch=str(batch),frozen_archive_verified=True,current_differences=changed,
                exception_scope='offline only; unused concurrent B8; no blue phase')


def audit(row,data):
    run=Path(row['attempt']);doc=base.read(run/'exploration_diagnostics.json')
    with np.load(run/'exploration_stages.npz') as f:arrays={k:f[k] for k in f.files}
    pred=np.asarray(base.read(next((run/'artifacts').glob('*_exploration.json')))['grid'],bool)
    wall,_=base.MapExplorationScorer().split_regions(data['truth']);wall=wall.astype(bool)
    w,h=data['w'],data['h'];ledger=arrays['endpoint_ledger']
    x=np.rint(ledger[:,2]+w/2).astype(int);y=np.rint(h/2-ledger[:,3]).astype(int)
    good=(x>=0)&(x<w)&(y>=0)&(y<h)
    ledger=ledger[good];x=x[good];y=y[good]
    good=wall[y,x];ledger=ledger[good];x=x[good];y=y[good]
    observed=np.zeros((h,w),bool);own_present=observed.copy();local_union=observed.copy()
    observed[y,x]=True
    stats=Counter();roles=Counter();finish=Counter();repeat=0;task_rows=[];source_rows=[]
    submitted=doc['submissions'][0] if doc['submissions'] else None
    for ident,drone in doc['drones'].items():
        stats.update(drone['branch_stats']);finish.update(drone['finish_stats']);roles.update(drone['roles'])
        repeat+=drone['repeat_samples'];index=ledger[:,0]==int(ident)
        local=arrays[f'local_final_pixels_{ident}'].astype(bool)
        local_union|=local
        xx=x[index];yy=y[index];accepted=local[yy,xx];own_present[yy[accepted],xx[accepted]]=True
        source_observed=np.zeros_like(observed);source_observed[yy,xx]=True
        if submitted:
            current=arrays[f'local_final_{ident}'];key=f"submitter_{submitted['drone']}_source_{ident}"
            received=arrays.get(key,np.zeros_like(current));seq=submitted['sources'].get(ident)
            source_rows.append(dict(source=int(ident),sequence=seq,age=None if seq is None else submitted['step']-seq,
                own_latest_cells=int(current.sum()),received_cells=int(received.sum()),
                own_final_absent_from_source_snapshot=int((current&~received).sum()),
                sampled_observed_pixels=int(source_observed.sum()),
                sampled_own_present_final_absent=int((source_observed&local&~pred).sum())))
        events=drone['branch_events']
        for event in events:
            if event['event']!='selected':continue
            task=event['task'];related=[e for e in events if e.get('task')==task]
            entries=[e['step'] for e in related if e['event']=='entered']
            ends=[e for e in related if e['event']=='task_end']
            entered=min(entries) if entries else None;end=ends[0]['step'] if ends else row['red_steps']
            samples=index&(ledger[:,1]>=entered)&(ledger[:,1]<=end) if entered is not None else np.zeros(len(ledger),bool)
            tx=x[samples];ty=y[samples]
            task_rows.append(dict(drone=int(ident),task=task,selected_step=event['step'],target=event['point'],
                estimated_observation_point_entered=entered,end_step=end,
                end_reason=ends[0]['reason'] if ends else 'no_end_in_bounded_log',
                sampled_wall_rays_after_entry=int(samples.sum()),
                own_final_present_rays=int(local[ty,tx].sum()),submitted_present_rays=int(pred[ty,tx].sum())))
    error=np.linalg.norm(ledger[:,4:6]-ledger[:,2:4],axis=1) if len(ledger) else np.array([])
    metrics=dict(sampled_true_surface_pixels=int(observed.sum()),
        own_final_present=int(own_present.sum()),own_final_absent=int((observed&~own_present).sum()),
        submitted_present=int((observed&pred).sum()),submitted_absent=int((observed&~pred).sum()),
        own_present_but_submission_absent=int((own_present&~pred).sum()),
        raw_endpoint_error_median=float(np.median(error)) if len(error) else None,
        raw_endpoint_error_p90=float(np.percentile(error,90)) if len(error) else None)
    region_maps={}
    for name,(left,top,right,bottom) in REGIONS[row['map']].items():
        sl=np.s_[top:bottom+1,left:right+1]
        region_maps[name]=dict(true_surface=int(wall[sl].sum()),submitted_surface_hits=int((wall&pred)[sl].sum()),
            sampled_observed=int(observed[sl].sum()),sampled_own_present_final_absent=int((own_present&~pred)[sl].sum()))
    oracle=base.MapExplorationScorer().score(data['truth'],local_union.astype(np.uint8),row['red_walltime'])
    row.update(branch_stats=dict(stats),finish_stats=dict(finish),role_ticks=dict(roles),
        repeated_exploration_samples=repeat,observation_chain=metrics,source_snapshots=source_rows,
        diagnostic_seconds=doc['diagnostic_seconds'],diagnostic_errors=doc['errors'],task_evidence=task_rows,
        region_drone_frames=regions(data,row['map']),region_map_evidence=region_maps,
        offline_local_union=dict(map_formula_value=oracle.s_map*100,
            meaning='Offline union of all actual final local rendered maps, ignoring communication; NOT an official/achievable score'))
    data.update(pred=pred,observed=observed,own_present=own_present,diagnostics=doc)


def trajectories(ax,data):
    w,h=data['w'],data['h'];trace=data['trajectories']
    ax.imshow(data['truth'],cmap='Greys',vmin=0,vmax=1);base.areas(ax,data['spec'])
    for slot in range(trace.shape[1]):
        good=trace['active'][:,slot]
        x=np.where(good,trace['x'][:,slot]+w/2,np.nan);y=np.where(good,h/2-trace['y'][:,slot],np.nan)
        jump=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40];x[jump]=np.nan;y[jump]=np.nan
        ax.plot(x,y,lw=.65,alpha=.75)


def figures(name,number,prefix,rows,assets,out):
    columns=2;lines=math.ceil(len(rows)/columns)
    for kind in ('mapping','trajectories','walls','bombs','observations'):
        fig,axes=base.plt.subplots(lines,columns,figsize=(16,6.2*lines),squeeze=False)
        fig.subplots_adjust(left=.02,right=.99,top=.89,bottom=.075,wspace=.04,hspace=.18)
        for ax in axes.ravel():ax.axis('off')
        for ax,row in zip(axes.ravel(),rows):
            data=assets[(name,number,row['controller'])];w,h=data['w'],data['h']
            if kind=='mapping':ax.imshow(data['merged'],interpolation='nearest')
            elif kind=='walls':ax.imshow(data['wall_panel'],interpolation='nearest')
            elif kind=='trajectories':trajectories(ax,data)
            else:
                ax.imshow(data['truth'],cmap='Greys',vmin=0,vmax=1)
                if kind=='bombs':
                    base.areas(ax,data['spec'])
                    for k,bomb in enumerate(data['bombs'],1):
                        x,y=bomb['x']+w/2,h/2-bomb['y']
                        ax.add_patch(Circle((x,y),data['radius'],fill=False,color='royalblue',alpha=.65,lw=.9))
                        ax.plot(x,y,'o',color='royalblue',ms=3);ax.annotate(str(k),(x,y),xytext=(3,3),textcoords='offset points',fontsize=9)
                else:
                    for mask,colour in ((data['observed']&data['pred'],'seagreen'),(data['observed']&~data['pred'],'crimson')):
                        yy,xx=np.nonzero(mask);ax.scatter(xx,yy,s=.5,c=colour,linewidths=0)
            title=f"{NAMES[row['controller']]} | M {row['map_score']:.2f} | W {row['wall_score']:.2f}\n{row['red_steps']}步 / {row['red_walltime']:.1f}秒 / {row['placed_count']}枚弹"
            ax.set_title(title,fontproperties=FONT,fontsize=13)
            ax.set(xlim=(0,w),ylim=(h,0),aspect='equal')
        label=dict(mapping='提交地图与真实墙体差异',trajectories='完整实际探索轨迹',walls='官方墙体覆盖',bombs='实际炸弹位置',observations='实际雷达采样与最终提交命中')[kind]
        fig.suptitle(f"{name}：{label} / 同图同种子，各一次红阶段",fontproperties=FONT,fontsize=18)
        notes=dict(mapping='绿色命中；蓝色漏墙；深红误画扣分；黄色误画免扣；灰色无分内部。',
            trajectories='轨迹不是雷达覆盖。灰色无GPS，黄色无通信；出生青色，处置区红色。',
            walls='绿色覆盖内墙，蓝色覆盖边界墙，灰色未覆盖，红色爆炸区域；分数沿用官方公式。',
            bombs='编号为实际落点产物顺序，不能推断机号或投放时间；圆重叠不等于墙面收益重复。',
            observations='每40步实际射线端点采样：绿色最终命中、红色最终漏画。只是观察下界，不是完整传感器覆盖。')
        fig.text(.5,.035,notes[kind],ha='center',fontproperties=FONT,fontsize=12)
        fig.text(.5,.012,f"r{number} / seed {rows[0]['red_seed']}；蓝队阶段未运行，完整官方总分未评测。",ha='center',fontproperties=FONT,fontsize=11)
        base.save(fig,out/f'{prefix}_{kind}.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);rows=[];assets={};failures=[];source_audits=[]
    for batch in args.batches:
        plan=base.read(batch/'plan.json');source_audits.append(verify_frozen_batch(batch,plan));sha=base.digest(batch/'plan.json')
        for raw in base.read(batch/'results.json'):
            try:
                row,data=base.review_row(raw,sha);audit(row,data)
                key=(row['map'],row['round'],row['controller'])
                if key in assets:raise ValueError(f'Duplicate map/round/controller: {key}')
                rows.append(row);assets[key]=data
            except Exception as exc:
                failures.append(dict(map=raw['map'],controller=raw['controller'],error=repr(exc),run=raw['red_run']))
    groups=[]
    for name,number in dict.fromkeys((r['map'],r['round']) for r in rows):
        prefix=name if sum(1 for n,k in dict.fromkeys((r['map'],r['round']) for r in rows) if n==name)==1 else f'{name}_r{number:02d}'
        groups.append((name,number,prefix))
        selected=[r for r in rows if r['map']==name and r['round']==number]
        figures(name,number,prefix,selected,assets,args.output)
    write(args.output/'summary.json',dict(rows=rows,review_failures=failures,blue_evaluated=False,source_audits=source_audits))
    doc=['# RX10：原版、路口深入与自适应收尾实测图册','',
        'RN7-B是本轮开发基线，不宣称全面优于进阶2。每个本轮版本在对应地图只跑一次，均使用同一被动诊断。仅红阶段；难度、完整官方原始及有效总分全部未评测。',
        '同图同种子不等于严格相同传感器输入；引擎随机UID等未完全由种子固定。','',
        '|地图/版本|红有效|M|W|T|弹数|步数|墙钟秒|墙/机接触|','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:doc.append(f"|{r['map']}/r{r['round']}/{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['red_steps']}|{r['red_walltime']:.2f}|{r['contacts_wall']}/{r['contacts_drone']}|")
    if failures:doc+=['','复核失败（未隐藏）：',json.dumps(failures,ensure_ascii=False)]
    for name,number,prefix in groups:
        doc+=['',f'## {name} / r{number}','','面板按上表该地图该轮顺序，从左到右、从上到下。所有图只读已有录像/官方产物。']
        for kind,label in [('mapping','建图差异'),('trajectories','实际探索轨迹'),('walls','官方墙体覆盖'),('bombs','实际落弹'),('observations','实际射线观察与提交')]:
            doc+=['',label,'',f'![{name} r{number} {label}]({(args.output.resolve()/f"{prefix}_{kind}.png").as_posix()})']
    doc+=['','## 探索任务与证据链','',
        '前沿任务“进入”指估计机体位置到所选安全观察点35px以内，并非离线真值确认穿过物理门。真实区域进入另用录像统计。重复探索样本指每24步取样中40px区域已访问至少3次且本地自由格无新增；返程与布弹排除，但不能把所有重复样本都称无意义。','',
        '|地图/版本|发现/选中/到观察点/续段|重复探索样本|区域机体帧数|','|---|---|---:|---|']
    for r in rows:
        s=r['branch_stats'];counts='/'.join(str(s.get(k,0)) for k in ('discovered','selected','entered','continued')) if s else '原版未记录前沿任务'
        doc.append(f"|{r['map']}/r{r['round']}/{NAMES[r['controller']]}|{counts}|{r['repeated_exploration_samples']}|{r['region_drone_frames']}|")
    doc+=['','|地图/版本|采样真墙像素|本机终图有|最终提交有|本机有而提交无|原始端点误差P90 px|','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        a=r['observation_chain'];doc.append(f"|{r['map']}/r{r['round']}/{NAMES[r['controller']]}|{a['sampled_true_surface_pixels']}|{a['own_final_present']}|{a['submitted_present']}|{a['own_present_but_submission_absent']}|{a['raw_endpoint_error_p90']}|")
    doc+=['','“本机有而提交无”指对应实际观察者的最终可提交像素与最终全队提交图比较；涉及源快照时效、传播、合并和后处理，不能一概命名为通信丢包。原始端点误差同时包含定位、朝向和雷达距离噪声，不是纯定位误差。未记录每次擦除的完整历史，仍不能精确拆分所有地图处理损失。来源/版本/年龄、最终本机格与提交者来源格差异、逐任务采样证据详见summary.json。',
        '离线局部图并集指标：仅用于检查地图是否留在各机本地。忽略真实通信约束，直接合并五机末帧可提交像素再调用原地图公式；不是本轮官方成绩，也不是策略运行时可读取的结果。值及区域命中诊断见summary.json。',
        '','## 本轮红回放','']
    for r in rows:
        doc += [f"{r['map']}/r{r['round']}/{NAMES[r['controller']]} / seed {r['red_seed']}",'','```powershell',
            "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{r['attempt']}'",'```','']
    doc+=['蓝队阶段未运行，无本轮蓝录像，不拼接历史蓝分。',
        '', '源码复核：冻结ZIP及每个条目哈希均验证；全部红队、评分及共享源码匹配。并发蓝队B8文件变化单列，未用于本轮红阶段；该例外只供离线审阅，不用于继续运行旧批次。',json.dumps(source_audits,ensure_ascii=False)]
    with (args.output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(doc)+'\n')
    for path in args.output.glob('*.png'):
        with Image.open(path) as img:img.verify()
    protected=base.read('output/sf13/rx10_dev_20261114/protected_before.json')
    changed=[v['path'] for v in protected if not Path(v['path']).exists() or base.digest(v['path'])!=v['sha256']]
    write(args.output/'verification.json',dict(protected=len(protected),changed=changed,
        result_count=len(rows),review_failures=failures,diagnostic_errors=[r['diagnostic_errors'] for r in rows],
        images_verified=len(list(args.output.glob('*.png'))),receipts_verified=not failures,map_scores_reproduced=not failures,source_audits=source_audits))
    print(json.dumps([{k:r[k] for k in ('map','controller','map_score','wall_score','time_score','red_valid','region_drone_frames','observation_chain')} for r in rows],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
