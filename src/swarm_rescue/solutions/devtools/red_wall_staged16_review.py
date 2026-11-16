"""Offline RL16 six-run audit and complete atlas; never runs a controller."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_rx10_review as evidence
from swarm_rescue.solutions.devtools import red_lines11_review as lines
from swarm_rescue.solutions.devtools import red_consistent12_review as panels

NAMES={'rl11_lines':'RL11 原版','rl16_wall':'A 单墙对齐','rl16_staged':'B 验证后提交'}
KINDS={'exploration':'雷达观察范围','mapping':'提交建图差异','trajectories':'完整实际轨迹','bombs':'实际炸弹位置','walls':'官方墙体覆盖'}
NOTES={'exploration':'绿：每8步/4px射线可见区采样估计，含射线间插值；白：该采样未覆盖。',
    'mapping':'绿：真墙面命中；蓝：漏墙；深红：受罚误画；黄：免罚误画。',
    'trajectories':'轨迹不是雷达覆盖；颜色为录像槽位。灰：无GPS/罗盘；黄：无通信。',
    'bombs':'编号是实际落点产物顺序；圆为爆炸范围。圆重叠不等于墙面收益重复。',
    'walls':'绿：覆盖内墙；蓝：覆盖边界墙；灰：未覆盖墙；红：爆炸范围。'}


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def audit(row,data):
    run=Path(row['attempt']);doc=base.read(run/'wall_staged16_diagnostics.json')
    row['experiment_diagnostic_errors']=doc['errors'];row['experiment_diagnostic_seconds']=doc['seconds']
    ws=Counter();ss=Counter();grades=Counter();reasons=Counter();errors=[];cells=conflicts=withheld=0
    maxima={'blocks':0,'frames':0};blocks_memory=0
    for d in doc['drones'].values():
        ws.update(d['wall_stats']);ss.update(d['stage_stats'])
        maxima['blocks']=max(maxima['blocks'],d['stage_stats'].get('maximum_blocks',0))
        maxima['frames']=max(maxima['frames'],d['stage_stats'].get('maximum_frames',0))
        for e in d['wall_events']:
            if e['event']=='alignment_verified' and e['relative_error'] is not None:errors.append(abs(np.degrees(e['relative_error'])))
        for e in d['stage_events']:
            if e['event']=='block_validated':
                grades[e['grade']]+=1;cells+=e['cells'];conflicts+=e['conflict_cells']
            if e['event']=='block_not_submitted':reasons[e['reason']]+=1;withheld+=e['cells']
    row.update(alignment_stats=dict(ws),alignment_verified_error_degrees=errors,
        staging_stats=dict(ss),validation_grades=dict(grades),withheld_reasons=dict(reasons),
        validated_cell_events=cells,conflict_cell_events=conflicts,withheld_cell_events=withheld,
        maximum_per_drone_cache=maxima,submission_receipts=doc['submission_receipts'])
    with np.load(run/'local13_diagnostics.npz') as z:
        v=np.logical_or.reduce([z[k].astype(bool) for k in z.files])
    data['visible']=np.repeat(np.repeat(v,4,0),4,1)[:data['h'],:data['w']]
    with np.load(run/'wall_staged16_diagnostics.npz') as z:
        for category in ('hit','local','trusted','staged','validated'):
            found=[z[k].astype(bool) for k in z.files if k.startswith(category+'_')]
            data[category]=np.logical_or.reduce(found) if found else np.zeros_like(data['truth'],bool)
        data['per_drone_layers']={k:z[k].astype(bool) for k in z.files if k.startswith(('local_','staged_','validated_'))}
        blocks_memory=sum(z[k].nbytes for k in z.files if k.startswith('block_'))
    row['offline_local_archive_uncompressed_bytes']=blocks_memory
    surface,_=base.MapExplorationScorer().split_regions(data['truth']);surface=surface.astype(bool)
    seen=data['hit']&surface;free=~data['truth'].astype(bool)
    row['dense_observed_surface']=int(seen.sum())
    row['observed_surface_in_submission']=int((seen&data['pred']).sum())
    row['sampled_visible_free_pixels']=int((free&data['visible']).sum())
    row['sampled_visible_free_fraction']=float((free&data['visible']).sum()/free.sum())
    row['local_staged_pixels']=int(data['staged'].sum());row['local_validated_pixels']=int(data['validated'].sum())
    # These masks are coordinate projections, not a causal partition of losses.
    row['local_staged_absent_submission_pixels']=int((data['staged']&~data['pred']).sum())
    gray=np.zeros_like(free)
    for region in row['gray_regions']:
        l,t,r,b=region['bounds'];mask=np.zeros_like(free);mask[max(0,t):b+1,max(0,l):r+1]=True;gray|=mask
        region.update(sampled_visible_free=int((free&mask&data['visible']).sum()),
            actual_observed_surface=int((seen&mask).sum()),submitted_observed_surface=int((seen&mask&data['pred']).sum()))
    row['gray_observed_surface']=int((seen&gray).sum())
    row['gray_visible_free']=int((free&gray&data['visible']).sum())
    pred=data['pred'].astype(bool);scorer=base.MapExplorationScorer()
    # The official image uses a red ramp, not a single red color. Reuse its
    # scorer distance field/config to classify all charged predictions.
    charged=pred&(scorer.distance_field(data['truth'])>scorer.config.dead_zone)
    if int(charged.sum())!=row['charged_pixels']:
        raise ValueError('Official charged-pixel mask does not match breakdown')
    row['gray_charged_pixels']=int((charged&gray).sum())
    data['dense_seen']=seen;data['experiment']=doc
    if row['map']=='Map08':
        # Reuse existing OFFLINE true-map gate audit, not its RC15 controller.
        from swarm_rescue.solutions.devtools.red_clearance15_review import crossings,GATES
        row['offline_gates']={k:crossings(data['trajectories'],data['w'],data['h'],g) for k,g in GATES.items()}
    if row['map']=='Map09':
        t=data['trajectories'];x=t['x']+data['w']/2;y=data['h']/2-t['y']
        row['offline_rooms']={name:dict(body_center_frames=int((t['active']&(x>l)&(x<r)&(y>top)&(y<b)).sum()))
            for name,(l,top,r,b) in panels.ROOMS.items()}


def panel(ax,kind,row,data):
    panels.panel(ax,'mapping' if kind=='exploration' else kind,row,data)
    if kind=='exploration':
        img=np.ones((*data['truth'].shape,3))*.97;img[data['visible']]=[.55,.84,.69]
        img[data['truth'].astype(bool)]=[.12,.12,.12];ax.imshow(img);base.areas(ax,data['spec'])


def compare_inputs(old,new):
    """Compare recorded legal observations, never infer equality from seed."""
    result={}
    with np.load(Path(old['attempt'])/'mapper_legal_inputs.npz') as a,np.load(Path(new['attempt'])/'mapper_legal_inputs.npz') as b:
        for k in a.files:
            if not k.startswith('drone_'):continue
            x,y=a[k],b[k];n=min(len(x),len(y));equal=np.all((x[:n]==y[:n])|(np.isnan(x[:n])&np.isnan(y[:n])),axis=1)
            wrong=np.flatnonzero(~equal)
            result[k]=dict(baseline_frames=len(x),candidate_frames=len(y),equal_prefix_frames=int(wrong[0]) if len(wrong) else n,
                first_different_step=float(x[wrong[0],0]) if len(wrong) else None,
                exactly_equal=bool(len(x)==len(y) and equal.all()))
        result['semantics_exactly_equal']=bool(np.array_equal(a['semantic'],b['semantic'],equal_nan=True))
    return result


def figures(rows,assets,out):
    panels.NAMES=NAMES
    for kind,title in KINDS.items():
        fig,axes=base.plt.subplots(2,3,figsize=(24,15),squeeze=False)
        fig.subplots_adjust(left=.015,right=.99,top=.87,bottom=.065,hspace=.46,wspace=.05)
        for y,m in enumerate(('Map08','Map09')):
            for x,c in enumerate(NAMES):
                r=next((r for r in rows if r['map']==m and r['controller']==c),None)
                if r:panel(axes[y,x],kind,r,assets[m,c])
                else:axes[y,x].axis('off')
        fig.suptitle('RL11 / A 单墙对齐 / B 验证后提交 · '+title,fontproperties=lines.FONT,fontsize=24,y=.98)
        fig.text(.5,.942,'上：Map08；下：Map09。各图各版一次完整红阶段；蓝队难度和完整总分未评测。',ha='center',fontproperties=lines.FONT,fontsize=14)
        fig.text(.5,.025,NOTES[kind],ha='center',fontproperties=lines.FONT,fontsize=13)
        base.save(fig,out/f'comparison_{kind}.png')
    for m in ('Map08','Map09'):
        for kind in ('trajectories','bombs','walls'):
            fig,axes=base.plt.subplots(1,3,figsize=(24,7.2));fig.subplots_adjust(left=.015,right=.99,top=.81,bottom=.08,wspace=.05)
            for ax,c in zip(axes,NAMES):
                r=next((r for r in rows if r['map']==m and r['controller']==c),None)
                if r:panel(ax,kind,r,assets[m,c])
                else:ax.axis('off')
            fig.suptitle(m+' · '+KINDS[kind],fontproperties=lines.FONT,fontsize=22,y=.98)
            fig.text(.5,.025,NOTES[kind]+' 每版1轮；0蓝队。',ha='center',fontproperties=lines.FONT,fontsize=12)
            base.save(fig,out/f'{m}_{kind}.png')
        fig,axes=base.plt.subplots(2,3,figsize=(24,14));fig.subplots_adjust(left=.015,right=.99,top=.88,bottom=.07,hspace=.38,wspace=.05)
        for x,c in enumerate(NAMES):
            r=next((r for r in rows if r['map']==m and r['controller']==c),None)
            for y,kind in enumerate(('exploration','mapping')):
                if r:panel(axes[y,x],kind,r,assets[m,c])
                else:axes[y,x].axis('off')
        fig.suptitle(m+' · 上：雷达观察范围；下：实际提交建图差异',fontproperties=lines.FONT,fontsize=22,y=.98)
        fig.text(.5,.027,'观察图是8步/4px采样估计。建图图：绿命中、蓝漏墙、深红受罚误画、黄免罚误画。',ha='center',fontproperties=lines.FONT,fontsize=13)
        base.save(fig,out/f'{m}_exploration_mapping.png')
    for r in rows:
        d=assets[r['map'],r['controller']];prefix=f"{r['map']}_{r['controller']}";w,h=d['w'],d['h']
        fig,axes=base.plt.subplots(2,3,figsize=(19,12));fig.subplots_adjust(left=.02,right=.98,top=.9,bottom=.08,hspace=.16,wspace=.06)
        temporary=d['staged'] if r['controller']=='rl16_staged' else d['local']
        labels=['真实墙体','实际雷达命中端点（下界）','累计局部暂存入口投影' if r['controller']=='rl16_staged' else '各机本机最终图并集（离线）',
            '各机准入内容并集（离线）' if r['controller']=='rl16_staged' else '各机原可信层并集（离线）','实际提交图','官方误画 / 漏画差异']
        images=[d['truth'],d['dense_seen'],temporary,d['validated'] if r['controller']=='rl16_staged' else d['trusted'],d['pred'],d['merged']]
        for ax,img,label in zip(axes.ravel(),images,labels):
            ax.imshow(img,cmap='Greys',vmin=0,vmax=1) if img.ndim==2 else ax.imshow(img)
            ax.axis('off');ax.set_title(label,fontproperties=lines.FONT,fontsize=14)
        fig.suptitle(f"{r['map']} / {NAMES[r['controller']]} / seed {r['red_seed']} / r1 / 红有效={r['red_valid']}",fontproperties=lines.FONT,fontsize=20)
        fig.text(.5,.025,'局部暂存投影不是提交图；准入并集不代表全部送达。可信层保持原坐标，临时块恢复后可重投。',ha='center',fontproperties=lines.FONT,fontsize=12)
        base.save(fig,out/f'{prefix}_layers.png')
        fig,axes=base.plt.subplots(2,3,figsize=(19,12));trace=d['trajectories']
        for slot,ax in enumerate(axes.ravel()):
            if slot>=trace.shape[1]:ax.axis('off');continue
            ax.imshow(d['truth'],cmap='Greys',vmin=0,vmax=1);base.areas(ax,d['spec'])
            active=trace['active'][:,slot];x=np.where(active,trace['x'][:,slot]+w/2,np.nan);y=np.where(active,h/2-trace['y'][:,slot],np.nan)
            jumps=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40];x[jumps]=np.nan;y[jumps]=np.nan
            ax.plot(x,y,lw=.7,color=base.plt.get_cmap('tab10')(slot));ax.axis('off');ax.set_title(f'录像槽位 {slot}',fontproperties=lines.FONT)
        fig.suptitle(f"{r['map']} / {NAMES[r['controller']]} · 各机完整实际轨迹",fontproperties=lines.FONT,fontsize=20)
        fig.tight_layout(rect=(0,0,1,.94));base.save(fig,out/f'{prefix}_per_drone.png')
        for kind in ('exploration','trajectories','bombs','walls'):
            fig,ax=base.plt.subplots(figsize=(11,8));panel(ax,kind,r,d);fig.tight_layout();base.save(fig,out/f'{prefix}_{kind}.png')


def report(rows,deltas,failures,out,planhash):
    doc=['# RL16：单墙对齐与暂存验证，六局完整红队图册','',
        'Map08 / Map09 × RL11 / A / B ×1轮，串行6红0蓝。蓝队难度、完整官方原始/有效总分均未评测。真值仅用于离线审计。',
        f'冻结计划 SHA256：`{planhash}`。同种子不等于相同观测或相同轨迹。','',
        '|地图/版本|红有效|M|W|T|实际步数|墙钟秒|弹数|平均健康|墙/机接触|真墙命中|受罚误画|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in rows:doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['red_steps']}|{r['red_walltime']:.3f}|{r['placed_count']}|{r['health']:.2f}|{r['contacts_wall']}/{r['contacts_drone']}|{r['wall_hits']}|{r['charged_pixels']}|")
    doc+=['','接触为引擎接触开始事件，不等于伤害次数。时间分取官方仿真步数。墙钟包含同一诊断链，单CPU、4GB、5400步/180秒上限。','',
        '|候选减 RL11|Δ地图分|Δ墙体分|Δ时间分|Δ真墙命中|Δ受罚误画|Δ墙钟秒|','|---|---:|---:|---:|---:|---:|---:|']
    for d in deltas:doc.append(f"|{d['map']}/{NAMES[d['controller']]}|{d['map_score']:+.4f}|{d['wall_score']:+.4f}|{d['time_score']:+.4f}|{d['wall_hits']:+d}|{d['charged_pixels']:+d}|{d['red_walltime']:+.3f}|")
    doc+=['','## 特殊区与触发证据','','|地图/版本|失效区位置误差均值px|朝向误差均值°|灰区真墙命中|灰区受罚误画|灰区实测墙面端点|灰区可见自由面积采样|','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        e=r['line_accuracy'];p=e['outage_position_summary']['mean'];a=e['outage_heading_summary']['mean']
        doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{p:.3f}|{a:.3f}|{r['gray_surface_hits']}|{r['gray_charged_pixels']}|{r['gray_observed_surface']}|{r['gray_visible_free']}|")
    doc+=['','不同轨迹/失效机步分布不能仅用均值作因果证明；空间漏搜、姿态误差、擦除、通信损失不能凭这些图精确互斥分类。','',
        '|地图/A|触发|停稳|选墙|对齐复测通过|完成短行|通过时相对角误差°|回退原因|','|---|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        if r['controller']!='rl16_wall':continue
        s=r['alignment_stats'];fallback={k.removeprefix('fallback_'):v for k,v in s.items() if k.startswith('fallback_')}
        doc.append(f"|{r['map']}|{s.get('triggered',0)}|{s.get('stopped',0)}|{s.get('wall_selected',0)}|{s.get('alignment_verified',0)}|{s.get('movement_complete',0)}|{r['alignment_verified_error_degrees']}|{fallback}|")
    doc+=['','相对角误差来自合法墙段观测，不是全局朝向真值。通过复测也不意味着沿墙方向的位置可观测。','',
        '|地图/B|暂存块|验证类型与次数|未提交原因与次数|验证格事件数|冲突格事件数|未提交格事件数|最高缓存块/帧|','|---|---:|---|---|---:|---:|---:|---|']
    for r in rows:
        if r['controller']!='rl16_staged':continue
        doc.append(f"|{r['map']}|{r['staging_stats'].get('block_staged',0)}|{r['validation_grades']}|{r['withheld_reasons']}|{r['validated_cell_events']}|{r['conflict_cell_events']}|{r['withheld_cell_events']}|{r['maximum_per_drone_cache']}|")
    doc+=['','格事件数含跨块重复，不能相加当成唯一面积。单格3×3px；累计暂存/最终准入投影的唯一像素量见 summary.json。容量检查前可瞬时出现第25块，处理后最多24块，另有一个≤24扫描的开放块。',
        '没有新增ACK。下表是离线观察到的提交端实际接收版本，不能宣称发送方知道已送达。','',
        '|地图/版本|提交机/步|实际收到的来源版本|来源序列年龄|','|---|---|---|---|']
    for r in rows:
        for rec in r['submission_receipts']:
            ages={k:rec['step']-v for k,v in rec['received_sources'].items()}
            doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{rec['submitter']}/{rec['step']}|{rec['received_sources']}|{ages}|")
    doc+=['','逐机合法观测输入的实际相等前缀/首次差异步见 summary.json 的 legal_input_comparison；包含GPS、罗盘、里程计、雷达和语义观测比较，不以种子代替检查。']
    doc+=['','## 入口及房间离线核对','',
        'Map08沿用原生真值G1中间入口、G2下方入口、G3最右下入口离线截面；中心越线与前后60帧内达到两侧±17px分开，后者为保守完整侧向通过，不代表零接触证明。Map09房间框为此前用户截图对应区域。策略未读取这些坐标。','',
        '|地图/版本|门/房间|中心越线或机身中心到访帧|完整侧向通过|','|---|---|---:|---:|']
    for r in rows:
        for k,g in r.get('offline_gates',{}).items():doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{k}|{len(g['center_crossings'])}|{g['complete_side_changes']}|")
        for k,g in r.get('offline_rooms',{}).items():doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{k}|{g['body_center_frames']}|不适用|")
    doc+=['','未到访不等于没有雷达观察；入口规划、估计坐标和真实截面不同，不能仅凭本离线表断言是窄路或罗盘导致。']
    for kind,label in KINDS.items():doc+=['',f'![{label}]({(out/f"comparison_{kind}.png").resolve().as_posix()})','',NOTES[kind]]
    for r in rows:
        prefix=f"{r['map']}_{r['controller']}";run=Path(r['attempt'])
        doc+=['',f"## {r['map']} / {NAMES[r['controller']]} / r1 / seed {r['red_seed']}",'',
            f"红有效={r['red_valid']}；回调={r['callback_errors']}；崩溃={r['controller_crashed']}；超时={r['walltime_limited']}；诊断异常={r['experiment_diagnostic_errors']}。"]
        for suffix in ('exploration','per_drone','layers','bombs','walls'):doc+=['',f'![{suffix}]({(out/f"{prefix}_{suffix}.png").resolve().as_posix()})']
        doc+=['','```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{run.resolve()}'",'```','',
            '蓝队未运行，无本局蓝回放。实际送达、每机原始事件、局部坐标扫描归档与完整合法观测输入保存在 attempt_001 内。']
    doc+=['','## 失败保留','',json.dumps(failures,ensure_ascii=False),
        '', '正式采用需更多训练/验证地图和完整红蓝对局；本图册不作全地图最优声明。']
    with (out/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(doc)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);planhash=lines.frozen(args.batch);rows=[];assets={};failures=[]
    for raw in base.read(args.batch/'results.json'):
        try:
            row,data=base.review_row(raw,planhash);evidence.audit(row,data);lines.accuracy(row,data);audit(row,data)
            rows.append(row);assets[row['map'],row['controller']]=data
        except Exception as exc:failures.append(dict(map=raw['map'],controller=raw['controller'],run=raw['red_run'],error=repr(exc)))
    deltas=[]
    for m in ('Map08','Map09'):
        old=next((r for r in rows if r['map']==m and r['controller']=='rl11_lines'),None)
        if old is None:continue
        for r in rows:
            if r['map']!=m or r['controller']=='rl11_lines':continue
            assert old['red_seed']==r['red_seed']
            r['legal_input_comparison']=compare_inputs(old,r)
            deltas.append(dict(map=m,controller=r['controller'],**{k:r[k]-old[k] for k in ('map_score','wall_score','time_score','wall_hits','charged_pixels','red_walltime','red_steps','contacts_wall','contacts_drone','health','gray_surface_hits','gray_charged_pixels','gray_visible_free')}))
    write(args.output/'summary.json',dict(rows=rows,deltas=deltas,failures=failures,blue_evaluated=False,plan_sha256=planhash))
    if len(rows)!=6 or failures:raise RuntimeError('Incomplete offline review; failures preserved in summary.json')
    figures(rows,assets,args.output);report(rows,deltas,failures,args.output,planhash)
    print(json.dumps(dict(reviewed=len(rows),failures=failures,deltas=deltas),ensure_ascii=False))


if __name__=='__main__':main()
