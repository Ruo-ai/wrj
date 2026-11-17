"""Read-only paired statistics and new-directory all-round red atlas."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from swarm_rescue.solutions.devtools.red_pair19_batch import read, write, digest, validate_plan, CONTROLLERS, REPO
from swarm_rescue.solutions.devtools.optimization.evaluation import runner

METRICS = ('map_score','wall_hits','wall_missed','charged_pixels','wall_score','time_score','bombs','wall_seconds')


def evaluate_row(item, attempt, plan, plan_sha):
    row = dict(map=item['name'], controller=item['controller'], round=item['round'], seed=item['seed'],
        attempt=str(attempt), map_sha256=plan['source_sha256'][item['map_file']],
        difficulty_score=None, full_score=None, valid=False, execution_ok=False,
        status='not_run', reasons=[], actual_zones=None, steps=None, wall_seconds=None,
        process_seconds=None, bombs=None, map_score=None, wall_score=None, time_score=None,
        wall_hits=None, wall_missed=None, charged_pixels=None)
    if not attempt.exists():
        row['reasons']=['not_run']; return row
    try:
        result, checked = runner.load_done(attempt.parent, plan_sha)
        if result is None or checked != attempt:
            raise ValueError('Incomplete attempt; no verified done receipt')
        request=read(attempt/'request.json'); resource=read(attempt/'resources.json')
        row.update(status=result.get('status'), steps=result.get('elapsed_steps'),
            wall_seconds=result.get('elapsed_walltime'),process_seconds=resource.get('process_seconds'),
            bombs=result.get('placed_count'), actual_zones=result.get('actual_conditions',{}).get('instantiated_zone_types'),
            resources=resource, actual_conditions=result.get('actual_conditions'), exit_code=result.get('exit_code'))
        reasons=[]; smoke=plan['configuration']['smoke']
        expected=plan['configuration']['controller_identities'][item['controller']]
        if any(request.get(k)!=v or result.get(k)!=v for k,v in expected.items()): reasons.append('controller_identity')
        if any(request.get(k)!=item[k] for k in ('map','round','seed')): reasons.append('pair_identity')
        if result.get('status')!='ok' or result.get('exit_code')!=0: reasons.append('process_or_engine_failure')
        if result.get('controller_crashed') or result.get('engine_callback_errors'): reasons.append('controller_or_callback_exception')
        if result.get('walltime_limited'): reasons.append('walltime_limit')
        if resource.get('cpu_count')!=1 or resource.get('memory_limit_bytes')!=4_000_000_000: reasons.append('resource_contract')
        actual=result.get('actual_conditions',{})
        if actual.get('max_steps')!=(16 if smoke else 5400) or actual.get('max_walltime')!=(20 if smoke else 180): reasons.append('engine_limits')
        if actual.get('drones')!=5 or actual.get('instantiated_zone_types')!=item['scenario']['zones']: reasons.append('actual_zones_or_team_size')
        identity=read(attempt/'loaded_identity.json')
        if identity['controller']!=expected['controller_module']+'.'+expected['controller_class']: reasons.append('loaded_controller')
        expected_mapper=('swarm_rescue.solutions.my_drone_red_lines11.WallLineMapper' if item['controller']=='rl11_lines' else
                         'swarm_rescue.solutions.my_drone_red_repair17.RepairMapper17')
        observed=identity.get('observed_instances',{})
        if len(observed)!=5 or any(v['controller']!=identity['controller'] or v['mapper']!=expected_mapper for v in observed.values()):
            reasons.append('actual_drone_or_mapper_class')
        if any(plan['source_sha256'].get(k)!=h for k,h in identity['loaded_sources'].items()): reasons.append('loaded_source_hash')
        row['loaded_controller']=identity['controller']
        row['controller_source_sha256']=plan['source_sha256']['src/'+expected['controller_module'].replace('.','/')+'.py']
        log=(attempt/'run.log').read_text(encoding='utf-8',errors='replace')
        if any(token in log for token in ('Traceback (most recent call last)', 'Exception ignored', 'AssertionError')):
            reasons.append('exception_in_log')
        for filename in ('exploration_diagnostics.json','line_diagnostics.json','local13_diagnostics.json','p2l17_diagnostics.json'):
            doc=read(attempt/filename)
            if doc.get('errors'): reasons.append(filename+':errors')
        if not read(attempt/'line_diagnostics.json').get('mapper_inputs_complete'): reasons.append('incomplete_legal_inputs')
        if not smoke and not runner.phase_valid(result,'red'): reasons.append('invalid_red_stage')
        row['execution_ok']=not reasons
        row['valid']=not smoke and not reasons
        if smoke: reasons.append('diagnostic_16_steps_not_score')
        row['reasons']=reasons
        phase=result.get('phase1') or {}; b=phase.get('map_breakdown') or {}
        row.update(map_score=phase.get('score_map'),wall_score=phase.get('score_wall'),time_score=phase.get('score_time'),
            wall_hits=b.get('wall_hits'), wall_missed=b['wall_pixels']-b['wall_hits'] if 'wall_pixels' in b else None,
            charged_pixels=b.get('charged_pixels'))
    except Exception as error:
        row['reasons'].append(repr(error))
    return row


def aggregate(rows):
    result={}
    for metric in METRICS:
        values=[r[metric] for r in rows if r.get(metric) is not None]
        result[metric]=dict(n=len(values),mean=statistics.mean(values) if values else None,
            median=statistics.median(values) if values else None,min=min(values) if values else None,
            max=max(values) if values else None)
    return result


def summarize(batch, destination=None):
    plan=read(batch/'plan.json'); validate_plan(batch,plan,check_live=False)
    plan_sha=digest(batch/'plan.json'); rows=[]
    for index,item in enumerate(plan['schedule'],1):
        attempt=batch/'runs'/f"{item['name']}_r{item['round']:02d}"/item['controller']/'red/attempt_001'
        row=evaluate_row(item,attempt,plan,plan_sha)
        record=batch/f'run_{index:02d}.json'
        if record.exists() and read(record).get('changed_during_run'):
            row['valid']=row['execution_ok']=False;row['reasons'].append('source_changed_during_run')
        rows.append(row)
    grouped=defaultdict(dict)
    for r in rows: grouped[r['map'],r['round'],r['seed']][r['controller']]=r
    pairs=[]
    for (name,number,seed), arms in grouped.items():
        a,b=arms['rl11_lines'],arms['rl17_repair']; valid=a['valid'] and b['valid'] and a['actual_zones']==b['actual_zones']
        raw={k:b[k]-a[k] if b[k] is not None and a[k] is not None else None for k in METRICS}
        pairs.append(dict(map=name,round=number,seed=seed,valid=valid,raw_delta=raw,
            delta=raw if valid else None, regression=valid and (any(raw[k]<0 for k in ('map_score','wall_score','time_score','bombs','wall_hits')) or
                any(raw[k]>0 for k in ('wall_missed','charged_pixels'))),
            reasons=[] if valid else [a['reasons'],b['reasons']]))
    valid_pairs=[r['delta'] for r in pairs if r['valid']]
    report=dict(scope='red stage only; difficulty/full score not evaluated',rows=rows,pairs=pairs,
        expected_count=len(plan['schedule']), valid_count=sum(r['valid'] for r in rows),
        invalid_count=sum(not r['valid'] for r in rows),
        all_execution_checks_pass=all(r['execution_ok'] for r in rows),
        paired_aggregate=aggregate(valid_pairs),
        unpaired_valid_aggregate={c:aggregate([r for r in rows if r['controller']==c and r['valid']]) for c in CONTROLLERS},
        per_map_paired_aggregate={name:aggregate([p['delta'] for p in pairs if p['valid'] and p['map']==name]) for name in sorted({p['map'] for p in pairs})},
        worst_map_score_by_version={c:min([r for r in rows if r['controller']==c and r['valid']],key=lambda r:r['map_score'],default=None) for c in CONTROLLERS},
        regressions=[p for p in pairs if p['regression']],limitations=['12 paired training cases, six correlated maps; no significance or universal superiority claim.'])
    write((destination or batch)/'summary.json',report)
    return report


def atlas(batch, output):
    output.mkdir(parents=True,exist_ok=False)
    report=summarize(batch,output)
    plan=read(batch/'plan.json')
    import numpy as np
    from PIL import Image
    from swarm_rescue.solutions.devtools import red_stopturn_match_report as b
    from swarm_rescue.solutions.devtools.red_recovery18_audit import Metrics
    from swarm_rescue.solutions.devtools.red_lines11_review import FONT
    def fmt(v): return '未评测/缺失' if v is None else (f'{v:.4f}' if isinstance(v,float) else str(v))
    lines=['# RL11 / RL17-R 配对红队阶段图册','',
        '蓝队难度分、完整总分：未评测。所有失败与未运行局保留；仅双方有效的配对进入差值汇总。',
        f"{len(plan['map_rows'])}张训练图、{len(report['pairs'])}配对的小样本；不作统计显著性或独立测试集结论。",
        '**仅16步冒烟，暂无新完整飞行图、无有效比赛成绩。**' if plan['configuration']['smoke'] else '以下是本批实际完整阶段记录，失败同列。',
        '', '|地图/轮次/版本|种子|实际区域|M|命中/漏画/受罚|W|T|弹|步/秒|有效/状态|',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in report['rows']:
        lines.append(f"|{r['map']}/r{r['round']}/{r['controller']}|{r['seed']}|{r['actual_zones']}|{fmt(r['map_score'])}|{r['wall_hits']}/{r['wall_missed']}/{r['charged_pixels']}|{fmt(r['wall_score'])}|{fmt(r['time_score'])}|{r['bombs']}|{r['steps']}/{fmt(r['wall_seconds'])}|{r['valid']}/{r['status']}|")
    lines+=['','## 配对差值：RL17-R 减 RL11','','|地图/轮次|有效|ΔM|Δ命中|Δ漏画|Δ受罚|ΔW|ΔT|Δ弹|Δ秒|','|---|---|---|---|---|---|---|---|---|---|']
    for p in report['pairs']:
        lines.append(f"|{p['map']}/r{p['round']}|{p['valid']}|"+'|'.join(fmt(p['delta'][k]) if p['delta'] else '无有效配对' for k in METRICS)+'|')
    lines+=['','## 有效配对差值分布','','|指标|n|均值|中位数|最小|最大|','|---|---|---|---|---|---|']
    for k,v in report['paired_aggregate'].items():lines.append('|'+k+'|'+'|'.join(fmt(v[x]) for x in ('n','mean','median','min','max'))+'|')
    lines+=['','## 最差有效局与所有退步','']
    for c,r in report['worst_map_score_by_version'].items():
        lines.append(f"- {c} 最低地图分局："+(f"{r['map']} / r{r['round']} / seed {r['seed']} / M={r['map_score']:.4f}" if r else '无有效成绩'))
    for p in report['regressions']:
        lines.append(f"- 退步：{p['map']} / r{p['round']} / seed {p['seed']}；{p['delta']}")
    figure_errors=[]
    for r in report['rows']:
        run=Path(r['attempt']); label=f"{r['map']} / {r['controller']} / r{r['round']} / seed {r['seed']} / valid={r['valid']}"
        lines+=['',f'## {label}','',f"状态：{r['status']}；原因：{r['reasons']}；实际区域：{r['actual_zones']}。"]
        try:
            raw=read(run/'result.json'); h,w=raw['expected_shape']
            png=np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
            truth=np.all(png[b.COMPARISON_HEADER_HEIGHT:b.COMPARISON_HEADER_HEIGHT+h,:w]==40,axis=2).astype(np.uint8)
            submitted=list((run/'artifacts').glob('*_exploration.json'))
            pred=np.asarray(read(submitted[0])['grid'],np.uint8) if submitted else None
            metric=Metrics(truth,(0,0,w-1,h-1))
            if pred is not None and abs(metric.count(pred)['map_score']-r['map_score'])>1e-6: raise ValueError('Official map score does not reproduce')
            request=read(run/'request.json'); spec=read(request['map_file'])
            with np.load(next((run/'artifacts/recordings').glob('*.npz')),allow_pickle=False) as z: trajectories=z['drone_states']
            bombs=read(next((run/'artifacts').glob('*_bombs.json')))['bombs']
            with np.load(run/'local13_diagnostics.npz',allow_pickle=False) as z:
                visible=np.logical_or.reduce([z[k].astype(bool) for k in z.files])
            visible=np.repeat(np.repeat(visible,4,0),4,1)[:h,:w]
            notes={'mapping':'绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。',
                'exploration':'绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率。',
                'trajectories':'颜色=录像槽位；灰=无GPS/罗盘；黄=无通信；青=出生区。',
                'walls':'绿=内墙覆盖；蓝=边界覆盖；灰=未覆盖；红=爆炸范围。',
                'bombs':'蓝圆=实际落弹/爆炸半径；编号=产物顺序，非机号。灰/黄=特殊区。'}
            for kind,note in notes.items():
                fig,ax=b.plt.subplots(figsize=(12,8))
                if kind=='mapping':
                    if pred is None:
                        ax.imshow(truth,cmap='Greys',vmin=0,vmax=1)
                        ax.text(.5,.5,'NO SUBMITTED GRID',transform=ax.transAxes,ha='center',color='red',fontsize=20)
                        note='未提交地图；仅显示真值背景，不能视为建图结果或有效地图分。'
                    else:ax.imshow(metric.rgb(pred))
                elif kind=='exploration':
                    img=np.full((*truth.shape,3),.97);img[visible]=[.55,.84,.69];img[truth.astype(bool)]=.12
                    ax.imshow(img);b.areas(ax,spec)
                elif kind=='walls':
                    wall=np.asarray(Image.open(next((run/'artifacts').glob('*_wall_destruction.png'))).convert('RGB'))
                    ax.imshow(wall[b.COMPARISON_HEADER_HEIGHT:b.COMPARISON_HEADER_HEIGHT+h,:w])
                else:
                    ax.imshow(truth,cmap='Greys',vmin=0,vmax=1);b.areas(ax,spec)
                    if kind=='bombs':
                        for i,bomb in enumerate(bombs,1):
                            x,y=bomb['x']+w/2,h/2-bomb['y'];ax.plot(x,y,'bo',ms=3)
                            ax.add_patch(b.Circle((x,y),raw['phase1']['wall_breakdown']['blast_radius'],fill=False,color='royalblue'))
                            ax.annotate(str(i),(x,y),fontsize=9)
                    else:
                        for slot in range(trajectories.shape[1]):
                            t=trajectories[:,slot];good=t['active']&np.isfinite(t['x'])&np.isfinite(t['y'])
                            x=np.where(good,t['x']+w/2,np.nan);y=np.where(good,h/2-t['y'],np.nan)
                            jump=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40];x[jump]=np.nan;y[jump]=np.nan
                            ax.plot(x,y,lw=.7,label=f'slot {slot}')
                        ax.legend(fontsize=8)
                ax.set(xlim=(0,w),ylim=(h,0),aspect='equal');ax.set_title(label,fontproperties=FONT,fontsize=12)
                fig.text(.5,.02,note,fontproperties=FONT,ha='center',fontsize=10)
                image=output/f"{r['map']}_r{r['round']}_{r['controller']}_{kind}.png"
                b.save(fig,image);lines+=['',f'![{label} {kind}]({image.as_posix()})',note]
        except Exception as error:
            figure_errors.append(dict(attempt=str(run),error=repr(error)))
            lines+=['',f'图件缺口（原始失败保留）：{error!r}']
        lines+=['','```powershell',f"Set-Location -LiteralPath '{REPO}'",f".\replay_map.cmd '{run}'",'```']
    lines+=['','事实：见逐局原始记录与有效配对。原因与后续机制：尚未归因，不自动采用或切换入口。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(output/'figure_errors.json',figure_errors)
    print(output/'report.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atlas(a.batch.resolve(),a.output.resolve())


if __name__=='__main__':main()
