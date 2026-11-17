"""R20 three-arm validity/statistics. Only existing artifacts are analyzed."""
from pathlib import Path
import argparse
import numpy as np

from swarm_rescue.solutions.devtools.red_pair19_batch import read,write,digest,validate_plan
from swarm_rescue.solutions.devtools.red_pair19_report import evaluate_row,aggregate,METRICS
from swarm_rescue.solutions.devtools.red_explore20_diagnostics import verify_decisions
from swarm_rescue.solutions.devtools.red_pair19_results_review import truth_metric

CONTROLLERS=('rl11_lines','rl17_repair','r20_budget')


def summarize(batch,destination=None):
    plan=read(batch/'plan.json');validate_plan(batch,plan,False);rows=[]
    for index,item in enumerate(plan['schedule'],1):
        run=batch/'runs'/f"{item['name']}_r{item['round']:02d}"/item['controller']/'red/attempt_001'
        row=evaluate_row(item,run,plan,digest(batch/'plan.json'))
        if plan['configuration']['smoke']:
            actual=row.get('actual_conditions',{})
            if actual.get('max_steps')==128 and actual.get('max_walltime')==20:
                row['reasons']=[r for r in row['reasons'] if r!='engine_limits']
            row['reasons']=[r.replace('diagnostic_16_steps_not_score','diagnostic_128_steps_not_score') for r in row['reasons']]
        try:
            record=read(batch/f'run_{index:02d}.json')
            if record['changed_during_run']:row['reasons'].append('source_changed_during_run')
            trace=read(run/'explore20_diagnostics.json');checks=verify_decisions(run/'explore20_diagnostics.json')
            row['planning_replay']=checks
            if checks['mismatches'] or checks['errors'] or checks['truncated']:row['reasons'].append('planning_trace_invalid')
            row['stalled_steps']=sum(x['stalled_steps'] for x in trace['final'].values())
            row['yield_steps']=sum(x['yield_steps'] for x in trace['final'].values())
            row['lifetime_passage_episodes']=sum(x['lifetime_episodes'] for x in trace['final'].values())
            row['diagnostic_seconds']=trace['diagnostic_seconds']
            raw,truth,_=truth_metric(run)
            with np.load(run/'local13_diagnostics.npz',allow_pickle=False) as z:
                visible=np.logical_or.reduce([z[k].astype(bool) for k in z.files])
            h,w=truth.shape;v=np.repeat(np.repeat(visible,4,0),4,1)[:h,:w]
            row['sampled_lidar_free_pixels']=int((v&~truth.astype(bool)).sum())
            row['sampled_lidar_free_percent']=100*row['sampled_lidar_free_pixels']/int((~truth.astype(bool)).sum())
            row['collision_contacts']=read(run/'motion_diagnostics.json')['collision_contacts']
        except Exception as e:row['reasons'].append('trace_or_coverage:'+repr(e))
        row['execution_ok']=not [r for r in row['reasons'] if not r.startswith('diagnostic_')]
        row['valid']=row['execution_ok'] and not plan['configuration']['smoke']
        rows.append(row)
    pairs=[]
    extra=('sampled_lidar_free_percent','stalled_steps','yield_steps','lifetime_passage_episodes')
    for name,number,seed in sorted({(r['map'],r['round'],r['seed']) for r in rows}):
        arms={r['controller']:r for r in rows if (r['map'],r['round'],r['seed'])==(name,number,seed)}
        for baseline,candidate in [('rl17_repair','r20_budget'),('rl11_lines','rl17_repair')]:
            if baseline not in arms or candidate not in arms:continue
            a,b=arms[baseline],arms[candidate];valid=a['valid'] and b['valid'] and a['actual_zones']==b['actual_zones']
            delta={k:b[k]-a[k] if a.get(k) is not None and b.get(k) is not None else None for k in (*METRICS,*extra)}
            pairs.append(dict(map=name,round=number,seed=seed,baseline=baseline,candidate=candidate,valid=valid,
                              delta=delta if valid else None,raw_delta=delta))
    result=dict(scope='red stage only; sampled physical lidar visibility is approximate; no blue/full score',rows=rows,pairs=pairs,
                expected_count=len(rows),valid_count=sum(r['valid'] for r in rows),invalid_count=sum(not r['valid'] for r in rows),
                all_execution_checks_pass=all(r['execution_ok'] for r in rows),
                aggregate_by_contrast={c:aggregate([p['delta'] for p in pairs if p['valid'] and p['candidate']==c]) for c in ('r20_budget','rl17_repair')})
    write((destination or batch)/'summary.json',result)
    return result


def atlas(batch,out,result):
    from swarm_rescue.solutions.devtools import red_pair19_results_review as draw
    from swarm_rescue.solutions.devtools.red_pair19_batch import REPO
    draw.LABELS['r20_budget']='R20-B (renew RN7 allowance)'
    plan=read(batch/'plan.json');smoke=plan['configuration']['smoke']
    lines=['# R20 三版本红阶段图册','',
           '**仅128步冒烟；无有效完整飞行成绩，暂无新完整飞行图。**' if smoke else '实际完整红阶段；逐局失败原样保留。',
           '蓝队难度分及完整总分均未评测。实际雷达覆盖为每8步/4px栅格的物理射线采样近似，非精确覆盖率。',
           '真值仅用于离线评分和绘图；候选轨迹改变时不假设同种子等于同输入。','']
    headers=['地图/轮/版本','种子','实际区','M','H/L/C','W','T','弹','步/秒','雷达采样%','卡住步计数','状态/有效/原因']
    lines+=draw.table(headers,[[f"{r['map']}/r{r['round']}/{r['controller']}",r['seed'],r['actual_zones'],draw.fmt(r['map_score']),
        f"{r['wall_hits']}/{r['wall_missed']}/{r['charged_pixels']}",draw.fmt(r['wall_score']),draw.fmt(r['time_score']),r['bombs'],
        f"{r['steps']}/{draw.fmt(r['wall_seconds'])}",draw.fmt(r.get('sampled_lidar_free_percent')),r.get('stalled_steps'),
        f"{r['status']}/{r['valid']}/{r['reasons']}"] for r in result['rows']])
    lines+=['','## 配对差值及退步（候选减对照）','']
    for p in result['pairs']:lines.append(f"- {p['map']} r{p['round']} seed{p['seed']} / {p['candidate']}−{p['baseline']} / 有效={p['valid']}：{p['delta'] if p['valid'] else '无有效差值；原始诊断差值见JSON'}")
    errors=[]
    for name,number,seed in sorted({(r['map'],r['round'],r['seed']) for r in result['rows']}):
        rows=[r for c in CONTROLLERS for r in result['rows'] if r['map']==name and r['round']==number and r['controller']==c]
        lines+=['',f'## {name} / r{number} / seed {seed}','']
        try:
            arms=[]
            for row in rows:
                run=Path(row['attempt']);raw,truth,metric=draw.truth_metric(run);h,w=truth.shape
                with np.load(draw.only(run/'artifacts/recordings','*.npz'),allow_pickle=False) as z:trajectories=z['drone_states']
                with np.load(run/'local13_diagnostics.npz',allow_pickle=False) as z:visible=np.logical_or.reduce([z[k].astype(bool) for k in z.files])
                maps=list((run/'artifacts').glob('*_exploration.json'));walls=list((run/'artifacts').glob('*_wall_destruction.png'))
                from PIL import Image
                wall=np.asarray(Image.open(walls[0]).convert('RGB'))[draw.plot.COMPARISON_HEADER_HEIGHT:draw.plot.COMPARISON_HEADER_HEIGHT+h,:w] if walls else None
                arms.append(dict(row=row,raw=raw,truth=truth,metric=metric,spec=read(read(run/'request.json')['map_file']),
                    trajectories=trajectories,visible=np.repeat(np.repeat(visible,4,0),4,1)[:h,:w],
                    pred=np.asarray(read(maps[0])['grid'],np.uint8) if maps else None,
                    bombs=read(draw.only(run/'artifacts','*_bombs.json'))['bombs'],walls=wall))
            for kind in draw.KINDS:
                fig,axes=draw.plot.plt.subplots(1,len(arms),figsize=(8*len(arms),6.5),squeeze=False)
                for ax,arm in zip(axes[0],arms):
                    if (kind=='mapping' and arm['pred'] is None) or (kind=='walls' and arm['walls'] is None):
                        ax.imshow(arm['truth'],cmap='Greys');ax.axis('off')
                        ax.set_title(draw.LABELS[arm['row']['controller']]+' / NO SUBMISSION OR SCORE IMAGE')
                        ax.text(.5,.5,'Truth background only / missing result',transform=ax.transAxes,ha='center',color='red')
                    else:draw.draw_arm(ax,arm,kind)
                label=f'{name} / r{number} / seed {seed} / {kind} / '+('SMOKE128, NO VALID FLIGHT SCORE' if smoke else 'RED STAGE ONLY')
                fig.suptitle(label,fontsize=12);fig.text(.5,.015,draw.NOTES[kind],fontproperties=draw.FONT,ha='center')
                fig.tight_layout(rect=(0,.04,1,.93));path=out/f'{name}_r{number}_{kind}.png';draw.plot.save(fig,path)
                lines+=draw.embed(path,label)
        except Exception as e:errors.append(dict(map=name,round=number,error=repr(e)));lines+=['图件缺口：'+repr(e)]
        for r in rows:lines+=['',r['controller']+' 原始录像：','','```powershell',f"Set-Location -LiteralPath '{REPO}'",fr".\replay_map.cmd '{r['attempt']}'",'```']
    draw.put_text(out/'report.md',lines);write(out/'figure_errors.json',errors)


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);r=summarize(a.batch.resolve(),a.output.resolve())
    atlas(a.batch.resolve(),a.output.resolve(),r)
    print('valid',r['valid_count'],'/',r['expected_count'],a.output)


if __name__=='__main__':main()
