"""Per-map/round scores, motion, layout figures and reproducible replay commands."""
import math
from pathlib import Path
import statistics

from ... import evaluate_verified_revision as ev
from ..map_generator.official_format import read_spec
from ..dataset_manager.index import REPO,resolve_path


def statistics_of(values):
    if not values:
        return None
    ordered = sorted(values)
    at = .1*(len(ordered)-1)
    lo,hi = math.floor(at),math.ceil(at)
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
        minimum=min(values),p10=ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo),
        population_stdev=statistics.pstdev(values))


def flatten(pair):
    from .runner import phase_valid
    red,blue = pair['red'],pair['blue']
    valid = bool(phase_valid(red,'red') and blue and phase_valid(blue,'blue') and not red.get('smoke'))
    full = blue['phase1'] if valid else {}
    partial = red.get('phase1') or {}
    raw_full = (blue or {}).get('phase1') or {}
    telemetry_path = Path(pair['red_run'])/'telemetry.json'
    telemetry = ev.read(telemetry_path) if telemetry_path.is_file() else {}
    failure = None
    if not valid:
        if red.get('smoke'):
            failure = 'smoke_diagnostic'
        else:
            phase = 'red' if not phase_valid(red,'red') else 'blue'
            result = red if phase=='red' else blue or {}
            reason = (result.get('error') or
                      ('engine_callback_error' if result.get('engine_callback_errors') else None) or
                      ('walltime_limit' if result.get('walltime_limited') else None) or
                      ('controller_crash' if result.get('controller_crashed') else None) or
                      ('not_submitted' if phase=='red' and not result.get('submitted') else None) or
                      ('invalid_submission' if phase=='red' and not result.get('valid_shape') else None) or
                      ('missing_final_score' if result.get('status')=='ok' else result.get('status','skipped_no_layout')))
            failure = phase+'_'+str(reason)
    return dict(map=pair['name'],map_id=pair['map_id'],family=pair['family'],controller=pair['controller'],
        round=pair['round'],seed=pair['seed'],valid_score=valid,total_score=full.get('round_score'),
        map_score=partial.get('score_map'),wall_score=partial.get('score_wall'),time_score=partial.get('score_time'),
        difficulty_score=full.get('score_bomb'),red_partial_score=partial.get('round_score'),
        raw_total_score=raw_full.get('round_score'),raw_difficulty_score=raw_full.get('score_bomb'),
        blue_seed=(blue or {}).get('seed'),red_seed=red.get('seed',pair['seed']),
        blue_score=(blue or {}).get('blue_score'),placed_count=red.get('placed_count'),expected_count=red.get('expected_count'),
        submitted=red.get('submitted',False),valid_shape=red.get('valid_shape',False),
        red_steps=red.get('elapsed_steps'),red_walltime=red.get('elapsed_walltime'),
        blue_steps=(blue or {}).get('elapsed_steps'),blue_walltime=(blue or {}).get('elapsed_walltime'),
        timed_out=any(p and (p.get('walltime_limited') or p.get('status')=='process_timeout') for p in (red,blue)),
        task_failed=not valid or red.get('placed_count')!=red.get('expected_count'),
        failure_reason=failure,telemetry=telemetry,red_run=pair['red_run'],blue_run=pair['blue_run'])


def layout_plot(pair,output):
    # Offline visualization only; official wall coverage/diff images remain unmodified.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    areas=[]
    if pair.get('official_map'):
        import json
        from swarm_rescue.tools.map_wall_editor import load_elements,render_elements_to_image
        import cv2
        doc=json.loads(resolve_path(pair['map_file']).read_text(encoding='utf-8'))
        w,h=doc['canvas']['width'],doc['canvas']['height']
        background=cv2.cvtColor(render_elements_to_image(w,h,load_elements(doc,w,h)),cv2.COLOR_BGR2RGB)
        spec={'size':[w,h],'walls':[],'center':[-100,-100], 'zones':[]}
        for element in doc['elements']:
            if element['type'] in ('no_com_zone','no_gps_zone'):
                spec['zones'].append(dict(type=element['type'].upper(),rect=element['top_left']+element['bottom_right']))
            if element['type'] in ('start_area','disposal_center'):
                # Use the map editor's actual geometry; never invent center coordinates.
                if 'top_left' in element and 'bottom_right' in element:
                    left,top=element['top_left'];right,bottom=element['bottom_right']
                    areas.append((left,top,right,bottom,element['type']))
    else:
        spec = read_spec(resolve_path(pair['map_file']))
        background=None
    w,h = spec['size']
    fig,ax = plt.subplots(figsize=(10,7))
    if background is not None: ax.imshow(background,extent=(0,w,h,0))
    for left,top,right,bottom,kind in areas:
        ax.add_patch(Rectangle((left,top),right-left,bottom-top,fill=False,
            edgecolor='cyan' if kind=='start_area' else 'tomato',linewidth=2))
    for zone in spec.get('zones',[]):
        l,t,r,b=zone['rect'];ax.add_patch(Rectangle((l,t),r-l,b-t,color='gold' if zone['type']=='NO_COM_ZONE' else 'gray',alpha=.18))
    for _,l,t,r,b in spec['walls']:
        ax.add_patch(Rectangle((l,t),r-l,b-t,color='black'))
    x,y = spec['center'];cw,ch = spec.get('center_size',[48,48])
    ax.add_patch(Rectangle((x-cw/2,y-ch/2),cw,ch,color='tomato'))
    radius = (pair['red'].get('score_configuration') or {}).get('blast_radius',.12)*min(w,h)
    layouts = list((Path(pair['red_run'])/'artifacts').glob('*_bombs.json'))
    if len(layouts)==1:
        for index,bomb in enumerate(ev.read(layouts[0])['bombs'],1):
            x,y = bomb['x']+w/2,h/2-bomb['y']
            ax.add_patch(Circle((x,y),radius,fill=False,color='royalblue',alpha=.45))
            ax.plot(x,y,'o',color='royalblue',markersize=4)
            ax.text(x,y,str(index),fontsize=7)
    ax.set(xlim=(0,w),ylim=(h,0),aspect='equal',title=f"{pair['name']} / {pair['controller']} / round {pair['round']}")
    fig.tight_layout()
    fig.savefig(output,dpi=110)
    plt.close(fig)


def save_report(output,plan,pairs):
    rows = [flatten(p) for p in pairs]
    summaries = {}
    for controller in plan['configuration']['controllers']:
        selected = [r for r in rows if r['controller']==controller]
        summaries[controller] = dict(completed_pairs=len(selected),
            scheduled_pairs=sum(i['controller']==controller for i in plan['schedule']),
            valid_scores=sum(r['valid_score'] for r in selected),
            failure_rate=sum(r['task_failed'] for r in selected)/len(selected) if selected else None,
            total=statistics_of([r['total_score'] for r in selected if r['valid_score']]),
            by_family={f:statistics_of([r['total_score'] for r in selected if r['valid_score'] and r['family']==f])
                       for f in sorted({r['family'] for r in selected})},
            by_map={m:statistics_of([r['total_score'] for r in selected if r['valid_score'] and r['map_id']==m])
                    for m in sorted({r['map_id'] for r in selected})})
    comparison = []
    baseline = plan['configuration']['controllers'][0]
    reference = {(r['map_id'],r['round'],r['seed']):r for r in rows if r['controller']==baseline and r['valid_score']}
    for row in rows:
        key = row['map_id'],row['round'],row['seed']
        if row['controller']!=baseline and row['valid_score'] and key in reference:
            comparison.append(dict(map=row['map'],round=row['round'],seed=row['seed'],controller=row['controller'],
                baseline=baseline,delta=row['total_score']-reference[key]['total_score']))
    comparison.sort(key=lambda r:-abs(r['delta']))
    ev.atomic_json(output/'results.json',rows)
    ev.atomic_json(output/'summary.json',summaries)
    ev.atomic_json(output/'comparison.json',comparison)
    figures = output/'figures';figures.mkdir(exist_ok=True)
    lines = ['# G3 红蓝同布局评测','',
        '完整分是本地用户蓝队参照结果，不等同平台盲测分。缺参照、触限、进程/控制器异常不伪造完整分。',
        '短步数 smoke 只检查链路与行为，所有完整分强制为空。失败率按已完成任务对统计；未运行项另列。','',
        '|地图|控制器|轮次|地图分|墙体分|时间分|难度分|完整分|放弹|状态|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    def display(value):
        return '-' if value is None else f'{value:.4f}'
    for row in rows:
        lines.append(f"|{row['map']}|{row['controller']}|{row['round']}|"+
            '|'.join(display(row[k]) for k in ('map_score','wall_score','time_score','difficulty_score','total_score'))+
            f"|{row['placed_count']}/{row['expected_count']}|{'valid' if row['valid_score'] else row['failure_reason']}|")
    lines += ['', '## 每图每轮图件与回放','']
    from .atlas import round_section
    for pair,row in zip(pairs,rows):
        lines += round_section(pair,output)
        telemetry = row['telemetry']
        lines += [f"种子：{row['seed']}；红队步数：{row['red_steps']}；红队墙钟秒：{display(row['red_walltime'])}。",
                  f"提交步：{telemetry.get('submission_step')}；末弹步：{telemetry.get('last_bomb_step')}；"
                  f"提交墙钟秒：{display(telemetry.get('submission_walltime'))}；末弹墙钟秒：{display(telemetry.get('last_bomb_walltime'))}。",'',
                  '|无人机|初始携弹|已放弹|剩余弹|移动距离（像素）|',
                  '|---:|---:|---:|---:|---:|']
        for drone in telemetry.get('per_drone',[]):
            lines.append(f"|{drone['id']}|{drone['initial_inventory']}|{drone['placed_count']}|"
                         f"{drone['remaining_inventory']}|{display(drone['distance_px'])}|")
        lines.append('')
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
