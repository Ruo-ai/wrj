"""Offline S6 full-match review. Reads frozen results, never runs a simulation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Circle
from swarm_rescue.simulation.reporting.bombs_io import COMPARISON_HEADER_HEIGHT, COMPARISON_SEPARATOR_WIDTH
from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer
from swarm_rescue.solutions.devtools.optimization.evaluation.runner import load_done, validate_sources, phase_valid

NAMES={'advanced2_mapping':'A3 baseline','s6_motion':'B stop-turn','s6_radar':'C + radar'}
ORDER=list(NAMES)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rgb(bgr):return np.array(bgr[::-1])/255.


def areas(ax,spec,labels=False):
    colours={'no_gps_zone':'gray','no_com_zone':'gold','start_area':'cyan','disposal_center':'tomato'}
    for el in spec['elements']:
        kind=el['type']
        if kind not in colours or 'top_left' not in el:continue
        left,top=el['top_left'];right,bottom=el['bottom_right']
        zone=kind.startswith('no_')
        ax.add_patch(Rectangle((left,top),right-left,bottom-top,edgecolor=colours[kind],
            facecolor=colours[kind] if zone else 'none',alpha=.15 if zone else .9,lw=1.5))
        if labels and not zone:ax.text(left,top,kind,fontsize=7,color='black')


def save(fig,path):
    fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)


def review_row(row,plan_hash):
    run=Path(row['red_run']);original,checked=load_done(run.parent,plan_hash)
    assert checked==run and original is not None
    assert len(list(run.parent.glob('attempt_*')))==1,'Only one attempt permitted'
    request=read(run/'request.json');result=read(run/'result.json');resource=read(run/'resources.json')
    diagnostics=read(run/'motion_diagnostics.json')
    stats=Counter();states=Counter();errors={};samples=[];pose_stats=Counter()
    for drone in diagnostics['drones'].values():
        stats.update(drone.get('motion_stats',{}));states.update(drone['states']);pose_stats.update(drone.get('pose_stats',{}))
        samples+=drone['samples']
        for key,value in drone['outage_errors'].items():
            aggregate=errors.setdefault(key,dict(count=0,sum=0.,maximum=0.))
            aggregate['count']+=value['count'];aggregate['sum']+=value['sum']
            aggregate['maximum']=max(aggregate['maximum'],value['maximum'])
    for value in errors.values():value['mean']=value['sum']/value['count']
    contacts=diagnostics['collision_contacts'];b=result['phase1']['map_breakdown']
    summary={k:row[k] for k in ('map','controller','round','red_seed','map_score','wall_score','time_score','placed_count','red_steps','red_walltime')}
    summary.update(red_valid=phase_valid(result,'red'),blue_evaluated=False,difficulty_score=None,raw_total=None,effective_total=None,
        attempt=str(run),actual_conditions=result['actual_conditions'],callback_errors=result['engine_callback_errors'],
        controller_crashed=result['controller_crashed'],walltime_limited=result['walltime_limited'],
        health=result['mean_health'],contacts_wall=contacts.get('wall',0),contacts_drone=contacts.get('drone',0),
        wall_hits=b['wall_hits'],charged_pixels=b['charged_pixels'],map_credit=b['credit'],map_penalty=b['penalty'],
        trajectory_metric=result['trajectory_exploration'],outage_errors=errors,states=dict(states),
        motion_stats=dict(stats),pose_stats=dict(pose_stats),resources=resource,
        layout_sha256=digest(next((run/'artifacts').glob('*_bombs.json'))),
        submitted_grid_sha256=digest(next((run/'artifacts').glob('*_exploration.json'))))
    h,w=result['expected_shape'];header=COMPARISON_HEADER_HEIGHT;sep=COMPARISON_SEPARATOR_WIDTH
    png=np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
    # Decode the original lossless truth panel; verify by independently rescoring
    # the original submitted grid. No new map geometry or engine is constructed.
    truth=np.all(png[header:header+h,:w]==40,axis=2).astype(np.uint8)
    pred=np.asarray(read(next((run/'artifacts').glob('*_exploration.json')))['grid'],np.uint8)
    score=MapExplorationScorer().score(truth,pred,row['red_walltime'])
    assert int(truth.sum())==b['truth_pixels'] and abs(score.s_map*100-row['map_score'])<1e-6
    merged=png[header:header+h,2*(w+sep):2*(w+sep)+w].copy()
    wall_png=np.asarray(Image.open(next((run/'artifacts').glob('*_wall_destruction.png'))).convert('RGB'))
    wall_panel=wall_png[header:header+h,:w].copy()
    bombs=read(next((run/'artifacts').glob('*_bombs.json')))['bombs']
    with np.load(next((run/'artifacts/recordings').glob('*.npz')),allow_pickle=True) as recording:
        trajectories=recording['drone_states'].copy()
    assets=dict(truth=truth,merged=merged,wall_panel=wall_panel,bombs=bombs,trajectories=trajectories,
        samples=samples,radius=result['phase1']['wall_breakdown']['blast_radius'],
        spec=read(request['map_file']),w=w,h=h)
    return summary,assets


def figures(map_name,rows,assets,output):
    selected=[r for r in rows if r['map']==map_name]
    selected.sort(key=lambda r:ORDER.index(r['controller']))
    for kind in ('mapping','walls','bombs','trajectories'):
        fig,axes=plt.subplots(1,3,figsize=(18,5.4))
        fig.subplots_adjust(left=.03,right=.995,top=.85,bottom=.15,wspace=.05)
        for ax,row in zip(axes,selected):
            data=assets[(map_name,row['controller'])];w,h=data['w'],data['h']
            if kind=='mapping':
                ax.imshow(data['merged'],interpolation='nearest')
                title=f"M {row['map_score']:.4f} | wall hits {row['wall_hits']:,}"
            elif kind=='walls':
                ax.imshow(data['wall_panel'],interpolation='nearest')
                title=f"W {row['wall_score']:.4f} | {row['placed_count']}/10 bombs"
            else:
                ax.imshow(data['truth'],cmap='Greys',vmin=0,vmax=1,interpolation='nearest');areas(ax,data['spec'])
                if kind=='bombs':
                    for i,bomb in enumerate(data['bombs'],1):
                        x,y=bomb['x']+w/2,h/2-bomb['y']
                        ax.add_patch(Circle((x,y),data['radius'],fill=False,color='royalblue',alpha=.65,lw=1))
                        ax.plot(x,y,'o',color='royalblue',markersize=3)
                        ax.annotate(str(i),(x,y),xytext=(3,3),textcoords='offset points',fontsize=8,fontweight='bold')
                    title=f"{len(data['bombs'])} actual drops | radius {data['radius']} px"
                else:
                    t=data['trajectories']
                    for slot in range(t.shape[1]):
                        valid=t['active'][:,slot]&np.isfinite(t['x'][:,slot])&np.isfinite(t['y'][:,slot])
                        x=np.where(valid,t['x'][:,slot]+w/2,np.nan);y=np.where(valid,h/2-t['y'][:,slot],np.nan)
                        # Break possible removal/reindex discontinuities; do not
                        # draw a fictitious long flight between recording slots.
                        jump=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40]
                        x[jump]=np.nan;y[jump]=np.nan
                        ax.plot(x,y,lw=.55,alpha=.65)
                    title=f"{row['red_steps']} steps | physical trajectories"
            ax.set_title(NAMES[row['controller']]+'\n'+title,fontsize=12)
            ax.set(xlim=(0,w),ylim=(h,0));ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        seed=selected[0]['red_seed']
        fig.suptitle(f'{map_name} | round 1 | seed {seed} | full red stage, one attempt per controller',fontsize=15)
        if kind=='mapping':
            legend=[(rgb((60,170,60)),'Wall credit'),(rgb((190,150,110)),'Missed truth'),
                (rgb((110,200,240)),'Uncharged false drawing'),(rgb((0,0,105)),'Charged false drawing'),
                (rgb((175,172,168)),'Unscored wall interior')]
        elif kind=='walls':
            legend=[(rgb((70,180,70)),'Covered inner wall'),(rgb((200,90,60)),'Covered boundary'),
                ('dimgray','Uncovered wall'),(rgb((60,60,220)),'Blast area')]
        else:
            legend=[('gray','No GPS'),('gold','No communication'),('cyan','Spawn'),('tomato','Disposal area')]
        fig.legend(handles=[Patch(facecolor=c,label=label) for c,label in legend],loc='lower center',ncol=len(legend),fontsize=10)
        if kind=='bombs':fig.text(.5,.08,'Numbers are artifact order, not verified drone IDs or drop timestamps.',ha='center',fontsize=10)
        if kind=='trajectories':fig.text(.5,.08,'Colors are recording slots. A physical trajectory is not lidar observation coverage.',ha='center',fontsize=10)
        save(fig,output/f'{map_name}_{kind}.png')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();batch=args.batch.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    plan=read(batch/'plan.json');validate_sources(plan['source_sha256']);plan_hash=digest(batch/'plan.json')
    source_rows=read(batch/'results.json');assert len(source_rows)==6
    rows=[];assets={}
    for row in source_rows:
        summary,data=review_row(row,plan_hash);rows.append(summary);assets[(row['map'],row['controller'])]=data
    for map_name in ('Map08','Map09'):figures(map_name,rows,assets,output)
    deltas=[]
    for name in ('Map08','Map09'):
        baseline=next(r for r in rows if r['map']==name and r['controller']==ORDER[0])
        for row in rows:
            if row['map']==name and row['controller']!=ORDER[0]:
                deltas.append(dict(map=name,controller=row['controller'],
                    **{key:row[key]-baseline[key] for key in ('map_score','wall_score','time_score','red_steps','red_walltime')}))
    with (output/'summary.json').open('x',encoding='utf-8') as f:
        json.dump(dict(plan_sha256=plan_hash,source_batch=str(batch),rows=rows,deltas=deltas,
            receipt_verified=True,map_formula_reproduced=True,attempts=6,blue_phases=0),f,indent=2,ensure_ascii=False)
    lines=['# S6：Map08/Map09完整红阶段单轮实测','',
        '2026-11-13。用户明确要求真实地图、只测一次。本批Map08/Map09各A3/B/C一次，共6红、0蓝，无重试。',
        '全部红阶段有效、10弹；无超时、崩溃、回调异常。官方难度D和完整红蓝原始/有效总分未评测，不拼接历史蓝分。',
        f'冻结计划SHA256：`{plan_hash}`；全部attempt收据校验通过，原始提交图重新调用原地图公式与结果一致。',
        'Map08种子200806601，NO_GPS_ZONE；Map09种子2477227837，NO_GPS_ZONE+NO_COM_ZONE。5架红机、上限5400步/180秒、串行1CPU/4GB。',
        '源码A3/MyDroneRedMapping；B/MyDroneRedStopTurn；C/MyDroneRedStopTurnRadar。未切换正式入口，未改策略、蓝队、共享源码或旧结果。',
        '', '|图/版本|红有效|M|W|T|弹数|步数|红墙钟秒|墙/机碰撞接触|平均生命|',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in rows:
        lines.append(f"|{r['map']}/{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|10|{r['red_steps']}|{r['red_walltime']:.3f}|{r['contacts_wall']}/{r['contacts_drone']}|{r['health']:.2f}|")
    lines+=['','碰撞为接触开始事件，墙与机分列，不是伤害次数。墙钟包含同一诊断流程；官方时间分依仿真步数。',
        '', '|图/候选减A3|ΔM|ΔW|ΔT|Δ步数|Δ墙钟秒|','|---|---:|---:|---:|---:|---:|']
    for r in deltas:
        lines.append(f"|{r['map']}/{NAMES[r['controller']]}|{r['map_score']:+.4f}|{r['wall_score']:+.4f}|{r['time_score']:+.4f}|{r['red_steps']:+d}|{r['red_walltime']:+.3f}|")
    lines+=['','## 失效区诊断','',
        '下表是控制后、物理推进前对齐的真实位置误差；对所有有效失效机步按样本数加权，不是每机均值的平均。每组失效时段/轨迹不同，不能称相同观测输入实验。',
        '', '|图/版本|失效机步|导航均值px|地图器均值px|运动私有均值px|墙命中|收费误画像素|地图信用/扣分|TURN机步|',
        '|---|---:|---:|---:|---|---:|---:|---|---:|']
    for r in rows:
        e=r['outage_errors'];private=f"{e['motion']['mean']:.3f}" if 'motion' in e else '无'
        lines.append(f"|{r['map']}/{NAMES[r['controller']]}|{e['navigation']['count']}|{e['navigation']['mean']:.3f}|{e['mapper']['mean']:.3f}|{private}|{r['wall_hits']}|{r['charged_pixels']}|{r['map_credit']:.6f}/{r['map_penalty']:.6f}|{r['states'].get('TURN',0)}|")
    lines+=['','Map08：B地图、墙体和时间均退步。Map09：B地图增分，但墙体下降、导航与地图器误差上升，不能证明降低漂移。C两图地图分都下降，Map09更早提交换来时间分，不能归为建图改善。',
        '**结论：本轮不采用B/C，不调整参数、不重跑挑结果。** 单图单种子只能作本次观察；现有证据不支持把方案作为稳定降低定位漂移或提高综合表现的正式改动。',
        '可能因素包括停转/回退改变探索和落弹路线、更长的独立角噪声累计、A3本身匹配/缓存对新观测分布的响应；目前没有完整逐帧雷达/地图证据，不能把这些假设写成已证实因果。',
        '所有详细状态/回退/雷达接受计数、峰值Job内存、进程秒、原始布局哈希见summary.json。真值只用于离线诊断。',
        '', '## 两图对比图','',
        '每类图使用同图相同视野和图例；建图差异/墙覆盖面板直接取原始官方PNG的地图区，原图未改写。炸弹图读取本轮实际落点JSON，不使用规划目标。']
    for name in ('Map08','Map09'):
        lines += ['',f'### {name}','',f'![{name}地图提交与真值差异]({name}_mapping.png)',
            f'![{name}墙体覆盖]({name}_walls.png)',f'![{name}实际落弹]({name}_bombs.png)',
            f'![{name}实际轨迹]({name}_trajectories.png)']
    lines+=['','## 每场原始图册与回放','',f'[六场完整原始图册]({(batch/"report.md").as_posix()})',
        '原图册包含每场三类图、原始分项、资源、身份与回放。所有蓝阶段均未运行，没有本轮蓝录像。']
    for r in rows:
        lines+=['',f"{r['map']} / {NAMES[r['controller']]} / r1 / seed {r['red_seed']}",'','```powershell',
            "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{r['attempt']}'",'```']
    lines+=['','本轮任务结束，不再启动任何评测；本报告新增离线工具仅用于绘图/核验，不改变冻结策略。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),valid=sum(r['red_valid'] for r in rows),deltas=deltas),indent=2))


if __name__=='__main__':main()
