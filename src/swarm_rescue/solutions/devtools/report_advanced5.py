"""Offline block1 motion, score, figure and replay audit; never imported online."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib
from pathlib import Path
import json
import statistics
import numpy as np
from swarm_rescue.solutions.devtools import evaluate_verified_revision as evaluation

ROOT = evaluation.ROOT
LABELS = {'advanced2': '进阶2基线', 'advanced5': '进阶5板块1候选'}
FIELDS = ('score_map', 'score_wall', 'score_time', 'score_bomb', 'blue_score', 'round_score')
read = evaluation.read


def events(run):
    return [json.loads(line.split(' ', 1)[1])
            for line in (run/'run.log').read_text(encoding='utf-8').splitlines()
            if line.startswith(('V1_EVENT ', 'V2_EVENT ', 'V3_EVENT '))]


def inspect_run(run):
    row=read(run/'result.json'); ev=events(run)
    with np.load(next((run/'artifacts/recordings').glob('*.npz')), allow_pickle=False) as z:
        s=z['drone_states']; xy=np.stack((s['x'],s['y']),axis=-1)
        health=s['health'][-1].tolist()
    horizon=min(len(xy),int(.68*row['max_steps']))
    motion=[]
    for drone in range(xy.shape[1]):
        p=xy[:horizon,drone]; covered=np.zeros(len(p),bool);broad=np.zeros(len(p),bool)
        windows=stationary=0; candidates=[]
        for start in range(0,max(0,len(p)-79),10):
            segment=p[start:start+80]
            travel=float(np.linalg.norm(np.diff(segment,axis=0),axis=1).sum())
            extent=float(np.ptp(segment,axis=0).max())
            net=float(np.linalg.norm(segment[-1]-segment[0])); windows+=1
            stationary+=travel<24
            if travel>=80 and net<=20 and extent<=70:
                covered[start:start+80]=True
                candidates.append(dict(start=start+1,end=start+80,travel=travel,extent=extent,net=net))
        broad_candidates=[]
        for start in range(0,max(0,len(p)-119),10):
            q=p[start:start+120]
            travel=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
            net=float(np.linalg.norm(q[-1]-q[0]));extent=float(np.linalg.norm(np.ptp(q,axis=0)))
            if travel>260 and net<80 and extent<220:
                broad[start:start+120]=True
                broad_candidates.append(dict(start=start+1,end=start+120,travel=travel,extent=extent,net=net))
        motion.append(dict(drone=drone,windows=windows,suspected_oscillation=candidates,
                           broad_cycles=broad_candidates,broad_frames=int(broad.sum()),
                           combined_frames=int((covered|broad).sum()),
                           suspected_frames=int(covered.sum()),stationary_windows=stationary))
    verified=[e for e in ev if e['event']=='loop_recovery_verified']
    return dict(map=row['map'],round=row['round'],run=str(run.resolve()),status=row['status'],
        score=row['phase1'],placed=row['placed_count'],submitted=row['submitted'],health=health,
        mean_health=statistics.mean(health),exploration_horizon=horizon,motion=motion,
        event_counts=dict(Counter(e['event'] for e in ev)),recovery_verifications=verified,
        suspected_frames=sum(m['suspected_frames'] for m in motion),available_frames=horizon*xy.shape[1],
        broad_frames=sum(m['broad_frames'] for m in motion),
        combined_frames=sum(m['combined_frames'] for m in motion),
        suspected_windows=sum(len(m['suspected_oscillation']) for m in motion),
        windows=sum(m['windows'] for m in motion),
        controller_confirmed_samples=sum(bool(e.get('confirmed')) for e in ev if e['event']=='loop_progress'),
        elapsed_steps=row['elapsed_steps'],walltime=row['elapsed_walltime'])


def audit(base):
    records={s:[inspect_run(p.parent) for p in sorted((base/s/'red').glob('*/result.json'))
                if read(p).get('status')=='ok' and (p.parent/'artifacts').exists()] for s in LABELS}
    (base/'report').mkdir(exist_ok=True)
    evaluation.atomic_json(base/'report/movement_audit.json',records)
    for s,rows in records.items():
        for r in rows:
            print(s,r['map'],r['round'],'frames',r['suspected_frames'],'health',r['mean_health'],
                  'partial',round(r['score']['round_score'],3),'steps',r['elapsed_steps'],
                  'broad',r['broad_frames'],'combined',r['combined_frames'])
    return records


def paired(base,selector,expected):
    reds={(r['map'],r['round']):r for r in read(base/selector/'red/results.json')}
    blues={(r['map'],r['round']):r for r in read(base/selector/'blue/results.json')}
    if set(reds)!=expected or set(blues)!=expected: raise ValueError('Incomplete rounds')
    rows=[]
    for key in sorted(expected):
        r,b=reds[key],blues[key]
        if r['status']!='ok' or b['status']!='ok' or not r['submitted']: raise ValueError(f'Failed round {key}')
        if r['seed']!=b['seed'] or b['phase1']['score_status']!='final': raise ValueError('Mismatch')
        run=base/selector/'red'/evaluation.round_name(r)
        layout=next((run/'artifacts').glob('*_bombs.json'))
        if evaluation.digest(layout)!=b['layout_sha256']: raise ValueError('Layout changed')
        if evaluation.digest(run/'result.json')!=b['red_result_sha256']: raise ValueError('Red result changed')
        score=b['phase1']
        recomputed=evaluation.blue.merge(r,b['blue_score'],b['disposed_percent'],b['full_disposal_timestep'],b['controller_crashed'])
        if recomputed!=score: raise ValueError('Original score manager disagrees')
        rows.append(dict(map=key[0],round=key[1],seed=r['seed'],
            **{f:score[f] for f in FIELDS if f!='blue_score'},blue_score=b['blue_score'],
            red_walltime_limited=r['walltime_limited'],blue_walltime_limited=b['walltime_limited']))
    return rows


def graphics(base,maps,records):
    evaluation.red.configure_ray_backend('cpu')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreManager,walls_and_boundary_grid
    wall_audit=[]
    for name in maps:
        cls=getattr(importlib.import_module(f'swarm_rescue.maps.map_{name[-2:]}'),name)
        world=cls(drone_type=DroneMotionless,number_drones=1)
        walls,boundary=walls_and_boundary_grid(world._playground)
        weight=np.where(boundary,.5,1.);h,w=walls.shape
        for selector,rows in records.items():
            folder=base/'report'/selector;folder.mkdir(exist_ok=True)
            for record in rows:
                if record['map']!=name:continue
                run=Path(record['run']);stem=run.name
                bombs=read(next((run/'artifacts').glob('*_bombs.json')))['bombs']
                breakdown=record['score']['wall_breakdown'];radius=breakdown['blast_radius']
                count=np.zeros(walls.shape,np.uint8)
                for bomb in bombs:
                    count+=PlaceScoreManager.wall_destruction_masks(
                        walls,[[bomb['x'],bomb['y']]],radius,boundary)[2].astype(np.uint8)
                union=float(weight[count>0].sum())
                assert int(walls.sum())==breakdown['total_wall_pixels']
                assert int((count>0).sum())==breakdown['destroyed_wall_pixels']
                assert union==breakdown['weighted_destroyed']
                wall_audit.append(dict(controller=selector,map=name,round=record['round'],
                    weighted_union=union,overlap_percent=100*(1-union/max(1.,float((weight*count).sum())))))
                fig,ax=plt.subplots(figsize=(10,7))
                ax.imshow(walls,cmap=ListedColormap(['#fafafa','#555555']),origin='upper',
                          extent=(-w/2,w/2,-h/2,h/2),interpolation='nearest')
                for i,b in enumerate(bombs,1):
                    ax.add_patch(Circle((b['x'],b['y']),radius,fill=False,linestyle='--',color='#bd7a17',alpha=.7))
                    ax.scatter(b['x'],b['y'],s=170,c='#f6b43c',edgecolors='black',zorder=3)
                    ax.text(b['x'],b['y'],str(i),ha='center',va='center',fontsize=9,zorder=4)
                ax.set(title=f'{selector} / {stem}: exported bombs, offline truth background',
                       xlabel='World X (px)',ylabel='World Y (px)')
                ax.set_aspect('equal');fig.tight_layout()
                fig.savefig(folder/(stem+'_bomb_layout.png'),dpi=130);plt.close(fig)
                with np.load(next((run/'artifacts/recordings').glob('*.npz')),allow_pickle=False) as data:
                    s=data['drone_states'];xy=np.stack((s['x'],s['y']),axis=-1)
                fig,ax=plt.subplots(figsize=(10,7))
                ax.imshow(walls,cmap=ListedColormap(['#ffffff','#dddddd']),origin='upper',
                          extent=(-w/2,w/2,-h/2,h/2),interpolation='nearest')
                for i,m in enumerate(record['motion']):
                    p=xy[:record['exploration_horizon'],i]
                    ax.plot(p[::3,0],p[::3,1],linewidth=.7,alpha=.65,label=f'Drone {i}')
                    for e in m['suspected_oscillation']+m['broad_cycles']:
                        segment=p[e['start']-1:e['end']]
                        ax.plot(segment[:,0],segment[:,1],color='red',linewidth=1.2,alpha=.45)
                ax.set(title=f'{selector} / {stem}: exploration; red = suspected loop windows',
                       xlabel='World X (px)',ylabel='World Y (px)')
                ax.set_aspect('equal');ax.legend();fig.tight_layout()
                fig.savefig(folder/(stem+'_motion.png'),dpi=130);plt.close(fig)
        world._playground.window.close()
        print('FIGURES',name,flush=True)
    evaluation.atomic_json(base/'report/wall_overlap.json',wall_audit)


def link(path):return Path(path).resolve().as_posix()


def final_report(base,document,records):
    plan=read(base/'plan.json');completion=read(base/'completion.json')
    if not completion['complete'] or not completion['sources_unchanged']:raise ValueError('Batch incomplete')
    if evaluation.digest(base/'source_snapshot.zip')!=plan['source_snapshot_sha256']:raise ValueError('Archive changed')
    for selector in LABELS:
        for phase in ('red','blue'):
            manifest=base/selector/phase/'manifest.json'
            evaluation.verify_sources(read(manifest))
            for r in read(base/selector/phase/'results.json'):
                evaluation.validate_receipt(base/selector/phase/evaluation.round_name(r),evaluation.digest(manifest))
    freeze=read(ROOT/'output/red_advanced5/block1/freeze.json')
    changed=[p for p,sha in freeze['protected_sha256'].items() if evaluation.digest(ROOT/p)!=sha]
    if changed:raise ValueError(f'Protected files changed: {changed}')
    expected={(m,r) for m in plan['maps'] for r in range(1,plan['rounds']+1)}
    scores={s:paired(base,s,expected) for s in LABELS}
    if [(r['map'],r['round'],r['seed']) for r in scores['advanced2']]!=[(r['map'],r['round'],r['seed']) for r in scores['advanced5']]:raise ValueError('Seeds differ')
    means={s:{m:{f:statistics.mean(r[f] for r in rows if m=='overall' or r['map']==m)
        for f in FIELDS} for m in [*plan['maps'],'overall']} for s,rows in scores.items()}
    evaluation.atomic_json(base/'report/full_scores.json',dict(scores=scores,means=means))
    a,b=means['advanced2']['overall'],means['advanced5']['overall']
    lines=['# 进阶阶段5：板块1——循环与徘徊优化总结','',
        '本次按《进阶5优化.md》的顺序，只实施第一个板块。窄路通过和动态协同等待用户确认后再做。以当前最好已采用的进阶2为基线，新增自包含候选 `my_drone_red_advanced5.py`；原三版红队、用户蓝队、地图、引擎和默认入口均保留。','',
        f'同期{len(plan["maps"])}图{plan["rounds"]}轮完整均分：进阶2 **{a["round_score"]:.4f}** → 板块1 **{b["round_score"]:.4f}**，变化 **{b["round_score"]-a["round_score"]:+.4f}**。当前默认入口仍为进阶2，供用户先查看本板块效果。','',
        '## 方法与代码逻辑','',
        '旧版的局部循环、低速、锚点位移、路点距离超时会分别触发恢复。`no_progress` 会取消正在执行的脱困动作，路点更换又会重置距离计时。本次把进展判断、恢复执行和恢复验证连接起来。','',
        '1. 连续观察：每6步采样，最多21个样本，监测净位移、行程、范围、路线投影进度、新观察格子、反向和目标切换。路点切换不清空运动历史。',
        '2. 分类确认：区分静止、小范围往返、目标切换和局部绕圈；有地图信息或路线进展时避免急于脱困，短暂原有让行、布弹制动单独处理。',
        '3. 有期限地恢复：不反复重启尚未执行完的脱困；确认徘徊后尝试短时局部修正，仍无进展再调用原方向搜索。',
        '4. 验证而非仅计数：记录恢复前的位置和信息量，检查是否离开问题区域或取得新信息；失败记录有期限，不向地图写假墙。',
        '5. 限定条件下保留目标：若已经出现低收益频繁换目标，新墙使路线失效时尝试重算到原目标的安全路径。承诺期最多90步且不因修路续期，每个目标最多修两次；不可达、到期或安全检查不通过仍释放。失败逃逸方向只在同一位置附近、记忆有效期内受罚，几何安全约束不放宽。',
        '6. 本次边界：不改布弹收益函数，不新增精细窄缝规划和多机任务调配。局部动力学预测只用于已确认的徘徊，是有预算的近似检查，不是完整MPC。','',
        '## 评测条件与评分口径','',
        f'CPU射线；5架无人机，每机2枚；{len(plan["maps"])}图×{plan["rounds"]}轮×2版本×红蓝两侧，共{completion["completed"]}/{completion["expected"]}次。按轮交替版本顺序，同地图同轮同种子，单仿真串行，保留原时间步及墙钟限制。','',
        '总分=地图×0.5+墙体×0.2+时间×0.1+红队难度×0.2。各分项100分；难度由原评分器根据用户蓝队原分及布弹完成度换算，逐轮计算再平均。这里只测试现有用户蓝队，不等于平台参考组成绩。','',
        '## 每张地图均分与前版对比','',
        '| 地图 | 进阶2总分 | 板块1总分 | 变化 | 地图分 | 墙体分 | 时间分 | 红队难度 | 蓝队原分 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in [*plan['maps'],'overall']:
        x,y=means['advanced2'][m],means['advanced5'][m]
        lines.append(f'| {m} | {x["round_score"]:.4f} | {y["round_score"]:.4f} | {y["round_score"]-x["round_score"]:+.4f} | '+' | '.join(f'{y[f]:.2f}' for f in FIELDS[:-1])+' |')
    lines+=['','## 徘徊与损伤的独立审计','',
        '两版使用同一录像判据，限探索前68%时间预算：①每10步取80步窗口，行程≥80像素、首尾位移≤20像素、最大坐标跨度≤70像素，筛小范围往返；②取120步窗口，行程>260像素、首尾位移<80像素、范围对角线<220像素，筛较大局部绕圈。下表为两类时间并集占比，重叠不重复累计。此指标可能包含必要绕行，不能证明所有循环已消失；后段集合与停机不计入。各类原始计数见运动审计JSON。','',
        '| 地图 | 进阶2疑似占比 | 板块1疑似占比 | 进阶2平均生命 | 板块1平均生命 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for m in [*plan['maps'],'overall']:
        v=[]
        for s in LABELS:
            rows=[r for r in records[s] if m=='overall' or r['map']==m]
            v.extend([100*sum(r['combined_frames'] for r in rows)/sum(r['available_frames'] for r in rows),statistics.mean(r['mean_health'] for r in rows)])
        lines.append(f'| {m} | {v[0]:.2f}% | {v[2]:.2f}% | {v[1]:.2f}/50 | {v[3]:.2f}/50 |')
    v=[e for r in records['advanced5'] for e in r['recovery_verifications']]
    lines+=['',f'候选有{len(v)}次完成恢复验证，其中{sum(bool(e["successful"]) for e in v)}次达到局部位移/信息阈值。短期达标不代表以后不再回到原处，仍需结合跨窗口指标和录像。','',
        '## 每轮完整成绩','',
        '| 版本 | 地图/轮次 | 地图 | 墙体 | 时间 | 难度 | 蓝队原分 | 总分 | 墙钟触限 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for s,rows in scores.items():
        for r in rows:
            lines.append(f'| {LABELS[s]} | {r["map"]}/{r["round"]} | '+' | '.join(f'{r[f]:.2f}' for f in FIELDS)+f' | {"是" if r["red_walltime_limited"] or r["blue_walltime_limited"] else "否"} |')
    lines+=['','## CPU提交图、墙体覆盖图、炸弹布局与轨迹','',
        '下表展示第一轮，各图三轮的完整图件在本批 artifacts 和 report 文件夹。CPU与墙体图由原引擎输出；炸弹按实际导出顺序编号，虚线为爆炸半径。轨迹红色为独立判据筛出的疑似片段。真实墙体仅用于离线核验与画图，在线控制器不读取。']
    for title,pattern in [('CPU提交图','*_exploration_diff.png'),('墙体覆盖图','*_wall_destruction.png'),('炸弹布局图','bomb_layout'),('徘徊轨迹图','motion')]:
        lines+=['',f'### {title}','','| 地图 | 进阶2 | 进阶5板块1 |','| --- | --- | --- |']
        for m in plan['maps']:
            images=[]
            for s in LABELS:
                path=(base/'report'/s/(m+'_r01_'+pattern+'.png') if '*' not in pattern else
                      next((base/s/'red'/f'{m}_r01/artifacts').glob(pattern)))
                images.append(f'![{s} {m} {title}]({link(path)})')
            lines.append(f'| {m} | '+' | '.join(images)+' |')
    lines+=['','## 问题解决状态与后续安排','',
        '| 问题 | 本板块处理 | 剩余安排 |','| --- | --- | --- |',
        '| 换路点导致进展历史丢失 | 增加跨路点连续监测 | 全局任务调度在第三板块处理 |',
        '| 恢复动作互相取消 | 当前脱困执行完再检查结果 | 仍需持续审查重复返回问题 |',
        '| 正常让行/制动误触发 | 有限等待豁免，不无限延期 | 双向通道仲裁不在本板块 |',
        '| 脱困后再次失败 | 临时失败记录、恢复验证 | 并非所有循环都已消除 |',
        '| 窄路通过 | 保留原逻辑 | 用户确认后做第二板块 |',
        '| 多机动态协同 | 保留原消息和让行机制 | 用户确认后做第三板块 |','',
        '## 演示运行命令','',
        '每段独立复制。回放显示实际评测录像；实时重跑会生成新一局。图形窗口需本机桌面支持。']
    for s in LABELS:
        for phase,label in [('red','红队'),('blue','用户蓝队处置')]:
            lines+=['',f'### {LABELS[s]}：{label}逐图逐轮回放']
            for m in plan['maps']:
                for r in range(1,plan['rounds']+1):
                    recording=next((base/s/phase/f'{m}_r{r:02}/artifacts/recordings').glob('*.npz'))
                    lines+=['',f'{m} 第{r}轮','','```powershell',f"Set-Location -LiteralPath '{ROOT}'",
                        '.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.evaluate_red --ray-backend cpu --replay "'+link(recording)+'"','```']
    lines+=['','### 候选实时重跑']
    for m in plan['maps']:
        lines+=['',m,'','```powershell',f"Set-Location -LiteralPath '{ROOT}'",
            '.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.evaluate_retained --live --controllers advanced5 --maps '+m,'```']
    lines+=['','## 证据与边界','',
        f'- [完整成绩]({link(base/"report/full_scores.json")})',
        f'- [运动审计]({link(base/"report/movement_audit.json")})',
        f'- [完成凭据]({link(base/"completion.json")})',
        f'- [冻结源码]({link(base/"source_snapshot.zip")})','',
        '已核验源码身份、结果与布局关联、原评分器合并结果。原项目文件、三版保留红队、用户蓝队、用户方案及问题文件和原提交包与本次开始时一致。未生成新提交包。','']
    document.write_text('\n'.join(lines),encoding='utf-8')
    evaluation.atomic_json(base/'report/verification.json',dict(protected_unchanged=True,
        source_archive_verified=True,receipts_verified=True,original_scoring_recomputed=True,matched_seeds=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--final',action='store_true')
    parser.add_argument('--document',type=Path,default=ROOT/'进阶阶段5_板块1_循环与徘徊优化总结.md')
    args=parser.parse_args();base=args.batch.resolve();records=audit(base)
    if args.final:
        graphics(base,read(base/'plan.json')['maps'],records)
        final_report(base,args.document,records)


if __name__=='__main__':main()
