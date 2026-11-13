"""Offline block2 paired report and per-map figures. Never imported online."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np
from swarm_rescue.solutions.devtools import report_advanced5 as previous
from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev

LABELS={'advanced5':'进阶5板块1基线','advanced5_narrow':'进阶5板块2窄路候选'}
ROOT=ev.ROOT


def passage_audit(base, records):
    result={}
    for selector,rows in records.items():
        result[selector]=[]
        for row in rows:
            run=Path(row['run']);events=previous.events(run)
            starts=[e for e in events if e['event']=='narrow_started']
            finishes=[e for e in events if e['event']=='narrow_finished']
            motions=[e for e in events if e['event']=='narrow_motion']
            record=dict(map=row['map'],round=row['round'],started=len(starts),finished=len(finishes),
                reasons=dict(Counter(e['reason'] for e in finishes)),
                retreats=dict(Counter(e['reason'] for e in events if e['event']=='narrow_retreat')),
                traffic=dict(Counter(e['action'] for e in events if e['event']=='narrow_traffic')),
                speed_samples=len(motions),mean_sampled_speed=statistics.mean(e['velocity'] for e in motions) if motions else None,
                median_duration=statistics.median(e['duration'] for e in finishes) if finishes else None,
                health=row['mean_health'],loop_frames=row['combined_frames'],
                submitted=row['submitted'],placed=row['placed'],walltime=row['walltime'])
            if row['map']=='Map07':
                with np.load(next((run/'artifacts/recordings').glob('*.npz'))) as z:
                    states=z['drone_states'];crossings=[]
                    # OFFLINE ROI from the historical diagnostic figure, never
                    # fed into the controller. Check continuous in-channel tracks.
                    for i in range(states.shape[1]):
                        x,y=states['x'][:,i],states['y'][:,i]
                        inside=(x>140)&(x<320)&(y>-290)&(y<-253)
                        indices=np.flatnonzero(inside);segments=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
                        good=[s for s in segments if len(s)>1 and x[s].min()<160 and x[s].max()>300]
                        crossings.append(dict(drone=i,frames=int(inside.sum()),crossings=len(good),
                            durations=[len(s) for s in good]))
                    record['map07_observed_passage']=crossings
            result[selector].append(record)
    ev.atomic_json(base/'report/passage_audit.json',result)
    return result


def graphics(base,name,records):
    import matplotlib
    matplotlib.use('Agg')
    from cycler import cycler
    matplotlib.rcParams['axes.prop_cycle']=cycler(color=['#2471a3','#e09b20','#269965','#8656aa','#4d6571'])
    previous.LABELS=LABELS
    from swarm_rescue.simulation.reporting import place_score_manager
    original=place_score_manager.walls_and_boundary_grid;cache=[]
    def capture(playground):
        result=original(playground);cache.append(result);return result
    place_score_manager.walls_and_boundary_grid=capture
    try:previous.graphics(base,[name],records)
    finally:place_score_manager.walls_and_boundary_grid=original
    overlap=ev.read(base/'report/wall_overlap.json')
    ev.atomic_json(base/'report'/f'wall_overlap_{name}.json',overlap)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    # Reuse the same offline background; do not create a second OpenGL world.
    walls,_=cache[0];h,w=walls.shape
    for selector,rows in records.items():
        for row in rows:
            if row['map']!=name:continue
            run=Path(row['run']);events=previous.events(run)
            with np.load(next((run/'artifacts/recordings').glob('*.npz'))) as z:
                s=z['drone_states'];xy=np.stack((s['x'],s['y']),axis=-1)
            fig,ax=plt.subplots(figsize=(11,8))
            ax.imshow(walls,cmap=ListedColormap(['white','#bbbbbb']),origin='upper',
                      extent=(-w/2,w/2,-h/2,h/2),interpolation='nearest')
            for i in range(xy.shape[1]):
                p=xy[:row['exploration_horizon'],i]
                ax.plot(p[::3,0],p[::3,1],lw=.7,alpha=.45,label=f'Drone {i}')
            for event in events:
                if event['event']=='narrow_started':
                    ax.scatter(*event['entry'],marker='o',facecolors='none',edgecolors='#176b45',s=24)
                elif event['event']=='narrow_retreat':
                    ax.scatter(*event['position'],marker='x',color='#c33434',s=24)
            ax.set(title=f'{selector} / {run.name}: green circles = passage starts; red crosses = retreats',xlabel='World X (px)',ylabel='World Y (px)')
            ax.set_aspect('equal');ax.legend();fig.tight_layout()
            fig.savefig(base/'report'/selector/(run.name+'_passages.png'),dpi=125);plt.close(fig)
    return overlap


def link(p):return Path(p).resolve().as_posix()


def final(base,records,passages,document):
    plan=ev.read(base/'plan.json');completion=ev.read(base/'completion.json')
    if not completion['complete']:raise ValueError('Incomplete or invalid runs; do not report final scores')
    if ev.digest(base/'source_snapshot.zip')!=plan['source_snapshot_sha256']:raise ValueError('Source snapshot changed')
    for selector in LABELS:
        for phase in ('red','blue'):
            path=base/selector/phase;manifest=path/'manifest.json'
            ev.verify_sources(ev.read(manifest))
            for row in ev.read(path/'results.json'):
                ev.validate_receipt(path/ev.round_name(row),ev.digest(manifest))
    expected={(m,r) for m in plan['maps'] for r in range(1,plan['rounds']+1)}
    scores={s:previous.paired(base,s,expected) for s in LABELS}
    identities=[[(r['map'],r['round'],r['seed']) for r in rows] for rows in scores.values()]
    if identities[0]!=identities[1]:raise ValueError('Unpaired seeds')
    fields=previous.FIELDS
    means={s:{m:{f:statistics.mean(r[f] for r in rows if m=='overall' or r['map']==m) for f in fields}
        for m in [*plan['maps'],'overall']} for s,rows in scores.items()}
    ev.atomic_json(base/'report/full_scores.json',dict(scores=scores,means=means))
    freeze=ev.read(ROOT/'output/red_advanced5/block2/freeze.json')
    allowed={'src/swarm_rescue/solutions/devtools/evaluate_verified_revision.py','优化建议.md'}
    changed=[p for p,h in freeze['before_sha256'].items() if ev.digest(ROOT/p)!=h]
    unexpected=[p for p in changed if p not in allowed]
    if unexpected:raise ValueError(f'Unexpected protected changes: {unexpected}')
    a,b=means['advanced5']['overall'],means['advanced5_narrow']['overall']
    lines=['# 进阶阶段5：板块2——窄路通过优化总结','',
        '更新时间：2026-10-29。本板块在当前最好进阶5板块1上累积优化，进阶4只作为局部精细通行的参考。原三版、板块1基线、蓝队和默认提交入口保留；没有进入第三板块动态任务调配。','',
        f'本批{len(plan["maps"])}图{plan["rounds"]}轮完整均分：板块1 **{a["round_score"]:.4f}** → 窄路候选 **{b["round_score"]:.4f}**，变化 **{b["round_score"]-a["round_score"]:+.4f}**。这是当前用户蓝队的本地对照结果，不是平台成绩。','',
        '## 基线、方法与代码逻辑','',
        '基线为 `my_drone_red_advanced5.py / MyDroneRedAdvanced5`；候选为自包含 `my_drone_red_advanced5_narrow.py / MyDroneRedAdvanced5Narrow`。第一板块原文件保持不变，候选复制其完整逻辑后增加局部通行。后续优化使用届时最好版本，而不是依据历史默认入口选择进阶2。','',
        '1. **观测与精细规划**：正常20像素导航保留，窄路使用3像素已观测自由空间。墙体与未知格都参与净空检查；八邻接搜索不切对角墙角，路径简化还要检查整段机身扫过的空间。规划节点与距离都有上限。局部导航图不写回提交墙图。',
        '2. **机体与误差余量**：离线核验原机体凸包包络半径为10像素，窄路实时轨迹检查用12像素包络；细网格规划再考虑半格对角线误差。通道宽于机身不等于任意噪声、姿态和速度下都可通过，首轮不承诺所有只多2～4像素的缝隙。',
        '3. **直段和转角分别控制**：直段速度上限6像素/步，实际还受制动距离、侧向净空和残余侧向速度约束；可信平行墙用于居中，斜向入口按通道方向计算。转角按路线夹角提前制动，路线点切换必须保证到下一点的连线仍可行。',
        '4. **有限通行任务**：记录入口、路线、目标、预算、净推进、走过的位置与退出原因。路线遇阻允许有限修路，失败后制动并沿已走路线返回；已观察到前方封闭且左右无出口时退回。安全退路本身若被新障碍挡住也不能盲退。长直通道允许有限延伸，总任务期限有硬上限。',
        '5. **局部通行顺序**：消息携带通行状态、入口、终点与近期进展；先进入者优先，同向保留间距，相向时低优先级一方退让。过期消息不继续占用通道；本机语义传感器仍承担实时避碰。全队任务竞价和探索分工未在本板块实施。',
        '6. **与现有策略衔接**：只在已观测窄区触发，普通区域沿用板块1。通行期间保留探索观测和地图广播，防止常规恢复、布弹趋近控制覆盖窄路指令；到预算或截止时退出。布弹收益函数、第一板块循环遗留算法及用户蓝队没有改写。','',
        '最终接管条件：有正在执行的探索目标时，只搜索通向同一目标的细路径；必须经过双侧实测墙体形成的窄区，且能明显缩短原路线或修复局部停滞。没有粗路线且低速时，才自行搜索未访问的出口。成功后尝试恢复原目标。直接识别的直窄道还要求方向匹配，防止几何模块擅自重排探索任务。','',
        '## 开发中发现的问题与修正','',
        '首轮重点试跑pilot01中Map05出现193次通行启动、79次到达、100次退回、14次退回停滞。虽然运动采样均速约3.26像素/步，但频繁接管与退回不能视为完成。随后收紧了触发间隔，排除只由未知边界形成的假窄区，增加失败方向冷却；侧墙改用平行拟合，避免将前方墙误当成左右侧壁；转角切点增加连线验证，长通道允许有上限地续行。试跑分数不混入下表。','',
        '早期formal01在5/72项后停止：Map01第一轮墙体分92.54→74.91，23次接管扰乱原路线。随后发现仅以双侧墙作为机会仍会改乱任务，而仅等停住又会漏掉入口。最终采用上面的同目标修复与无路线出口搜索边界，另起正式批次。pilot08三张重点图均放满并提交；Map01恢复基线，Map07该轮没有新通行启动，说明自主入口利用仍需真实评测，不能把pilot05的39步跨越直接当作最终源码已经稳定做到。','',
        '## 评测与评分口径','',
        '实现冻结前89项测试通过：原保留策略39项、板块1循环23项、窄路专项27项。窄路专项覆盖斜向、转角、过窄拒绝、连续扫掠、速度制动、横向稳定、通行顺序与有限退回，并检查未知边界和单侧墙不能充当双侧实测通道。测试通过不代替下面的整图实测。','',
        f'两版同地图同轮同种子，CPU射线、5机每机2枚，用户蓝队，原时间步和墙钟限制；单仿真串行，按轮交替版本顺序。有效运行{completion["completed"]}/{completion["expected"]}。先用原评分器逐轮计算，再平均。','',
        '完整分=地图×0.5+墙体×0.2+时间×0.1+红队难度×0.2。分项均以100分报告；蓝队原分先按原规则转换为红队难度，不能直接将蓝队原分加给红队。','',
        '## 每图均分与前一板块对比','',
        '| 地图 | 板块1总分 | 窄路候选总分 | 变化 | 候选地图 | 候选墙体 | 候选时间 | 候选难度 | 蓝队原分 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in [*plan['maps'],'overall']:
        x,y=means['advanced5'][name],means['advanced5_narrow'][name]
        lines.append(f'| {name} | {x["round_score"]:.4f} | {y["round_score"]:.4f} | {y["round_score"]-x["round_score"]:+.4f} | '+' | '.join(f'{y[f]:.2f}' for f in fields[:-1])+' |')
    lines+=['','## 通行、徘徊与损伤','',
        '通行到达表示到达本次局部路线终点，不等于每次都穿过完整物理通道。运动均速为每15步的通行日志采样；徘徊沿用板块1相同几何筛查，并非逐帧人工判定。Map07指定窄缝的完整跨越另外从真实录像检查，不能用任务启动次数代替。','',
        '| 地图 | 通行启动 | 到达 | 退回/其他终止 | 运动采样均速 | 板块1生命 | 候选生命 | 板块1疑似循环步 | 候选疑似循环步 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in plan['maps']:
        p=[r for r in passages['advanced5_narrow'] if r['map']==name]
        old=[r for r in records['advanced5'] if r['map']==name];new=[r for r in records['advanced5_narrow'] if r['map']==name]
        count=sum(r['speed_samples'] for r in p)
        speed=sum((r['mean_sampled_speed'] or 0)*r['speed_samples'] for r in p)/max(1,count)
        arrived=sum(r['reasons'].get('arrived',0) for r in p)
        lines.append(f'| {name} | {sum(r["started"] for r in p)} | {arrived} | {sum(r["finished"] for r in p)-arrived} | {speed:.2f} | {statistics.mean(r["mean_health"] for r in old):.2f} | {statistics.mean(r["mean_health"] for r in new):.2f} | {sum(r["combined_frames"] for r in old)} | {sum(r["combined_frames"] for r in new)} |')
    for selector,rows in passages.items():
        for r in rows:
            if r['map']=='Map07':
                cross=r['map07_observed_passage']
                lines+=['',f'{LABELS[selector]}，Map07第{r["round"]}轮指定窄缝：完整跨越{sum(x["crossings"] for x in cross)}次；进入该局部窗口的无人机编号：'+', '.join(str(x['drone']) for x in cross if x['frames'])+'。']
    lines+=['','## 每轮完整成绩','',
        '| 版本 | 地图/轮次 | 地图 | 墙体 | 时间 | 难度 | 蓝队原分 | 总分 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for selector,rows in scores.items():
        for r in rows:lines.append(f'| {LABELS[selector]} | {r["map"]}/{r["round"]} | '+' | '.join(f'{r[f]:.3f}' for f in fields)+' |')
    lines+=['','## CPU提交图、墙体覆盖图、炸弹布局与通行轨迹','',
        '下面对照第一轮；随后索引保留全部三轮图件。CPU提交图和墙体覆盖由原引擎生成；炸弹布局使用实际导出数据；真实地图仅作离线背景，在线策略不读取。']
    kinds=[('CPU提交图','*_exploration_diff.png'),('墙体覆盖图','*_wall_destruction.png'),('炸弹布局图','bomb_layout'),('窄路任务轨迹','passages')]
    def figure(selector,name,number,pattern):
        stem=f'{name}_r{number:02}'
        return (next((base/selector/'red'/stem/'artifacts').glob(pattern)) if '*' in pattern else base/'report'/selector/(stem+'_'+pattern+'.png'))
    for title,pattern in kinds:
        lines+=['',f'### {title}','','| 地图 | 板块1 | 窄路候选 |','| --- | --- | --- |']
        for name in plan['maps']:
            lines.append(f'| {name} | '+' | '.join(f'![{s} {name} {title}]({link(figure(s,name,1,pattern))})' for s in LABELS)+' |')
    lines+=['','### 全图三轮图件索引','','| 地图/轮次 | 板块1 | 窄路候选 |','| --- | --- | --- |']
    for name in plan['maps']:
        for n in range(1,plan['rounds']+1):
            lines.append(f'| {name}/{n} | '+' | '.join(' · '.join(f'[{title}]({link(figure(s,name,n,pattern))})' for title,pattern in kinds) for s in LABELS)+' |')
    lines+=['','## 问题解决状态与后续安排','',
        '| 问题 | 本板块处理 | 验证及后续 |','| --- | --- | --- |',
        '| 以旧进阶2继续优化 | 改为当前进阶5板块1基线 | 基线原文件保留；所有配对身份明确 |',
        '| 窄直路固定过慢 | 净空允许时提速，转角单独制动 | 实测速度见上表；非所有窄路都可满速 |',
        '| 斜向小入口/粗网格封闭 | 精细八邻接与扫掠检查 | 真实入口仍受观测、定位和安全余量约束 |',
        '| 前方死路还持续尝试 | 有限修路、制动、沿已走路径退出 | 分类退出日志见通行审计，不能保证后方动态阻塞时立即退出 |',
        '| 多机相向/跟车 | 近期通行占用与本机传感器避碰 | 全队入口分配、任务接替留第三板块 |',
        '| 循环板块遗留 | 原基线逻辑保留 | 第二轮更新方向汇总继续维护，不在本轮偷偷重写 |',
        '| 探索后统一布弹/更强蓝队 | 已建立后续讨论文档 | 三个主板块首轮完成后由用户决定 |','',
        '## 演示运行命令','',
        '回放显示本次真实录像；实时命令会生成新一局。每条分别复制，无需拼接。']
    for selector in LABELS:
        for phase,label in [('red','红队'),('blue','用户蓝队')]:
            lines+=['',f'### {LABELS[selector]}：{label}']
            for name in plan['maps']:
                for number in range(1,plan['rounds']+1):
                    recording=next((base/selector/phase/f'{name}_r{number:02}/artifacts/recordings').glob('*.npz'))
                    lines+=['',f'{name} 第{number}轮','','```powershell',f"Set-Location -LiteralPath '{ROOT}'",
                        f'.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.evaluate_red --ray-backend cpu --replay "{link(recording)}"','```']
    lines+=['','### 窄路候选实时演示']
    for name in plan['maps']:
        lines+=['',name,'','```powershell',f"Set-Location -LiteralPath '{ROOT}'",
            f'.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.evaluate_retained --live --controllers advanced5_narrow --maps {name}','```']
    lines+=['','## 证据与限制','',
        f'- [完整成绩]({link(base/"report/full_scores.json")})',f'- [通行审计]({link(base/"report/passage_audit.json")})',
        f'- [运动审计]({link(base/"report/movement_audit.json")})',f'- [完整运行凭据]({link(base/"completion.json")})',
        f'- [冻结源码]({link(base/"source_snapshot.zip")})','',
        '独立离线核验已检查原评分器重算、配对种子、布局关联和运行收据。源码保护核验只允许新增候选/测试/报告、离线控制器注册，以及按用户要求填写《优化建议》的解答；原地图、引擎、原红队、蓝队、提交入口及ZIP保留。没有生成新提交包。','']
    document.write_text('\n'.join(lines),encoding='utf-8')
    ev.atomic_json(base/'report/verification.json',dict(complete=True,receipts_verified=True,
        matched_seeds=True,original_scoring_recomputed=True,protected_changes=changed,
        unauthorized_changes=unexpected,baseline='advanced5',candidate='advanced5_narrow'))
    print('FULL MEANS',a['round_score'],b['round_score'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path);parser.add_argument('--final',action='store_true')
    parser.add_argument('--graphics-map');parser.add_argument('--document',type=Path,
        default=ROOT/'进阶阶段5_板块2_窄路通过优化总结.md')
    args=parser.parse_args();base=args.batch.resolve();previous.LABELS=LABELS
    if args.graphics_map:
        graphics(base,args.graphics_map,ev.read(base/'report/movement_audit.json'));return
    records=previous.audit(base);passages=passage_audit(base,records)
    if args.final:
        # Fresh process per map avoids cross-map OpenGL context invalidation on Windows.
        for name in ev.read(base/'plan.json')['maps']:
            subprocess.run([sys.executable,'-B','-m','swarm_rescue.solutions.devtools.report_advanced5_narrow',str(base),'--graphics-map',name],
                           cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        final(base,records,passages,args.document)


if __name__=='__main__':main()
