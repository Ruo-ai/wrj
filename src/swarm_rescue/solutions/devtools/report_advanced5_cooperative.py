"""Offline block3 score, team-task and figure audit. Never used online."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np
from swarm_rescue.solutions.devtools import report_advanced5 as previous
from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev

ROOT=ev.ROOT
LABELS={'advanced5_narrow':'板块2最好基线','advanced5_cooperative':'板块3协同候选'}
FIELDS=previous.FIELDS


def link(path):
    return Path(path).resolve().as_posix()


def sustained_departure(xy, origin, radius=180, duration=30):
    outside=np.linalg.norm(xy-origin,axis=1)>radius
    if len(outside)<duration:return None
    hits=np.flatnonzero(np.convolve(outside.astype(int),np.ones(duration,dtype=int),'valid')==duration)
    return int(hits[0])+1 if len(hits) else None


def team_audit(base,records):
    result={}
    for selector,rows in records.items():
        result[selector]=[]
        for row in rows:
            run=Path(row['run']);events=previous.events(run)
            with np.load(next((run/'artifacts/recordings').glob('*.npz')),allow_pickle=False) as z:
                st=z['drone_states'];xy=np.stack((st['x'],st['y']),axis=-1)
            depart=[];revisits=[];unique=[]
            for i in range(xy.shape[1]):
                depart.append(sustained_departure(xy[:,i],xy[0,i]))
                cells=np.floor(xy[:row['exploration_horizon'],i]/60).astype(int)
                changes=np.r_[True,(np.diff(cells,axis=0)!=0).any(axis=1)]
                route=[tuple(c) for c in cells[changes]]
                unique.append(set(route));revisits.append(len(route)-len(set(route)))
            all_cells=set().union(*unique)
            completed=[e for e in events if e['event']=='team_task_end' and e['reason']=='arrived']
            assignments=[e for e in events if e['event'] in ('team_reassigned','team_plan_alternative','team_offer_taken')]
            result[selector].append(dict(map=row['map'],round=row['round'],
                departure_180_for_30_steps=depart,
                last_departure=max(depart) if all(d is not None for d in depart) else None,
                revisit_60px_cells=revisits,
                shared_visited_cells=sum(map(len,unique))-len(all_cells),
                unique_visited_cells=len(all_cells),
                completed_tasks=len(completed),assignments=assignments,
                task_end_reasons=dict(Counter(e['reason'] for e in events if e['event']=='team_task_end')),
                events=dict(Counter(e['event'] for e in events if e['event'].startswith('team_'))),
                traffic_wait_steps=sum(e.get('wait_ticks',0) for e in events if e['event']=='team_traffic_end'),
                health=row['health'],loop_frames=row['combined_frames'],
                score=row['score'],elapsed_steps=row['elapsed_steps'],walltime=row['walltime']))
    ev.atomic_json(base/'report/team_audit.json',result)
    return result


def graphics(base,name):
    previous.LABELS=LABELS
    records=ev.read(base/'report/movement_audit.json')
    previous.graphics(base,[name],records)
    ev.atomic_json(base/'report'/f'wall_overlap_{name}.json',ev.read(base/'report/wall_overlap.json'))


def final(base,document,records,team):
    plan=ev.read(base/'plan.json');completion=ev.read(base/'completion.json')
    assert completion['complete'] and completion['placement_complete'] and completion['sources_unchanged']
    assert set(plan['controllers'])==set(LABELS)
    expected={(m,r) for m in plan['maps'] for r in range(1,plan['rounds']+1)}
    assert len(plan['schedule'])==len(expected)*4, 'A red-only pilot is not a full evaluation'
    assert ev.digest(base/'source_snapshot.zip')==plan['source_snapshot_sha256']
    for selector in LABELS:
        for phase in ('red','blue'):
            folder=base/selector/phase;manifest=folder/'manifest.json'
            ev.verify_sources(ev.read(manifest))
            for row in ev.read(folder/'results.json'):
                assert not row.get('walltime_limited') and not row.get('controller_crashed')
                ev.validate_receipt(folder/ev.round_name(row),ev.digest(manifest))
    scores={s:previous.paired(base,s,expected) for s in LABELS}
    a,b=LABELS
    assert [(r['map'],r['round'],r['seed']) for r in scores[a]]==[(r['map'],r['round'],r['seed']) for r in scores[b]]
    means={s:{m:{f:statistics.mean(r[f] for r in rows if m=='overall' or r['map']==m)
                  for f in FIELDS} for m in [*plan['maps'],'overall']} for s,rows in scores.items()}
    ev.atomic_json(base/'report/full_scores.json',dict(rounds=scores,means=means))
    protected=ev.read(ROOT/'output/red_advanced5/block3/baseline_protection.json')
    changed=[p for p,sha in protected.items() if ev.digest(ROOT/p)!=sha]
    allowed={'src/swarm_rescue/solutions/devtools/evaluate_verified_revision.py',
             'src/swarm_rescue/solutions/my_drone_red_advanced4.py'}
    assert not set(changed)-allowed,changed
    overlaps=[r for m in plan['maps'] for r in ev.read(base/'report'/f'wall_overlap_{m}.json')]
    ev.atomic_json(base/'report/wall_overlap.json',overlaps)
    delta=means[b]['overall']['round_score']-means[a]['overall']['round_score']
    lines=['## 同期正式结果','',
        f"本批完成{len(plan['schedule'])}项，所有红队放满10枚并合法提交。两版完整均分 **{means[a]['overall']['round_score']:.4f}→{means[b]['overall']['round_score']:.4f}（{delta:+.4f}）**。本地用户蓝队口径，不等同于平台成绩。",
        '', '各分项满分100。完整分按原评分器计算：地图×0.5＋墙体×0.2＋时间×0.1＋红队难度×0.2；蓝队原分经项目原规则转换为红队难度。表中同时列蓝队原分，避免混淆。',
        '', '| 地图 | 基线完整分 | 协同完整分 | 变化 | 协同地图 | 墙体 | 时间 | 红队难度 | 蓝队原分 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for m in [*plan['maps'],'overall']:
        x,y=means[a][m],means[b][m]
        lines.append(f"| {m} | {x['round_score']:.4f} | {y['round_score']:.4f} | {y['round_score']-x['round_score']:+.4f} | "+' | '.join(f'{y[f]:.2f}' for f in FIELDS[:-1])+' |')
    lines+=['','### 每图每轮分项','','| 版本 | 地图 | 轮次 | 地图 | 墙体 | 时间 | 红队难度 | 蓝队原分 | 完整分 |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for s,rows in scores.items():
        for r in rows:
            lines.append(f"| {LABELS[s]} | {r['map']} | {r['round']} | "+' | '.join(f'{r[f]:.3f}' for f in FIELDS)+' |')
    lines+=['','### 协同、重复路线与可靠性','',
        '出发指标统一定义为离各自首帧位置超过180像素且连续保持30帧，不等于人工划定的Map07出生区出口。重复路线为探索阶段60像素网格重访，包含必要返程；跨机重复格子不是重复观测面积。疑似循环沿用前两板块的同一几何筛查，不能直接等同于无效动作。',
        '', '| 地图/轮次 | 最后一架稳定离开 基线→候选 | 重访格次数 基线→候选 | 疑似循环步 基线→候选 | 候选改派次数 | 候选任务到达 | 候选最终生命 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    indexed={s:{(r['map'],r['round']):r for r in rows} for s,rows in team.items()}
    for key in sorted(expected):
        x,y=indexed[a][key],indexed[b][key]
        lines.append(f"| {key[0]}/{key[1]} | {x['last_departure']}→{y['last_departure']} | {sum(x['revisit_60px_cells'])}→{sum(y['revisit_60px_cells'])} | {x['loop_frames']}→{y['loop_frames']} | {len(y['assignments'])} | {y['completed_tasks']} | {y['health']} |")
    lines+=['','## 每张地图图件对比','',
            'CPU图为原评分器生成的提交地图对比；墙体覆盖图为原爆炸覆盖结果；布弹图由导出的实际落点绘制。所有真实地图背景只用于离线报告，没有提供给在线控制器。']
    for m in plan['maps']:
        lines+=['',f'### {m}','','| 轮次/图件 | 板块2基线 | 板块3候选 |','| --- | --- | --- |']
        for number in range(1,plan['rounds']+1):
            for title,suffix in [('CPU提交图','exploration_diff'),('墙体覆盖图','wall_destruction'),('炸弹布置图','bomb_layout'),('运动图','motion')]:
                paths=[]
                for s in LABELS:
                    run=base/s/'red'/f'{m}_r{number:02}'
                    p=(next((run/'artifacts').glob('*_'+suffix+'.png')) if suffix in ('exploration_diff','wall_destruction')
                       else base/'report'/s/f'{m}_r{number:02}_{suffix}.png')
                    assert p.is_file(),p
                    paths.append(f'![{LABELS[s]} {m} 第{number}轮 {title}]({link(p)})')
                lines.append(f'| 第{number}轮 {title} | '+' | '.join(paths)+' |')
    lines+=['','## 演示运行命令','', '每条命令独立复制。回放播放这次已有录像；实时演示会重新运行，不复用正式成绩。']
    for s,label in LABELS.items():
        for phase in ('red','blue'):
            lines+=['',f'### {label} {phase}']
            for m in plan['maps']:
                for number in range(1,plan['rounds']+1):
                    p=next((base/s/phase/f'{m}_r{number:02}'/'artifacts/recordings').glob('*.npz'))
                    lines+=['',f'{m} 第{number}轮','','```powershell',f"Set-Location -LiteralPath '{ROOT}'",
                        f'.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.evaluate_red --ray-backend cpu --replay "{link(p)}"','```']
    lines+=['','### 协同候选实时演示']
    for m in plan['maps']:
        lines+=['',m,'','```powershell',f"Set-Location -LiteralPath '{ROOT}'",
            f'.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.evaluate_retained --live --controllers advanced5_cooperative --maps {m}','```']
    lines+=['','## 本批证据','',
        f'- [逐轮原评分与均分]({link(base/"report/full_scores.json")})',
        f'- [任务与出发审计]({link(base/"report/team_audit.json")})',
        f'- [运动审计]({link(base/"report/movement_audit.json")})',
        f'- [运行完成记录]({link(base/"completion.json")})',
        f'- [冻结源码]({link(base/"source_snapshot.zip")})','']
    begin='<!-- BLOCK3_RESULTS_BEGIN -->';end='<!-- BLOCK3_RESULTS_END -->'
    existing=document.read_text(encoding='utf-8')
    generated=begin+'\n'+'\n'.join(lines)+'\n'+end
    if begin in existing:
        left,rest=existing.split(begin,1);_,right=rest.split(end,1);existing=left+generated+right
    else:existing=existing.rstrip()+'\n\n'+generated+'\n'
    document.write_text(existing,encoding='utf-8')
    ev.atomic_json(base/'report/verification.json',dict(complete=True,receipts_verified=True,
        original_scoring_recomputed=True,protected_changes=changed,baseline=a,candidate=b,delta=delta))
    print('FULL MEANS',means[a]['overall']['round_score'],means[b]['overall']['round_score'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path);parser.add_argument('--final',action='store_true')
    parser.add_argument('--graphics-map');parser.add_argument('--document',type=Path,
        default=ROOT/'进阶阶段5_板块3_动态协同优化总结.md')
    args=parser.parse_args();base=args.batch.resolve();previous.LABELS=LABELS
    if args.graphics_map:
        graphics(base,args.graphics_map);return
    records=previous.audit(base);team=team_audit(base,records)
    if args.final:
        for name in ev.read(base/'plan.json')['maps']:
            subprocess.run([sys.executable,'-B','-m','swarm_rescue.solutions.devtools.report_advanced5_cooperative',
                            str(base),'--graphics-map',name],cwd=ROOT,check=True,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        final(base,args.document,records,team)


if __name__=='__main__':main()
