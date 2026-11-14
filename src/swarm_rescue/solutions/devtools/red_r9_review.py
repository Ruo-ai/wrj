"""Map08 focused E9/P9 follow-up atlas, retained RN7-B reference reused."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools.red_rn7_review import regions
from swarm_rescue.solutions.devtools.optimization.evaluation.runner import validate_sources


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    names={'rn7_b':'RN7-B reference','e9_n':'E9 entrance task','p9_m':'P9 incremental 1px'}
    rows=[];assets={}
    for batch in (Path('output/sf13/rn7_m89_once'),args.batch):
        plan=base.read(batch/'plan.json');validate_sources(plan['source_sha256']);sha=base.digest(batch/'plan.json')
        for raw in base.read(batch/'results.json'):
            if raw['map']!='Map08' or raw['controller'] not in names:continue
            r,a=base.review_row(raw,sha);r['region_drone_frames']=regions(a,'Map08')
            stats=Counter();run=Path(r['attempt'])
            for d in base.read(run/'narrow_diagnostics.json')['drones'].values():stats.update(d['stats'])
            r['narrow_stats']=dict(stats)
            if (run/'r8_evidence.json').exists():r['evidence']=base.read(run/'r8_evidence.json')
            rows.append(r);assets[('Map08',r['controller'])]=a
    base.NAMES=names;base.ORDER=list(names);base.figures('Map08',rows,assets,args.output)
    with (args.output/'summary.json').open('x',encoding='utf-8') as f:json.dump(dict(rows=rows),f,indent=2)
    lines=['# Map08：入口任务与1 px增量缓存针对性验证','',
        '旧RN7-B复用，E9和P9各一次新红局。E9仍用A3建图；P9仍用RN7-B运动。两者未组合。难度及完整官方总分未评测。','',
        '|版本|有效|M|W|T|弹数|步数|秒|底部区域机体帧数|','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"|{names[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['red_steps']}|{r['red_walltime']:.2f}|{r['region_drone_frames']}|")
    for kind,label in [('mapping','提交建图差异'),('walls','官方墙体覆盖'),('bombs','实际炸弹位置'),('trajectories','实际探索轨迹')]:
        lines+=['',label+'（左旧RN7-B，中E9，右P9）','',f'![{label}]({(args.output.resolve()/f"Map08_{kind}.png").as_posix()})']
    for r in rows:
        lines+=['',f"Map08 / {r['controller']} / seed {r['red_seed']}",'','```powershell',
                "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{r['attempt']}'",'```']
    lines+=['','未运行蓝阶段，无本轮蓝回放。P8/P9像素等价测试仅验证建图缓存语义；独立比赛轨迹与时长仍必须分别核查。']
    with (args.output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in ('controller','map_score','wall_score','red_steps','region_drone_frames')} for r in rows],indent=2))


if __name__=='__main__':main()
