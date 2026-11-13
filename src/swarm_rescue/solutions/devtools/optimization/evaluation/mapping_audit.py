"""Offline map-error decomposition, using frozen attempts; no controller input."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .runner import REPO,read,write,file_hash,receipt
from .replay import validate_replay_sources


def audit(attempts, output):
    from ...evaluate_red import configure_ray_backend
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    from swarm_rescue.simulation.reporting.place_score_manager import walls_and_boundary_grid
    from swarm_rescue.tools.map_exploration_scoring import scorer_for_walltime,load_prediction_grid
    from .scenarios import resolve_map
    configure_ray_backend('cpu')
    output.mkdir(parents=True,exist_ok=True)
    records=[];panels={};truths={}
    for attempt in attempts:
        attempt=attempt.resolve()
        batch=next(p for p in attempt.parents if (p/'plan.json').is_file())
        plan=read(batch/'plan.json');validate_replay_sources(plan['source_sha256'])
        if read(attempt/'integrity.json')!=receipt(attempt,file_hash(batch/'plan.json')):
            raise ValueError('Frozen attempt changed')
        req=read(attempt/'request.json');r=read(attempt/'result.json');name=req.get('official_map') or req['map']
        if name not in truths:
            cls=resolve_map(req)
            world=cls(DroneMotionless,tuple(ZoneType[z] for z in req['zones']),5)
            try:truths[name]=walls_and_boundary_grid(world.playground)[0]
            finally:world.playground.window.close()
        truth=truths[name]
        pred=load_prediction_grid(next((attempt/'artifacts').glob('*_exploration.json')))
        scorer=scorer_for_walltime(180);score=scorer.score(truth,pred,r['elapsed_walltime'])
        expected=r['phase1']['score_map']
        if abs(score.s_map*100-expected)>1e-5:
            raise ValueError(f'Offline score mismatch: {score.s_map*100} vs {expected}')
        surface,_=scorer.split_regions(truth)
        distance=scorer.distance_field(pred)
        misses=(surface>0)&(pred==0)
        bins={f'{lo}-{hi}px':int((misses&(distance>lo)&(distance<=hi)).sum())
              for lo,hi in ((0,10),(10,30),(30,60),(60,100000))}
        telemetry=read(attempt/'telemetry.json')
        pose=telemetry.get('pose_samples',[]);mapped=telemetry.get('mapping_samples',[])
        events=[]
        for line in (attempt/'run.log').read_text(encoding='utf-8',errors='replace').splitlines():
            if line.startswith('V3_EVENT '):
                event=json.loads(line[9:])
                if event.get('event')=='region_pose':events.append(event)
        record=dict(map=name,controller=req['controller_selector'],round=req['round'],
            seed=req['seed'],run=str(attempt),map_score=expected,credit=100*score.credit,
            penalty=100*score.penalty,wall_pixels=score.wall.pixels,wall_hits=score.wall.hits,
            missed_wall_pixels=int(misses.sum()),miss_distance_bins=bins,
            charged_pixels=score.charged_pixels,trajectory=r.get('trajectory_exploration'),
            pose_sample_mean=float(np.mean([p['error'] for p in pose])) if pose else None,
            mapping_sample_mean=float(np.mean([p['error'] for p in mapped])) if mapped else None,
            gate_sample_count=len(events),gate_samples_false=sum(not e['trusted'] for e in events),
            last_mapping_stats={str(p['drone']):p['stats'] for p in mapped})
        records.append(record)
        image=np.full((*truth.shape,3),.97)
        image[truth>0]=(.65,.65,.65)
        image[misses]=(.2,.45,.85)
        image[(pred>0)&(surface>0)]=(.1,.65,.2)
        image[(pred>0)&(truth==0)]=(1.,.65,.35)
        image[(pred>0)&(scorer.distance_field(truth)>10)]=(.75,.1,.1)
        panels.setdefault(name,[]).append((record,pred,image))
    write(output/'audit.json',records)
    lines=['# 建图误差离线审计','',
        '只读冻结产物；官方墙真值与误差统计均不输入在线策略。近邻距离桶表示提交地图是否在附近，不足以独自证明错位或未探索。',
        '位姿误差来自控制后稀疏采样，可能含一次状态更新的时点差；只作同口径诊断，不冒充精确激光配准误差。','',
        '|地图/版本/轮|信用|扣分|地图分|轨迹覆盖%|漏墙0–10px|10–30px|30–60px|>60px|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        values=[r['credit'],r['penalty'],r['map_score'],r['trajectory']]
        lines.append(f"|{r['map']}/{r['controller']}/r{r['round']}|"+'|'.join(f'{v:.4f}' for v in values)+'|'+
                     '|'.join(str(v) for v in r['miss_distance_bins'].values())+'|')
    for name,rows in panels.items():
        fig,axes=plt.subplots(len(rows),3,figsize=(15,4.8*len(rows)),squeeze=False)
        for i,(r,pred,diff) in enumerate(rows):
            for ax,data,title in zip(axes[i],[truths[name],pred,diff],
                    ['Offline wall truth',f"{r['controller']}  M={r['map_score']:.2f}",
                     f"Credit {r['credit']:.2f} - penalty {r['penalty']:.2f}"]):
                ax.imshow(data,cmap='gray_r',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
        fig.suptitle(f'{name}: green=hit, blue=missed surface, orange=near prediction, red=charged prediction')
        fig.tight_layout();path=output/(name+'_comparison.png');fig.savefig(path,dpi=120);plt.close(fig)
        lines += ['',f'![{name} 建图对比](<{path.as_posix()}>)','']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempts',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();audit(a.attempts,a.output.resolve())


if __name__=='__main__':main()
