"""B5 offline report: all micro-scenes and every full-game attempt retained."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .blue_maze_audit import audit


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def coverage_figure(telemetry,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    maps=telemetry.get('lifecycle',{}).get('coverage_maps',{})
    if not maps:return None
    example=next(iter(maps.values()));shape=example['shape'];w,h=example['size']
    united={k:np.zeros(shape,bool) for k in ('body','lidar','walls','semantic_opportunity')}
    has_sem=False
    for m in maps.values():
        for key,value in m['layers'].items():
            united[key]|=np.unpackbits(np.frombuffer(base64.b64decode(value),np.uint8))[:np.prod(shape)].reshape(shape).astype(bool)
            has_sem|=key=='semantic_opportunity'
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,key,label in zip(axes,('body','lidar','semantic_opportunity'),
            ('Body visited (estimated frame)','Lidar free (estimated frame)','Semantic opportunity (not proof of detection)')):
        image=np.full((*shape,3),.85);image[united[key]]=[.35,.75,.9];image[united['walls']]=[.05,.05,.05]
        ax.imshow(image,extent=(0,w,h,0));ax.set_title(label,fontsize=9)
        if key=='semantic_opportunity' and not has_sem:ax.text(w/2,h/2,'NOT RECORDED',ha='center',color='red')
    fig.suptitle('Offline union of local estimated grids; drift/remote overlap can distort coverage')
    fig.tight_layout();fig.savefig(destination,dpi=120);plt.close(fig)
    return destination


def micro_figures(rows,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    configs=list(dict.fromkeys((r['width'],r['length'],r['offset'],r['angle'],r['noise'],r['dropout']) for r in rows))
    fig,axes=plt.subplots(4,4,figsize=(15,12),squeeze=False)
    # Interleave matching baseline/candidate panels; fixed scale in each pair.
    for index,(variant,cfg) in enumerate((v,c) for c in configs for v in ('compute_steering','narrow')):
        ax=axes.flat[index];width,length,offset,angle,noise,dropout=cfg
        group=[r for r in rows if r['variant']==variant and (r['width'],r['length'],r['offset'],r['angle'],r['noise'],r['dropout'])==cfg]
        for r in group:ax.plot([p['x'] for p in r['trace']],[p['y'] for p in r['trace']],label=f"{r['seed']}: contacts={r['contact_episodes']}",lw=.8)
        for side in (-1,1):ax.plot([0,length],[side*width/2]*2,'k',lw=4)
        ax.set(xlim=(-70,length+35),ylim=(-65,65))
        ax.set_title(f"{variant}: {sum(r['passed'] for r in group)}/{len(group)} pass\nW={width} L={length}; offset={offset}, yaw={angle}\nnoise={int(noise)}; GPS/compass dropout={int(dropout)}",fontsize=9)
        ax.legend(fontsize=6);ax.set_aspect('equal')
    fig.tight_layout();fig.savefig(destination,dpi=125);plt.close(fig)


def map_comparisons(summary,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .optimization.evaluation.runner import REPO
    groups={}
    for row in summary:
        r=row['result'];groups.setdefault((r['map'],r['round']),[]).append(row)
    outputs=[]
    for (name,number),group in groups.items():
        # The plan binds the actual map image; no online policy reads this file.
        plan_path=next(p for p in Path(group[0]['attempt']).parents if (p/'plan.json').exists())/'plan.json'
        plan=read(plan_path);item=next(i for i in plan['schedule'] if i['name']==name and i['round']==number)
        doc=read(REPO/item['map_file']);w,h=doc['canvas']['width'],doc['canvas']['height']
        background=plt.imread((REPO/item['map_file']).with_suffix('.png'))
        fig,axes=plt.subplots((len(group)+1)//2,2,figsize=(13,5*((len(group)+1)//2)),squeeze=False)
        for ax,row in zip(axes.flat,group):
            ax.imshow(background,extent=(0,w,h,0));r=row['result']
            # The stock recorder compacts columns after removals; never join
            # those columns into an alleged same-drone trajectory. Telemetry
            # has stable identifiers; use NPZ only for all-agent occupancy sums.
            telemetry=read(Path(row['attempt'])/'telemetry.json')
            samples=telemetry.get('pose_samples',[])
            for ident in sorted({s['drone'] for s in samples}):
                points=np.asarray([s['truth'] for s in samples if s['drone']==ident])
                if len(points):ax.plot(points[:,0]+w/2,h/2-points[:,1],lw=.65)
            ax.set(xlim=(0,w),ylim=(h,0),title=f"{r['blue_variant']}: score {r.get('blue_score')}, disposed {r.get('disposed_count')}/10")
        for ax in list(axes.flat)[len(group):]:ax.axis('off')
        fig.suptitle(f'{name} round {number}: true trajectories sampled every 30 steps, OFFLINE ONLY')
        fig.tight_layout();out=destination/f'{name}_r{number:02d}_comparison.png'
        fig.savefig(out,dpi=125);plt.close(fig);outputs.append(out)
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--micro',type=Path,required=True);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=read(a.micro/'results.json');lines=['# B5 空载窄路与前沿初验','',
      '小场景为原生物理和传感器、固定方向导航组件对照，不是完整B0策略或正式得分。未运行场景不能从单元测试推断通过。',
      '所有原生场景失败及异常均保留；passed是物理跨过整段通道，不等于零接触。','',
      '|版本|宽度/实测|长度|偏心/角度|噪声/失定位|种子|通过|接触段|生命损失|步数|仿真秒(步/30)|墙钟秒|有效|退出|',
      '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('|'+ '|'.join(map(str,[r['variant'],f"{r['width']}/{r.get('actual_gap')}",r['length'],
          f"{r['offset']}/{r['angle']}",f"{r['noise']}/{r['dropout']}",r['seed'],r['passed'],r['contact_episodes'],
          r['health_lost'],r.get('steps'),round(r.get('steps',0)/30,3),r.get('seconds'),r.get('valid'),r.get('reason')]))+'|')
    micro_figures(rows,a.output/'micro_comparison.png')
    lines+=['','![全部小场景轨迹](<'+(a.output/'micro_comparison.png').resolve().as_posix()+'>)','',
            '资源限制与源码：每局results.json含Windows Job单CPU/4GB峰值；sources.json/source_snapshot.zip为本批原始源码。']
    summary=[]
    for file in sorted(a.batch.rglob('attempt_*/result.json')):
        attempt=file.parent;t=read(attempt/'telemetry.json') if (attempt/'telemetry.json').exists() else {}
        result=read(file);l=t.get('lifecycle',{});gates={}
        for f in l.get('frames',[]):
            if f.get('narrow'):gates[str(f['drone'])]=f['narrow']
        aggregate=Counter()
        for g in gates.values():aggregate.update(g['stats'])
        summary.append(dict(attempt=str(attempt),result=result,narrow=dict(aggregate),
                            discovered=l.get('discovered'),bombs=l.get('bombs')))
        name=str(attempt.relative_to(a.batch)).replace('\\','_').replace('/','_')
        dest=a.output/(name+'_coverage.png');coverage_figure(t,dest)
        lines+=['',f'## {name}','',f"窄路末次采样累计：{dict(aggregate)}；成功计数是控制器自判，需与物理轨迹核验。"]
        if dest.exists():lines+=['',f'![机身/雷达/语义机会三类覆盖](<{dest.resolve().as_posix()}>)','']
    for file in sorted(a.batch.glob('*/report.md')):
        lines+=['',file.read_text(encoding='utf-8')]
    history=audit(a.batch)
    for figure in map_comparisons(summary,a.output):lines+=['',f'![同布局轨迹对照](<{figure.resolve().as_posix()}>)','']
    lines+=['','## 完整回放全帧区域核查','',
            '|版本/路径|灰框内机步|x≥1100深处机步|灰框内最深x|','|---|---:|---:|---:|']
    for row in history:
        replay=row.get('replay') or {}
        lines.append(f"|{row['source']}|{replay.get('inside_drone_steps')}|{replay.get('deep_drone_steps')}|{replay.get('max_region_x')}|")
    lifecycle=a.output/'lifecycle'/'report.md'
    if lifecycle.exists():lines+=['',lifecycle.read_text(encoding='utf-8')]
    (a.output/'analysis.json').write_text(json.dumps(dict(micro=rows,full=summary,maze=history),indent=2),encoding='utf-8')
    (a.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(a.output/'report.md')


if __name__=='__main__':main()
