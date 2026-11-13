"""Offline fixed-layout reference audit, figures and complete replay commands."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from .optimization.evaluation.runner import read,write,file_hash,receipt,phase_valid,REPO
from .optimization.evaluation.replay import validate_replay_sources
from .optimization.evaluation.scenarios import verify_recording


def checked_run(run,plan,batch):
    if read(run/'integrity.json')!=receipt(run,file_hash(batch/'plan.json')):
        raise ValueError('Changed artifacts: '+str(run))
    r=read(run/'result.json');q=read(run/'request.json')
    recordings=list((run/'artifacts/recordings').glob('*.npz'))
    if len(recordings)!=1:
        raise ValueError('Expected a single recording')
    verify_recording(recordings[0],q)
    callbacks=(run/'run.log').read_text(encoding='utf-8',errors='replace').count('Exception ignored from cffi callback')
    return r,callbacks,bool(phase_valid(r,'blue') and not callbacks)


def figure(row,destination,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment
    run=Path(row['run']);red=Path(row['red_run'])
    bombs=read(next((red/'artifacts').glob('*_bombs.json')))['bombs']
    starts=np.array([(b['x'],b['y']) for b in bombs])
    with np.load(next((run/'artifacts/recordings').glob('*.npz')),allow_pickle=False) as z:
        ds,bs=z['drone_states'],z['bomb_states']
    with np.load(source/'diagnostics'/(row['map']+'_wall_geometry.npz')) as z:
        walls=z['walls']
    assert int(walls.sum())==read(red/'result.json')['phase1']['wall_breakdown']['total_wall_pixels']
    hh,ww=walls.shape
    records=[dict(bomb=i+1,first_motion_frame=None,disappearance_frame=None,max_displacement=0.) for i in range(len(starts))]
    ids=np.arange(len(starts));last=starts.copy();ambiguous=False
    for frame_index,frame in enumerate(bs,1):
        active=frame[frame['active']];current=np.column_stack((active['x'],active['y']))
        cost=np.linalg.norm(last[:,None]-current,axis=2)
        rows,cols=linear_sum_assignment(cost)
        if len(cols)!=len(current) or (len(cols) and np.max(cost[rows,cols])>30):
            ambiguous=True;break
        for lost in set(ids)-set(ids[rows]):records[lost]['disappearance_frame']=frame_index
        ids,last=ids[rows],current[cols]
        for i,distance in zip(ids,np.linalg.norm(last-starts[ids],axis=1)):
            records[i]['max_displacement']=max(records[i]['max_displacement'],float(distance))
            if distance>35 and records[i]['first_motion_frame'] is None:records[i]['first_motion_frame']=frame_index
    if not ambiguous:
        assert sum(b['disappearance_frame'] is not None for b in records)==row['result']['disposed_count']
    audit=dict(identity_method='Sequential minimum-distance assignment; close crossings can exchange identities. Motion is diagnostic, not a claim of exact pickup ownership.',ambiguous=ambiguous,bombs=records,
        active_drone_count_final=int(ds[-1]['active'].sum()),active_bomb_count_final=int(bs[-1]['active'].sum()))
    write(destination.with_suffix('.json'),audit)
    fig,ax=plt.subplots(figsize=(11,7))
    ax.imshow(~walls.astype(bool),cmap='gray',vmin=0,vmax=1,extent=(-ww/2,ww/2,-hh/2,hh/2))
    sampled=ds[::6];active=sampled['active']
    ax.scatter(sampled['x'][active],sampled['y'][active],s=.6,color='royalblue',alpha=.25,label='drone positions (sampled)')
    ax.scatter(starts[:,0],starts[:,1],s=22,facecolors='none',edgecolors='darkorange',label='initial bombs')
    for i,(x,y) in enumerate(starts,1):ax.text(x,y,str(i),fontsize=7,color='darkorange')
    final=bs[-1][bs[-1]['active']]
    ax.scatter(final['x'],final['y'],s=28,marker='x',color='crimson',label='remaining bombs')
    ax.set(title=f"{row['variant']} / {row['map']} / red {row['controller']}\nDisposed {row['result']['disposed_count']}/10; valid={row['valid']}",xlim=(-ww/2,ww/2),ylim=(-hh/2,hh/2),aspect='equal')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(destination,dpi=110);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'output/semifinal_reference_blue_20261111')
    parser.add_argument('--source',type=Path,default=REPO/'output/semifinal_r1_official_pilot_20261111')
    parser.add_argument('--output',type=Path,required=True,help='New report directory; historical data are never overwritten')
    args=parser.parse_args();root=args.root.resolve();source=args.source.resolve()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);figures=output/'figures';figures.mkdir()
    from .optimization.evaluation.atlas import round_section
    schedule=read(source/'plan.json')['schedule']
    original=read(source/'results.json');rows=[];archives=[]
    for batch in [source]+sorted(p for p in root.iterdir() if p.is_dir() and (p/'plan.json').is_file()):
        if not (batch/'results.json').is_file():continue
        plan=read(batch/'plan.json');validate_replay_sources(plan['source_sha256'])
        assert file_hash(batch/'source_snapshot.zip')==plan['source_snapshot_sha256']
        archives.append(dict(batch=str(batch),archive_sha256=plan['source_snapshot_sha256']))
        for item in read(batch/'results.json'):
            old=batch==source
            variant='original_R1' if old else batch.name
            run=Path(item['blue_run'] if old else item['run']);red=Path(item['red_run'])
            r,callbacks,valid=checked_run(run,plan,batch)
            layout=next((red/'artifacts').glob('*_bombs.json'))
            assert r['layout_sha256']==file_hash(layout) and r['red_result_sha256']==file_hash(red/'result.json')
            baseline=next(x for x in original if x['map']==item['map'] and x['controller']==item['controller'] and x['round']==item['round'])
            assert item['seed']==baseline['seed'] and red==Path(baseline['red_run'])
            row=dict(map=item['map'],controller=item['controller'],round=item['round'],seed=item['seed'],variant=variant,
                run=str(run),red_run=str(red),result=r,valid=valid,engine_callback_errors=callbacks,
                resources=read(run/'resources.json'))
            key=f"{variant}_{item['map']}_{item['controller']}_r{item['round']:02d}"
            image=figures/(key+'.png')
            if not image.exists():figure(row,image,source)
            row['figure']=str(image);rows.append(row)
            print(key,'verified',flush=True)
    groups={}
    for variant in sorted({r['variant'] for r in rows}):
        selected=[r for r in rows if r['variant']==variant]
        groups[variant]=dict(completed=len(selected),valid=sum(r['valid'] for r in selected),
            disposed_sum=sum(r['result']['disposed_count'] for r in selected),
            raw_blue_mean=statistics.mean(r['result']['blue_score'] for r in selected),
            walltime_mean=statistics.mean(r['result']['elapsed_walltime'] for r in selected),
            timeouts=sum(r['result']['walltime_limited'] for r in selected),
            callback_runs=sum(bool(r['engine_callback_errors']) for r in selected))
    paired=[]
    base={(r['map'],r['controller'],r['round']):r for r in rows if r['variant']=='adapted01'}
    for r in rows:
        key=r['map'],r['controller'],r['round']
        if r['variant'].startswith('enhanced') and key in base:
            b=base[key]
            paired.append(dict(variant=r['variant'],map=r['map'],controller=r['controller'],valid=bool(r['valid'] and b['valid']),
                disposed_delta=r['result']['disposed_count']-b['result']['disposed_count'],blue_delta=r['result']['blue_score']-b['result']['blue_score']))
    write(output/'audit.json',dict(archives=archives,rows=rows,groups=groups,paired=paired))
    fmt=lambda v:'—' if v is None else f'{v:.4f}'
    lines=['# 参考蓝队固定弹图完整对照','',
        '每行均为第1轮，同图／红布局使用相同种子与弹图。原始分保留；触限或引擎回调异常不计有效完整分。原始R1与适配版的旧结果未改写，日志补审计在本报告单列。', '',
        '|地图/红布局|蓝批次|处置|蓝分|地图分|墙体分|时间分|原始难度|原始红总分|有效红总分|蓝秒|存活|回调异常|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        r=row['result'];p=r['phase1'];a=read(Path(row['figure']).with_suffix('.json'))
        values=[row['map']+'/'+row['controller'],row['variant'],str(r['disposed_count']),fmt(r['blue_score']),
            *[fmt(p.get(k)) for k in ['score_map','score_wall','score_time','score_bomb','round_score']],
            fmt(p['round_score']) if row['valid'] else '无效',fmt(r['elapsed_walltime']),str(a['active_drone_count_final']),str(row['engine_callback_errors'])]
        lines.append('|'+ '|'.join(values)+'|')
    for row in rows:
        r=row['result'];red=Path(row['red_run']);image=Path(row['figure']);a=read(image.with_suffix('.json'))
        lines += ['',f"## {row['variant']} / {row['map']} / {row['controller']} / 第{row['round']}轮",'',
            f"种子：{row['seed']}；蓝队{r['elapsed_steps']}步；墙钟{r['elapsed_walltime']:.2f}秒；峰值Job内存{row['resources']['peak_job_commit_bytes']/1048576:.2f}MiB。",'',
            f'![蓝队轨迹点云、初始弹图与剩余弹](<figures/{image.name}>)','',
            '点云不假定录像槽位在无人机损毁后保持身份。逐弹运动按连续位置匹配，近邻交叉可能交换编号，仅作为诊断。','',
            '|弹编号|首次移动超过35px的录像帧|消失帧|最大位移px|','|---:|---:|---:|---:|']
        for b in a['bombs']:lines.append(f"|{b['bomb']}|{b['first_motion_frame']}|{b['disappearance_frame']}|{b['max_displacement']:.2f}|")
        layout_image=source/'figures'/f"{row['map']}_{row['controller']}_r{row['round']:02d}_{red.name}.png"
        item=next(s for s in schedule if s['name']==row['map'] and s['controller']==row['controller'] and s['round']==row['round'])
        pair=dict(item,red=read(red/'result.json'),blue=r,red_run=str(red),blue_run=row['run'])
        # Old raw results predate callback accounting; audit a copy, never rewrite receipts.
        pair['blue']=dict(r,engine_callback_errors=row['engine_callback_errors'])
        lines+=round_section(pair,output,layout_image,historical=True)
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(groups,ensure_ascii=False,indent=2));print('paired',paired)


if __name__=='__main__':main()
