"""RL11 offline full atlas; official artifacts and native truth never enter policy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_rx10_review as evidence

NAMES={'rn7_b':'RN7-B原版','rl11_lines':'RL11 墙段校正','rl11_slow':'RL11 单独慢速',
       'rl11_combined':'RL11 组合（实验）','rl11_off':'RL11 关闭对照'}
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')


def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False)


def frozen(batch):
    plan=base.read(batch/'plan.json');archive=batch/'source_snapshot.zip'
    assert base.digest(archive)==plan['source_snapshot_sha256']
    with zipfile.ZipFile(archive) as z:
        for name,digest in plan['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    changed=[n for n,h in plan['source_sha256'].items() if not Path(n).is_file() or base.digest(n)!=h]
    if changed:raise ValueError(f'Frozen source changed: {changed}; preserve results and explicitly audit before offline review')
    return base.digest(batch/'plan.json')


def accuracy(row,data):
    run=Path(row['attempt']);doc=base.read(run/'line_diagnostics.json')
    motion=base.read(run/'motion_diagnostics.json');gray=np.zeros_like(data['truth'],bool)
    for area in data['spec']['elements']:
        if area['type']=='no_gps_zone':
            left,top=area['top_left'];right,bottom=area['bottom_right']
            gray[max(0,int(top)):min(data['h'],int(bottom)+1),max(0,int(left)):min(data['w'],int(right)+1)]=True
    surface,_=base.MapExplorationScorer().split_regions(data['truth']);surface=surface.astype(bool)
    errors={};stats=Counter();guard=Counter();distance=0.;ticks=0
    for drone in doc['drones'].values():
        stats.update(drone['line_stats']);guard.update(drone['motion_guard'])
        for key in ('outage_heading_summary','normal_heading_summary','outage_position_summary'):
            item=drone[key];r=errors.setdefault(key,dict(count=0,total=0.,maximum=0.))
            if item['count']:
                r['count']+=item['count'];r['total']+=item['count']*item['mean'];r['maximum']=max(r['maximum'],item['maximum'])
    for r in errors.values():r['mean']=r['total']/r['count'] if r['count'] else None
    for drone in motion['drones'].values():distance+=drone['outage_distance'];ticks+=drone['outage_ticks']
    pred=data['pred'];region_rows=[]
    for area in data['spec']['elements']:
        if area['type']!='no_gps_zone':continue
        left,top=map(int,area['top_left']);right,bottom=map(int,area['bottom_right'])
        sl=np.s_[max(0,top):min(data['h'],bottom+1),max(0,left):min(data['w'],right+1)]
        region_rows.append(dict(bounds=[left,top,right,bottom],surface=int(surface[sl].sum()),
            hits=int((surface&pred)[sl].sum()),sampled_seen=int(data['observed'][sl].sum())))
    row.update(line_accuracy=errors,line_stats=dict(stats),slow_stats=dict(guard),gray_regions=region_rows,
        gray_true_surface=int((surface&gray).sum()),gray_surface_hits=int((surface&gray&pred).sum()),
        gray_sampled_observed=int((data['observed']&gray).sum()),
        actual_outage_speed=distance/ticks if ticks else None,
        input_tape_bytes=(run/'mapper_legal_inputs.npz').stat().st_size,
        input_tape_sha256=base.digest(run/'mapper_legal_inputs.npz'),input_tape_complete=doc['mapper_inputs_complete'],
        line_diagnostic_seconds=doc['diagnostic_seconds'],line_diagnostic_errors=doc['errors'])


def panel(ax,kind,row,data):
    ax.axis('off');w,h=data['w'],data['h']
    if kind=='mapping':ax.imshow(data['merged'],interpolation='nearest')
    elif kind=='walls':ax.imshow(data['wall_panel'],interpolation='nearest')
    elif kind=='trajectories':evidence.trajectories(ax,data)
    else:
        ax.imshow(data['truth'],cmap='Greys',vmin=0,vmax=1,interpolation='nearest');base.areas(ax,data['spec'])
        for i,b in enumerate(data['bombs'],1):
            x,y=b['x']+w/2,h/2-b['y']
            ax.add_patch(Circle((x,y),data['radius'],fill=False,color='royalblue',lw=.9))
            ax.plot(x,y,'o',color='royalblue',ms=3);ax.annotate(str(i),(x,y),xytext=(3,3),textcoords='offset points',fontsize=9)
    title=f"{row['map']} · {NAMES[row['controller']]}\n地图 {row['map_score']:.2f} / 墙体 {row['wall_score']:.2f} / {row['placed_count']}枚弹\n{row['red_steps']}步 · {row['red_walltime']:.1f}秒 · {'红有效' if row['red_valid'] else '红无效'}"
    ax.set_title(title,fontproperties=FONT,fontsize=14);ax.set(xlim=(0,w),ylim=(h,0),aspect='equal')


def figures(rows,assets,out):
    order=list(dict.fromkeys(r['controller'] for r in rows));maps=list(dict.fromkeys(r['map'] for r in rows))
    for kind,label in [('mapping','建图与真墙差异'),('trajectories','完整实际轨迹'),('bombs','实际落弹位置'),('walls','官方墙体覆盖')]:
        fig,axes=base.plt.subplots(len(maps),len(order),figsize=(8*len(order),6.2*len(maps)),squeeze=False)
        fig.subplots_adjust(left=.015,right=.99,bottom=.06,top=.86,wspace=.03,hspace=.32)
        for y,name in enumerate(maps):
            for x,controller in enumerate(order):
                row=next((r for r in rows if r['map']==name and r['controller']==controller),None)
                if row:panel(axes[y,x],kind,row,assets[(name,controller)])
                else:axes[y,x].axis('off')
        note=dict(mapping='绿色命中，蓝色漏墙，深红收费误画，黄色免罚误画。免罚不等于命中。',
            trajectories='轨迹不是雷达覆盖；灰色为GPS及罗盘缺测，黄色为无通信。颜色为录像槽位。',
            bombs='实际落点产物顺序；圆为官方爆炸半径，重叠不等于墙面收益重复。',
            walls='绿色覆盖内墙，蓝色覆盖边界墙，灰色未覆盖，红色爆炸范围；沿用官方公式。')[kind]
        fig.suptitle('RN7-B 与特殊区独立候选：'+label,fontproperties=FONT,fontsize=22,y=.985)
        fig.text(.5,.025,note+' 两图每版各一次完整红阶段；蓝队及完整总分未评测。',ha='center',fontproperties=FONT,fontsize=13)
        base.save(fig,out/f'comparison_{kind}.png')
    region_list=[]
    for name in maps:
        reference=next(r for r in rows if r['map']==name and r['controller']=='rn7_b')
        region_list.extend((name,i,region['bounds']) for i,region in enumerate(reference['gray_regions']))
    fig,axes=base.plt.subplots(len(region_list),len(order),figsize=(7*len(order),5.3*len(region_list)),squeeze=False)
    fig.subplots_adjust(left=.02,right=.99,top=.92,bottom=.04,hspace=.27,wspace=.04)
    for y,(name,index,(left,top,right,bottom)) in enumerate(region_list):
        for x,controller in enumerate(order):
            row=next((r for r in rows if r['map']==name and r['controller']==controller),None);ax=axes[y,x];ax.axis('off')
            if row is None:continue
            data=assets[(name,controller)];region=row['gray_regions'][index]
            ax.imshow(data['merged'],interpolation='nearest');ax.set(xlim=(left-8,right+8),ylim=(bottom+8,top-8),aspect='equal')
            ax.set_title(f"{name}灰区{index+1} · {NAMES[controller]}\n真实墙面命中 {region['hits']} / {region['surface']} px",fontproperties=FONT,fontsize=14)
    fig.suptitle('灰色特殊区建图放大：同一真实墙体范围',fontproperties=FONT,fontsize=22)
    fig.text(.5,.015,'绿色命中、蓝色漏墙、深红收费误画、黄色免罚误画；区域命中不是独立官方分数。',ha='center',fontproperties=FONT,fontsize=14)
    base.save(fig,out/'special_regions.png')
    for name in maps:
        selected=[r for r in rows if r['map']==name]
        fig,axes=base.plt.subplots(1,2,figsize=(13,4.8))
        for row in selected:
            samples=assets[(name,row['controller'])]['samples'];bins={}
            for sample in samples:
                if sample['gps_available'] and sample['compass_available']:continue
                estimate=sample.get('mapper')
                if estimate is None:continue
                bins.setdefault(sample['tick']//120,[]).append([estimate['position_error'],estimate['heading_error']*180/np.pi])
            ts=sorted(bins);values=np.array([np.mean(bins[t],axis=0) for t in ts])
            for col,ax in enumerate(axes):
                if len(values):ax.plot(np.array(ts)*120,values[:,col],label=NAMES[row['controller']])
        for ax in axes:ax.set_xlabel('仿真步',fontproperties=FONT);ax.legend(prop=FONT);ax.grid(alpha=.2)
        axes[0].set_ylabel('地图器位置误差 / px',fontproperties=FONT);axes[1].set_ylabel('地图器朝向误差 / 度',fontproperties=FONT)
        fig.suptitle(name+' 缺测期间误差（12步采样，120步分箱；运动版轨迹不同）',fontproperties=FONT)
        fig.tight_layout();base.save(fig,out/f'{name}_accuracy.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);sha=frozen(args.batch)
    rows=[];assets={};failures=[]
    for raw in base.read(args.batch/'results.json'):
        try:
            row,data=base.review_row(raw,sha);evidence.audit(row,data);accuracy(row,data)
            rows.append(row);assets[(row['map'],row['controller'])]=data
        except Exception as exc:failures.append(dict(map=raw['map'],controller=raw['controller'],error=repr(exc)))
    figures(rows,assets,args.output);deltas=[];paths=[]
    for name in dict.fromkeys(r['map'] for r in rows):
        reference=next(r for r in rows if r['map']==name and r['controller']=='rn7_b')
        old=assets[(name,'rn7_b')]['trajectories']
        for row in rows:
            if row['map']!=name or row['controller']=='rn7_b':continue
            deltas.append(dict(map=name,controller=row['controller'],**{k:row[k]-reference[k] for k in (
                'map_score','wall_score','time_score','wall_hits','charged_pixels','red_walltime','gray_surface_hits')}))
            current=assets[(name,row['controller'])]['trajectories'];n=min(len(old),len(current))
            difference=np.hypot(old['x'][:n]-current['x'][:n],old['y'][:n]-current['y'][:n])
            with np.load(Path(reference['attempt'])/'mapper_legal_inputs.npz') as left, np.load(Path(row['attempt'])/'mapper_legal_inputs.npz') as right:
                same_inputs=set(left.files)==set(right.files) and all(np.array_equal(left[k],right[k],equal_nan=True) for k in left.files)
            paths.append(dict(map=name,controller=row['controller'],same_frame_count=len(old)==len(current),
                max_common_position_delta=float(np.nanmax(difference)),same_bomb_layout=row['layout_sha256']==reference['layout_sha256'],
                identical_recorded_mapper_inputs=same_inputs))
    write(args.output/'summary.json',dict(rows=rows,deltas=deltas,path_checks=paths,review_failures=failures))
    doc=['# RL11 特殊区准确性：两图逐组完整红阶段图册','',
        '只读原始结果与录像；六局各一次，无蓝阶段。地图分复算和每局收据已核对；蓝队难度及完整官方原始/有效总分均未评测。',
        '', '|地图/版本|红有效|M|W|T|弹数|步数|墙钟秒|墙/机接触|缺测位姿误差px/度|',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        a=r['line_accuracy'];doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['red_steps']}|{r['red_walltime']:.2f}|{r['contacts_wall']}/{r['contacts_drone']}|{a['outage_position_summary']['mean']:.2f}/{a['outage_heading_summary']['mean']:.2f}|")
    if failures:doc+=['','复核失败保留：',json.dumps(failures,ensure_ascii=False)]
    for kind,label in [('mapping','建图'),('trajectories','实际轨迹'),('walls','官方墙体覆盖'),('bombs','实际落弹')]:
        doc+=['',label,'',f'![{label}]({(args.output.resolve()/f"comparison_{kind}.png").as_posix()})']
    for name in dict.fromkeys(r['map'] for r in rows):doc+=['',f'![{name}定位误差]({(args.output.resolve()/f"{name}_accuracy.png").as_posix()})']
    doc+=['',f'![灰区建图放大]({(args.output.resolve()/"special_regions.png").as_posix()})']
    doc+=['','## 真实回放','', '逐次运行；本轮蓝队未运行，没有蓝回放路径。']
    for r in rows:
        doc += ['',f"{r['map']}/{NAMES[r['controller']]} / r{r['round']} / seed {r['red_seed']}",'','```powershell',
            "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{r['attempt']}'",'```']
    with (args.output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(doc)+'\n')
    for path in args.output.glob('*.png'):
        with Image.open(path) as img:img.verify()
    protected=base.read('output/sf13/rl11_dev_20261114/protected_before.json')
    changed=[r['path'] for r in protected if not Path(r['path']).is_file() or base.digest(r['path'])!=r['sha256']]
    write(args.output/'verification.json',dict(protected=len(protected),changed=changed,source_archive_verified=True,
        receipts_verified=not failures,formula_reproduced=not failures,review_failures=failures,
        images=len(list(args.output.glob('*.png'))),all_valid=all(r['red_valid'] for r in rows),
        diagnostic_errors=[(r['diagnostic_errors'],r['line_diagnostic_errors']) for r in rows]))
    print(json.dumps(dict(deltas=deltas,paths=paths,failures=failures),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
