"""R8 official-red atlas + independent component evidence, offline only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools.red_rn7_review import regions,REGIONS
from swarm_rescue.solutions.devtools.optimization.evaluation.runner import validate_sources

NAMES={'rn7_b':'RN7-B reference','r8_n':'N8: footprint + task','r8_m':'P8: 1 px mapping'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[];assets={};plans={}
    for batch in (Path('output/sf13/rn7_m89_once'),args.batch):
        plan=base.read(batch/'plan.json');validate_sources(plan['source_sha256'])
        sha=base.digest(batch/'plan.json');plans[str(batch)]=sha
        for raw in base.read(batch/'results.json'):
            if raw['controller'] not in NAMES:continue
            row,data=base.review_row(raw,sha);run=Path(row['attempt'])
            row['region_drone_frames']=regions(data,row['map'])
            counts=Counter()
            for d in base.read(run/'narrow_diagnostics.json')['drones'].values():counts.update(d['stats'])
            row['narrow_stats']=dict(counts)
            if (run/'r8_evidence.json').exists():
                row['evidence']=base.read(run/'r8_evidence.json')
                observed=np.zeros(data['truth'].shape,bool)
                with np.load(run/'observed_endpoints.npz') as file:
                    for key in file.files:observed|=file[key]
                data['observed_endpoints']=observed
                row['sampled_observed_truth_endpoints']=int((observed&data['truth'].astype(bool)).sum())
            rows.append(row);assets[(row['map'],row['controller'])]=data
    base.NAMES=NAMES;base.ORDER=list(NAMES)
    for name in ('Map08','Map09'):base.figures(name,rows,assets,args.output)
    # Explicitly highlight requested regions in the trajectory panels.
    for name in ('Map08','Map09'):
        selected=sorted([r for r in rows if r['map']==name],key=lambda r:base.ORDER.index(r['controller']))
        fig,axes=base.plt.subplots(1,3,figsize=(18,5.4));fig.subplots_adjust(left=.02,right=.99,top=.83,bottom=.1,wspace=.04)
        for ax,r in zip(axes,selected):
            a=assets[(name,r['controller'])];w,h=a['w'],a['h'];t=a['trajectories']
            ax.imshow(a['truth'],cmap='Greys',vmin=0,vmax=1);base.areas(ax,a['spec'])
            for slot in range(t.shape[1]):
                good=t['active'][:,slot];x=np.where(good,t['x'][:,slot]+w/2,np.nan);y=np.where(good,h/2-t['y'][:,slot],np.nan)
                jump=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40];x[jump]=np.nan;y[jump]=np.nan
                ax.plot(x,y,lw=.6,alpha=.7)
            for l,top,right,bottom in REGIONS[name].values():ax.add_patch(Rectangle((l,top),right-l,bottom-top,fill=False,color='magenta',ls='--'))
            ax.set_title(f"{NAMES[r['controller']]}\nM {r['map_score']:.4f} | W {r['wall_score']:.4f}",fontsize=12)
            ax.set(xlim=(0,w),ylim=(h,0),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(name+' | physical exploration paths | reference reused; one new attempt per ablation',fontsize=14)
        fig.text(.5,.025,'Magenta: diagnostic regions. Paths are not lidar coverage. Same seed is not exact sensor replay.',ha='center')
        base.save(fig,args.output/f'{name}_regions.png')
    # Same INPUT mapping evidence, including failed early implementation.
    native=base.read('output/sf13/r8_mapping_native_v2/results.json')
    fig,axes=base.plt.subplots(2,3,figsize=(15,8))
    for rr,row in enumerate(native):
        with np.load(Path('output/sf13/r8_mapping_native_v2')/(row['case']+'.npz')) as arrays:
            for cc,key in enumerate(('truth','prediction','pixel_prediction')):
                ax=axes[rr,cc];ax.imshow(arrays[key],cmap='Greys',vmin=0,vmax=1)
                label=['Truth (offline)','A3 mapper','P8 pixel faces'][cc]
                score='' if cc==0 else f" | M formula {100*row['map_formula' if cc==1 else 'pixel_map_formula']['s_map']:.3f}"
                ax.set_title(f"{'Noise' if row['noise'] else 'Ideal'} / {label}{score}",fontsize=10);ax.axis('off')
    fig.suptitle('Identical legal sensor inputs, synthetic corner fixture. Not official Map08/Map09 scores.')
    base.save(fig,args.output/'same_input_mapping.png')
    native_motion=base.read('output/sf13/r8_native_final/results.json')
    fig,axes=base.plt.subplots(1,3,figsize=(15,4))
    for ax,row in zip(axes,native_motion):
        width=row['width']
        for y in (-width/2,width/2):ax.plot([0,160],[y,y],color='black',lw=3)
        if row['deadend']:ax.plot([160,160],[-width/2,width/2],color='black',lw=3)
        ax.plot([r['x'] for r in row['trace']],[r['y'] for r in row['trace']],lw=1)
        ax.set(xlim=(-70,210),ylim=(-85,85),aspect='equal')
        ax.set_title(f"{row['case']} | passed={row['passed']}\ncontacts={row['contacts']} | {row['steps']} steps")
    fig.suptitle('Final native noisy fixtures; rejected/unfinished passage attempts retained')
    base.save(fig,args.output/'native_passages.png')
    deltas=[]
    for row in rows:
        if row['controller']=='rn7_b':continue
        old=next(r for r in rows if r['map']==row['map'] and r['controller']=='rn7_b')
        assert row['red_seed']==old['red_seed']
        deltas.append(dict(map=row['map'],controller=row['controller'],**{key:row[key]-old[key] for key in ('map_score','wall_score','time_score','wall_hits','charged_pixels','red_walltime')}))
    result=dict(rows=rows,deltas=deltas,plans=plans,blue_evaluated=False,native_mapping=native,native_motion=native_motion)
    # Drop bulky traces only in this summary; original native logs remain intact.
    for r in result['native_mapping']+result['native_motion']:r.pop('trace',None)
    with (args.output/'summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    lines=['# R8：1 px墙面融合与机身轮廓窄路独立对照','',
        'RN7-B历史参考直接复用；N8仅改变窄路运动/任务预算，P8仅改变提交地图及红队专用快照载荷。两项未默认组合。',
        '每图每个新单项候选一轮。相同地图、种子和官方资源约束；引擎随机UID并未完全受固定种子控制，因此完整比赛不是相同输入回放。蓝阶段未运行：难度、官方原始及有效总分未评测。','',
        '|地图|版本|红有效|M|W|T|弹数|墙/机接触|步数|秒|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['map']}|{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['contacts_wall']}/{r['contacts_drone']}|{r['red_steps']}|{r['red_walltime']:.2f}|")
    for name in ('Map08','Map09'):
        lines+=['',f'## {name}','', '从左到右：RN7-B历史参考、N8窄路单项、P8建图单项。']
        for kind,title in [('mapping','提交建图差异'),('walls','官方墙体覆盖'),('bombs','实际落弹位置'),('regions','实际探索轨迹与关注区域')]:
            lines+=['',title,'',f'![{name} {title}]({(args.output.resolve()/f"{name}_{kind}.png").as_posix()})']
    lines+=['','## 区域、地图像素与诊断','',
        '区域帧数是机体中心在框内的无人机帧数，不是传感器覆盖率。新增端点诊断每4步采样引擎实际射线命中，真值访问在控制器保护边界外；端点落在真值墙像素的计数只是稀疏端点下界，不是完整墙面观察率。','',
        '|地图/版本|关注区域机体帧数|真实墙命中|误画扣分像素|采样实际墙端点像素|','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['map']}/{r['controller']}|{r['region_drone_frames']}|{r['wall_hits']}|{r['charged_pixels']}|{r.get('sampled_observed_truth_endpoints','历史未记录')}|")
    lines+=['','## 相同输入建图组件对照','',f'![同输入建图]({(args.output.resolve()/"same_input_mapping.png").as_posix()})',
        '这两个建图器在同一架组件测试飞机上接收完全相同的合法GPS、罗盘、里程计、雷达和语义输入；分数是原官方地图公式作用于合成场景，不是正式地图比赛成绩。',
        '',f'![原生窄路]({(args.output.resolve()/"native_passages.png").as_posix()})','',
        '46px带噪通道最终未完整穿出，50px带噪通过，死胡同保留真实结果；不能宣布所有门已保证通过。早期测试版本与失败均保留在r8_native_v1/v2/v3、r8_mapping_native_v1。','',
        '## 回放','', '没有本轮蓝阶段录像，不拼接历史蓝分。']
    for r in rows:
        lines+=['',f"{r['map']}/{r['controller']} / r1 / seed {r['red_seed']}",'','```powershell',
                "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{r['attempt']}'",'```']
    with (args.output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    protected=json.loads(Path('output/sf13/r8_dev/protected_before.json').read_text(encoding='utf-8-sig'))
    changed=[p['path'] for p in protected if not Path(p['path']).exists() or base.digest(p['path']).lower()!=p['sha256'].lower()]
    image_paths=re.findall(r'!\[[^\]]*\]\(([^)]+)\)','\n'.join(lines))
    for path in image_paths:
        with Image.open(path) as image:image.verify()
    verification=dict(protected_count=len(protected),changed=changed,images_verified=len(image_paths),
        new_attempts=len(list((args.batch/'runs').rglob('attempt_*'))),source_receipts_verified=True,
        diagnostic_errors=[r['evidence']['diagnostic_errors'] for r in rows if 'evidence' in r])
    with (args.output/'verification.json').open('x',encoding='utf-8') as f:json.dump(verification,f,indent=2)
    print(json.dumps(dict(deltas=deltas,verification=verification),indent=2))


if __name__=='__main__':main()
