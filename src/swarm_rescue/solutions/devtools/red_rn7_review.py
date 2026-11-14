"""Offline RN7 atlas: official scoring + actual trajectories, no evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from matplotlib.patches import Rectangle
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base

NAMES={'rn7_a':'A3 + RN7','rn7_b':'B + RN7','rn7_c':'C + RN7'}
CONTROL={'rn7_a':'advanced2_mapping','rn7_b':'s6_motion','rn7_c':'s6_radar'}
REGIONS={'Map08':{'bottom_lanes':(956,1027,1654,1116),'bottom_right':(1365,1064,1654,1116)},
         'Map09':{'upper_left':(6,6,920,316),'middle_left':(6,325,920,559)}}


def regions(asset,map_name):
    t=asset['trajectories'];x=t['x']+asset['w']/2;y=asset['h']/2-t['y']
    return {name:int((t['active']&(x>=l)&(x<=r)&(y>=top)&(y<=b)).sum())
            for name,(l,top,r,b) in REGIONS[map_name].items()}


def comparison(map_name,new_rows,old_rows,assets,old_assets,output):
    fig,axes=base.plt.subplots(2,3,figsize=(18,11))
    for col,controller in enumerate(NAMES):
        old=CONTROL[controller]
        for row_i,selector,rows,data_source,label in (
            (0,old,old_rows,old_assets,'Original'),(1,controller,new_rows,assets,'+ RN7')):
            row=next(r for r in rows if r['map']==map_name and r['controller']==selector)
            data=data_source[(map_name,selector)];ax=axes[row_i,col]
            ax.imshow(data['truth'],cmap='Greys',vmin=0,vmax=1,interpolation='nearest');base.areas(ax,data['spec'])
            t=data['trajectories'];w,h=data['w'],data['h']
            for slot in range(t.shape[1]):
                valid=t['active'][:,slot]&np.isfinite(t['x'][:,slot])&np.isfinite(t['y'][:,slot])
                x=np.where(valid,t['x'][:,slot]+w/2,np.nan);y=np.where(valid,h/2-t['y'][:,slot],np.nan)
                jump=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40];x[jump]=np.nan;y[jump]=np.nan
                ax.plot(x,y,lw=.6,alpha=.65)
            for name,(l,top,r,b) in REGIONS[map_name].items():
                ax.add_patch(Rectangle((l,top),r-l,b-top,fill=False,color='magenta',lw=1.2,ls='--'))
            ax.set(xlim=(0,w),ylim=(h,0));ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f"{label} {'ABC'[col]} | M {row['map_score']:.4f} | W {row['wall_score']:.4f}")
    fig.suptitle(f'{map_name}: same recorded seed, one full red attempt each | magenta: offline diagnostic regions',fontsize=15)
    fig.text(.5,.02,'Physical trajectories are NOT lidar coverage. Engine random UIDs are not fully controlled by the recorded seed.',ha='center')
    base.save(fig,output/f'{map_name}_trajectory_before_after.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('output/sf13/rs6_m89_once'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    new=[];old=[];assets={};old_assets={}
    for batch,rows,data in ((args.batch,new,assets),(args.baseline,old,old_assets)):
        plan_hash=base.digest(batch/'plan.json')
        for raw in base.read(batch/'results.json'):
            row,item=base.review_row(raw,plan_hash)
            row['region_drone_frames']=regions(item,row['map'])
            if row['controller'] in NAMES:
                stats=Counter()
                for drone in base.read(Path(row['attempt'])/'narrow_diagnostics.json')['drones'].values():stats.update(drone['stats'])
                row['narrow_stats']=dict(stats)
            rows.append(row);data[(row['map'],row['controller'])]=item
    base.NAMES=NAMES;base.ORDER=list(NAMES)
    deltas=[]
    for row in new:
        control=next(r for r in old if r['map']==row['map'] and r['controller']==CONTROL[row['controller']])
        assert row['red_seed']==control['red_seed']
        deltas.append(dict(map=row['map'],controller=row['controller'],
            **{key:row[key]-control[key] for key in ('map_score','wall_score','time_score','wall_hits','charged_pixels','contacts_wall','contacts_drone','red_walltime')}))
    for map_name in ('Map08','Map09'):
        base.figures(map_name,new,assets,args.output)
        comparison(map_name,new,old,assets,old_assets,args.output)
    with (args.output/'summary.json').open('x',encoding='utf-8') as f:
        json.dump(dict(rows=new,baseline=old,deltas=deltas,blue_evaluated=False),f,indent=2)
    lines=['# RN7 红队窄路独立候选：Map08 / Map09 单轮实测','',
        '每个新候选每图仅运行一次完整红阶段；原版来自 rs6_m89_once，未重跑挑分。蓝队难度、官方原始/有效总分均未评测。',
        '同地图、种子、官方参数、资源限制；引擎随机 UID 未完全受种子控制，不能声称严格相同观测输入回放。','',
        '|地图|版本|红有效|M|ΔM|W|ΔW|T|弹数|墙/机接触|步数|秒|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r,d in zip(new,deltas):
        lines.append(f"|{r['map']}|{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{d['map_score']:+.4f}|{r['wall_score']:.4f}|{d['wall_score']:+.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['contacts_wall']}/{r['contacts_drone']}|{r['red_steps']}|{r['red_walltime']:.2f}|")
    for map_name in ('Map08','Map09'):
        lines+=['',f'## {map_name}','',
            '下列三列依次为 A3+RN7、B+RN7、C+RN7；建图差异不能冒充完整激光观测覆盖。']
        for kind,title in [('mapping','提交建图差异'),('trajectories','新候选实际轨迹'),('walls','官方墙体覆盖'),('bombs','实际落弹位置'),('trajectory_before_after','原版与候选轨迹对照')]:
            lines+=['',f'### {title}','',f'![{map_name} {title}]({(args.output.resolve()/f"{map_name}_{kind}.png").as_posix()})']
    lines+=['','## 关注区域的实际进入证据','',
        '下面是原始录像中机体中心位于离线框选区域的无人机帧数；不是激光覆盖率，也不是独立访问次数。未记录完整逐帧传感器，不能精准区分所有漏墙成因。','',
        '|地图|版本|原版区域帧数|RN7区域帧数|局部任务进入/完成/活跃步|','|---|---|---|---|---|']
    for r in new:
        o=next(x for x in old if x['map']==r['map'] and x['controller']==CONTROL[r['controller']]);s=r['narrow_stats']
        lines.append(f"|{r['map']}|{NAMES[r['controller']]}|{o['region_drone_frames']}|{r['region_drone_frames']}|{s.get('entered',0)}/{s.get('completed',0)}/{s.get('active_ticks',0)}|")
    lines+=['','## 原始图册与回放','',f'[全六局原始结果与完整图册]({(args.batch.resolve()/"report.md").as_posix()})',
        '蓝阶段未运行，因此没有本轮蓝回放；不以历史蓝录像代替。']
    for r in new:
        lines+=['',f"{r['map']} / {NAMES[r['controller']]} / r1 / seed {r['red_seed']}",'','```powershell',
                "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{r['attempt']}'",'```']
    with (args.output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(valid=sum(r['red_valid'] for r in new),deltas=deltas),indent=2))


if __name__=='__main__':main()
