"""RL12 red-only atlas and offline doorway audit, no simulator execution."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import cv2
import numpy as np
from matplotlib.patches import Rectangle, Circle
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_rx10_review as evidence
from swarm_rescue.solutions.devtools import red_lines11_review as lines

NAMES={'rl11_lines':'RL11 原版','rl12_nav':'RL12-N 导航墙段校正'}
FONT=lines.FONT
# Offline truth-coordinate regions selected from the user's screenshot only.
ROOMS={'上左房间':(540,75,655,215),'上右房间':(735,75,845,215),'下方房间':(650,340,948,545)}
DOORS={'上左入口':(626,235),'上右入口':(813,235),'下方入口':(815,319)}


def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False)


def reachable(free,obstacle,position,w,h,cell):
    blocked=cv2.dilate((obstacle>=2).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))).astype(bool)
    allowed=free&~blocked
    r=int(np.clip((h/2-position[1])/cell,0,allowed.shape[0]-1))
    c=int(np.clip((position[0]+w/2)/cell,0,allowed.shape[1]-1))
    allowed[r,c]=True;seen=np.zeros_like(allowed);seen[r,c]=True;q=deque([(r,c)])
    while q:
        r,c=q.popleft()
        for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=rr<seen.shape[0] and 0<=cc<seen.shape[1] and allowed[rr,cc] and not seen[rr,cc]:
                seen[rr,cc]=True;q.append((rr,cc))
    return seen,blocked


def doorway_audit(row,data):
    run=Path(row['attempt']);doc=base.read(run/'navigation_diagnostics.json')
    row['navigation_diagnostic_errors']=doc['errors'];row['navigation_diagnostic_seconds']=doc['diagnostic_seconds']
    assert not doc['errors'],doc['errors']
    if row['map']!='Map09':return
    t=data['trajectories'];x=t['x']+data['w']/2;y=data['h']/2-t['y'];audit={}
    for label,(left,top,right,bottom) in ROOMS.items():
        inside=t['active']&(x>left)&(x<right)&(y>top)&(y<bottom)
        audit[label]=dict(actual_frames=int(inside.sum()),per_slot=[int(inside[:,i].sum()) for i in range(t.shape[1])])
    snapshots=[];navigation=Counter()
    with np.load(run/'navigation_diagnostics.npz') as f:
        for sample in doc['rows']:
            px,py=sample['truth'][0]+data['w']/2,data['h']/2-sample['truth'][1]
            if sample['step']>3600 or not (680<px<1000 and 200<py<395):continue
            key=sample['key'];free=f[key+'_free'];obstacle=f[key+'_obstacle'];cell=sample['cell']
            seen,blocked=reachable(free,obstacle,sample['position'],data['w'],data['h'],cell)
            probes={}
            for label,(dx,dy) in DOORS.items():
                rr,cc=int(dy/cell),int(dx/cell)
                probes[label]=dict(free=bool(free[rr,cc]),wall=bool(obstacle[rr,cc]>=2),
                    inflated=bool(blocked[rr,cc]),reachable=bool(seen[rr,cc]))
            goals=[]
            if sample['route']:
                rr,cc=sample['route'][-1];gx,gy=(cc+.5)*cell,(rr+.5)*cell
                goals=[label for label,(l,tp,r,b) in ROOMS.items() if l<gx<r and tp<gy<b]
            snapshots.append(dict(drone=sample['drone'],step=sample['step'],truth_pixel=[px,py],
                navigation_position_error=float(np.linalg.norm(np.asarray(sample['position'])-sample['truth'])),
                heading_error_degrees=abs(float(np.degrees((sample['heading']-sample['true_heading']+np.pi)%(2*np.pi)-np.pi))),
                probes=probes,selected_room_goals=goals,gathering=sample['gathering'],holding=sample['holding'],
                bomb_task=sample['bomb_target'] is not None,narrow_active=sample['narrow_active'],key=key))
        # Choose a snapshot near the actual lower doorway, with deterministic ties.
        if snapshots:
            chosen=min(snapshots,key=lambda s:(np.linalg.norm(np.asarray(s['truth_pixel'])-[815,285]),s['step']))
            sample=next(s for s in doc['rows'] if s['key']==chosen['key']);key=chosen['key']
            seen,blocked=reachable(f[key+'_free'],f[key+'_obstacle'],sample['position'],data['w'],data['h'],sample['cell'])
            data['nav_snapshot']=dict(sample=sample,free=f[key+'_free'].copy(),wall=(f[key+'_obstacle']>=2),
                                      blocked=blocked,reachable=seen,audit=chosen)
    row['doorway_audit']=dict(rooms=audit,nearby_snapshots=snapshots,
        limitation='Grid probes are at offline true doorway coordinates. Drift means these are not necessarily the controller believed doorway. Samples every 60 steps; missed short tasks cannot be ruled out.')


def panel(ax,kind,row,data):
    w,h=data['w'],data['h'];ax.axis('off')
    if kind=='mapping':ax.imshow(data['merged'])
    elif kind=='walls':ax.imshow(data['wall_panel'])
    elif kind=='trajectories':evidence.trajectories(ax,data)
    else:
        ax.imshow(data['truth'],cmap='Greys',vmin=0,vmax=1);base.areas(ax,data['spec'])
        for i,b in enumerate(data['bombs'],1):
            x,y=b['x']+w/2,h/2-b['y'];ax.add_patch(Circle((x,y),data['radius'],fill=False,color='royalblue',lw=.9))
            ax.plot(x,y,'o',color='royalblue',ms=3);ax.annotate(str(i),(x,y),xytext=(3,3),textcoords='offset points')
    ax.set(xlim=(0,w),ylim=(h,0),aspect='equal')
    ax.set_title(f"{row['map']} · {NAMES[row['controller']]}\nM {row['map_score']:.4f} / W {row['wall_score']:.2f} / {row['placed_count']}弹\n{row['red_steps']}步 / {row['red_walltime']:.1f}秒 / {'有效' if row['red_valid'] else '无效'}",fontproperties=FONT,fontsize=13)


def figures(rows,assets,out):
    maps=list(dict.fromkeys(r['map'] for r in rows));controllers=list(dict.fromkeys(r['controller'] for r in rows))
    for kind,label in [('mapping','提交建图'),('trajectories','实际完整轨迹'),('walls','官方墙体覆盖'),('bombs','实际炸弹位置')]:
        fig,axes=base.plt.subplots(len(maps),len(controllers),figsize=(8*len(controllers),6*len(maps)),squeeze=False)
        for y,m in enumerate(maps):
            for x,c in enumerate(controllers):
                row=next(r for r in rows if r['map']==m and r['controller']==c);panel(axes[y,x],kind,row,assets[m,c])
        fig.suptitle('RL11 与独立导航校正对照：'+label,fontproperties=FONT,fontsize=19)
        fig.tight_layout(rect=(0,.055,1,.96))
        note={'mapping':'绿：命中；蓝：漏墙；深红：收费误画；黄：免罚误画。',
              'trajectories':'轨迹不是雷达覆盖；灰：无GPS/罗盘；黄：无通信；线色为录像槽位。',
              'walls':'绿：覆盖内墙；蓝：覆盖边界墙；灰：未覆盖；红：爆炸范围。',
              'bombs':'落点编号按产物顺序；圆为实际爆炸半径，重叠不等于墙面收益重复。'}[kind]
        fig.text(.5,.025,note+' 各图各版一次完整红队；蓝队与完整总分未评测。',ha='center',fontproperties=FONT,fontsize=12)
        base.save(fig,out/f'comparison_{kind}.png')
    fig,axes=base.plt.subplots(2,len(controllers),figsize=(8*len(controllers),11),squeeze=False)
    for x,c in enumerate(controllers):
        row=next(r for r in rows if r['map']=='Map09' and r['controller']==c);data=assets['Map09',c]
        panel(axes[0,x],'trajectories',row,data);panel(axes[1,x],'mapping',row,data)
        for ax in axes[:,x]:
            ax.set(xlim=(495,1000),ylim=(580,35))
            for label,(l,t,r,b) in ROOMS.items():
                ax.add_patch(Rectangle((l,t),r-l,b-t,fill=False,color='magenta',lw=1.2,ls='--'))
            for dx,dy in DOORS.values():ax.plot(dx,dy,'o',color='deepskyblue',ms=7,mfc='none')
    fig.suptitle('用户截图区域：三个房间及入口（仅离线标注）',fontproperties=FONT,fontsize=19)
    fig.tight_layout(rect=(0,.03,1,.96));base.save(fig,out/'Map09_entrances.png')
    fig,axes=base.plt.subplots(1,len(controllers),figsize=(8*len(controllers),7),squeeze=False)
    for x,c in enumerate(controllers):
        data=assets['Map09',c];nav=data.get('nav_snapshot');ax=axes[0,x]
        if nav is None:ax.axis('off');continue
        s=nav['sample'];h,w=data['h'],data['w'];image=np.ones((*nav['free'].shape,3))*.9
        image[nav['free']]=[.65,.85,1];image[nav['blocked']]=[1,.65,.3];image[nav['wall']]=[.1,.1,.1]
        image[nav['reachable']]=[.45,.85,.6];cell=s['cell']
        ax.imshow(image,extent=(0,image.shape[1]*cell,image.shape[0]*cell,0),interpolation='nearest')
        tx,ty=s['truth'][0]+w/2,h/2-s['truth'][1];nx,ny=s['position'][0]+w/2,h/2-s['position'][1]
        ax.plot(tx,ty,'o',color='red',ms=8,label='实际机身');ax.plot(nx,ny,'x',color='blue',ms=10,label='导航估计')
        ax.plot([tx,nx],[ty,ny],'r--',lw=1)
        if s['route']:
            r=np.array(s['route']);ax.plot((r[:,1]+.5)*cell,(r[:,0]+.5)*cell,'m-',lw=2,label='当前路径')
        for dx,dy in DOORS.values():ax.plot(dx,dy,'o',mfc='none',color='cyan',ms=9)
        ax.set(xlim=(495,1100),ylim=(620,20),aspect='equal')
        ax.set_title(f"{NAMES[c]} · 机{s['drone']} · 步{s['step']}\n导航位置误差 {nav['audit']['navigation_position_error']:.1f}px",fontproperties=FONT,fontsize=14)
        ax.legend(prop=FONT,fontsize=10)
    fig.suptitle('实际记录的导航图：黑障碍、橙膨胀、绿可达、蓝已知空地、灰未知',fontproperties=FONT,fontsize=16)
    fig.tight_layout(rect=(0,0,1,.94));base.save(fig,out/'Map09_navigation_grid.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);ph=lines.frozen(a.batch)
    rows=[];assets={}
    for raw in base.read(a.batch/'results.json'):
        row,data=base.review_row(raw,ph);evidence.audit(row,data);lines.accuracy(row,data)
        doorway_audit(row,data);rows.append(row);assets[row['map'],row['controller']]=data
    figures(rows,assets,a.output)
    write(a.output/'summary.json',dict(rows=rows,source_batch=str(a.batch.resolve()),plan_sha256=ph))
    report=['# RL12：RL11基础上的独立导航墙段校正','',
        '2026-11-15；各图各版一次完整红阶段。非同输入地图器实验：导航候选会改变真实轨迹。蓝队难度及官方完整总分未评测。','',
        '|地图|版本|有效|M|W|T|弹数|步|秒|墙/机接触|','|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        report.append(f"|{r['map']}|{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['red_steps']}|{r['red_walltime']:.3f}|{r['contacts_wall']}/{r['contacts_drone']}|")
    for name in ('mapping','trajectories','walls','bombs'):
        report+=['',f'![{name}]({(a.output/f"comparison_{name}.png").resolve().as_posix()})']
    for name in ('Map09_entrances','Map09_navigation_grid'):
        report+=['',f'![{name}]({(a.output/f"{name}.png").resolve().as_posix()})']
    report+=['','## 身份与回放','']
    for r in rows:
        run=Path(r['attempt']).resolve()
        report += [f"{r['map']} / {r['controller']} / r{r['round']} / seed {r['red_seed']} / zones {r['actual_conditions']}",
            '', '```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{run}'",'```','']
    report+=['无本轮蓝录像。全部资源、源码身份、地图误差、碰撞和入口记录见summary.json；原始每组图册保留在批次report.md。']
    with (a.output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    print(json.dumps([dict(map=r['map'],controller=r['controller'],M=r['map_score'],W=r['wall_score'],valid=r['red_valid'],doors=r.get('doorway_audit',{}).get('rooms')) for r in rows],ensure_ascii=True))


if __name__=='__main__':main()
