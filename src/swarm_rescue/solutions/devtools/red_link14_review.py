"""RL14 offline chain/region atlas. Truth coordinates are confined to devtools."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_rx10_review as evidence
from swarm_rescue.solutions.devtools import red_lines11_review as lines
from swarm_rescue.solutions.devtools import red_consistent12_review as panels
from swarm_rescue.solutions.devtools import red_local13_finalize as previous

NAMES={'rl11_lines':'RL11 原版','rl14_delivery':'RL14-D 送达退路','rl14_opening':'RL14-E 入口任务','rl14_relative':'RL14-R 相对通道'}
KINDS={'exploration':'探索观察范围','mapping':'完整提交建图','trajectories':'实际轨迹','bombs':'实际炸弹位置','walls':'墙体覆盖'}
NOTES={'exploration':'绿：每8步/4px可见区估计，含射线间插值；白：未覆盖估计；轨迹不是雷达观察。',
    'mapping':'绿：真墙面命中；蓝：漏墙；深红：受罚误画；黄：免罚误画。',
    'trajectories':'线色是录像槽位；灰：无GPS/罗盘；黄：无通信；实际坐标只用于离线诊断。',
    'bombs':'编号为实际产物顺序；圆为爆炸半径，圆重叠不等于有效墙面收益重复。',
    'walls':'绿：覆盖内墙；蓝：覆盖边界墙；灰：未覆盖墙；红：爆炸范围。'}


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def partition(truth):
    """Exhaustive offline geometric partition, not a claim of semantic rooms.

    8px free cells, cores with >=48px clearance, then obstacle-respecting
    multi-source flood. Small disconnected free pieces get their own IDs.
    """
    h,w=truth.shape;scale=8
    padded=np.pad(truth.astype(bool),((0,(-h)%scale),(0,(-w)%scale)),constant_values=True)
    blocked=padded.reshape(padded.shape[0]//scale,scale,padded.shape[1]//scale,scale).any(axis=(1,3))
    free=~blocked;distance=cv2.distanceTransform(free.astype(np.uint8),cv2.DIST_L2,5)*scale
    _,labels,stats,_=cv2.connectedComponentsWithStats((distance>=48).astype(np.uint8),connectivity=4)
    for k in range(1,len(stats)):
        if stats[k,cv2.CC_STAT_AREA]<8:labels[labels==k]=0
    unique=np.unique(labels);renumber={int(k):i for i,k in enumerate(unique) if k}
    labels=np.array([renumber.get(int(v),0) for v in labels.ravel()],np.int32).reshape(labels.shape)
    queue=deque(map(tuple,np.argwhere(labels>0)))
    while queue:
        r,c=queue.popleft()
        for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0<=rr<labels.shape[0] and 0<=cc<labels.shape[1] and free[rr,cc] and labels[rr,cc]==0:
                labels[rr,cc]=labels[r,c];queue.append((rr,cc))
    n,isolated=cv2.connectedComponents((free&(labels==0)).astype(np.uint8),connectivity=4)
    off=int(labels.max())
    labels[isolated>0]=isolated[isolated>0]+off
    return np.repeat(np.repeat(labels,scale,0),scale,1)[:h,:w]


def inside(point,box,w,h):
    x,y=float(point[0])+w/2,h/2-float(point[1]);l,t,r,b=box
    return l<=x<=r and t<=y<=b


def chain_audit(row,data):
    run=Path(row['attempt']);doc=base.read(run/'link14_diagnostics.json');row['chain_errors']=doc['errors']
    row['chain_seconds']=doc['seconds'];row['link_stats']=dict(sum((Counter(d['stats']) for d in doc['drones'].values()),Counter()))
    w,h=data['w'],data['h'];truth=data['truth'].astype(bool);wall,_=base.MapExplorationScorer().split_regions(data['truth']);wall=wall.astype(bool)
    with np.load(run/'link14_diagnostics.npz') as f:
        hits={k[4:]:f[k].astype(bool) for k in f.files if k.startswith('hit_')}
    with np.load(run/'exploration_stages.npz') as f:
        locals_={k.removeprefix('local_final_pixels_'):f[k].astype(bool) for k in f.files if k.startswith('local_final_pixels_')}
    seen=np.logical_or.reduce(list(hits.values()))&wall
    own_seen=np.zeros_like(wall)
    for ident,mask in hits.items():own_seen|=mask&wall&locals_[ident]
    data['dense_seen']=seen;data['dense_own']=own_seen
    row['dense_ray_chain']=dict(actual_hit_surface_pixels=int(seen.sum()),seen_own_final=int(own_seen.sum()),
        seen_submitted=int((seen&data['pred']).sum()),seen_own_but_missing_submission=int((own_seen&~data['pred']).sum()))
    trace=data['trajectories'];x=trace['x']+w/2;y=h/2-trace['y'];valid=trace['active']&np.isfinite(x)&np.isfinite(y)
    xx=np.clip(np.rint(x).astype(int),0,w-1);yy=np.clip(np.rint(y).astype(int),0,h-1)
    visited=np.zeros_like(wall);visited[yy[valid],xx[valid]]=True
    labels=partition(data['truth']);data['region_labels']=labels;regions=[]
    counts=np.bincount(labels[yy[valid],xx[valid]],minlength=int(labels.max())+1)
    for k in range(1,int(labels.max())+1):
        mask=labels==k;rr,cc=np.nonzero(mask)
        if not len(rr):continue
        area=int(mask.sum());visible=int((mask&data['visible']).sum())
        boundary=wall&(cv2.dilate(mask.astype(np.uint8),np.ones((11,11),np.uint8))>0)
        regions.append(dict(id=k,bbox=[int(cc.min()),int(rr.min()),int(cc.max()),int(rr.max())],free_pixels=area,
            actual_frames=int(counts[k]),sampled_visible_fraction=visible/area,
            wall_surface=int(boundary.sum()),actual_hit_surface=int((boundary&seen).sum()),
            submitted_wall_hits=int((boundary&data['pred']).sum())))
    row['all_geometric_regions']=regions
    row['unvisited_region_ids']=[r['id'] for r in regions if r['actual_frames']==0]
    row['zero_sampled_visibility_region_ids']=[r['id'] for r in regions if r['sampled_visible_fraction']==0]
    row['low_sampled_visibility_region_ids']=[r['id'] for r in regions if r['sampled_visible_fraction']<.5]
    rooms={}
    # Prior user-screenshot rooms are OFFLINE only; all other regions above
    # are enumerated automatically on both maps, without strategy knowledge.
    if row['map']=='Map09':
        for name,box in panels.ROOMS.items():
            l,t,r,b=box;mask=np.zeros_like(wall);mask[t:b+1,l:r+1]=True
            near=wall&(cv2.dilate(mask.astype(np.uint8),np.ones((31,31),np.uint8))>0)
            enter=valid&(x>l)&(x<r)&(y>t)&(y<b)
            discoveries=[];selected=[];route_samples=[]
            for ident,d in doc['drones'].items():
                for e in d['discoveries']:
                    if inside(e['actual_point'],box,w,h):discoveries.append(dict(drone=int(ident),**e))
                for e in d['events']:
                    if 'actual_point' in e and inside(e['actual_point'],box,w,h):selected.append(dict(drone=int(ident),**e))
                for s in d['samples']:
                    if s['route']:
                        rr,cc=s['route'][-1];px=(cc+.5)*s['cell'];py=(rr+.5)*s['cell']
                        if l<=px<=r and t<=py<=b:route_samples.append(dict(drone=int(ident),**s))
            rooms[name]=dict(bbox=box,actual_frames=int(enter.sum()),frames_per_slot=enter.sum(axis=0).tolist(),
                candidate_discoveries=len(discoveries),selected_entry_events=len(selected),coarse_goal_samples=len(route_samples),
                wall_surface=int(near.sum()),actual_observed_surface=int((near&seen).sum()),
                seen_own_final=int((near&own_seen).sum()),seen_submitted=int((near&seen&data['pred']).sum()),
                seen_own_missing_submission=int((near&own_seen&~data['pred']).sum()),
                discoveries=discoveries,selected=selected,coarse_goals=route_samples)
    row['screenshot_rooms']=rooms;row['task_chains']=[]
    for ident,d in doc['drones'].items():
        for e in d['events']:
            if e['event']!='opening_selected':continue
            related=[v for v in d['events'] if v.get('task')==e['task']]
            row['task_chains'].append(dict(drone=int(ident),task=e['task'],selected=e,
                reached_estimated_waypoint=any(v['event']=='opening_waypoint_reached' for v in related),
                ended=next((v for v in related if v['event']=='opening_end'),None),events=related))
    row['delivery_events']=[dict(drone=int(k),**e) for k,d in doc['drones'].items() for e in d['events'] if e['event'].startswith('return') or e['event']=='trail_invalid']
    data['chain']=doc


def plot_panel(ax,kind,row,data):
    panels.panel(ax,'mapping' if kind=='exploration' else kind,row,data)
    if kind=='exploration':
        pixels=np.ones((*data['truth'].shape,3))*.95;pixels[data['visible']]=[.57,.85,.73]
        pixels[data['truth'].astype(bool)]=[.1,.1,.1];ax.imshow(pixels);base.areas(ax,data['spec'])


def make_figures(rows,assets,out):
    panels.NAMES=NAMES
    for mapname in dict.fromkeys(r['map'] for r in rows):
        selected=[r for r in rows if r['map']==mapname]
        for kind,label in KINDS.items():
            fig,axes=base.plt.subplots(2,2,figsize=(18,15),squeeze=False)
            fig.subplots_adjust(left=.02,right=.98,top=.85,bottom=.08,wspace=.08,hspace=.45)
            for ax,r in zip(axes.ravel(),selected):plot_panel(ax,kind,r,assets[mapname,r['controller']])
            fig.suptitle(mapname+' · RL11 与三项独立候选：'+label,fontproperties=lines.FONT,fontsize=22,y=.98)
            fig.text(.5,.935,'左上 RL11，右上 D送达；左下 E入口，右下 R相对通道。每版一次完整红阶段。',ha='center',fontproperties=lines.FONT,fontsize=12)
            fig.text(.5,.035,NOTES[kind]+' 蓝队难度及完整总分未评测。',ha='center',fontproperties=lines.FONT,fontsize=11)
            base.save(fig,out/f'{mapname}_{kind}.png')
    for r in rows:
        d=assets[r['map'],r['controller']];prefix=f"{r['map']}_{r['controller']}"
        fig,axes=base.plt.subplots(1,3,figsize=(21,6))
        for ax,image,title in zip(axes,[d['truth'],d['pred'],d['merged']],['真实地图（离线）','实际提交地图','命中、漏画与误画']):
            ax.imshow(image,cmap='Greys',vmin=0,vmax=1);ax.axis('off');ax.set_title(title,fontproperties=lines.FONT)
        fig.suptitle(r['map']+' / '+NAMES[r['controller']],fontproperties=lines.FONT,fontsize=19)
        fig.tight_layout(rect=(0,.03,1,.93));base.save(fig,out/f'{prefix}_truth_submission.png')
        fig,axes=base.plt.subplots(2,3,figsize=(18,10));trace=d['trajectories'];w,h=d['w'],d['h']
        for slot,ax in enumerate(axes.ravel()):
            if slot>=trace.shape[1]:ax.axis('off');continue
            ax.imshow(d['truth'],cmap='Greys',vmin=0,vmax=1);base.areas(ax,d['spec'])
            active=trace['active'][:,slot];x=np.where(active,trace['x'][:,slot]+w/2,np.nan);y=np.where(active,h/2-trace['y'][:,slot],np.nan)
            jump=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40];x[jump]=np.nan;y[jump]=np.nan
            ax.plot(x,y,lw=.7,color=f'C{slot}');ax.set(xlim=(0,w),ylim=(h,0),aspect='equal');ax.axis('off')
            ax.set_title(f'录像槽位 {slot}',fontproperties=lines.FONT)
        fig.suptitle(r['map']+' / '+NAMES[r['controller']]+' · 各机完整实际轨迹',fontproperties=lines.FONT,fontsize=19)
        fig.tight_layout(rect=(0,.02,1,.94));base.save(fig,out/f'{prefix}_per_drone.png')
        fig,ax=base.plt.subplots(figsize=(11,8));image=np.ones((*d['truth'].shape,3))*.96;lab=d['region_labels']
        palette=base.plt.get_cmap('tab20')
        for reg in r['all_geometric_regions']:
            k=reg['id'];image[lab==k]=palette(k%20)[:3]
            rr,cc=np.nonzero(lab==k);i=len(rr)//2
            ax.text(cc[i],rr[i],str(k),fontsize=9,color='black',bbox=dict(facecolor='white',alpha=.7,pad=.6))
        image[d['truth'].astype(bool)]=[.1,.1,.1];ax.imshow(image);ax.axis('off')
        ax.set_title(r['map']+'：离线几何分区编号（不是已识别的房间或在线任务）',fontproperties=lines.FONT)
        fig.tight_layout();base.save(fig,out/f'{prefix}_regions.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);ph=lines.frozen(a.batch)
    rows=[];assets={};failed=[]
    for raw in base.read(a.batch/'results.json'):
        try:
            r,d=base.review_row(raw,ph);evidence.audit(r,d);lines.accuracy(r,d)
            extra=previous.assets(r);d['visible']=extra['visible'];chain_audit(r,d)
            rows.append(r);assets[r['map'],r['controller']]=d
        except Exception as exc:failed.append(dict(raw=raw,error=repr(exc)))
    comparisons=[]
    for r in rows:
        baseline=next((v for v in rows if v['map']==r['map'] and v['controller']=='rl11_lines'),None)
        if baseline is None or r is baseline:continue
        with np.load(Path(baseline['attempt'])/'mapper_legal_inputs.npz') as aa,np.load(Path(r['attempt'])/'mapper_legal_inputs.npz') as bb:
            equal=set(aa.files)==set(bb.files) and all(np.array_equal(aa[k],bb[k],equal_nan=True) for k in aa.files)
        comparisons.append(dict(map=r['map'],controller=r['controller'],legal_inputs_equal=equal,
            trajectories_equal=bool(np.array_equal(assets[r['map'],'rl11_lines']['trajectories'],assets[r['map'],r['controller']]['trajectories'])),
            map_delta=r['map_score']-baseline['map_score']))
    write(a.output/'summary.json',dict(rows=rows,failures=failed,input_comparisons=comparisons,plan_sha256=ph,blue_phases=0))
    make_figures(rows,assets,a.output)
    doc=['# RL14：RL11 独立送达、入口与相对导航完整实测','','各图各版一次，串行；蓝队难度和完整原始/有效总分未评测。',
        '相同种子不等于相同输入；输入逐数组比较记录在summary.json。回执只确认来源版本，不保证每个像素正确。',
        '几何分区按真值自由区8px栅格、48px净空核心与障碍约束扩展自动生成，供离线穷举；不等同真实房间或真实入口。细小分区可能来自栅格化，不直接判作应探索任务。',
        '雷达可见面积为每8步4px采样估计；真实墙面端点逐控制步记录，端点交集是观测下界。24步入口/路径快照可漏掉短任务，零记录不证明从未发现。',
        '|地图/版|有效|M|W|T|弹|实际步|墙钟秒|墙/机接触|平均健康|墙命中|受罚误画|',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in rows:doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['red_steps']}|{r['red_walltime']:.3f}|{r['contacts_wall']}/{r['contacts_drone']}|{r['health']:.2f}|{r['wall_hits']}|{r['charged_pixels']}|")
    for m in dict.fromkeys(r['map'] for r in rows):
        doc+=['',f'## {m} 完整对照','']
        for kind,label in KINDS.items():doc+=['',f'![{label}]({(a.output/f"{m}_{kind}.png").resolve().as_posix()})',NOTES[kind]]
    for r in rows:
        run=Path(r['attempt']);prefix=f"{r['map']}_{r['controller']}"
        doc+=['',f"## {r['map']} / {NAMES[r['controller']]} / seed {r['red_seed']}",'',
            f"实际条件 {r['actual_conditions']}；红有效 {r['red_valid']}；回调错误 {r['callback_errors']}；新增诊断错误 {r['chain_errors']}。",
            f"入口/送达计数：`{json.dumps(r['link_stats'],ensure_ascii=False)}`。",
            f"完整列出未到访几何分区：{r['unvisited_region_ids']}；零采样可见分区：{r['zero_sampled_visibility_region_ids']}；采样可见不足50%分区：{r['low_sampled_visibility_region_ids']}。这些是观测定义下的事实，不等于所有区域物理可达。",'']
        for suffix,label in [('truth_submission','真实/提交/差异'),('per_drone','各机轨迹'),('regions','全图分区编号')]:
            doc+=[f'![{label}]({(a.output/f"{prefix}_{suffix}.png").resolve().as_posix()})','']
        # Each run's original bomb and wall PNG are shown inline as required.
        for pattern,label in [('*_wall_destruction.png','本局墙体覆盖'),('*_exploration_diff.png','本局官方建图三联图')]:
            doc+=[f'![{label}]({next((run/"artifacts").glob(pattern)).resolve().as_posix()})','']
        # Per-run bomb plot instead of only a shared tiny overview.
        fig,ax=base.plt.subplots(figsize=(11,8));plot_panel(ax,'bombs',r,assets[r['map'],r['controller']]);fig.tight_layout()
        destination=a.output/f'{prefix}_bombs.png';base.save(fig,destination)
        doc+=[f'![本局实际落弹]({destination.resolve().as_posix()})','',
            '|来源|收到版本|落后步数|本机终图格|收到图格|本机终图未含于对应收到快照的格|','|---|---:|---:|---:|---:|---:|']
        for s in r['source_snapshots']:doc.append(f"|{s['source']}|{s['sequence']}|{s['age']}|{s['own_latest_cells']}|{s['received_cells']}|{s['own_final_absent_from_source_snapshot']}|")
        doc+=['','### 截图房间链路（仅Map09离线）','',
            '|房间|实际进入帧|可见候选记录|入口选中记录|粗目标样本|墙面端点命中|已观测且本机画中|已观测且提交画中|本机有但提交无|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,v in r['screenshot_rooms'].items():doc.append(f"|{name}|{v['actual_frames']}|{v['candidate_discoveries']}|{v['selected_entry_events']}|{v['coarse_goal_samples']}|{v['actual_observed_surface']}|{v['seen_own_final']}|{v['seen_submitted']}|{v['seen_own_missing_submission']}|")
        doc+=['','### 全部几何分区','',
            '|ID|像素包围框|自由区像素|到访帧|采样可见%|邻近真墙面|实际墙端点命中|提交墙命中|','|---|---|---:|---:|---:|---:|---:|---:|']
        for v in r['all_geometric_regions']:doc.append(f"|{v['id']}|{v['bbox']}|{v['free_pixels']}|{v['actual_frames']}|{100*v['sampled_visible_fraction']:.2f}|{v['wall_surface']}|{v['actual_hit_surface']}|{v['submitted_wall_hits']}|")
        doc+=['','### 每项入口任务与释放原因','',
            '发现候选来自当前雷达可达观察点与导航未知的交界，不是已证实物理门。位置、当前路径、导航误差及全部发现记录在本局link14_diagnostics.json；本表列出全部选中入口任务。',
            '|机/任务|选中步|导航目标|到达估计观察点|结束步/原因|','|---|---:|---|---|---|']
        for t in r['task_chains']:
            end=t['ended'] or {};doc.append(f"|{t['drone']}/{t['task']}|{t['selected']['step']}|{t['selected']['point']}|{t['reached_estimated_waypoint']}|{end.get('step')}/{end.get('reason','阶段结束时未释放')}|")
        doc+=['','```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{run.resolve()}'",'```']
    doc+=['','## 有效性与局限','',f'完整产物读取/审计失败：{failed}。全部原始attempt及失败记录保留；不把退出码0直接判为有效。',
        '同源回执只证明某版本曾到0号机，不能保证阶段结束时最新地图全送达。来源快照缺格也不代表缺的都是正确墙。',
        '轨迹变化组无法仅凭分数拆分定位、探索与送达因果。逐端点与最终本机图的交集不能区分从未写入和后来擦除；精确归因需要逐证据重放。',
        '本批没有蓝队阶段，没有蓝回放或完整比赛总分。没有扩到其他地图、参数搜索或100图。']
    with (a.output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(doc)+'\n')
    print(json.dumps(dict(rows=[dict(map=r['map'],controller=r['controller'],M=r['map_score'],valid=r['red_valid'],stats=r['link_stats']) for r in rows],failures=failed),ensure_ascii=True))


if __name__=='__main__':main()
