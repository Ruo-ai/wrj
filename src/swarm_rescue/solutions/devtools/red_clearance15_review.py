"""Offline RC15 four-run audit/atlas. No simulation or online map access."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from matplotlib.patches import Rectangle
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_rx10_review as evidence
from swarm_rescue.solutions.devtools import red_lines11_review as lines
from swarm_rescue.solutions.devtools import red_consistent12_review as panels
from swarm_rescue.solutions.my_drone_red_rn7 import rotate

NAMES={'rl11_lines':'RL11 原版','rc15_clearance':'RC15 小净空实验'}
KINDS={'exploration':'探索观察范围','mapping':'完整提交建图','trajectories':'完整实际轨迹','bombs':'实际炸弹位置','walls':'官方墙体覆盖'}
NOTES={'exploration':'绿：每8步/4px雷达可见区采样估计，含射线间插值；白：该采样未覆盖。',
       'mapping':'绿：真墙面命中；蓝：漏墙；深红：受罚误画；黄：免罚误画。',
       'trajectories':'线色是录像槽位；灰：无GPS/罗盘；黄：无通信。轨迹不是雷达覆盖。',
       'bombs':'编号按实际产物顺序；圆为爆炸范围，圆重叠不等于墙面收益重复。',
       'walls':'绿：覆盖内墙；蓝：覆盖边界墙；灰：未覆盖墙；红：爆炸范围。'}
# Pixel gates read OFFLINE from Map08 native truth/replay. Never imported online.
GATES={'G1中间入口':dict(axis=0,at=956.,lo=754.,hi=892.,width=139),
       'G2下方入口':dict(axis=1,at=1024.,lo=1301.,hi=1356.,width=56),
       'G3最右下入口':dict(axis=1,at=1059.,lo=1608.,hi=1653.,width=46)}

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def crossings(trace,w,h,gate):
    """Actual center crossings plus conservative full-body 17px confirmations."""
    coordinates=np.stack((trace['x']+w/2,h/2-trace['y']),axis=-1);axis=gate['axis'];other=1-axis
    result=[];near_frames=0
    for slot in range(trace.shape[1]):
        p=coordinates[:,slot];valid=trace['active'][:,slot]&np.isfinite(p).all(1)
        near=valid&(np.abs(p[:,axis]-gate['at'])<60)&(p[:,other]>=gate['lo']-40)&(p[:,other]<=gate['hi']+40)
        near_frames+=int(near.sum())
        for t in range(1,len(p)):
            if not(valid[t-1] and valid[t]) or np.linalg.norm(p[t]-p[t-1])>40:continue
            a=p[t-1,axis]-gate['at'];b=p[t,axis]-gate['at']
            if a*b>=0 or a==b:continue
            q=p[t-1]+(-a/(b-a))*(p[t]-p[t-1])
            if not gate['lo']<=q[other]<=gate['hi']:continue
            sign=1 if b>a else -1
            before=p[max(0,t-60):t];after=p[t:min(len(p),t+60)]
            confirmed=bool(np.any(sign*(before[:,axis]-gate['at'])<=-17) and np.any(sign*(after[:,axis]-gate['at'])>=17))
            result.append(dict(slot=slot,frame=t,pixel=q.tolist(),direction=sign,full_body_side_change=confirmed))
    return dict(raster_open_width=gate['width'],near_drone_frames=near_frames,center_crossings=result,
                complete_side_changes=sum(x['full_body_side_change'] for x in result))

def audit(row,data):
    run=Path(row['attempt']);doc=base.read(run/'clearance15_diagnostics.json')
    row['clearance_diagnostic_errors']=doc['errors'];row['clearance_diagnostic_seconds']=doc['seconds']
    row['clearance_stats']=dict(sum((Counter(x['stats']) for x in doc['drones'].values()),Counter()))
    with np.load(run/'local13_diagnostics.npz') as f:
        visible=np.logical_or.reduce([f[k].astype(bool) for k in f.files])
    data['visible']=np.repeat(np.repeat(visible,4,0),4,1)[:data['h'],:data['w']]
    with np.load(run/'clearance15_diagnostics.npz') as f:hits={k[4:]:f[k].astype(bool) for k in f.files}
    with np.load(run/'exploration_stages.npz') as f:
        local={k.removeprefix('local_final_pixels_'):f[k].astype(bool) for k in f.files if k.startswith('local_final_pixels_')}
    wall,_=base.MapExplorationScorer().split_regions(data['truth']);wall=wall.astype(bool)
    seen=np.logical_or.reduce(list(hits.values()))&wall;own=np.zeros_like(wall)
    for k,v in hits.items():own|=v&wall&local[k]
    data['dense_seen']=seen;data['dense_own']=own
    free=~data['truth'].astype(bool)
    row['sampled_visible_free_pixels']=int((free&data['visible']).sum())
    row['sampled_visible_free_fraction']=float((free&data['visible']).sum()/free.sum())
    row['dense_chain']=dict(observed_true_surface=int(seen.sum()),observed_and_own_final=int(own.sum()),
        observed_and_submitted=int((seen&data['pred']).sum()),own_observed_missing_submission=int((own&~data['pred']).sum()))
    if row['map']=='Map08':
        row['gates']={name:crossings(data['trajectories'],data['w'],data['h'],gate) for name,gate in GATES.items()}
        # Lower passage interior beyond G2; measured body-centre visits vs rays.
        t=data['trajectories'];x=t['x']+data['w']/2;y=data['h']/2-t['y']
        box=np.zeros_like(wall);box[1028:1116,960:1357]=True
        roi=np.zeros_like(wall);roi[1008:1116,950:1364]=True
        row['lower_lane']=dict(actual_frames=int((t['active']&(x>=960)&(x<1357)&(y>=1028)&(y<1116)).sum()),
            sampled_visible_free_pixels=int((box&free&data['visible']).sum()),
            observed_wall_pixels=int((roi&seen).sum()),own_wall_pixels=int((roi&own).sum()),
            submitted_observed_wall_pixels=int((roi&seen&data['pred']).sum()))
        row['gate_tasks']={}
        for name,g in GATES.items():
            found=[]
            for ident,d in doc['drones'].items():
                for event in d['audit']:
                    if event['reason']!='fine_goal_selected' or event.get('goal_body') is None:continue
                    p=np.array(event['position'])+rotate(event['goal_body'],event['heading'])
                    px=np.array([p[0]+data['w']/2,data['h']/2-p[1]])
                    if abs(px[g['axis']]-g['at'])<100 and g['lo']-65<=px[1-g['axis']]<=g['hi']+65:
                        found.append(dict(drone=ident,**event,estimated_goal_pixel=px.tolist()))
            row['gate_tasks'][name]=found
    data['narrow_doc']=doc

def marks(ax):
    for name,g in GATES.items():
        if g['axis']==0:xs=[g['at']]*2;ys=[g['lo'],g['hi']]
        else:xs=[g['lo'],g['hi']];ys=[g['at']]*2
        ax.plot(xs,ys,color='magenta',lw=2.2)
        ax.text(np.mean(xs)+7,np.mean(ys)-10,name[:2],fontsize=10,color='darkmagenta',bbox=dict(facecolor='white',alpha=.8,pad=1))

def panel(ax,kind,row,data):
    panels.panel(ax,'mapping' if kind=='exploration' else kind,row,data)
    if kind=='exploration':
        image=np.ones((*data['truth'].shape,3))*.96;image[data['visible']]=[.56,.84,.72]
        image[data['truth'].astype(bool)]=[.1,.1,.1];ax.imshow(image);base.areas(ax,data['spec'])
    if row['map']=='Map08' and kind in ('exploration','trajectories'):marks(ax)

def pictures(rows,assets,out):
    panels.NAMES=NAMES
    for kind,label in KINDS.items():
        fig,axes=base.plt.subplots(2,2,figsize=(18,15),squeeze=False)
        fig.subplots_adjust(left=.02,right=.98,top=.88,bottom=.07,hspace=.45,wspace=.07)
        for y,m in enumerate(('Map08','Map09')):
            for x,c in enumerate(NAMES):
                row=next(r for r in rows if r['map']==m and r['controller']==c)
                panel(axes[y,x],kind,row,assets[m,c])
        fig.suptitle('RL11 与 RC15 · '+label,fontproperties=lines.FONT,fontsize=24,y=.98)
        fig.text(.5,.945,'左：RL11；右：RC15。上：Map08；下：Map09。各一次完整红阶段，蓝队与完整总分未评测。',ha='center',fontproperties=lines.FONT,fontsize=13)
        fig.text(.5,.025,NOTES[kind],ha='center',fontproperties=lines.FONT,fontsize=12)
        base.save(fig,out/f'comparison_{kind}.png')
    fig,axes=base.plt.subplots(2,2,figsize=(17,13))
    for col,c in enumerate(NAMES):
        r=next(r for r in rows if r['map']=='Map08' and r['controller']==c);d=assets['Map08',c]
        for y,kind in enumerate(('trajectories','mapping')):
            panel(axes[y,col],kind,r,d);marks(axes[y,col])
            axes[y,col].set(xlim=(880,1660),ylim=(1122,690))
    fig.suptitle('Map08 中间入口 G1 / 下方入口 G2 / 最右下入口 G3',fontproperties=lines.FONT,fontsize=20)
    fig.tight_layout(rect=(0,.06,1,.94));fig.text(.5,.025,'G1约139px；G2约56px；G3约46px（真值栅格截面，仅离线）。标线不是策略目标。',ha='center',fontproperties=lines.FONT)
    base.save(fig,out/'Map08_gate_focus.png')
    for r in rows:
        d=assets[r['map'],r['controller']];prefix=f"{r['map']}_{r['controller']}";w,h=d['w'],d['h']
        fig,axes=base.plt.subplots(1,4,figsize=(24,6))
        for ax,img,title in zip(axes,[d['truth'],d['dense_seen'],d['pred'],d['merged']],['真实墙体','实际雷达墙面端点（下界）','最终提交图','官方误画/漏画差异']):
            ax.imshow(img,cmap='Greys',vmin=0,vmax=1) if img.ndim==2 else ax.imshow(img)
            ax.axis('off');ax.set_title(title,fontproperties=lines.FONT)
        fig.suptitle(f"{r['map']} / {NAMES[r['controller']]} / seed {r['red_seed']} / r1 / red_valid={r['red_valid']}",fontproperties=lines.FONT)
        fig.tight_layout(rect=(0,0,1,.94));base.save(fig,out/f'{prefix}_truth_detected_submission.png')
        fig,axes=base.plt.subplots(2,3,figsize=(18,11));trace=d['trajectories']
        for slot,ax in enumerate(axes.ravel()):
            if slot>=trace.shape[1]:ax.axis('off');continue
            ax.imshow(d['truth'],cmap='Greys',vmin=0,vmax=1);base.areas(ax,d['spec'])
            active=trace['active'][:,slot];x=np.where(active,trace['x'][:,slot]+w/2,np.nan);y=np.where(active,h/2-trace['y'][:,slot],np.nan)
            jumps=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40];x[jumps]=np.nan;y[jumps]=np.nan
            ax.plot(x,y,lw=.7,color=base.plt.get_cmap('tab10')(slot));ax.axis('off')
            if r['map']=='Map08':marks(ax)
            ax.set_title(f'录像槽位 {slot}',fontproperties=lines.FONT)
        fig.suptitle(f"{r['map']} / {NAMES[r['controller']]} · 各机完整轨迹",fontproperties=lines.FONT,fontsize=20)
        fig.tight_layout(rect=(0,0,1,.94));base.save(fig,out/f'{prefix}_per_drone.png')
        for kind in ('exploration','bombs','walls'):
            fig,ax=base.plt.subplots(figsize=(11,8));panel(ax,kind,r,d);fig.tight_layout()
            base.save(fig,out/f'{prefix}_{kind}.png')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);planhash=lines.frozen(a.batch);rows=[];assets={};failures=[]
    for raw in base.read(a.batch/'results.json'):
        try:
            row,data=base.review_row(raw,planhash);evidence.audit(row,data);lines.accuracy(row,data);audit(row,data)
            rows.append(row);assets[row['map'],row['controller']]=data
        except Exception as exc:failures.append(dict(raw=raw,error=repr(exc)))
    deltas=[]
    for m in ('Map08','Map09'):
        old=next((r for r in rows if r['map']==m and r['controller']=='rl11_lines'),None)
        new=next((r for r in rows if r['map']==m and r['controller']=='rc15_clearance'),None)
        if old and new:
            assert old['red_seed']==new['red_seed']
            deltas.append(dict(map=m,**{k:new[k]-old[k] for k in ('map_score','wall_score','time_score','wall_hits','charged_pixels','red_walltime','contacts_wall','contacts_drone','health','sampled_visible_free_pixels')}))
    write(a.output/'summary.json',dict(rows=rows,deltas=deltas,failures=failures,blue_evaluated=False,plan_sha256=planhash))
    if len(rows)!=4:raise RuntimeError(f'Incomplete review, failures preserved: {failures}')
    pictures(rows,assets,a.output)
    doc=['# RC15：RL11 最小净空实验完整四局图册','',
        'Map08/09 × RL11/RC15 ×1轮，串行4红0蓝。真值仅在本离线图册使用。完整比赛原始/有效总分与蓝队难度均未评测。','',
        '|地图/版本|红有效|M|W|T|步数|墙钟秒|弹数|平均健康|墙/机接触|命中|受罚误画|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in rows:doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['red_steps']}|{r['red_walltime']:.3f}|{r['placed_count']}|{r['health']}|{r['contacts_wall']}/{r['contacts_drone']}|{r['wall_hits']}|{r['charged_pixels']}|")
    for kind,label in KINDS.items():doc+=['',f'![{label}]({(a.output/f"comparison_{kind}.png").resolve().as_posix()})','',NOTES[kind]]
    doc+=['',f'![Map08入口]({(a.output/"Map08_gate_focus.png").resolve().as_posix()})','',
        '入口宽度为真实墙栅格截面估计；中心越线与机身通过分开。完整侧向通过按前后60帧内达到距分界线±17px确认，是保守离线定义；非无接触证明。','',
        '|版本/门|附近机帧|中心越线|完整侧向通过|近门细目标选择|','|---|---:|---:|---:|---:|']
    for r in rows:
        for k,g in r.get('gates',{}).items():doc.append(f"|{NAMES[r['controller']]}/{k}|{g['near_drone_frames']}|{len(g['center_crossings'])}|{g['complete_side_changes']}|{len(r['gate_tasks'][k])}|")
    doc+=['','未记录到选择不能称入口未识别。RL11没有RC15细任务事件；其粗目标/路线每16步采样，可能漏掉短暂决定。RC15保留目标、估计位姿、完整局部路径、余量、规划/释放事件；没有伪造在线门宽或绝对定位置信度。',
        '实际墙面端点是逐步记录下界；传感器看到附近墙不等于完整看见墙面。多机本机图/提交版本证据见summary.json；不能从轨迹推断精确擦除因果。']
    for r in rows:
        prefix=f"{r['map']}_{r['controller']}";run=Path(r['attempt'])
        doc+=['',f"## {r['map']} / {NAMES[r['controller']]} / seed {r['red_seed']} / r1",'',
            f"有效={r['red_valid']}；回调={r['callback_errors']}；控制器崩溃={r['controller_crashed']}；超时={r['walltime_limited']}。"]
        for suffix in ('exploration','per_drone','truth_detected_submission','walls','bombs'):
            doc+=['',f'![{suffix}]({(a.output/f"{prefix}_{suffix}.png").resolve().as_posix()})']
        doc+=['','```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{run.resolve()}'",'```','',
            '本轮蓝阶段未运行，无对应蓝回放。', '', '局部事件统计：', '```json',json.dumps(r['clearance_stats'],ensure_ascii=False,indent=2),'```']
    (a.output/'report.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
    print(json.dumps(dict(valid=sum(r['red_valid'] for r in rows),deltas=deltas,failures=failures),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
