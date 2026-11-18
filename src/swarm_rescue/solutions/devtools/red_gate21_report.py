"""R21 historical execution/portal figures; reads R20 without running flights."""
import argparse
from collections import Counter
from pathlib import Path
import math

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from swarm_rescue.solutions.devtools.red_pair19_batch import read,write,digest
from swarm_rescue.solutions.devtools.red_pair19_results_review import only,truth_metric
from swarm_rescue.solutions.devtools.red_gate21_audit import world_gate,RELEASES,rotate,section
from swarm_rescue.solutions.devtools.red_recovery18_audit import load_npz,semantic_objects,legal_arguments
from swarm_rescue.solutions.my_drone_red_rn7 import LocalPassage
from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
from swarm_rescue.solutions.devtools.red_explore20_audit import navigation_graph


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+[
        '| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows]


def fmt(x):return '未记录' if x is None else f'{x:.3f}' if isinstance(x,float) else str(x)


def crossings(record,gate,w,h,owner,offset,events):
    center,tangent,normal=world_gate(gate,w,h);s=record[:,owner];p=np.c_[s['x'],s['y']]
    signed=(p-center)@normal;side=None;pending=None;result=[]
    radius=LocalPassage.TRACK_RADIUS
    for i in range(1,len(p)):
        if signed[i]*signed[i-1]<=0 and abs(signed[i]-signed[i-1])>1e-9:
            at=p[i-1]+(p[i]-p[i-1])*(-signed[i-1]/(signed[i]-signed[i-1]))
            pending=(i,float(abs((at-center)@tangent)))
        nextside=1 if signed[i]>=radius else -1 if signed[i]<=-radius else None
        if nextside is not None:
            if side is not None and nextside!=side and pending is not None and pending[1]<=gate['nominal_gap']/2-radius:
                tick=i-offset;before=[e for e in events if e['tick']<=tick and (e['reason']=='entered' or e['reason'] in RELEASES)]
                result.append(dict(owner=owner,gate=gate['id'],step=tick,plane_step=pending[0]-offset,
                    lateral_at_plane=pending[1],rn7_active=bool(before and before[-1]['reason']=='entered')))
            side=nextside;pending=None
    return result


def annotate_entries(doc,record,w,h,narrow):
    result=[]
    for row in doc['entries']:
        owner,tick=row['owner'],row['step'];gate=next(g for g in doc['gates'] if g['id']==row['gate'])
        center,tangent,normal=world_gate(gate,w,h);start=tick+doc['recording_tick_offset']
        events=narrow['drones'][str(owner)]['events'];end=next((e for e in events if e['tick']>=tick and e['reason'] in RELEASES),None)
        stop=min(len(record)-1,(end['tick'] if end else tick+171)+doc['recording_tick_offset'])
        q=record[:,owner];positions=np.c_[q['x'],q['y']];side=1 if (positions[start]-center)@normal>=0 else -1
        future=positions[start:min(len(record),start+201)];signed=(future-center)@normal*side
        result.append(dict(**row,end=end,actual_end_signed_distance=float((positions[stop]-center)@normal*side),
            actual_min_signed_next200=float(signed.min()),actual_max_signed_next200=float(signed.max()),
            task_path_length=float(np.linalg.norm(np.diff(positions[start:stop+1],axis=0),axis=1).sum()),
            post200_crossed_plane=bool((signed<0).any())))
    return result


def entry_figure(doc,row,out):
    run=Path(doc['source']);inputs=load_npz(run/'mapper_legal_inputs.npz');objects=semantic_objects(inputs,read(run/'line_diagnostics.json'))
    owner,tick=row['owner'],row['step'];args=legal_arguments(inputs['drone_'+str(owner)][tick-1],inputs['angles_'+str(owner)],objects[owner,tick])
    fine=LocalPassage(args[3],args[4],args[5]);_,truth,_=truth_metric(run);h,w=truth.shape
    gate=next(g for g in doc['gates'] if g['id']==row['gate']);c,t,n=world_gate(gate,w,h)
    tp=np.array(row['true_position']);th=row['true_heading'];edge=np.array([c-t*gate['nominal_gap']/2,c+t*gate['nominal_gap']/2])
    chosen=np.array(row['selected_path']);alternative=np.array(row['best_cross']['path'])
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    values=np.zeros(fine.shape,int);values[fine.free]=1;values[fine.clearance>=fine.PLAN_RADIUS]=2;values[fine.hits>0]=3
    axes[0].imshow(values,cmap=ListedColormap(['#343846','#b9d8e7','#d9f0c1','#ee6754']),vmin=0,vmax=3,extent=(-220,220,-220,220))
    for points,color,label in [(chosen,'#ed9322','Recovered original target'),(alternative,'#00ad69','Existing safe alternative')]:
        if len(points):axes[0].plot(points[:,0],points[:,1],'-o',color=color,label=label,markersize=3)
    eb=rotate(edge-tp,-th);axes[0].plot(eb[:,0],eb[:,1],'m--',label='Offline gate label')
    axes[0].plot(0,0,'kx');axes[0].set(xlim=(-150,150),ylim=(-150,150),title='Legal current scan + swept paths\nDark unknown / blue seen / green clearance / red hits')
    axes[0].legend(fontsize=7)
    for ax in axes[1:]:
        ax.imshow(truth,cmap='Greys',vmin=0,vmax=1,extent=(-w/2,w/2,-h/2,h/2))
        ax.plot(edge[:,0],edge[:,1],'m--',label='Offline gate cross-section')
        ax.set(xlim=(c[0]-160,c[0]+160),ylim=(c[1]-140,c[1]+180),aspect='equal')
    for path,color,label in [(chosen,'#ed9322','Original selected path'),(alternative,'#00ad69','Alternative; not flown')]:
        if len(path):
            p=tp+rotate(path,th);axes[1].plot(p[:,0],p[:,1],'-o',color=color,label=label,markersize=3)
    axes[1].set_title('Truth only for offline geometry\nAlternative is NOT a flight result');axes[1].legend(fontsize=7)
    with np.load(only(run/'artifacts/recordings','*.npz'),allow_pickle=False) as z:states=z['drone_states'][:,owner]
    begin=tick+doc['recording_tick_offset'];end=min(len(states),begin+201)
    axes[2].plot(states['x'][begin:end],states['y'][begin:end],color='#2468b4',label='Actual next 200 ticks')
    axes[2].scatter(*tp,c='k',marker='x');axes[2].set_title('Historical execution\nTask ended: '+str(row['end']))
    axes[2].legend(fontsize=7)
    for ax in axes:ax.set_xlabel('px');ax.set_ylabel('px')
    fig.suptitle(f"R20 history | {doc['map']} r{doc['round']} seed {doc['seed']} | RL17-R d{owner} tick {tick}")
    path=out/f"{doc['map']}_r{doc['round']}_d{owner}_t{tick}_entry.png";fig.savefig(path,dpi=150);plt.close(fig);return path


def coarse_figure(doc,row,out):
    run=Path(doc['source']);trace=read(run/'explore20_diagnostics.json');plan=trace['plans'][row['plan_index']]
    _,truth,_=truth_metric(run);h,w=truth.shape;grid=_M19_SurveyGrid((w,h),lambda *a,**k:None)
    with np.load(run/'explore20_diagnostics.npz',allow_pickle=False) as z:
        for key in ('free','obstacle','visits'):setattr(grid,key,z[plan['state'][key]['array']])
    raw=grid.obstacle>=2;inflated=cv2.dilate(raw.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))).astype(bool)
    _,_,reachable=navigation_graph(grid,np.array(row['position']));gate=next(g for g in doc['gates'] if g['id']==row['gate'])
    c,t,n=world_gate(gate,w,h);edge=np.array([c-t*gate['nominal_gap']/2,c+t*gate['nominal_gap']/2])
    fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
    for ax,blocked,title in zip(axes,[raw,inflated,inflated],['Raw obstacle evidence >= 2','Original cross-cell inflation','Start connected component (green)']):
        img=np.zeros(grid.shape,int);img[grid.free]=1;img[blocked]=2
        if ax is axes[2]:img[reachable]=3
        ax.imshow(img,extent=(-w/2,-w/2+grid.shape[1]*20,h/2-grid.shape[0]*20,h/2),
            cmap=ListedColormap(['#656575','#f7f7f7','#d74343','#68bd81']),vmin=0,vmax=3,interpolation='nearest')
        ax.plot(edge[:,0],edge[:,1],'m-',linewidth=3);ax.plot(*row['position'],'bx',markersize=9,label='Navigation pose')
        ax.plot(*row['true_position'],'k+',markersize=9,label='Truth (offline)')
        ax.set(xlim=(c[0]-110,c[0]+110),ylim=(c[1]-110,c[1]+130),title=title,aspect='equal');ax.legend(fontsize=7)
    fig.suptitle(f"R20 history | {doc['map']} r{doc['round']} seed {doc['seed']} RL17-R d{row['owner']} t{row['step']}\n"
        'Magenta = whole labelled aperture; gray unknown. Safe local alternative exists, but initial cooldown is still active.')
    path=out/'map72_r1_t105_coarse.png';fig.savefig(path,dpi=150);plt.close(fig);return path


def return_figure(doc,out):
    run=Path(doc['source']);_,truth,_=truth_metric(run);h,w=truth.shape;gate=doc['gates'][0]
    c,t,n=world_gate(gate,w,h);edge=np.array([c-t*gate['nominal_gap']/2,c+t*gate['nominal_gap']/2])
    with np.load(only(run/'artifacts/recordings','*.npz'),allow_pickle=False) as z:r=z['drone_states'][:,1]
    trace=read(run/'explore20_diagnostics.json')['drones']['1'];fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for k,(start,stop,entered,completed) in enumerate([(2255,2360,2265,2315),(3325,3440,3336,3387)]):
        ticks=np.arange(start,stop+1);ix=ticks+doc['recording_tick_offset'];p=np.c_[r['x'][ix],r['y'][ix]]
        axes[k,0].imshow(truth,cmap='Greys',extent=(-w/2,w/2,-h/2,h/2),vmin=0,vmax=1)
        axes[k,0].plot(edge[:,0],edge[:,1],'m--',label='Gate plane')
        axes[k,0].plot(p[:,0],p[:,1],color='#2471ba',label='Actual out-and-back path')
        axes[k,0].scatter(*p[0],c='k',marker='x',label='Start')
        axes[k,0].set(xlim=(c[0]-100,c[0]+100),ylim=(c[1]-100,c[1]+160),aspect='equal',title=f'Actual trajectory t{start}..{stop}')
        axes[k,0].legend(fontsize=8)
        ax=axes[k,1];ax.plot(ticks,(p-c)@n,label='Actual center signed distance',color='#2471ba')
        samples=[s for s in trace if start<=s['step']<=stop and s['goal'] is not None]
        ax.plot([s['step'] for s in samples],[(np.array(s['goal'])-c)@n for s in samples],'o--',color='#df8a1e',label='Recorded parent goal (24-tick samples)')
        ax.axhline(0,color='m',ls='--');ax.axhspan(-18.5,18.5,color='gray',alpha=.15)
        ax.axvspan(entered,completed,color='#69bb87',alpha=.2,label='RN7 active: entered to completed')
        ax.set(title='Parent goal remains on original side',xlabel='Controller tick',ylabel='Signed distance to gate / px')
        ax.legend(fontsize=8)
    fig.suptitle('R20 history | Map72 r1 seed2026111701 | R20-B d1\nCrossing alone did not persist: no bomb target / gathering in surrounding recorded samples')
    path=out/'map72_r1_r20_d1_cross_and_return.png';fig.savefig(path,dpi=150);plt.close(fig);return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--sections',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    docs=[read(f) for f in sorted(a.sections.glob('*/analysis.json'))];details=read(a.review/'run_details.json');summ=read(a.review/'atlas/summary.json')
    figures=[];all_cross=[];all_entries=[];rollup=[];drone_rows=[];checks=[]
    for d in docs:
        run=Path(d['source']);_,truth,_=truth_metric(run);h,w=truth.shape;narrow=read(run/'narrow_diagnostics.json')
        with np.load(only(run/'artifacts/recordings','*.npz'),allow_pickle=False) as z:record=z['drone_states']
        detail=next(v for v in details if (v['map'],v['round'],v['controller'])==(d['map'],d['round'],d['controller']))
        cross=[]
        for owner in range(5):
            events=narrow['drones'][str(owner)]['events']
            for gate in d['gates']:cross.extend(crossings(record,gate,w,h,owner,d['recording_tick_offset'],events))
        entries=annotate_entries(d,record,w,h,narrow)
        for x in cross:all_cross.append(dict(map=d['map'],round=d['round'],controller=d['controller'],**x))
        for x in entries:all_entries.append(dict(map=d['map'],round=d['round'],controller=d['controller'],**x))
        safe=[r for r in d['rows'] if r['physical_disc_safe_cross_candidates']]
        chosen=[r for r in entries if r['physical_disc_safe_cross_candidates'] and r['selected_path_reconstructed']]
        known_miss=[r for r in chosen if not r['chosen_endpoint_beyond_gate']]
        rollup.append(dict(map=d['map'],round=d['round'],controller=d['controller'],cases=len(d['rows']),
            safe=len(safe),inflated_full=sum(r['planes'][1]['coarse_inflated_blocked']==r['planes'][1]['samples'] for r in d['rows']),
            raw_reachable_but_inflated_zero=sum(r['planes'][1]['coarse_raw_reachable']>0 and r['planes'][1]['coarse_reachable']==0 for r in d['rows']),
            active=sum(r['task_state']['active_by_events'] for r in safe),cooldown=sum(r['task_state']['certainly_in_cooldown'] for r in safe),
            budget_exhausted=sum(r['episodes_already_entered']>=8 for r in safe) if d['controller']=='rl17_repair' else None,
            entries=len(entries),matched=sum(r['selected_path_reconstructed'] for r in entries),
            safe_entry_matches=len(chosen),selected_not_beyond=len(known_miss),actual_crossings=len(cross)))
        checks.append(dict(map=d['map'],round=d['round'],controller=d['controller'],pose_visits=d['checks'],
            record_offset=d['recording_tick_offset'],record_error=d['recording_max_error'],source_hash=d['source_hash'],
            rn7_entered_events={k:dict(events=sum(e['reason']=='entered' for e in v['events']),
                aggregate=v['stats'].get('entered',0)) for k,v in narrow['drones'].items()}))
        for dr in detail['drones']:
            owner=dr['owner'];events=[r for r in cross if r['owner']==owner];rr=[r for r in d['rows'] if r['owner']==owner]
            drone_rows.append([d['map']+'/r'+str(d['round']),d['controller'],owner,len(rr),
                sum(r['physical_disc_safe_cross_candidates']>0 for r in rr),len(events),len({r['gate'] for r in events}),
                fmt(dr['sampled_visible_free_percent']),fmt(dr['navigation_error_p95']),
                str({k:v for k,v in dr['rn7'].items() if k in RELEASES}),
                dr['local']['hits'],dr['received']['hits'],dr['local']['charged'],dr['received']['charged'],
                dr['source_version'],dr['source_age'],dr['placement_engine_steps']])
        if d['map']=='yh_map_fs_72' and d['controller']=='rl17_repair':
            targets=[(0,866)] if d['round']==1 else [(4,341),(3,599)]
            for owner,tick in targets:
                r=next(x for x in entries if x['owner']==owner and x['step']==tick and x['gate']=='G01')
                figures.append(entry_figure(d,r,out))
            if d['round']==1:figures.append(coarse_figure(d,next(x for x in d['rows'] if x['owner']==4 and x['step']==105),out))
        if d['map']=='yh_map_fs_72' and d['controller']=='r20_budget' and d['round']==1:figures.append(return_figure(d,out))
    write(out/'execution_crossings.json',all_cross);write(out/'actual_entries.json',all_entries);write(out/'rollup.json',rollup);write(out/'replay_checks.json',checks)
    lines=['# R21 入口链路历史数据诊断附录','',
        '仅离线复算 R20 原局，没有重新运行完整飞行。离线安全候选不代表实际执行成功或最终提分。',
        '同一入口横截面按1px采样，粗栅格20px；样点不独立。安全候选计数是存在性检查，不是所有端点的穷举数量。',
        '跨入口：历史真实中心从入口两侧±18.5px通过，过平面处留18.5px横向余量；这是保守几何事件定义，不替代物理接触记录。',
        '已有碰撞记录只能可靠给出团队总数，不能按原局逐机归因。已接收来源分项不是各机单独的团队得分。','']
    lines+=table(['地图/轮','版本','入口快照','全膨胀阻挡','原始可达→膨胀不可达','安全跨口快照','其中正在任务/冷却/已8次','实际入口任务快照/可还原','可还原且有安全跨口/未选跨侧','实际跨口'],
        [[r['map']+'/r'+str(r['round']),r['controller'],r['cases'],r['inflated_full'],r['raw_reachable_but_inflated_zero'],r['safe'],
          f"{r['active']}/{r['cooldown']}/{r['budget_exhausted']}",f"{r['entries']}/{r['matched']}",
          f"{r['safe_entry_matches']}/{r['selected_not_beyond']}",r['actual_crossings']] for r in rollup])
    lines+=['','状态计数可以重叠；R20-B重新补充预算，不能用累计8次判断其剩余预算。未选某个入口的另一侧不一定是失误：可能在去另一个方向、另一个入口或处于任务抢占。','',
        '## 逐机：观察、执行、实际送达与放弹','']
    lines+=table(['地图/轮','版本','机','入口快照','安全跨口','实际跨口/次','不同入口','采样覆盖%','定位P95px','RN7退出原因',
        '本机真墙','已收到真墙','本机误画','已收到误画','收到版本','年龄/步','两次放弹/步'],drone_rows)
    lines+=['','## 原局完整分项（RL11继续作为历史对照）','']
    lines+=table(['地图/轮','版本','种子','实际特殊区','M','命中/漏画/受罚误画','W','T','弹','步/引擎秒','覆盖%','墙/机接触','有效/异常'],
        [[r['map']+'/r'+str(r['round']),r['controller'],r['seed'],str(r['actual_zones']),fmt(r['map_score']),
          f"{r['wall_hits']}/{r['wall_missed']}/{r['charged_pixels']}",fmt(r['wall_score']),fmt(r['time_score']),r['bombs'],
          f"{r['steps']}/{fmt(r['wall_seconds'])}",fmt(r['sampled_lidar_free_percent']),str(r['collision_contacts']),f"{r['valid']}/{r['reasons']}"] for r in summ['rows']])
    lines+=['','所有红阶段的蓝队难度分、完整比赛总分均未评测。','', '## R20-B 相对 RL17-R 全部分项差值，负值及所有失败均保留','']
    lines+=table(['地图/轮','M','命中','漏画','误画','W','T','弹','覆盖百分点','耗时秒','卡住步'],
        [[r['map']+'/r'+str(r['round'])]+[fmt(r['raw_delta'].get(k)) for k in ('map_score','wall_hits','wall_missed','charged_pixels','wall_score','time_score','bombs','sampled_lidar_free_percent','wall_seconds','stalled_steps')]
         for r in summ['pairs'] if r['candidate']=='r20_budget'])
    lines+=['','## 离线入口图（历史传感器、历史执行；绿色备选从未飞行）','']
    for f in figures:lines+=['!['+f.stem+']('+f.as_posix()+')','']
    lines+=['## 历史 R20 图册','', '每图上排r1 seed2026111701，下排r2 seed2026111702；左RL11、中RL17-R、右R20-B。图例见图内。']
    for name in ('72','37','24','55'):
        for kind in ('mapping','exploration','trajectories','bombs'):
            f=(a.review/'contact_sheets'/f'yh_map_fs_{name}_{kind}_r1_r2.png').resolve()
            lines+=['',f'历史来源：R20完整批次 / Map{name} / {kind}。','!['+f.stem+']('+f.as_posix()+')']
    (out/'历史诊断附录.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'figure_manifest.json',{f.name:digest(f) for f in figures})
    print('Historical analysis completed; no flights.',out,flush=True)


if __name__=='__main__':main()
