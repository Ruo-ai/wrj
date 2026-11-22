"""Read-only comparison and full figure delivery for the gated Delivery39 run."""
import importlib.util
import json
import os
from pathlib import Path
os.environ['PYGLET_SHADOW_WINDOW']='false';os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
BATCH=ROOT/'output/sf13/delivery39r2_run_20261122';OUT=BATCH/'review'
OLD=ROOT/'output/sf13/continuous38_random3_20261122'
REPORT=ROOT/'红队_Delivery39R2_通信交付与未知探索对照_20261122.md'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
helper=module('delivery39_report_helpers','output/sf13/continuous38_report_20261122.py')
atlas=helper.atlas;navplot=helper.navplot;read=atlas.read

def save(fig,path):
    fig.savefig(path,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    driver=module('delivery39_final_validate','src/swarm_rescue/solutions/devtools/delivery39r2_batch.py')
    plan=driver.validate(BATCH);done=read(BATCH/'completion.json');assert done['complete']
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();checks=[];all_records=[];parts=[];attempts=[]
    def check(name,good):
        checks.append(dict(check=name,passed=bool(good)))
        if not good:raise AssertionError(name)
    oldpairs=read(OLD/'pairs.json');newpairs=read(BATCH/'pairs.json')
    check('conditional schedule obeyed',len(newpairs)==(2 if done['gate_first_passed'] else 1))
    check('single attempt per scheduled run',len(list((BATCH/'runs').glob('**/attempt_*')))==len(newpairs))
    for p in newpairs:
        pairs=[next(x for x in oldpairs if x['seed']==p['seed']),p]
        names=['Continuous38 历史基线','Delivery39R2 本次新执行']
        loaded=[atlas.inspect(x,BATCH) for x in pairs];nav=[navplot.load(x) for x in pairs]
        diags=[read(Path(x['red_run'])/'dualmap32_diagnostics.json') for x in pairs]
        common=min(v['_f27_stats']['cutoff_elapsed'] for d in diags for v in d['drones'].values())
        folder=OUT/f"seed_{p['seed']}";folder.mkdir();records=[];curves=[]
        for j,(pair,(m,d),n,diag) in enumerate(zip(pairs,loaded,nav,diags)):
            run=Path(pair['red_run']);attempts.append(run);m['version']=names[j]
            check(f"{p['seed']} {j} native submission audit",not m['audit_issues'])
            integrity=read(run/'integrity.json')
            check(f"{p['seed']} {j} saved evidence hashes",all(atlas.sha(run/f)==v for f,v in integrity['files'].items()))
            resource=read(run/'resources.json');req=read(run/'request.json')
            check(f"{p['seed']} {j} official zones and resource limits",req['zones']==['NO_COM_ZONE','NO_GPS_ZONE'] and resource['cpu_count']==1 and resource['memory_limit_bytes']==4000000000)
            receipt=n['receipt'];snap=np.load(run/'dualmap32_snapshots.npz',allow_pickle=False)
            sources=[]
            for i in range(5):
                own=diag['snapshot_payloads'][str(i)][str(i)];got=diag['snapshot_payloads']['0'][str(i)]
                actual=snap[got['array']].tobytes();sealed=snap[own['array']].tobytes()
                fresh=(receipt['received_versions'][str(i)]==own['sequence']
                    and receipt['received_digests'][str(i)]==own['payload_sha256'] and actual==sealed)
                sources.append(dict(drone=i,received=receipt['received_versions'][str(i)],final=own['sequence'],
                    final_epoch=own['final_epoch'],latest=bool(fresh),own_sealed=own['own_sealed'],
                    received_digest=receipt['received_digests'][str(i)],final_digest=own['payload_sha256']))
            covered,curve=helper.h.coverage(d,common);curves.append(curve)
            samples,alignment=helper.observations(pair,d,common)
            outage=np.concatenate([a[a[:,1]>0] for a in samples.values()])
            native=read(run/'result.json')
            xy=np.stack([d['red']['x']+850,550-d['red']['y']],axis=-1)
            far=((xy[:common-1,:,0]<400)&(xy[:common-1,:,1]<320))
            zone=(xy[:,:,0]>=173)&(xy[:,:,0]<=962)&(xy[:,:,1]>=10)&(xy[:,:,1]<=569)
            comm=[]
            for i in range(5):
                started=diag['drones'][str(i)]['_f27_stats'].get('gather_started_control')
                comm.append(dict(drone=i,return_control=started,
                    final_xy=xy[-1,i].astype(float).tolist(),
                    post_return_center_no_com_frames=None if started is None else int(zone[max(0,started-1):,i].sum()),
                    post_return_frames=None if started is None else len(xy)-started+1))
            rec=dict(version=names[j],seed=p['seed'],historical=j==0,attempt=str(run),valid_red=m['valid_red'],
                map_score=m['map_score'],wall_score=m['wall_score'],time_score=m['time_score'],blue_score=None,total_score=None,
                bombs=m['placed_count'],steps=m['red_steps'],seconds=m['red_seconds'],
                wall_hits=m['error_metric']['wall_hits'],wall_misses=m['error_metric']['wall_misses'],penalty=m['error_metric']['penalty'],
                charged_pixels=native['phase1']['map_breakdown']['charged_pixels'],sources=sources,
                latest=sum(x['latest'] for x in sources),acknowledgements=receipt['acknowledged_sources'],
                source_maps_complete=receipt['source_maps_complete'],final_complete=receipt['final_complete'],
                submit_reason=receipt['reason'],submit_control=receipt['submit_control_step'],freeze_elapsed=receipt['epoch_elapsed'],
                coverage=covered,common_cutoff=common,northwest_far_drone_frames=far.sum(0).astype(int).tolist(),
                outage=helper.stats(outage),boundary=m['boundary'],return_delivery=comm,
                actual_submission_sha256=m['actual_submission_sha256'],audit_issues=m['audit_issues'],alignment=alignment)
            if j:
                identity=read(run/'loaded_identity.json')
                check(f"{p['seed']} five actual candidate instances",len(identity['instances'])==5 and all(s.endswith('.MyDroneRedDelivery39R2') for s in identity['instances'].values()))
                check(f"{p['seed']} loaded sources frozen",all(plan['source_sha256'][f]==v for f,v in identity['loaded_sources'].items()))
                cd=read(run/'continuous38_diagnostics.json')
                check(f"{p['seed']} shared Continuous38 pose",not cd['errors'] and max(s[k] for s in cd['samples'] for k in ('shared_position_difference','shared_heading_difference','stopturn_heading_difference'))==0)
                gd=read(run/'delivery39_diagnostics.json');check(f"{p['seed']} delivery observer",not gd['errors'])
                rec['delivery39']={i:{k:v for k,v in row.items() if k not in ('walk_points','walk_edges','samples')} for i,row in gd['drones'].items()}
            records.append(rec)
            Image.fromarray(np.uint8(~d['pred'])*255).save(folder/f'{j}_actual_submission.png')
            for i in range(5):Image.fromarray(navplot.nav_colors(n,str(i))).save(folder/f'{j}_drone{i}_navigation.png')
            # Individually readable submission, bombs and physical trajectories.
            for kind,draw in [('submission',lambda ax:atlas.draw_submission(ax,m,d)),('bombs',lambda ax:atlas.draw_bombs(ax,m,d)),('trajectory',lambda ax:atlas.draw_tracks(ax,m,d,'red'))]:
                fig,ax=plt.subplots(figsize=(10,7));draw(ax);fig.tight_layout();save(fig,folder/f'{j}_{kind}.png')
        check(f"{p['seed']} actual matching inputs",all(read(Path(pairs[0]['red_run'])/'request.json')[k]==read(Path(pairs[1]['red_run'])/'request.json')[k] for k in ('seed','round','map_file','zones','stage','drones','bombs','ray_backend','smoke','diagnostic_steps')))
        atlas.write(folder/'data.json',records);all_records.extend(records)
        fig,axs=plt.subplots(2,3,figsize=(21,11.7))
        for row,((m,d),rec) in enumerate(zip(loaded,records)):
            atlas.draw_submission(axs[row,0],m,d)
            axs[row,0].set_title(f"{names[row]} · 实际提交\n地图分 {rec['map_score']:.5f} · 终图 {rec['latest']}/5 · ACK {len(rec['acknowledgements'])}/5",fontsize=12)
            atlas.draw_bombs(axs[row,1],m,d);atlas.draw_tracks(axs[row,2],m,d,'red')
        fig.suptitle(f"Map09 / seed {p['seed']} / 官方特殊区域开启\n上：历史 Continuous38；下：本次冻结 Delivery39R2；均为实际提交、实际炸弹和原生轨迹",fontsize=16)
        fig.legend([Line2D([0],[0],color=atlas.COLORS[i]) for i in range(5)],[f'无人机{i}' for i in range(5)],loc='lower center',ncol=5,frameon=False)
        fig.subplots_adjust(top=.83,bottom=.06,left=.01,right=.99,hspace=.26,wspace=.055);save(fig,folder/'submission_bombs_trajectory.png')
        fig,axs=plt.subplots(2,5,figsize=(24,10))
        for row,((m,d),n) in enumerate(zip(loaded,nav)):
            for i in range(5):navplot.draw(axs[row,i],navplot.nav_colors(n,str(i)),names[row]+f'\n{i}号机',d['shape'])
        fig.suptitle(f"Map09 / seed {p['seed']} / 各机实际赛末导航网格\n灰未知、白观测自由、黑占据；20px规划格；未经过最终规整与补边",fontsize=16)
        fig.subplots_adjust(top=.85,bottom=.04,left=.01,right=.99,wspace=.05,hspace=.25);save(fig,folder/'navigation_all5.png')
        fig,axs=plt.subplots(2,2,figsize=(16,10))
        for row,(m,d) in enumerate(loaded):
            atlas.draw_errors(axs[row,0],m,d);axs[row,1].imshow(Image.open(d['native_wall']));axs[row,1].set_axis_off()
            axs[row,1].set_title(f"{names[row]} · 原生墙体分 {records[row]['wall_score']:.5f}")
        fig.suptitle('实际提交误差与实际炸弹墙体覆盖\n误差：绿命中／蓝漏画／红受罚／黄容差；右侧红色为原生爆炸覆盖',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.92));save(fig,folder/'errors_wall.png')
        fig,axs=plt.subplots(2,2,figsize=(15,8))
        for ax,key in zip(axs.flat,('all','north','north_deep','northwest_deep')):
            for j,c in enumerate(curves):ax.plot(np.arange(len(c[key]))+2,c[key],label=names[j])
            ax.set(title=key,xlabel='原生步数',ylabel='中心到访40px格');ax.grid(alpha=.2);ax.legend()
        fig.suptitle(f'共同截止 {common} 步的实际到访；不是雷达视野覆盖率',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.95));save(fig,folder/'coverage.png')
        fig,axs=plt.subplots(2,1,figsize=(13,8),sharex=True,sharey=True)
        for j,diag in enumerate(diags):
            rows=[x for x in diag['return_samples'] if x['drone']==0]
            for i in range(5):axs[j].step([x['control_step'] for x in rows],[x['received_versions'].get(str(i),0) for x in rows],where='post',color=atlas.COLORS[i],label=f'源{i}')
            axs[j].axvline(records[j]['freeze_elapsed']-1,ls='--',c='#555',label='源图冻结')
            axs[j].set(title=names[j],ylabel='提交者实收源版本');axs[j].grid(alpha=.2);axs[j].legend(ncol=6)
        axs[-1].set_xlabel('控制步数');fig.suptitle('实际消息产生的源图版本进度（30步采样；最终精确版本见表）')
        fig.tight_layout(rect=(0,0,1,.95));save(fig,folder/'delivery_timeline.png')
        lines=[f"**Map09 / seed {p['seed']} / 特殊区域开启**",'',
            '|版本|地图分|墙体分|时间分|实际弹|步数 / 秒|最新终图|源集合ACK|蓝分 / 完整总分|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in records:lines.append(f"|{r['version']}|{r['map_score']:.5f}|{r['wall_score']:.5f}|{r['time_score']:.5f}|{r['bombs']}/10|{r['steps']} / {r['seconds']:.2f}|{r['latest']}/5|{len(r['acknowledgements'])}/5|未执行 / 不报告|")
        lines+=['','|来源|原版实收 / 终版|候选实收 / 终版|','|---|---:|---:|']
        for i in range(5):
            a,b=records[0]['sources'][i],records[1]['sources'][i]
            lines.append(f"|{i}号|{a['received']} / {a['final']}|{b['received']} / {b['final']}|")
        lines+=['',f"共同截止 {common} 步；西北深处到访格：{records[0]['coverage']['northwest_deep']['cells40']} → {records[1]['coverage']['northwest_deep']['cells40']}。更左侧 x<400、y<320 区域累计无人机帧：{sum(records[0]['northwest_far_drone_frames'])} → {sum(records[1]['northwest_far_drone_frames'])}；该区域仅用于赛后诊断。",'']
        for filename,title in [('submission_bombs_trajectory.png','实际提交图、炸弹图、轨迹图'),('navigation_all5.png','五机导航图'),('errors_wall.png','提交误差与原生墙体覆盖'),('coverage.png','到访曲线'),('delivery_timeline.png','实际通信交付进度')]:
            lines += [f"![{title}]({(folder/filename).as_posix()})",'']
        lines += [f"[本组完整数据]({(folder/'data.json').as_posix()})",'']
        parts.append('\n'.join(lines))
    analysis=[]
    for seed in sorted(set(r['seed'] for r in all_records),reverse=True):
        old,new=[r for r in all_records if r['seed']==seed]
        analysis.append(f"种子{seed}：地图分 {old['map_score']:.5f} → {new['map_score']:.5f}；命中 {old['wall_hits']:,} → {new['wall_hits']:,}；误画扣分 {old['penalty']:.5f} → {new['penalty']:.5f}；全图到访 {old['coverage']['all']['cells40']} → {new['coverage']['all']['cells40']} 格；北侧深处 {old['coverage']['north_deep']['cells40']} → {new['coverage']['north_deep']['cells40']} 格。")
        analysis.append(f"源图冻结控制步 {new['freeze_elapsed']-1}；提交控制步 {new['submit_control']}，原因 `{new['submit_reason']}`。")
        if seed==4233377800:
            events=new['delivery39']['2']['events']
            returned=next(e['step'] for e in events if e['kind']=='return_started')
            contact=next((e['step'] for e in events if e['kind']=='contact_restored' and e['step']>=returned),None)
            arrived=next((e['step'] for e in events if e['kind']=='exchange_arrived'),None)
            analysis.append(f"关键2号机：第{returned}控制步开始返程，第{contact}步实际收到同伴消息，第{arrived}步进入集合交换。原版提交时仅有440版，本版以实际收到的封存字节验收。")
    parts.insert(0,'**结果解释**\n\n'+'\n\n'.join(analysis)+'\n\n实际飞行路线与失定位时长随策略改变；地图分变化不能全部归因于无线协议，墙体分与布弹效果单独列出。固定问题种子的通过不等于随机地图稳定性已验证。\n')
    atlas.write(OUT/'data.json',dict(completion=done,metrics=all_records,checks=checks))
    atlas.write(BATCH/'audit.json',dict(passed=True,count=len(checks),checks=checks))
    initial=read(ROOT/'output/sf13/delivery39_run_20261122/completion.json')
    intro=['# Delivery39R2：通信交付与未知区域探索对照','',
        '日期：2026-11-22。HEAD：'+plan['head']+'。本轮保留 Continuous38 定位、建图、原观测补边及布弹规则；候选修改返程、交付调度与未完成前沿管理。',
        '',f"最终R2新红阶段 {done['new_red_runs']} 场；加上初版、R1各1场开发验证，本轮共 {done['new_red_runs']+2} 场新红阶段。新蓝阶段0场，各冻结版本同种子均只执行1次。历史对照来自 Continuous38 随机三种子批次，未重跑。第一场通信门槛 {'通过' if done['gate_first_passed'] else '未通过'}；未执行种子：{done['skipped_seeds']}。只有红阶段实测分项，不报告完整总分。",'',
        '资源与环境：原生 Map09，NO_COM_ZONE / NO_GPS_ZONE 开启；1 CPU、4 GB、5400步／180秒，CPU雷达。各机只使用合法传感器和实际通信消息；地图真值和特殊区域几何只用于赛后分析。',
        '', '实现：记录实际走过的路径，当前雷达验证局部捷径；在占据网格断路时用历史通路返程。按路径长度、近期实耗时和交换余量调度；通过消息约定已观测的集合位置及最终窗口。60%进度后且布弹完成、交付预算允许时，延续未完成前沿。沿用原源图字节、manifest与源集合ACK校验，不以靠近、发出或中途收到冒充最终交付。',
        '', '开发记录：初版 Delivery39 另执行1场同种子4233377800，地图65.09993、终图4/5、ACK0/5，未通过交付门槛。R1修复节点推进后又执行1场，地图67.46369、终图4/5、ACK0/5，仍未过门槛。进一步定位到返程路径未同步GPS回锁修正；R2读取Continuous38现有回修结果更新历史路径。两场开发证据和图件均保留，不能计作成功或稳定性验证。',
        '', '局限：走过的路径仍依赖自身定位，当前雷达必须继续检查安全；消息约定使用初始已知集合区，尚无中途动态接应点优化。R2已将现有定位回修同步到路径记录，但没有改动定位或建图模块。只在最终交付完成后才允许进入第二个种子，不能把门槛未过的候选当作采用版本。',
        '',f"组件检查18项通过；产物核验{len(checks)}项通过。冻结计划SHA256：`{atlas.sha(BATCH/'plan.json')}`。",'']
    parts.append('**保留的初版失败场：Delivery39 / seed4233377800**\n\n初版实际地图分65.09993、4/5终图；不能算全机交付成功。\n\n![初版提交炸弹轨迹](E:/桌面/gitee/swarm-eod/output/sf13/delivery39_run_20261122/review/seed_4233377800/submission_bombs_trajectory.png)\n\n[初版完整导航、墙体、通信图册](E:/桌面/gitee/swarm-eod/红队_Delivery39_通信交付与未知探索对照_20261122.md)\n')
    attempts.append(ROOT/'output/sf13/delivery39_run_20261122/runs/Map09_r03/candidate_delivery39/red/attempt_001')
    parts.append('**保留的R1失败场：Delivery39R1 / seed4233377800**\n\nR1实际地图分67.46369、4/5终图。\n\n![R1提交炸弹轨迹](E:/桌面/gitee/swarm-eod/output/sf13/delivery39r1_run_20261122/review/seed_4233377800/submission_bombs_trajectory.png)\n\n[导航、墙体与交付图册](E:/桌面/gitee/swarm-eod/红队_Delivery39R1_通信交付与未知探索对照_20261122.md)\n')
    attempts.append(ROOT/'output/sf13/delivery39r1_run_20261122/runs/Map09_r03/candidate_delivery39r1/red/attempt_001')
    replay=['','**录像回放（不新增仿真）**','','本批未运行蓝阶段，因此无蓝阶段录像。','','```powershell',
        "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'","$env:PYTHONPATH='src'","$env:PYGLET_SHADOW_WINDOW='false'","$env:PYTHONIOENCODING='utf-8'"]
    replay += [f".venv/Scripts/python.exe -m swarm_rescue.solutions.devtools.optimization.evaluation.replay '{p.as_posix()}'" for p in attempts]
    replay += ['```','',f"[原始评测计划]({(BATCH/'plan.json').as_posix()}) · [门槛检查]({(BATCH/'gates.json').as_posix()}) · [核验]({(BATCH/'audit.json').as_posix()}) · [全部数据]({(OUT/'data.json').as_posix()})"]
    intro[2:2] = ['**验收结论：两个指定问题样本均通过，保留 Delivery39R2 作为下一轮候选；尚未替换正式比赛 ZIP。** 第一场先通过五源封存字节与五方源集合 ACK 检查，之后才运行第二场，同一份冻结 R2 代码没有再改。西北补探与最终交付在这两个样本成立，不代表随机种子、未知地图已经稳定。\n\n|问题种子|Continuous38地图分|R2地图分|地图分变化|最新终图：原→新|源集合ACK：原→新|全图到访40px格：原→新|\n|---|---:|---:|---:|---|---|---|\n|4233377800|41.25847|77.84078|+36.58231|3/5→5/5|0/5→5/5|600→652|\n|2315063155|61.60675|78.85950|+17.25275|5/5→5/5|5/5→5/5|569→623|\n\n第二场西北深处到访9→61格；更左侧诊断区（图像坐标x<400、y<320）从0帧到953帧，均来自2号机实际轨迹。补边新增像素反而从27,347降到14,472，因此不能把这次西北改善解释为只补了一条外框。五份终版均为控制4643版，控制4648完成提交；版本、摘要和源图实际字节一致。ACK仅是算法内部的源集合确认，不是新增的官方提交要求，也不表示五机各自的本地导航网格变成同一张图。\n\n退步与剩余问题：第一场误画扣分2.39529→5.43112，新增收到的北侧墙体仍有偏斜、重影；失定位样本航向误差P95为7.73°→7.99°，本轮没有改善定位器本身。第二场原生炸弹墙体分80.58041→73.40579，下降7.17462；布弹规则保留，但飞行和决策时序改变后实际弹位也会变化。第一场墙体分70.70319保持不变。两场都10/10弹，红阶段有效；本轮未跑蓝队，完整总分不报告。\n\n定位与建图原模块保持不变。返程记录需同步Continuous38已有的GPS回锁修正：离线复现中104个旧路径节点最大需移动75.23px，R2只同步这些历史路径坐标；同观测下定位结果差为0。这解决的是新返程模块与现有定位回修之间的坐标不一致。候选仍用当前雷达检查局部行驶安全，没有用地图真值、种子或禁通信区坐标规划。', '']
    REPORT.write_text('\n'.join(intro)+'\n\n'.join(parts)+'\n'+'\n'.join(replay),encoding='utf-8')
    print(json.dumps(dict(report=str(REPORT),checks=len(checks),completion=done,
        results=[{k:r[k] for k in ('version','seed','map_score','latest','acknowledgements','steps','seconds','coverage')} for r in all_records]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
