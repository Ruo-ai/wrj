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
BATCH=ROOT/'output/sf13/delivery39_run_20261122';OUT=BATCH/'review'
OLD=ROOT/'output/sf13/continuous38_random3_20261122'
REPORT=ROOT/'红队_Delivery39_通信交付与未知探索对照_20261122.md'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
helper=module('delivery39_report_helpers','output/sf13/continuous38_report_20261122.py')
atlas=helper.atlas;navplot=helper.navplot;read=atlas.read

def save(fig,path):
    fig.savefig(path,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    driver=module('delivery39_final_validate','src/swarm_rescue/solutions/devtools/delivery39_batch.py')
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
        names=['Continuous38 历史基线','Delivery39 本次新执行']
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
                check(f"{p['seed']} five actual candidate instances",len(identity['instances'])==5 and all(s.endswith('.MyDroneRedDelivery39') for s in identity['instances'].values()))
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
        fig.suptitle(f"Map09 / seed {p['seed']} / 官方特殊区域开启\n上：历史 Continuous38；下：本次冻结 Delivery39；均为实际提交、实际炸弹和原生轨迹",fontsize=16)
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
    atlas.write(OUT/'data.json',dict(completion=done,metrics=all_records,checks=checks))
    atlas.write(BATCH/'audit.json',dict(passed=True,count=len(checks),checks=checks))
    intro=['# Delivery39：通信交付与未知区域探索对照','',
        '日期：2026-11-22。HEAD：'+plan['head']+'。本轮保留 Continuous38 定位、建图、原观测补边及布弹规则；候选修改返程、交付调度与未完成前沿管理。',
        '',f"本次新红阶段 {done['new_red_runs']} 场，新蓝阶段0场，重试0次。历史对照来自 Continuous38 随机三种子批次，未重跑。第一场通信门槛 {'通过' if done['gate_first_passed'] else '未通过'}；未执行种子：{done['skipped_seeds']}。只有红阶段实测分项，不报告完整总分。",'',
        '资源与环境：原生 Map09，NO_COM_ZONE / NO_GPS_ZONE 开启；1 CPU、4 GB、5400步／180秒，CPU雷达。各机只使用合法传感器和实际通信消息；地图真值和特殊区域几何只用于赛后分析。',
        '', '实现：记录实际走过的路径，当前雷达验证局部捷径；在占据网格断路时用历史通路返程。按路径长度、近期实耗时和交换余量调度；通过消息约定已观测的集合位置及最终窗口。60%进度后且布弹完成、交付预算允许时，延续未完成前沿。沿用原源图字节、manifest与源集合ACK校验，不以靠近、发出或中途收到冒充最终交付。',
        '', '局限：走过的路径仍依赖自身定位，当前雷达必须继续检查安全；消息约定使用初始已知集合区，尚无中途动态接应点优化。只在最终交付完成后才允许进入第二个种子，不能把门槛未过的候选当作采用版本。',
        '',f"组件检查16项通过；产物核验{len(checks)}项通过。冻结计划SHA256：`{atlas.sha(BATCH/'plan.json')}`。",'']
    replay=['','**录像回放（不新增仿真）**','','本批未运行蓝阶段，因此无蓝阶段录像。','','```powershell',
        "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'","$env:PYTHONPATH='src'","$env:PYGLET_SHADOW_WINDOW='false'","$env:PYTHONIOENCODING='utf-8'"]
    replay += [f".venv/Scripts/python.exe -m swarm_rescue.solutions.devtools.optimization.evaluation.replay '{p.as_posix()}'" for p in attempts]
    replay += ['```','',f"[原始评测计划]({(BATCH/'plan.json').as_posix()}) · [门槛检查]({(BATCH/'gates.json').as_posix()}) · [核验]({(BATCH/'audit.json').as_posix()}) · [全部数据]({(OUT/'data.json').as_posix()})"]
    REPORT.write_text('\n'.join(intro)+'\n\n'.join(parts)+'\n'+'\n'.join(replay),encoding='utf-8')
    print(json.dumps(dict(report=str(REPORT),checks=len(checks),completion=done,
        results=[{k:r[k] for k in ('version','seed','map_score','latest','acknowledgements','steps','seconds','coverage')} for r in all_records]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
