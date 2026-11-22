"""Frozen paired flight evidence; figures cannot affect contestant navigation."""
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
BATCH=ROOT/'output/sf13/schedule40r1_run_20261122';OUT=BATCH/'review'
OLD=ROOT/'output/sf13/delivery39r2_random3_20261122'
REPORT=ROOT/'红队_Schedule40R1_探索调度冲突修复对照_20261122.md'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


h=module('schedule40_figures','src/swarm_rescue/solutions/devtools/delivery39r2_random3_report.py')
atlas=h.atlas;navplot=h.navplot;read=atlas.read;save=h.save
LABELS=['Delivery39R2 历史基线','Schedule40R1 本次新执行']


def main():
    driver=module('schedule40_validate','src/swarm_rescue/solutions/devtools/schedule40r1_batch.py')
    plan=driver.validate(BATCH);done=read(BATCH/'completion.json');assert done['complete']
    new=read(BATCH/'pairs.json');old=read(OLD/'pairs.json');assert len(new)==2
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();checks=[];records=[]
    def check(name,good):
        checks.append(dict(check=name,passed=bool(good)))
        if not good:raise AssertionError(name)
    check('one attempt per map',len(list((BATCH/'runs').glob('**/attempt_*')))==2)
    for p in new:
        name=p['map'];folder=OUT/name;folder.mkdir()
        pairs=[next(x for x in old if x['map']==name and x['seed']==p['seed']),p]
        diags=[read(Path(x['red_run'])/'dualmap32_diagnostics.json') for x in pairs]
        common=min(v['_f27_stats']['cutoff_elapsed'] for d in diags for v in d['drones'].values())
        loaded=[];navs=[];local=[];curves=[];observations=[]
        for j,(pair,diag) in enumerate(zip(pairs,diags)):
            run=Path(pair['red_run']);metric,data=atlas.inspect(pair,BATCH)
            metric['version']=LABELS[j];loaded.append((metric,data));nav=navplot.load(pair);navs.append(nav)
            receipt=nav['receipt'];identity=read(run/'loaded_identity.json')
            check(name+str(j)+' saved evidence',all(atlas.sha(run/f)==v for f,v in read(run/'integrity.json')['files'].items()))
            check(name+str(j)+' frozen loaded sources',all(plan['source_sha256'][f]==v for f,v in identity['loaded_sources'].items()))
            cls='MyDroneRedDelivery39R2' if j==0 else 'MyDroneRedSchedule40R1'
            check(name+str(j)+' actual five instances',len(identity['instances'])==5 and all(v.endswith('.'+cls) for v in identity['instances'].values()))
            req=read(run/'request.json');resource=read(run/'resources.json')
            check(name+str(j)+' native zones and resources',req['zones']==p['scenario']['zones'] and resource['cpu_count']==1 and resource['memory_limit_bytes']==4000000000)
            check(name+str(j)+' native actual submission score',not metric['audit_issues'])
            cd=read(run/'continuous38_diagnostics.json');dd=read(run/'delivery39_diagnostics.json')
            check(name+str(j)+' observers',not cd['errors'] and not dd['errors'] and not diag['return_sample_errors'])
            check(name+str(j)+' shared Continuous38 pose',max(s[k] for s in cd['samples'] for k in ('shared_position_difference','shared_heading_difference','stopturn_heading_difference'))==0)
            with np.load(run/'dualmap32_snapshots.npz',allow_pickle=False) as tape:sources,acks=h.delivery(diag,receipt,tape)
            check(name+str(j)+' ACK actual bytes',sorted(acks)==sorted(receipt['acknowledged_sources']))
            cover,curve=h.coverage(data,common,name);curves.append(curve)
            obs,alignment=h.helpers.observations(pair,data,common);observations.append(obs)
            outage=np.concatenate([a[a[:,1]>0] for a in obs.values()])
            schedule=read(run/'schedule40r1_diagnostics.json') if j else None
            if schedule:check(name+' schedule observer',not schedule['errors'])
            if schedule:check(name+' corrected dispatch in all instances',all('ExplicitMotionDispatch40' in v['mro'] for v in schedule['drones'].values()))
            record=dict(map=name,seed=p['seed'],round=p['round'],version=LABELS[j],historical=j==0,
                valid_red=metric['valid_red'],map_score=metric['map_score'],wall_score=metric['wall_score'],time_score=metric['time_score'],
                blue_score=None,total_score=None,bombs=metric['placed_count'],steps=metric['red_steps'],seconds=metric['red_seconds'],
                wall_hits=metric['error_metric']['wall_hits'],wall_misses=metric['error_metric']['wall_misses'],penalty=metric['error_metric']['penalty'],
                latest_sources=sum(s['latest'] for s in sources),acknowledgements=len(acks),sources=sources,
                delivery_complete=bool(all(s['latest'] for s in sources) and len(acks)==5 and receipt['final_complete']),
                submit_control=receipt['submit_control_step'],submit_reason=receipt['reason'],common_cutoff_elapsed=common,
                coverage=cover,outage=h.helpers.stats(outage),alignment=alignment,boundary=metric['boundary'],
                delivery_drones={i:{k:v for k,v in d.items() if k not in ('walk_points','walk_edges','samples')} for i,d in dd['drones'].items()},
                schedule_stats=None if schedule is None else {i:v['stats'] for i,v in schedule['drones'].items()},
                actual_zones=metric['actual_zones'],red_issues=metric['red_issues'],red_run=run.as_posix(),submission_sha256=metric['actual_submission_sha256'])
            records.append(record);local.append(record)
            arm='baseline' if j==0 else 'candidate'
            atlas.write(folder/f'{arm}_data.json',record)
            Image.fromarray(np.uint8(~data['pred'])*255).save(folder/f'{arm}_actual_submitted_pixels.png')
            for i in range(5):Image.fromarray(navplot.nav_colors(nav,str(i))).save(folder/f'{arm}_drone{i}_navigation.png')
        # Actual submissions and native physical trajectory, not an offline rescore.
        fig,axes=plt.subplots(2,3,figsize=(21,11.7))
        for j,((m,d),r) in enumerate(zip(loaded,local)):
            atlas.draw_submission(axes[j,0],m,d)
            axes[j,0].set_title(f"{LABELS[j]} · 实际提交\n地图分 {r['map_score']:.5f} · 最新图 {r['latest_sources']}/5 · ACK {r['acknowledgements']}/5")
            atlas.draw_bombs(axes[j,1],m,d);atlas.draw_tracks(axes[j,2],m,d,'red')
        fig.suptitle(f'{name} / 同种子2252748057 / 官方特殊区域开启\n上：历史R2；下：调度修复新执行；左提交、中炸弹、右轨迹',fontsize=16)
        fig.legend([Line2D([0],[0],color=atlas.COLORS[i]) for i in range(5)],[f'无人机{i}' for i in range(5)],loc='lower center',ncol=5,frameon=False)
        fig.subplots_adjust(top=.86,bottom=.045,left=.01,right=.99,wspace=.06,hspace=.28);save(fig,folder/'comparison.png')
        fig,axes=plt.subplots(2,5,figsize=(25,10))
        for j,((m,d),nav) in enumerate(zip(loaded,navs)):
            for i in range(5):navplot.draw(axes[j,i],navplot.nav_colors(nav,str(i)),f'{LABELS[j]} / {i}号',d['shape'])
        fig.suptitle(f'{name} 导航图：灰未知、白自由、黑占据；20px格，未经过最终补边',fontsize=15)
        fig.subplots_adjust(top=.87,bottom=.02,left=.01,right=.99,wspace=.04,hspace=.28);save(fig,folder/'navigation.png')
        fig,axes=plt.subplots(2,2,figsize=(16,12.7))
        for j,(m,d) in enumerate(loaded):
            atlas.draw_errors(axes[j,0],m,d);axes[j,1].imshow(Image.open(d['native_wall']));axes[j,1].set_axis_off()
            axes[j,1].set_title(f"{LABELS[j]} · 原生墙体分 {local[j]['wall_score']:.5f}")
        fig.suptitle(f'{name} 误差与布弹覆盖；绿命中、蓝漏画、红受罚、黄容差',fontsize=15)
        fig.subplots_adjust(top=.88,bottom=.025,left=.02,right=.99,wspace=.16,hspace=.28);save(fig,folder/'errors_wall.png')
        fig,axes=plt.subplots(2,5,figsize=(25,10))
        for j,(nav,d) in enumerate(zip(navs,local)):
            for i,s in enumerate(d['sources']):
                navplot.draw(axes[j,i],nav['sources'][str(i)],f"{LABELS[j]} / 来源{i}\n实收{s['received']} / 终版{s['final']}",loaded[j][1]['shape'],True)
        fig.suptitle(f'{name} 提交者实收各机源图',fontsize=16)
        fig.subplots_adjust(top=.86,bottom=.02,left=.01,right=.99,wspace=.04,hspace=.32);save(fig,folder/'sources.png')
        fig,axes=plt.subplots(2,1,figsize=(12,8))
        for j,diag in enumerate(diags):
            rows=[x for x in diag['return_samples'] if x['drone']==0]
            for i in range(5):axes[j].step([x['control_step'] for x in rows],[x['received_versions'].get(str(i),0) for x in rows],where='post',color=atlas.COLORS[i],label=f'来源{i}')
            axes[j].set(title=LABELS[j],xlabel='控制步数',ylabel='提交者实收版本');axes[j].legend(ncol=5);axes[j].grid(alpha=.2)
        fig.suptitle(f'{name} 实际收图进度（30步采样；精确终版另列）');fig.tight_layout(rect=(0,0,1,.94));save(fig,folder/'delivery.png')
        fig,axes=plt.subplots(2,2,figsize=(14,9))
        for ax,key in zip(axes.flat,curves[0]):
            for j,curve in enumerate(curves):ax.plot(np.arange(len(curve[key]))+2,curve[key],label=LABELS[j])
            ax.set(title=key,xlabel='原生步数',ylabel='机身中心到访40px格');ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f'{name} 到访曲线 / 统一截止{common}步 / 不是雷达覆盖率');fig.tight_layout(rect=(0,0,1,.94));save(fig,folder/'coverage.png')
        fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,sharey='col')
        for j,obs in enumerate(observations):
            for i,a in obs.items():
                part=a[a[:,1]>0]
                for c,col in enumerate((2,3)):axes[j,c].scatter(part[:,0],abs(part[:,col]),s=3,alpha=.45,color=atlas.COLORS[i],label=f'{i}号')
            for c in range(2):
                axes[j,c].set(title=LABELS[j],xlabel='控制步数',ylabel='位置误差px' if c==0 else '航向误差度')
                axes[j,c].legend(ncol=5,fontsize=8,markerscale=3);axes[j,c].grid(alpha=.2)
        fig.suptitle(f'{name} 失定位期间漂移 / 路径不同，仅作诊断');fig.tight_layout(rect=(0,0,1,.94));save(fig,folder/'drift.png')
    result=dict(completion=done,records=records,checks=checks)
    atlas.write(OUT/'data.json',result);atlas.write(BATCH/'audit.json',dict(passed=True,count=len(checks),checks=checks))
    write_report(result)
    print(json.dumps(dict(report=str(REPORT),checks=len(checks),results=[{k:r[k] for k in ('map','version','map_score','latest_sources','acknowledgements','coverage','schedule_stats')} for r in records]),ensure_ascii=False,indent=2))


def write_report(result):
    rows=result['records']
    text=['# Schedule40R1：探索调度冲突修复对照','',
        '日期：2026-11-22。用户授权修复冲突并继续测试。本轮新增Map08、Map09各一次红阶段，均用2252748057种子、原第2轮编号、原生特殊区开启；历史Delivery39R2同条件回执复用。1CPU、4GB、CPU雷达、5400步／180秒，每机两弹。没有新跑蓝队，不报告完整总分。','',
        '本候选修复普通导航调用入口，保证下述修复在完整调用链生效，并统一三项调度限制：探索时取消旧72%清空路线；布弹完成后持续未知任务不再等待60%；窄路执行不再受68%固定截止限制，仍受真实交付预算抢占。Continuous38定位、导航观测、最终建图和补边、通信协议、返程预算及执行器、源图封存时限、开局方向策略保持不变。22项回归通过，源码及驱动在两场飞行前冻结，未中途调参、换种子或重试。三个调度修改一起测试，不能把全部变化单独归因于某一个修改。','',
        '|地图|版本|有效地图分|墙体分|时间分|实际弹|步数／秒|最新终图／ACK|','|---|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        text.append(f"|{r['map']}|{r['version']}|{r['map_score']:.5f}|{r['wall_score']:.5f}|{r['time_score']:.5f}|{r['bombs']}/10|{r['steps']}／{r['seconds']:.2f}|{r['latest_sources']}/5、{r['acknowledgements']}/5|")
    initial=read(ROOT/'output/sf13/schedule40_run_20261122/review/data.json')['records']
    text+=['','**开发试验保留与实际执行次数**','','本轮先执行Schedule40两场，发现Advanced2静态super调用绕过72%修复方法；有效成绩保留，但不能据此声称该冲突修好。随后新增Schedule40R1独立版本修正调用入口，重新冻结并各跑一场。合计新红阶段4场，最终版本占2场；并非原代码失败后重试或择优选分。无新蓝队。','', '|开发版本|Map08地图分|Map09地图分|状态|','|---|---:|---:|---|']
    initial_new=[r for r in initial if not r['historical']]
    text.append(f"|Schedule40|{initial_new[0]['map_score']:.5f}|{initial_new[1]['map_score']:.5f}|调用冲突未完整修复，保留开发记录|")
    text+=['','**探索与误差**','','到访格按地图内存活飞机的中心统计，40px网格；不等同雷达覆盖率。Map08右下为最右1/3且最底100px带；Map09西北深处为x<622、y<320。分区仅用于赛后诊断，算法不读取地图真值。漂移按失GPS／罗盘时每10步采样，路径不同，不能当作同一输入定位对照。','',
        '|地图|版本|全图到访格|重点区到访格|真墙命中|误画扣分|位置P95|航向P95|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        region='southeast' if r['map']=='Map08' else 'northwest_deep'
        text.append(f"|{r['map']}|{r['version']}|{r['coverage']['all']['cells40']}|{r['coverage'][region]['cells40']}|{r['wall_hits']}|{r['penalty']:.5f}|{r['outage']['navigation_position']['p95']:.2f}px|{r['outage']['navigation_heading']['p95']:.2f}°|")
    for name in ('Map08','Map09'):
        text+=['',f'**{name} 同条件完整图示**','']
        for filename,label in [('comparison.png','提交图、炸弹图、轨迹图'),('navigation.png','两版五机导航图'),('errors_wall.png','误差及墙体覆盖'),('coverage.png','到访曲线'),('drift.png','漂移诊断'),('sources.png','实收源图'),('delivery.png','通信交付曲线')]:
            text+=['',f'![{label}]({(OUT/name/filename).as_posix()})']
        for r in (x for x in rows if x['map']==name):
            text+=['',f"**{r['version']}：交付与探索调度**",'',f"有效红阶段：{r['valid_red']}；提交控制步{r['submit_control']}，原因 `{r['submit_reason']}`。",'',
                '|机号|实收版／终版|字节一致|开始返程|首次到达集合区域|扩展未知目标选择|','|---|---|---|---:|---:|---:|']
            for s in r['sources']:
                i=str(s['drone']);d=r['delivery_drones'][i]
                ret=next((e['step'] for e in d['events'] if e['kind']=='return_started'),None)
                arr=next((e['step'] for e in d['events'] if e['kind']=='exchange_arrived'),None)
                text.append(f"|{i}|{s['received']}／{s['final']}|{s['latest']}|{ret}|{arr}|{d['frontier_stats'].get('frontier_selected',0)}|")
            if r['schedule_stats']:
                text+=['','|机号|60%前未知探索启用步|72%后保留路线步|68%后窄路动作步|','|---|---:|---:|---:|']
                for i,s in r['schedule_stats'].items():text.append(f"|{i}|{s.get('early_frontier_enabled_ticks',0)}|{s.get('legacy_route_clear_prevented',0)}|{s.get('late_narrow_command',0)}|")
            b=r['boundary'];chosen=b.get('diagnostic',{}).get('chosen',{})
            info=dict(inferred=b.get('inferred'),added_pixels=b.get('added_pixels'),bounds_pixels=chosen.get('bounds_pixels'),rejection=b.get('rejection'))
            text+=['',f"补边信息：`{json.dumps(info,ensure_ascii=False)}`"]
            if r['red_issues']:text+=['',f"有效性问题：{r['red_issues']}"]
    text+=['','**核验与录像**','',f"{len(result['checks'])}项证据核验通过。ACK是算法内确认，不是官方额外要求；使用实际接收版本、源图摘要与字节验证。每图一个固定问题种子不能证明随机稳定性。",'',
        f"[冻结计划]({(BATCH/'plan.json').as_posix()}) · [核验]({(BATCH/'audit.json').as_posix()}) · [完整数据]({(OUT/'data.json').as_posix()})",'',
        '以下仅回放录像，不新增仿真：','','```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'","$env:PYTHONPATH='src'","$env:PYGLET_SHADOW_WINDOW='false'"]
    for r in rows:text.append(f".venv/Scripts/python.exe -m swarm_rescue.solutions.devtools.optimization.evaluation.replay '{r['red_run']}'")
    text+=['```','']
    text[2:2]=['**结论：Schedule40R1的调度冲突已修复并在实测生效，Map09改善，Map08右下漏探索未解决。** 最终候选两场均有效、各10/10弹、最新终图与全队确认均5/5；没有运行蓝队，不报告完整总分。建议保留为后续开发基线，尚不能凭每图一个问题种子替换正式ZIP或认定随机稳定。\n\nMap09地图分48.89603→57.04372（+8.14769），全图到访528→548格，北侧深处44→86格，西北深处0→24格。0、3号机进入了原先缺失的西北分支，但最左上角与若干房间仍未覆盖。真墙命中60,263→68,924（+8,661），误画扣分5.14811→4.76765。墙体分71.66292→71.20364（−0.45928），不是所有分项都提高。\n\nMap08地图分52.32775→52.41928（+0.09152），全图到访542→545格，但右侧73→68、底部带23→22、右下最底区域仍0格。真墙命中54,059→55,512（+1,453），误画扣分3.60384→5.01564，收益大部分被误画抵消。补边沿用原策略，当前证据把右界误推到x=1498，真值外墙约1660；错误边框位置确认，证据究竟来自内墙误认还是漂移后的外墙尚待拆分。墙体分60.59411→61.40938（+0.81527）。\n\n修复生效证据：完整导航入口回归检查确认旧Schedule40会清空路线、R1会保留；真实Map09里1、2、4号机在72%后分别有647、647、391个控制步保留有效路线，合计1,685步。这不是1,685次独立规划成功，也不是新增探索格数。Map08所有飞机在72%前返程，所以这项取消清空本身没有改变该场晚期行为；提前未知任务等修改仍改变了路线。68%后窄路资格已放开，但最终两场专用窄路动作计数均为0，不能宣称窄路通过率已提升。\n\n开发过程须保留：先跑Schedule40两场，发现静态super调用绕过修复；其有效分为52.41928、56.44901，但修复不完整。修正调用入口后另冻结R1，各图再跑一次；本轮累计新红4场，最终候选2场，不把开发试验混入最终结果。两版间Map09又增加0.59471分，但整组相对R2的8.14769分包含三项调度修改，不全部归因于72%修复。所有原始回执保留。','']
    REPORT.write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':main()
