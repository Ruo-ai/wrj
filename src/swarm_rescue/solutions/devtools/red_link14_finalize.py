"""Offline RL14 conclusions, figure audit and workspace protection receipt."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from collections import Counter
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from swarm_rescue.solutions.devtools import red_link14_review as review
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_local13_finalize as prior


ROOT=Path.cwd().resolve()
BATCH=ROOT/'output/sf13/rl14_m89_once_20261115'
OUT=ROOT/'output/sf13/rl14_m89_review_20261115'
DEV=ROOT/'output/sf13/rl14_dev_20261115'
REPORT=ROOT/'复赛阶段_FS4_红队RL14_入口探索成图送达链路实测_20261115.md'


def main():
    assert not REPORT.exists(),'Refuse to overwrite existing report'
    s=base.read(OUT/'summary.json');rows=s['rows'];assert len(rows)==8 and not s['failures']
    protected=base.read(DEV/'protected_before.json')
    changed=[p for p,h in protected.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    for r in rows:
        assert r['red_valid'] and r['placed_count']==10 and r['callback_errors']==0
        assert not r['diagnostic_errors'] and not r['chain_errors']
        raw=base.read(Path(r['attempt'])/'result.json')
        assert raw['status']=='ok' and not raw['walltime_limited'] and not raw['controller_crashed']
    review.panels.NAMES=review.NAMES
    focus=OUT/'Map09_room_chain_focus.png';assert not focus.exists()
    fig,axes=base.plt.subplots(2,3,figsize=(18,15))
    fig.subplots_adjust(left=.025,right=.985,top=.87,bottom=.075,wspace=.12,hspace=.29)
    for rowindex,controller in enumerate(['rl11_lines','rl14_opening']):
        r=next(v for v in rows if v['map']=='Map09' and v['controller']==controller)
        d=prior.assets(r);run=Path(r['attempt'])
        pred=np.asarray(base.read(next((run/'artifacts').glob('*_exploration.json')))['grid'],bool)
        with np.load(run/'exploration_stages.npz') as f:
            union=np.logical_or.reduce([f[k].astype(bool) for k in f.files if k.startswith('local_final_pixels_')])
        review.evidence.trajectories(axes[rowindex,0],d)
        axes[rowindex,1].imshow(union,cmap='Greys',vmin=0,vmax=1)
        axes[rowindex,2].imshow(pred,cmap='Greys',vmin=0,vmax=1)
        for col,label in enumerate(['实际轨迹','各机终图并集（仅离线）','实际提交图']):
            ax=axes[rowindex,col];ax.set(xlim=(495,1000),ylim=(580,35),aspect='equal');ax.axis('off')
            ax.set_title(review.NAMES[controller]+' · '+label,fontproperties=review.lines.FONT,fontsize=13)
            for letter,box in zip('ABC',review.panels.ROOMS.values()):
                l,t,rr,b=box;ax.add_patch(Rectangle((l,t),rr-l,b-t,fill=False,color='magenta',ls='--',lw=1.2))
                ax.text(l+8,t+25,letter,color='magenta',fontsize=15)
    fig.suptitle('Map09：进入上右房间后，地图是否进入提交图',fontproperties=review.lines.FONT,fontsize=23,y=.98)
    fig.text(.5,.93,'A上左、B上右、C下方；审计框仅用于离线。上排RL11，下排入口候选E。',ha='center',fontproperties=review.lines.FONT,fontsize=13)
    fig.text(.5,.03,'E的B框附近：714个真实墙端点像素；526个在本机终图，提交图命中0。终图并集忽略通信，不能当官方成绩。',ha='center',fontproperties=review.lines.FONT,fontsize=11)
    base.save(fig,focus)
    pngs=list(OUT.glob('*.png'))
    for p in pngs:
        with Image.open(p) as im:im.verify()
    verification=dict(protected_files=len(protected),changed=changed,full_red_phases=8,valid=8,blue_phases=0,
        tests_passed=24,source_head_at_start='7e658b16e784728c92ab76149f1ee35f7aeaa474',
        tracked_diff=subprocess.check_output(['git','diff','--stat'],text=True),
        png_files=len(pngs),rl11_sha256=base.digest(ROOT/'src/swarm_rescue/solutions/my_drone_red_lines11.py'),
        root_package_sha256=base.digest(ROOT/'team742_preliminary.zip'),input_comparisons=s['input_comparisons'])
    review.write(OUT/'verification.json',verification)
    doc=['# RL14：入口探索、建图与送达链路单因素实测','',
        '2026-11-15。**继续保留RL11；本轮D/E/R均不替换开发基线，不切正式入口，不组合。** 完成8场完整红阶段、5组128步冒烟和24项行为测试。入口E在Map09进入了此前漏入的上右房间，但全图地图分下降；链路尚未打通。',
        '', '## 最新状态与诊断依据','',
        '接手HEAD为7e658b16e784728c92ab76149f1ee35f7aeaa474，工作区干净；RL13已经由用户提交。核对本地RL11/12/13完整结果：RL11地图分63.1415/73.3676；RL12-N为55.7578/50.3630；RL13三个单项均无双图提升。RL12/13已出现进入后来源地图长期未送达，故先实现带回执的独立退路实验。',
        f'本轮共{len(protected)}个既有文件哈希变化0；蓝队、共享模块、正式入口、旧文档/结果和根目录ZIP不变。正式place仍为进阶2，实验基线为独立RL11。',
        '运行前Get-Process可见列表没有其他Python评测；进程可见性受环境限制，不能据此断言整机绝无其他负载。本轮由同一驱动串行执行，simultaneous_simulations=1、单CPU亲和、4,000,000,000字节Job限制；未并行启动其他仿真或参数搜索。',
        '', '## 独立机制与文件','',
        '在线新文件：src/swarm_rescue/solutions/my_drone_red_link14.py。MyDroneRedLink14三个开关默认False；Delivery14只开DELIVERY，Opening14只开OPENING_TASKS，Relative14只开RELATIVE_CORRIDOR，Off全部关闭。三版继承RL11的提交地图器、导航定位、放弹与截止，不采用1px画图，不整体平移旧地图。',
        '**D送达退路：**收到有效同队消息时保存联络点；失联后每步使用实际距离/方向/转角把路标变换到当前机体系，28px间隔、最多192点，相邻边必须经当前雷达精细图确认。缺里程计、断边或容量不足使退路失效，不把未知空间改成可通。失联超过600步或进度超过0.54后，在失联至少120步且返程预算足够时尝试返回；预算按路线长度/2.4+240+90×余弹，进度0.79硬退出。遇到新障碍尝试当前已观察自由区局部重规划，无进展80步或总650步退出原策略。旧联络点最多120步等待回执且原导航继续运行。0号机本轮不主动退路，保留原汇图行为。',
        '回执是队内消息内容：0号机把自己实际接受的来源版本表发出，队友可转发原始时间戳；收到相应来源版本>=返程要求版本才记送达。发包、转发和通信未禁用都不会单独标送达。不修改communicator、距离/禁区规则，不读其他机内存。回执不会证明图像素正确或未来最新扫描也已送达。',
        '**E入口任务：**18个方向的实际雷达自由扇区内选60～145px观察点；完整当前扫掠路径须通过RN7保守精细净空检查，周围有未知、避开短期重访及实际收到的队友认领。任务目标按里程计维持，局部堵塞先重规划，200步任务预算、65步无进展退出，之后可重试；到点最多再接一个当时安全可达观察点。保留原导航图的已收到访问数据，只影响收益，不证明安全。到进度0.64/集合/紧急状态交还原策略，放弹优先。当前独立入口任务不额外广播新的任务认领，沿用原目标广播，跨机分工仍有限。',
        '**R相对通道：**缺测时在已有任务方向附近识别连续双侧有限平行墙段，不假设水平/垂直；3次方向稳定、宽度52～130px、70px扫掠净空通过才辅助局部运动。较大相对角先用实际里程计确认停稳再转，小角直接沿相对方向平移；75步执行预算、100步冷却。单侧、开口、动态遮挡或退化均回退，不写全局朝向。三项都使用实际里程计残余位移，命令不当位移。',
        '新增devtools：red_link14_batch.py（专用注册与原评测器串行适配）、red_link14_diagnostics.py（控制返回后的被动逐步墙面/24步入口及导航审计）、test_red_link14.py、red_link14_review.py、red_link14_finalize.py（本报告与图件）。所有新增Python位于solutions内；真值/地图名只出现在离线工具，不进入策略。',
        '', '## 实际测试与全部成绩','',
        '行为测试首轮22项通过1.81秒；加入消息兼容/离线分区检查后24项通过1.85秒。覆盖真实版本回执、陈旧/未来/重复包、原消息解析、加减速/侧滑/角环绕、断链/缓存上限、斜通道、未知不置通、动态队友和有限任务退出。理想与合成观测检查不作为比赛成绩。',
        'Map09基线/Off/D/E/R各128步冒烟，耗时3.540/3.382/3.219/3.349/3.354秒，全部诊断无异常。Off与基线完整输入数组和轨迹精确相同；没有把冒烟标为完整有效成绩。',
        'Map08 seed=200806601、NO_GPS_ZONE、1660×1122；Map09 seed=2477227837、NO_GPS_ZONE+NO_COM_ZONE、1700×1100。官方5400步/180秒，5红机×2弹，每图每版一轮且仅attempt_001，无重试。八局全部完整有效、10弹，无超时/崩溃/回调错误；蓝队难度、完整比赛原始/有效总分未评测。',
        '|地图/版本|M|W|T|步数|引擎墙钟秒|墙/机接触|平均健康|墙面命中|受罚误画|',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in rows:doc.append(f"|{r['map']}/{review.NAMES[r['controller']]}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['red_steps']}|{r['red_walltime']:.3f}|{r['contacts_wall']}/{r['contacts_drone']}|{r['health']:.1f}|{r['wall_hits']}|{r['charged_pixels']}|")
    doc+=['',f"Job峰值提交内存范围 {min(r['resources']['peak_job_commit_bytes'] for r in rows)/1e6:.1f}～{max(r['resources']['peak_job_commit_bytes'] for r in rows)/1e6:.1f} MB。墙钟包含相同诊断链路；新增链路诊断秒数逐局写入summary。实际CPU秒未单列，不能拿外层进程墙钟当CPU秒。",
        'Map09 R与RL11完整合法输入、实际轨迹逐数组相同，R实际上未介入；其他候选均改变输入与轨迹。不能将同种子说成同传感器输入，也不能把变化全部归于地图器。',
        '', '## 入口到提交：证据与限制','',
        '以下A/B/C是此前截图审计框：A上左、B上右、C下方。实际进入为完整录像中机身中心落入框内的帧数；墙像素统计仅在审计框外扩15px内，不代表整间物理房屋的全部墙。',
        '|版本|A进入帧|B进入帧|C进入帧|B真实墙端点像素|B端点且本机终图有|B端点且提交有|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['map']!='Map09':continue
        a,b,c=list(r['screenshot_rooms'].values())
        doc.append(f"|{review.NAMES[r['controller']]}|{a['actual_frames']}|{b['actual_frames']}|{c['actual_frames']}|{b['actual_observed_surface']}|{b['seen_own_final']}|{b['seen_submitted']}|")
    doc+=['',f'![Map09房间链路]({focus.as_posix()})','',
        '**已证实：**RL11未进入三个审计框，但A/B附近已有墙面端点观测。E进入B与C，B附近526个实际观测且本机终图有的真墙像素全部未进入提交图。E的2号来源最终版本1020，距提交步落后3515；D的2号来源仅450，落后4085。',
        '**不能过度归因：**E的B框没有采样到直接指向该框的入口任务选择记录；其进入可能由此前任务改变轨迹后，原导航接着完成。不能说已经证明该门被入口检测器直接识别并选中。发现列是独立被动检测器基于当时合法观测计算的候选可用性，尤其RL11没有该入口任务系统，不能把被动候选等同RL11实际任务意图。',
        '每24步保存的是原导航粗路径、位置估计与通信状态；新增局部重规划只记录次数、任务目标与释放原因，没有逐次保存精细路径节点。实际走过的路径由完整录像保留，但不能据此重建全部临时计划。逐步雷达端点是观测下界，不能把零端点记录直接当作从未看到该区域。未到访、未观察、错位/擦除和未送达的完整因果仍有缺口。',
        '', '|地图/版本|实际提交M|各机终图离线并集公式值|最旧来源延迟步|本机终图未包含于各对应来源快照的格数和|',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        age=max(v['age'] for v in r['source_snapshots'] if v['age'] is not None)
        missing=sum(v['own_final_absent_from_source_snapshot'] for v in r['source_snapshots'])
        doc.append(f"|{r['map']}/{review.NAMES[r['controller']]}|{r['map_score']:.4f}|{r['offline_local_union']['map_formula_value']:.4f}|{age}|{missing}|")
    doc+=['','离线并集忽略通信，不是官方成绩、可达成绩或理论上界；其值也会被额外假墙降低。Map08 E的并集68.7027说明本次有可供送达改善的地图内容；Map09 E并集仍仅46.5118，送达不是其唯一损失。格数按来源相加会重复，不是去重的真实漏墙面积。',
        '', '## 全图遗漏列表','',
        '8px保守自由栅格内按48px净空核心作障碍约束分区：Map08共5区、Map09共19区；以下ID在逐局分区图有对应坐标，完整包围框及每区观测/命中表见图册。分区不等于物理房间，保守栅格排除的近墙条带不另称未探索房间，狭小孤立区未证明物理可达。',
        '|地图/版本|未到访分区|零采样可见分区|采样可见不足50%|有墙边界但零实际端点记录分区|','|---|---|---|---|---|']
    for r in rows:
        nohits=[v['id'] for v in r['all_geometric_regions'] if v['wall_surface'] and not v['actual_hit_surface']]
        doc.append(f"|{r['map']}/{review.NAMES[r['controller']]}|{r['unvisited_region_ids']}|{r['zero_sampled_visibility_region_ids']}|{r['low_sampled_visibility_region_ids']}|{nohits}|")
    doc+=['','分区边界的墙面邻域可能重叠，不可把分区墙面计数直接相加。可见区仍是每8步4px、射线间插值的估计，零/低覆盖只用于定位证据缺口。',
        '', '## 采用结论与下一步','',
        '- D拒绝作为替换：Map08地图/墙体下降，虽然最大版本延迟635→175；Map09更严重退步。Map08七次返程四次获目标版本回执；Map09六次只有一次获回执，两次无进展、三次旧联络点同步超时。只记旧联络点和漂移路标不足以实现可靠返程。',
        '- E拒绝作为替换，保留入口研究：确有B/C进入和Map08墙体提升，但两图地图分大幅下降、Map09还有本机图质量损失。不能仅以进入帧或离线并集选优。',
        '- R拒绝作为替换：Map08只有103个停转控制步与4个通道平移步，地图/墙体/健康退步；Map09未介入，与基线相同，不是提升。',
        '- 下一步先在已有失败合法输入/路线记录上定位退路失效：区分路标坐标漂移、当前雷达真实阻挡、旧联络点已无队友。返程锚点需要可重新识别的局部几何与实际新鲜队友信息，再独立验证；在送达候选可靠前不组合E。入口候选的跨机任务广播和精细路径逐次日志属于待完善项。',
        '', '## 图件、原始结果与命令','',
        f'[逐局完整图册（每局各机轨迹、真实/提交/差异、炸弹和墙体图、全部入口任务、来源与完整回放命令）]({(OUT/"report.md").as_posix()})。',
        f'[机器可读完整审计]({(OUT/"summary.json").as_posix()})；[原始批次与源码快照]({(BATCH/"report.md").as_posix()})；[保护核验]({(OUT/"verification.json").as_posix()})。']
    for m in ['Map08','Map09']:
        doc+=['',f'### {m} 对照','']
        for kind,label in review.KINDS.items():doc.append(f'![{m}{label}]({(OUT/f"{m}_{kind}.png").as_posix()})')
    doc+=['','已实际运行的命令（已完成输出请勿复用）：','','```powershell',
        "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",
        '.\\.venv\\Scripts\\python.exe -B -m pytest src/swarm_rescue/solutions/devtools/test_red_link14.py src/swarm_rescue/solutions/devtools/test_red_lines11.py -q -p no:cacheprovider',
        '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_link14_batch --output output/sf13/rl14_m89_once_20261115 --maps Map08 Map09 --controllers rl11_lines rl14_delivery rl14_opening rl14_relative --rounds 1 --run',
        '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_link14_review --batch output/sf13/rl14_m89_once_20261115 --output output/sf13/rl14_m89_review_20261115','```',
        '', '每局回放：','']
    for r in rows:doc += [f"{r['map']} / {review.NAMES[r['controller']]}",'```powershell',f".\\replay_map.cmd '{Path(r['attempt']).resolve()}'",'```','']
    doc+=['本轮无蓝阶段，无蓝回放；蓝队版本只是原配置里的未执行字段，不称冻结强对手。',
        '', '后续只建议先分析这批失败，不扩大搜索。若用户需要原样复核，下列命令是8红0蓝、两图每版各1轮、同默认种子、新输出；本轮未自动执行第二批。删去--run只生成计划。',
        '', '```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",
        "$rl14Out = 'output/sf13/rl14_recheck_' + (Get-Date -Format 'yyyyMMdd_HHmmss')",
        "if (Test-Path -LiteralPath $rl14Out) { throw '输出目录已存在' }",
        '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_link14_batch --output $rl14Out --maps Map08 Map09 --controllers rl11_lines rl14_delivery rl14_opening rl14_relative --rounds 1 --run',
        "if ($LASTEXITCODE -ne 0) { throw '评测中断，保留所有失败产物' }",
        "$rl14Review = $rl14Out + '_review'",
        '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_link14_review --batch $rl14Out --output $rl14Review',
        "Invoke-Item (Join-Path $rl14Review 'report.md')",'```','',
        '未改100图及70/15/15划分，未提交、推送或打包。所有失败、低分和原始产物保留。本阶段结束。']
    with REPORT.open('x',encoding='utf-8') as f:f.write('\n'.join(doc)+'\n')
    # Strictly check all absolute embedded image targets in both new reports.
    for p in [REPORT,OUT/'report.md']:
        text=p.read_text(encoding='utf-8')
        for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):assert Path(target).is_file(),target
    print(json.dumps(dict(report=str(REPORT),protected=len(protected),png_files=len(pngs),valid=8),ensure_ascii=True))


if __name__=='__main__':main()
