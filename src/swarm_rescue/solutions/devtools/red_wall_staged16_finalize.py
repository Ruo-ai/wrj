"""Create a new RL16 delivery report; read-only verification of old files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def individual_layers(rows,review):
    import numpy as np
    from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
    from swarm_rescue.solutions.devtools.red_lines11_review import FONT
    from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper
    paths=[]
    for r in rows:
        if r['controller']!='rl16_staged':continue
        run=Path(r['attempt']);shape=read(run/'result.json')['expected_shape'];h,w=shape
        mapper=WallLineMapper((w,h));receipt=r['submission_receipts'][0]
        path=review/f"{r['map']}_B_individual_staging_delivery.png"
        if path.exists():raise FileExistsError(path)
        with np.load(run/'wall_staged16_diagnostics.npz') as z,np.load(run/'exploration_stages.npz') as received:
            fig,axes=base.plt.subplots(5,4,figsize=(23,24))
            fig.subplots_adjust(left=.015,right=.99,top=.94,bottom=.035,hspace=.24,wspace=.04)
            for ident in range(5):
                key=f"submitter_{receipt['submitter']}_source_{ident}"
                mask=mapper.to_pixels(received[key]) if key in received else np.zeros(shape,bool)
                seq=receipt['received_sources'].get(str(ident));age=None if seq is None else receipt['step']-seq
                images=[z[f'staged_{ident}'],z[f'validated_{ident}'],z[f'local_{ident}'],mask]
                titles=[f'{ident}号机：累计暂存入口投影',f'{ident}号机：已准入特殊区内容',f'{ident}号机：最终本机提交层',f'{ident}号机：提交端实收，版本年龄{age}步']
                for ax,img,title in zip(axes[ident],images,titles):
                    ax.imshow(img,cmap='Greys',vmin=0,vmax=1);ax.axis('off');ax.set_title(title,fontproperties=FONT,fontsize=11)
            fig.suptitle(r['map']+' · B 各机暂存、准入与实际送达',fontproperties=FONT,fontsize=23)
            fig.text(.5,.012,'暂存是累计合法观测的入口坐标投影。实收列为该来源在提交端的墙层，最终提交还会进行自由证据冲突否决。',ha='center',fontproperties=FONT,fontsize=12)
            base.save(fig,path);paths.append(path)
    return paths


def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    dev=Path('output/sf13/rl16_dev_20261116');before=read(dev/'protected_before.json')
    changed=[n for n,h in before.items() if not Path(n).is_file() or digest(n)!=h]
    audit=dict(protected_files=len(before),changed=changed,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    with (dev/'protected_after.json').open('x',encoding='utf-8') as f:json.dump(audit,f,ensure_ascii=False,indent=2)
    result=read(args.review/'summary.json');rows=result['rows'];calls=read(dev/'call_paths.json');plan=read(args.batch/'plan.json')
    native=read('output/sf13/rl16_native_20261116/results.json')
    extra_figures=individual_layers(rows,args.review)
    outcomes={c:[d for d in result['deltas'] if d['controller']==c] for c in ('rl16_wall','rl16_staged')}
    decision={c:('保留为后续验证候选，不切正式入口' if len(v)==2 and all(x['map_score']>0 for x in v)
        else '当前版本不采用；保留实验及失败/回退记录') for c,v in outcomes.items()}
    report=['# 复赛红队 RL16：单墙对齐 / 暂存地图验证，独立开发与六局实测','',
        '日期：2026-11-16。开发基线 RL11（rl11_lines）；本轮仅新增红队文件。A、B 分别实现、分别评测，没有组合 RL14-E、RC15 或 A+B。','',
        f"**A：{decision['rl16_wall']}。B：{decision['rl16_staged']}。** 结论仅限本轮两图单种子，完整蓝队难度与比赛总分未评测。",'',
        '## 1. 已核实的进度、原有机制与保护','',
        f"接手 HEAD `{audit['head']}`；原工作区有上一轮 RC15 的未提交新增文件，本轮完整保留。接手哈希清单共 {len(before)} 个既有文件，交付时变化：`{changed}`。未提交、推送、打包、切分支或正式入口。",'',
        '正式 my_drone_eval.py 仍以 MyDroneRedAdvanced2 为 place 入口；rl11_lines 通过红队专用 batch 注册到 MyDroneRedLines11。其实际继承链是 RN7-B 窄路 + S6 停转运动 + A3 提交地图 + advanced2 放弹/探索。RL11 墙段匹配只校正提交地图器，不替换导航位置。',
        'S6 已对每个实际里程计增量积分，包括起步、刹车、原地转向的残余平移；已有3次缺测确认、8次两传感器共同恢复确认。A没有重复实现这套定位器。',
        'A3 原缓存最多1400扫描，缺测片段恢复时重锚定；但 mask() 把高重复 pending 叠进对外地图，缓存不等于隔离。原快照按来源/序列取最新版本，导航/提交两层，提交侧并集，无接收ACK。',
        'RL13-E 用双侧墙入口引导；RL13-L 是48步/约0.18rad短块重投及24步入口提交保留，之后仍返回原 pending 提交行为。它不是本轮B的验证分层。RL14-D 有回执和退路，RL14-E/R 为另一个独立实验，本轮均未组合。',
        '已核对完整旧批：RL13-L Map08 63.1880、Map09 73.2211，RL11为63.1415/73.3676；RL14-D 为56.7935/46.8630，RL14-E为47.8816/43.3024。上一轮RC15小净空为46.4789/30.6557，不能因通过更多就采用。旧成绩仅交接证据，不代替本轮新实战。','',
        '基线及候选哈希：','', '|文件|SHA256|','|---|---|']
    for path,sha in calls['sha256'].items():report.append(f'|{path}|`{sha}`|')
    report+=['',f"完整源码冻结包 SHA256：`{plan['source_snapshot_sha256']}`。每局前由原评测器核对冻结源码，实际 MRO/方法调用文件记录在 `{(dev/'call_paths.json').resolve()}`。",'',
        '## 实测结论及下一步依据','',
        'A在Map08地图+1.8537、墙体+8.6666、时间+6.2698；但Map09地图−30.8522、墙体−8.8470。Map08虽然分数提高，G2下方入口和G3最右下入口仍未记录完整穿越。Map09 A进入上右房间297机帧、下方房间2704机帧，上左仍0；全图可见自由面积采样由1,637,754降至1,229,711px，失效区地图器位置误差均值30.13→41.63px，朝向7.88→13.65°。不同轨迹样本不可作为纯定位器消融，但明确不支持整体改善。',
        'B在Map08的五机GPS/罗盘/里程计/雷达输入和语义序列与RL11全部逐项一致，可以较干净地比较提交路径：误画受罚像素减少27,667，同时真墙命中减少2,914，地图仅+0.0348；灰区真墙命中也减少1,853。并非一味少画就更好。',
        'B在Map09地图−31.5707，不能全部归因于验证门槛。实际输入到约2807步前一致，2808步开始分叉；第2808步B进度0.551409、RL11为0.520185。原advanced2布弹逻辑的任务保持条件是进度<0.55，B此步记录额外bomb_target，基线没有。墙钟开销与原预算分支相互影响，原运动方法未改也会改变轨迹。B最终4226步、151.299秒，基线4451步、140.752秒。',
        '送达也没有解决：Map09提交时2号机版本，RL11为4380（年龄70步），A为810（年龄3651步），B为450（年龄3775步）。B所有本机最终图离线并集公式值65.1472，实际提交41.7968；这只是通信损失线索，不是官方成绩、可实现成绩或精确归因的23.3504分。A离线并集35.7658还低于其实际提交42.5154，说明当前本机图/历史快照的内容变化也重要，不能把所有损失都说成未送达。',
        'B实际跨机准入：Map08有1块、Map09有2块通过入口+独立队友检查；主要准入仍靠恢复锚定（74/16块）和短入口支持（8/3块）。截止高重复子集本轮没有触发。原生物理与单元测试证明功能能执行，不证明这些阈值已最优。',
        '下一步优先：在不改官方时限的前提下减少B的验证/快照开销，并用已保存合法观测做离线局部图消融；将未准入真墙与准入后未送达分开。A先审计入口动作改变任务/退路后的长期效果，避免把相对墙角通过等同全球定位准确。本轮不继续叠加A+B或自动参数搜索，等用户下一轮指示。','',
        '## 2. 实际接口与方案限制','',
        'NoGpsZone 同时关闭GPS、罗盘，合法读数为 None；禁通信由独立区域关闭。GPS位置噪声标准差5px、罗盘4°且为0.98相关噪声；里程计实际距离/方向/转角噪声分别0.2px/8°/1°。雷达181条360°射线、230px、2.5px噪声。记录上一时刻朝向只能提供有误差的初值，不能保证区内朝向或沿墙距离准确。',
        '原生机身为八边形 [(-13,-10),(-11,-12),(5,-12),(12,-11),(12,11),(10,13),(-10,13),(-13,10)]，0°宽25px，45°宽约32.53px，外接圆半径约16.40px。radius=10不是碰撞体。引擎每个控制步总物理积分为1个时间单位，不能把里程计再任意乘0.1。力300/质量50、旋转比例0.18rad/步；刹车基于实际里程计反馈，不假设瞬时停住。',
        '在线代码仅使用合法传感器、实际收到的信息与传统几何算法；真实位置、真实墙面、地图名和离线区域坐标仅在诊断工具中使用。未修改评分、物理、通信规则或共享定位。','',
        '## 3. A 新增机制与开关','',
        '`MyDroneRedWall16.SINGLE_WALL_ALIGN` 可关闭，关闭类 `MyDroneRedWall16Off`。只替换S6运动对象与入口调度；提交建图、原导航位置、探索目标和放弹均继承RL11。',
        '状态：IDLE → BRAKE → SELECT → TURN → VERIFY → MOVE → 原RL11。连续缺测3步、最近有可靠姿态、任务允许时触发；记录最近入口姿态、不确定性及墙段。停稳要求实际速度<0.45px/步、角增量<0.045rad连续3步。',
        '雷达拟合任意角度有限线段，不要求两侧墙。参考墙至少65px；连续3帧关联，法向距离<7px、角差<0.085rad、重叠>30px，候选关联接近多解时拒绝。围绕墙轴生成2平行+2垂直方向，与任务方向差≤0.9rad，结合近场净空选方向。',
        '检查真实八边形的平移/旋转连续空间近似：2px局部栅格，≤1px/3°采样，对扫掠凸包加6px保守膨胀（覆盖约2px额外净空及离散误差），并继承自由射线末端3px退缩。凸包会过度拒绝凹形空间，不把未知标自由；噪声无硬上界，不能宣称绝对安全。',
        '用实际转角搬运参考墙和累积目标角，连续墙观测闭环复测；容许相对角误差3°，连续3帧合格才动作，最多2次小修正。转向仍累计残余平移。短段目标速度3px/步、36px或28步结束，入口总时限125步；队友55px以内、墙段丢失、滑移/净空不足、任务抢占、角漂或恢复均有有界回退。RN7正在执行的窄路任务优先，A对齐时暂停新RN7接管，防止停稳指令被覆盖。',
        '本轮不是把无GPS区全程降速，也没有将墙相对方向当世界绝对方向；长走廊沿墙方向仍不可观测。','',
        '## 4. B 新增机制与开关','',
        '`MyDroneRedStaged16.STAGED_SUBMISSION` 可关闭，关闭类 `MyDroneRedStaged16Off`。运动/导航/放弹方法保持RL11；新增 StagedMapper 和私有四层快照协议。',
        '数据层为：原RL11可信hits层；特殊区入口局部坐标的扫描块；验证准入validated层。缺测新扫描从原pending移到局部块，避免pending无条件进入提交；本机即时雷达、原导航图、RL11墙段匹配参考继续运行，不因暂缓提交而看不见墙。既有可信图不整体平移。',
        '每块24次扫描（约96步）、最多24个封闭块及1个开放块，封闭块过期720步。检查前最多瞬时25块；容量处理后恢复上限。保存真实观测增量、入口锚、距离、匹配比、墙段与不确定性。私有位置风险量 sqrt(25+0.04×缺测年龄+(累计距离×4°)^2) 是启发式门槛，非标定概率或严格误差界。',
        '分级准入：①稳定恢复8帧、位置离散≤8px、角离散≤0.10rad、片段年龄≤600步、匹配比例≥0.35、闭合位置差≤70px/角差≤0.25rad，按现有 anchor_frames 重投本块；②入口有唯一重叠且匹配比≥0.6、位置风险≤18px，再加实际收到的独立绝对观测，两种非平行墙约束唯一匹配；③无第二机时，≤120步、风险≤12px、匹配比≥0.65且入口一致的短片段可以准入；④截止时仅保留风险≤18px、比≥0.75、入口双方向唯一一致、至少6次重复的子集。',
        '其他块到龄/容量/截止仍无证据则记录原因与格数，不强行提交。进度0.78及最终submit均进行截止处理；这是一套保守实验门槛，未用测试集调参。不是所有区域必须等第二架机，也不是所有缺测扫描一律丢弃。',
        '队友证据仅从实际收到的 red-staged16 包取得，要求源机自己连续8帧有效绝对定位生成、来源链仅该机、最多12墙段，拒绝自身/过期240步/重复或倒退版本/共同转发来源。每段检查法向距离、角差、有限重叠和次优差，平行走廊不能假装双向可观测。区内自身重复匹配不算独立真值。',
        '四层快照：原导航mask、原可信mask、新准入mask、可靠自由证据mask。保留原导航共享逻辑和来源序列去重；提交时队友实际收到的可靠自由证据可否决新准入墙，旧可信层沿用RL11。没有改官方无线规则。没有增加可靠ACK：报告只给离线观察到的提交端实际版本，不声称发送方知道已收到。',
        '限制：两机绝对传感器可能仍有相关偏差，来源过滤+唯一几何并不能证明完全独立；8帧稳定也不消除相关噪声。恢复初始7帧的原RL11正常区观测仍按基线处理；已经准入的短片段不做整张地图历史重优化。继承3px栅格、闭运算与小连通域过滤，最终像素效果不能等同原始墙点。','',
        '## 5. 实际测试、异常及公平性','',
        '合成合法观测行为测试25项实际通过（6.99秒）。覆盖任意斜墙/环绕、单墙、短墙/遮挡/多解拒绝、噪声、真实机身旋转扫掠、墙角、实际里程计残移、传感器分别恢复、动态队友和任务/观测/时限回退；地图覆盖错位/重复来源/平行走廊退化、无第二机、稳定恢复、容量与截止、自由冲突、快照版本。合成传感器测试不是完整实战。','',
        '|原生物理小场景|有效|噪声|实步|墙钟秒|墙接触|健康|复测通过|小修正|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in native:report.append(f"|{r['case']}|{r['valid']}|{r['noise']}|{r['steps']}|{r['seconds']:.3f}|{r['contacts']}|{r['health']}|{r['stats'].get('alignment_verified',0)}|{r['stats'].get('small_correction',0)}|")
    report+=['','原生理想/带噪单墙均实际转向、复测后移动。斜墙真方向约14.70°，通过时实际机身约12.93°/12.99°，仍有约1.7°误差，不宣称绝对对齐。short_noisy夹具还看到了边界长墙，实际选择了边界，因此不能把它当作“只有短墙”拒绝的证据；短墙拒绝由独立合成测试覆盖。',
        'Map08 128步五机冒烟：RL11、A、B、A关闭、B关闭各一次，均无回调/崩溃/诊断异常；两个关闭版本的完整合法输入数组与RL11逐项一致。冒烟无正式成绩意义。中断发生在原生测试完成后、补充测试写入前；已核查未发生部分文件覆盖，原生测试未重复运行。',
        '完整评测前检查可见进程及近期输出，未发现其他活动评测；6局由同一父进程顺序调度，单CPU/4GB，上限5400步/180秒。Map08 seed 200806601；Map09 seed 2477227837；每版每图1轮，每局仅attempt_001。源码快照、请求/收据/资源、原始日志/回放/图和所有异常记录均保留。',
        '同种子不保证相同输入；summary.json逐机比较实际GPS/罗盘/里程计/雷达数组及语义序列，给出首次差异步。A确实改变运动；B没有改运动方法，但必须按实测检查其输入、墙钟与截止是否变化。诊断时间计入实际资源，不隐藏开销。',
        '离线图册初版失败：把所有受罚误画误当单一深红色，实际官方采用浅红到深红梯度，导致6行校验失败（工具退出码虽0也未视为有效图册）。原目录 rl16_m89_review_20261116 完整保留；修正为直接调用官方 distance_field 与 dead_zone，逐局受罚像素计数和原分数全部一致，输出新目录 rl16_m89_review_v2_20261116，比赛未重跑。',
        '测试重跑命令（行为测试不生成或覆盖评测结果）：','', '```powershell',
        "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",
        '.\\.venv\\Scripts\\python.exe -B -m pytest src/swarm_rescue/solutions/devtools/test_red_wall_staged16.py -q -p no:cacheprovider','```','',
        '## 6. 新增文件、回退与下一批命令','',
        '新增候选：my_drone_red_wall16.py、my_drone_red_staged16.py；新增专用工具：devtools/test_red_wall_staged16.py、red_wall_staged16_native.py、red_wall_staged16_diagnostics.py、red_wall_staged16_batch.py、red_wall_staged16_review.py、red_wall_staged16_finalize.py。所有Python均在 src/swarm_rescue/solutions/ 下。旧源码、正式入口、蓝队及其输出、共享文档、地图、提交包不变。',
        '回退选择 rl11_lines 或候选off类即可；本轮未把任何候选接入正式入口。未运行A+B，避免在单项尚未证实稳定前混合归因。后续多图/多种子由用户启动，保留70/15/15划分，不重新生成100图，不以测试集反复调参。',
        '下面是可复制的同条件复核命令：Map08/09、3版本、各1轮，共6红0蓝；自动产生全新输出目录。需要空闲时手动运行，不是本轮自动追加。', '', '```powershell',
        "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",
        "$out = Join-Path 'output\\sf13' ('rl16_m89_manual_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))",
        "if (Test-Path -LiteralPath $out) { throw '输出目录已存在' }",
        '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_wall_staged16_batch --output $out --maps Map08 Map09 --controllers rl11_lines rl16_wall rl16_staged --rounds 1 --run',
        "$fig = $out + '_figures'",
        '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_wall_staged16_review --batch $out --output $fig','```','',
        '各局完整回放命令和全部图件见下文；本轮无蓝队对局，因此无蓝回放。待合并进共享总方案的内容：A实际触发和回退结果、B分层提交与真实收到版本的区别、六局官方分项及候选结论。当前仅写此新红队报告。','',
        '## 7. 本轮逐局官方分项与完整图册','']
    report.append((args.review/'report.md').read_text(encoding='utf-8'))
    report+=['','## B各机暂存与实际送达补图','']
    report.extend(f'![各机暂存与送达]({p.resolve().as_posix()})' for p in extra_figures)
    with args.report.open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    print(json.dumps(dict(report=str(args.report.resolve()),decisions=decision,protected=audit),ensure_ascii=False))


if __name__=='__main__':main()
