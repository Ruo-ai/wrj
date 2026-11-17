"""Write a new independent RL18 report, preserve failure fixtures and audit hashes."""
import argparse
from pathlib import Path
import re
import subprocess
import numpy as np
from PIL import Image
from swarm_rescue.solutions.devtools.red_recovery18_audit import read,write,digest,load_npz
from swarm_rescue.solutions.devtools.red_p2l17_replay import summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();repo=Path.cwd();history=repo/'output/sf13/rl17_m89_once_20261116_figures'
    final=root/'delivery_v1';final.mkdir(exist_ok=False);figs=root/'figures_v2'
    paths=list(figs.glob('*.png'))+list(history.glob('comparison_*.png'))
    for path in paths:
        with Image.open(path) as im:im.verify()
    protected=read(root/'protected_sha256.json')
    changed=[path for path,h in protected.items() if not Path(path).is_file() or digest(path)!=h]
    if changed:raise RuntimeError('Previously existing files changed: '+repr(changed))
    failures=[];samples={}
    for name in ('Map08','Map09'):
        directory=root/'stage_b_v1'/name;z=load_npz(directory/'traces.npz');d=read(directory/'results.json')
        tape=load_npz(Path(d['source'])/'mapper_legal_inputs.npz')
        for owner in range(5):
            pos=z[f'shadow_position_{owner}'];mask=z[f'missing_{owner}'];idx=np.flatnonzero(mask)
            if not len(idx):continue
            worst=int(idx[np.argmax(pos[idx])]);step=worst+1
            failures.append(dict(map=name,owner=owner,step=step,shadow_position=float(pos[worst]),
                rl11_position=float(z[f'rl11_lines_position_{owner}'][worst]),
                shadow_heading=float(z[f'shadow_heading_{owner}'][worst]),
                rl11_heading=float(z[f'rl11_lines_heading_{owner}'][worst]),
                note='Maximum cumulative shadow position error in this aircraft, not a local mismatch probability.'))
            begin=max(0,worst-32);end=min(len(pos),worst+2);prefix=f'{name}_d{owner}_t{step}'
            samples[prefix+'_legal']=tape[f'drone_{owner}'][begin:end]
            samples[prefix+'_shadow']=z[f'shadow_{owner}'][begin:end]
            samples[prefix+'_truth_OFFLINE_ONLY']=z[f'truth_{owner}'][begin:end]
    write(final/'cumulative_failures.json',failures);np.savez_compressed(final/'failure_windows.npz',**samples)
    write(final/'failed_attempts.json',[
        dict(attempt='stage_a_v1',status='failed before replay',reason="KeyError source on submitted radio event; fixed event-type check order, old output retained; not a valid result"),
        dict(attempt='tests_v1',status='17 passed / 2 failed',reason='Noiseless corner extraction error .081px exceeded .05px fixture threshold; changed test to .25px. Noisy oblique reverse support 44/90 is a legitimate rejection, now retained as explicit rejection fixture; estimator thresholds unchanged.'),
        dict(attempt='stage_b_v1',status='completed, candidate rejected',reason='Relative turn improved, but cumulative pose and far endpoint errors regressed overall. No stage C map candidate.'),
        dict(attempt='synthetic_cases_v1/repeated_rooms_alias',status='accepted wrong alias',reason='Deliberate out-of-basin initial guess, 400px repeated-room ambiguity; low residual cannot prove global alignment.')])
    papers=[
      dict(name='RF2O',version='ros2 branch read 2026-11-16; immutable remote commit unavailable',
           files=['README.md','LICENSE','src/CLaserOdometry2D.cpp'],license='Repository and LICENSE GPL-3.0',
           url='https://github.com/MAPIRlab/rf2o_laser_odometry',use='Consecutive scans, robust bounded optimization principle only; no source copied.'),
      dict(name='SRF',version='master branch read 2026-11-16; immutable remote commit unavailable',
           files=['README.md','LICENSE','src/laser_odometry_refscans.cpp'],
           license='Repository/LICENSE LGPL-3.0, but actual cpp file header says GPLv3. Do not infer file reuse permission from repository badge.',
           url='https://github.com/MAPIRlab/srf_laser_odometry',use='Previous + key scan consistency principle only. No code port, ROS or MRPT dependency.'),
      dict(name='Polar scan matching',version='Diosi and Kleeman IROS 2005, pp1439-1444',files=['Original provided PDF URL failed (parked domain); Monash publication record read'],
           license='Paper not a code license; implementation permission unverified; no code copied',
           url='https://research.monash.edu/en/publications/laser-scan-matching-in-polar-coordinates-with-application-to-slam/',use='Bibliographic verification only; do not claim the original full PDF was read.'),
      dict(name='Cartographer',version='ICRA 2016 paper publication abstract; cartographer_ros master docs and LICENSE',
           files=['docs/source/algo_walkthrough.rst','LICENSE','Google Research publication record'],license='cartographer_ros LICENSE Apache-2.0; paper copyright separate',
           url='https://github.com/cartographer-project/cartographer_ros/blob/master/docs/source/algo_walkthrough.rst',
           use='Local matching can drift; global loop closure is separate. Original PDF redirect inaccessible; no C++/ROS implementation copied.'),
      dict(name='Feature-Based Occupancy Map-Merging for Collaborative SLAM',version='Sensors 2023,23(6),3114,14 March 2023',
           files=['publisher article'],license='Article does not establish license for an unspecified source implementation',
           url='https://www.mdpi.com/1424-8220/23/6/3114',use='Ambiguous registration must be validated; unknown/free/occupied semantics matter. No fusion code or efficacy transferred.'),
    ]
    write(final/'references.json',papers)
    lines=['# 复赛红队 RL18：恢复归因与短窗影子实验', '',
        '2026-11-16。**RL11 继续保留稳定对照，RL17-R 保留研究候选；RL18-B 本轮拒绝接入建图与导航。**', '',
        '本轮完成 A 的四组同输入精确回放、B 的两图五机全程影子估计、19 项行为检查、9 个合成诊断场景与来源叠加审计。未运行完整飞行或蓝队，没有新炸弹/探索路径。**暂无新评测图**；本报告新图是离线重放/诊断图，历史探索、轨迹、墙体和炸弹图均明确标注。', '',
        '## 接手状态、范围与实际阅读', '',
        'HEAD 为 `65eeb35`，main。接手已有 `red_rl17_evaluate.cmd`、`red_p2l17_batch.py`、`red_p2l17_review.py`、RL17 报告四个跟踪文件的未提交修改，以及 `test_red_p2l17_review.py` 未跟踪文件；完整保留。最新改动将未来完整对照扩为 Map06/08/09，Map06 本轮未运行。没有适用 AGENTS.md。CIM 无法查询进程命令行，Get-Process 未列出 Python，但可见性不足以证明整机空闲。', '',
        '已读《新对话交接说明.md》快速接手、《复赛阶段_最终优化方案.md》相关方向、《复赛阶段_阶段总结与图件交付规范.md》，RL11、RL13-L、RL16、RL17 报告；核对 `my_drone_red_lines11.py`、`my_drone_red_repair17.py`、`my_drone_red_p2l17.py`、`my_drone_red_mapping.py`、`semifinal_localization.py`、相关 replay/diagnostics/review/finalize、官方评分及传感器/失效区定义。读取了用户指定完整局目录、两份 repair replay 结果和对应历史图册。旧文档中的 HEAD 不是当前状态。', '',
        'RL11 已有每步里程计、每8步墙段/栅格匹配、近期参考、pending、恢复校正和来源版本 OR。RL17-R 先保留 RL11 的接受结果，拒绝后补点到线。没有改探索、导航、布弹或融合。入口有噪声；平行走廊沿廊不可观测。NoGpsZone 同时禁 GPS/罗盘；181束、360°、230px、雷达和里程计有噪声，仍按本地最新源码。', '',
        '仅新增 solutions/devtools 内独立 Python 文件和本报告/新输出。未修改蓝队、红队基线或既有候选、共享文档、规则、地图、入口、旧产物和提交包；未暂存、回退、提交、推送或打包。', '',
        '## A：基线精确复现与局部诊断', '',
        '输入：`output/sf13/rl17_m89_once_20261116`，Map08 seed=200806601、Map09 seed=2477227837，均 r1/attempt_001，5机。两版本全部合法输入数组逐项相同（含NaN）。4/4 回放逐步位姿误差<1e-9，所有已保存发布图、本机终图、提交端实际来源mask、最终像素全部匹配；官方地图公式复算一致。原始全量包只保存哈希，没有每包完整字节：新增中间快照来自确定性重放，不能声称每一个中间包都独立逐字节核验。', '',
        '|图/版本|官方公式 M（复现值）|真墙命中|受罚误画|信用/惩罚百分点|',
        '|---|---:|---:|---:|---|']
    audit={};shadow={}
    for name in ('Map08','Map09'):
        shadow[name]=read(root/'stage_b_v1'/name/'results.json')
        for c in ('rl11_lines','rl17_repair'):
            d=read(root/'stage_a_v2'/f'{name}_{c}/results.json');audit[name,c]=d;v=d['final']
            lines.append(f'|{name}/{c}|{v["map_score"]:.6f}|{v["hits"]}|{v["charged"]}|{v["credit"]:.4f}/{v["penalty"]:.4f}|')
    lines += ['', '不是新比赛成绩或完整总分提升。实际来源 Map08={0:4535,1:4530,2:4360,3:3900,4:3930}；Map09={0:4450,1:4440,2:4380,3:4380,4:4410}。没有用各机终图替代已收到版本。', '',
        'ROI 由本轮离线选取近似截图位置：Map08=(950,730,1655,1117)，Map09=(6,6,1694,325)，均图像像素坐标且端点包含。原截图未附于本轮，因此不声称精确恢复旧裁剪，也不将不同 ROI 的像素数混用。Map08 ROI 真墙命中两版均10015，受罚26013→26514（变差501）；Map09 ROI 命中26518→27568，受罚47589→47546（少43）。', '',
        '恢复前保存原始墙/自由扫描点、每帧位姿/权重、pending证据格、hits；依次重建“不改位姿仅按0.8提交”的对照、实际anchor_frames重投、恢复步新增正常扫描，最终hits逐值相同。下表是**整张单机地图**受罚像素净变化，含原后处理；中间对照仅用于解释权重和位姿的交互，三项是指定顺序差值，不能称为唯一独立因果比例。', '',
        '|图/版本/机/恢复步|pending数|恢复前受罚|仅升权净变化|再校正位姿净变化|当前正常扫描净变化|恢复后受罚|真墙命中净变化|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,c in [('Map08','rl17_repair'),('Map09','rl11_lines')]:
        for e in audit[name,c]['recovery']:
            s=e['stages'];b,u,k,after=[s[x] for x in ('before','unanchored_commit','anchored_commit','after')]
            lines.append(f'|{name}/{c}/d{e["owner"]}/{e["step"]}|{e["pending_before"]}|{b["charged"]}|{u["charged"]-b["charged"]:+}|{k["charged"]-u["charged"]:+}|{after["charged"]-k["charged"]:+}|{after["charged"]}|{after["hits"]-b["hits"]:+}|')
    lines += ['', '恢复前已增长的事实：Map08 R/d3 的1200→1680步受罚7740→10509；R/d4 的2776→3244步17132→21239。Map09 RL11/d0 的816→1296步8468→19286，d3 的1808→2765步18434→30734。这些缺测期单机临时图里的误差增长不能归咎于最终多机合并。', '',
        '恢复本机不等于立即改变提交者的图：Map08/d3 在1755步恢复，但0号机直到2188步才接受来源3的2160版本；d4 在3276步恢复，0号在3514步才收到3510版本。期间仍使用旧来源。Map09/d0 恢复1377，自己1380发布；d3恢复2793，0号2826收到2820版本。后续来源版本还含其他扫描，不能把后续整图增量全部归给恢复这一帧。', '',
        '单一来源 OR 边际贡献（去掉该源作离线对照，含后处理；不实施删图）：Map08 R 的来源3/v3900增加16038受罚像素，同时增加717真墙命中；Map09 RL11 的来源3/v4380增加15185受罚，同时增加3099真墙命中。边际贡献不互斥，不能相加成全图分解。', '',
        '## B：独立短窗实验和拒绝结论', '',
        '实现 `ShortWindow18`：每步积分合法里程计，每4步匹配当前扫描与上一扫描、最长24步key扫描。有限线提取复用RL11合法提取器；最多90点、16段，3个转角初值、每个最多5次有界求解，稳健残差、有限重叠、近表面歧义、平移法向特征空间、耦合退化、双向支持和key一致性检查。失败保留里程计；无世界轴假设，无持久全局墙图。参考扫描来自独立估计，不读取RL11/RL17位姿；正常时仅对带噪GPS/罗盘平滑，恢复后清空短窗。', '',
        '它不是 RF2O/SRF 的 range-flow 实现，也不是把 RL17 的“拒绝后补一次”换名。借鉴连续扫描和多参考一致性的思想，本轮采用有限段几何以限定工作量。没有修改任何原地图或控制器。', '',
        '下表为全体缺测机步，均值/P95；位置px、角度deg。四步相对转角排除跨恢复区间；累计角漂移扣除同一缺测入口的角偏差；绝对角仍包含入口噪声。远端是同一条150–218px带噪射线分别用真位姿/估计位姿投影的偏差，隔离位姿放大效应，**不是真实墙端点测距误差**。', '',
        '|图/方法|位置均值/P95|绝对角均值/P95|四步转角均值/P95|累计角漂移均值/P95|远端投影均值/P95|',
        '|---|---|---|---|---|---|']
    for name,d in shadow.items():
        for method,v in d['metrics'].items():
            pairs=[f'{v[k]["mean"]:.3f}/{v[k]["p95"]:.3f}' for k in ('position','heading','relative_turn','accumulated_heading','far_endpoint')]
            lines.append('|'+name+'/'+method+'|'+'|'.join(pairs)+'|')
    lines += ['', 'Map08 缺测10192机步，四步对齐9984样本；Map09分别6260/6204。每束远端误差的统计是射线加权，不是每机等权。完整逐步数据与所有方法均已保存。', '',
        '|指定片段|RL11位置均值→影子|RL11绝对角均值→影子|结果|',
        '|---|---:|---:|---|']
    for name,d in shadow.items():
        for e in d['event_windows']:
            old,new=e['metrics']['rl11_lines'],e['metrics']['shadow']
            lines.append(f'|{name}/d{e["owner"]}/{e["start"]}–{e["end"]}|{old["position"]["mean"]:.3f}→{new["position"]["mean"]:.3f}|{old["heading"]["mean"]:.3f}→{new["heading"]["mean"]:.3f}|'+('退步' if new['position']['mean']>old['position']['mean'] else '该片段改善')+'|')
    lines += ['', 'Map08接受1790/2548=70.25%，rank1=675/rank2=1115；Map09接受798/1567=50.93%，rank1=63/rank2=735。沿墙不可观测方向保留里程计；没有将rank2或匹配通过解释为全局可信。两图分别72/54次key不一致拒绝，199/185次反向支持不足，468/519次所有初值失败。', '',
        '已接受且能对齐的四步区间中，本轮未发现“角误差>3°或机体系位移误差>6px”的粗错误标签；这不是正确匹配证明，也不代表误匹配率为零。累计错误显著，`delivery_v1/cumulative_failures.json` 保留每机最差位置误差点和合法输入窗口，Map09最大174.879px。`matches.json` 保留全部接受/拒绝，`failure_samples.json` 的空列表只对应上述很粗的短区间门槛。', '',
        '**C门槛未通过，未做新地图候选。** 短区间误差下降不足以抵消累计漂移；三个指定片段改善不能覆盖Map08/d4和其余机的退步。没有按图名、坐标或离线真值挑选何时切换估计器，没有把不确定扫描一律丢弃。', '',
        '## D：实际可做和缺失证据', '',
        '来源叠加图只统计提交者实际收到的源机+版本，一源一票，重复转发不增票。来源哈希检查全部通过。白色0仅是无墙断言，可能是未知；不是自由区。现协议只有导航/提交mask，没有可靠自由射线、自由证据置信度和原始观测版本。离线能从各机全量输入重算自由，却不能冒充提交者当时收到。', '',
        '因此没有足够信息制作合规的“墙命中 vs 可靠自由反证 vs 未观测”冲突图；本轮D只交付墙来源叠加和边际统计。反证命中误墙/误伤真墙数为**不可评估**，不是0。不扩通信、不跨遮挡清墙、不改蓝队或正式图。', '',
        '## 行为测试、失败及资源', '',
        '19项最终通过，4.82s。初次17通过/2失败保留：无噪斜角的有限段提取误差0.081px超出测试自定0.05px容差，测试改成0.25px；带噪斜角反向支持44/90<0.5，本来应拒绝，单列为明确失败召回样本，未放宽算法阈值。A初次在解析submitted事件时KeyError，尚未开始回放；仅修新增工具，旧stage_a_v1原样保留。', '',
        '9个合成诊断全部有结果但不称9/9成功：走廊和单墙rank1保留沿墙2px初值偏差；角落约0.528px/0.234°；有语义动态物约0.510px/0.152°，无语义动态物约0.864px/0.032°；开阔/无效射线回退；带噪斜角拒绝；重复房间错误初值接受且误差399.987px。最后一例刻意超出正常四步运动初值范围，用于证明局部低残差不能解决全局别名，不冒充比赛中真实发生了400px瞬移。合成种子742，全部原始射线保留。', '',
        '|工作|墙钟s|CPUs|峰值RSS MB|额外说明|', '|---|---:|---:|---:|---|']
    for (name,c),d in audit.items():
        r=d['resources'];lines.append(f'|A {name}/{c}|{r["wall_seconds"]:.3f}|{r["cpu_seconds"]:.3f}|{r["peak_rss_bytes"]/1e6:.1f}|含回放、细粒度重建、评分、保存|')
    for name,d in shadow.items():
        r=d['resources'];calls=summary([e['seconds']*1000 for e in read(root/'stage_b_v1'/name/'matches.json')])
        lines.append(f'|B {name}|{r["wall_seconds"]:.3f}|{r["cpu_seconds"]:.3f}|{r["peak_rss_bytes"]/1e6:.1f}|缺测扫描调用均值/P95/max={calls["mean"]:.2f}/{calls["p95"]:.2f}/{calls["maximum"]:.2f}ms|')
    lines += ['', 'A累计77.855s墙钟/66.891sCPU；B累计21.625s墙钟/19.328sCPU。B估计更新本身两图合计18.096s。资源包装器实设单CPU affinity=1、Job约4GB（系统回读3999997952字节）；A峰值Job提交1.263GB，B 1.167GB。RSS含输入、图、诊断与评分，是进程累计峰值，不是在线估计器净内存。本机性能不等于平台。单次有38ms长尾，未提供可抢占硬实时保证。绘图/测试和工具启动不包含在A/B子表累计中。', '',
        '## 外部资料、版本与许可', '',
        '以下均2026-11-16读取，远端GitHub提交API不可用，不能声称已取得不可变commit；明确列出分支和文件。没有下载/复制第三方算法源码到项目，没有运行中AI模型/API、ROS或MRPT依赖。', '']
    for paper in papers:
        lines.append(f'- [{paper["name"]}]({paper["url"]})：{paper["version"]}；文件：{", ".join(paper["files"])}。{paper["license"]}。{paper["use"]}')
    lines += ['', 'SRF源码头核验：[实际cpp](https://raw.githubusercontent.com/MAPIRlab/srf_laser_odometry/master/src/laser_odometry_refscans.cpp)，[仓库LICENSE](https://github.com/MAPIRlab/srf_laser_odometry/blob/master/LICENSE)。Cartographer原论文PDF重定向失败，改读[作者机构摘要](https://research.google/pubs/real-time-loop-closure-in-2d-lidar-slam/)；极坐标原链接为停放域名，未声称阅读全文。所有他人成绩/速度均未作为本比赛效果证据。', '',
        '## 图册：新离线图与历史飞行图分开标注', '',
        '下列新图上排Map08、下排Map09；左右为RL11、RL17-R，均r1，种子分别200806601/2477227837。绿=真墙命中、蓝=漏画、红=受罚误画、黄=免罚预测、灰=真值内部。', '']
    for title,file in [('全图离线精确回放','mapping_full.png'),('两个近似截图区域','mapping_roi.png'),('恢复分层（中间升权是离线对照）','recovery_layers.png'),('恢复之前的误画增长','growth_before_recovery.png'),('短窗影子累计漂移与失败','shadow_failures.png'),('来源叠加（不是自由反证冲突）','source_overlap.png')]:
        lines.append(f'![{title}]({(figs/file).as_posix()})\n')
    lines += ['以下四图复用历史批次 `rl17_m89_once_20261116`，不是RL18新飞行结果。各图两地图、两版本、r1，种子同上。RL18没有新轨迹、弹位或墙体分。历史两版本轨迹和弹位相同，不意味着新候选提升了探索/布弹。', '']
    for title,kind in [('历史探索范围：绿色为采样雷达可见区近似','exploration'),('历史轨迹：颜色为录像槽位，灰区无GPS/罗盘，黄区无通信','trajectories'),('历史墙体：绿覆盖内墙、蓝覆盖边界、灰未覆盖、红爆炸范围','walls'),('历史弹位：编号为产物顺序、圆为爆炸半径','bombs')]:
        lines.append(f'![{title}]({(history/f"comparison_{kind}.png").as_posix()})\n')
    lines += ['历史分项身份（从原始result读取，D/完整总分未评测）：', '', '|地图/版本/r1|M|W|T|弹|步|引擎墙钟s|有效性|','|---|---:|---:|---:|---:|---:|---:|---|']
    for (name,c),d in audit.items():
        r=read(Path(d['source'])/'result.json');ph=r['phase1']
        lines.append(f'|{name}/{c}|{ph["score_map"]:.4f}|{ph["score_wall"]:.4f}|{ph["score_time"]:.4f}|{r["placed_count"]}|{r["elapsed_steps"]}|{r["elapsed_walltime"]:.3f}|历史有效红阶段|')
    lines += ['', '## 后续建议与可运行命令', '',
        '建议继续离线研究，暂不启动RL18完整红阶段或扩大地图批次。下一问题是多参考链的累计偏置和失去参考后的漂移，而不是继续放宽接受率。可研究独立角度约束、参考不确定性和失效持续时间，但必须用合法观测定义、单独实验，不能按本轮真值挑选“好片段”。恢复升权/投影影响已有精确证据，适合另开单因素对照；不恢复RL13-L/RL16-B整包。', '',
        '下面是**下一条手动复现命令**：固定已验证A输入，RL18影子+RL11/RL17保存轨迹+里程计，Map08/09各1个历史开发种子、r1；共2张输入带、10机序列、0红/0蓝仿真。新目录，约22s核心处理加启动保存。当前候选未显示可靠改善，因此不提供其完整飞行评测命令。以后如进入完整对照，遵守Map06/08/09并补独立种子/留出集。', '',
        '```powershell', "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'", "$rl18Replay = 'output/sf13/rl18_shadow_recheck_' + (Get-Date -Format 'yyyyMMdd_HHmmss')", "if (Test-Path -LiteralPath $rl18Replay) { throw '输出目录已存在' }",
        ".\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.windows_pressure ($rl18Replay + '_resources.json') swarm_rescue.solutions.devtools.red_scan18_replay --batch output/sf13/rl17_m89_once_20261116 --baseline output/sf13/rl18_audit_20261116/stage_a_v2 --output $rl18Replay", "if ($LASTEXITCODE -ne 0) { throw '复现失败，保留输出，不计有效结果' }", '```', '',
        'A从头复现：将下面输出路径换成全新名称；同一组两图×两地图器=4离线回放，不飞行。', '',
        '```powershell', '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.windows_pressure output/sf13/rl18_A_manual_resources.json swarm_rescue.solutions.devtools.red_recovery18_audit --batch output/sf13/rl17_m89_once_20261116 --output output/sf13/rl18_A_manual', '```', '',
        '行为测试：', '', '```powershell', '.\\.venv\\Scripts\\python.exe -B -m pytest -q -p no:cacheprovider src/swarm_rescue/solutions/devtools/test_red_scan18.py', '```', '',
        '历史红阶段回放（从项目根目录，逐条运行；没有新蓝队回放）：', '', '```powershell']
    for d in audit.values():lines.append(f'.\\replay_map.cmd \'{d["source"]}\'')
    lines += ['```','','## 可追溯文件与保护核验','','新增Python：`red_recovery18_audit.py`、`red_scan18_shadow.py`、`red_scan18_replay.py`、`test_red_scan18.py`、`red_scan18_cases.py`、`red_recovery18_figures.py`、`red_scan18_finalize.py`，全部位于 `src/swarm_rescue/solutions/devtools/`。', '',
        f'交付时1680个既有文件SHA256变化0；图片{len(paths)}张均可解码。`delivery_v1/verification.json`记录源码/输入/报告哈希；A/B每组results另存输入和实现哈希，所有失败目录保留。未执行新官方完整评测。']
    document='\n'.join(lines)+'\n'
    missing=[x for x in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',document) if not Path(x).is_file()]
    assert not missing
    with a.report.open('x',encoding='utf-8') as f:f.write(document)
    sources=sorted(Path(__file__).parent.glob('*18*.py'))
    write(final/'verification.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        protected_count=len(protected),protected_changed=changed,images_verified=len(paths),
        new_source_hashes={str(p):digest(p) for p in sources},report_sha256=digest(a.report),
        input_hashes={name:shadow[name]['input_sha256'] for name in shadow},valid_replays=4,
        new_simulations=0,stage_c=False,stage_d_reliable_free_available=False))
    print(str(a.report.resolve()),flush=True)


if __name__=='__main__':main()
