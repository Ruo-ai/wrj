# 蓝队逐弹诊断与闭环优化（B4）

2026-11-13。本阶段已完成8局完整检查，7局有效、1局回调异常；未达到强蓝队目标，B0与正式入口不变。最终采用结论见本文结果表；所有原始结果保留。当前HEAD为09597d2，文档中旧6557940不是当前仓库版本。B0、本地参考入口、B2、B3、红A0/N0/A3和正式入口均未改；正式rescue仍是MyDroneRescueExample。B3的PAYLOAD_RECOVERY=False，transport末端开启、transport_compute末端关闭。规则以实际引擎为准：语义130/激光230/通信180/生命25，开发指南部分示例仍是初赛旧值。

固定现有oracle_layouts_v2强布局，离线22px空载连通净空、16px候选步长、逐弹加权墙面积贪心、10弹、最小弹间距32px；Map08爆炸半径135、边界墙权重0.5、加权覆盖62.9954%。不是已证实最难/全局最优布局。在线策略不访问布局、地图名、真实位姿、实体ID或隐藏处置计数。

官方炸弹与配对处置中心碰撞即处置，抓持时也能成功；释放不是必要条件，也不等于成功。公开开发API未提供逐弹官方处置回执，在线对载荷消失只记未确认，由离线官方标志核验。不得通过读取bomb.disposed或reward内部值伪造在线闭环。

官方墙钟从GuiSR记录，elapsed_walltime超过180即终止并封顶180；5400步与180秒取先触发。全弹完成可提前正常结束。外层240秒只允许启动/退出/导出，不放宽比赛墙钟。0退出码但超时、控制异常或回调异常仍无效。蓝分=0.6处置百分比+0.2健康分+0.2时间分，未全处置则时间0；没有红运行，所以强布局组红M/W/T/D/完整总分均不适用。

## 已证实的问题

新参考批b4_refs同Map08/r1、种子200806601，三局均有效：B0 21.44/2处置/173.23秒；compute 21.44/2/171.60秒；transport 6.56/0/164.21秒。B0/compute逐弹相同：7颗已发现并抓取，3颗未探测；4颗返程未完成，1颗未确认脱手。6/9号最后部分路径距离目标约561/239px，携弹末尾位置长期局限在约58/56px范围。不是所有未完成都发生在处置中心。

导航函数累计B0 17.99秒、compute 10.11秒；整体控制28.68→21.00秒，但整局只差1.63秒，不能夸大整局加速。离线观察器累计约0.96～0.99秒；这不等于全部诊断成本，包装器开销也在总墙钟中。没有扣除诊断时间制造有效成绩。

旧日志无法回溯准确逐弹发现；新增开关通过公开语义存在性与离线渲染ID对齐，识别每颗弹，真值不进入在线控制。认领按校正后目标在60px内唯一匹配，歧义标未知。记录抓取/脱手/中心接近/释放/官方处置及实际最终命令来源。采样每30步、状态变化限频，无逐步stdout。死亡机最后帧重复问题已修复；首批报告按机号/控制步去重，旧phase_steps不能当准确时长；首次事件、得分及实际控制不受影响。

## 独立机制及当前结论

- boundary：只改携弹不完整路径的有界沿障恢复。120步/36px停滞触发，当前激光43px裕度，720步退出、60步冷却；旧B3恢复仍关闭。实测有效11.36分/1处置/162.47秒，6颗发现、3颗最终未确认脱手，健康26.8；比B0退步，拒绝采用。移动范围增加不等于返程成功。
- contact：独立末端滞回、短失见容忍、实际转角、受阻保留导航、有限重试；公开API不能确认处置成功，任何释放保持未确认。单独检查中，不与boundary组合。
- semantic_search：仅改探索覆盖与搜索目标，区分激光已知空地和语义搜索机会。覆盖仅在语义可用且当前激光无遮挡的118px保守范围，同行通信只传播实际收到的覆盖；抓取/返程路线不变。原三颗未发现与230/130量程差提供动机，是否有效待完整局。

多机任务分配评估：B0仅欧氏最近且忙于其他目标的机也参与，存在不必要抑制风险；本轮已发现7颗均有抓取，优先修运输和漏搜，暂不追加拍卖/全局规划。现有PEER_TTL=90、BOMB_TTL=700与消息时间戳保持，无通信独立行动，断联仍可能重复任务。载荷净空只用保守43px局部圆形裕度，并非铰接载荷扫掠可行性证明；新沿障单因素失败，不默认启用。多人堵中心尚无专用预约方案，先验证最终控制不覆盖避障，不能声称拥堵已经解决。GPS/罗盘缺测仍用B0里程计，未解决长时间漂移，不恢复B2整包可信度冻结。

## 已运行检查

诊断/既有工具36项通过（2.82秒）；128步出生点单弹诊断通过、无异常，仅链路检查，不算成绩。boundary/contact/诊断/既有工具51项通过（2.89秒）。语义搜索测试及完整评测待完成后追加，未运行前不声称通过。

截至单因素前两局：boundary有效11.36/1处置、健康26.8、162.47秒，拒绝；contact有效19.84/2处置、健康39.2、153.85秒，仍低于B0 21.44/健康47.2，拒绝。contact发现8颗，但第8号从首次中心附近1770到处置2192，比B0的1900更晚；不能用发现数增长或更短整局运行时间掩盖运输/健康退步。独立semantic_search的5个新增检查与现有组合共56项已通过（2.59秒），第6局正在新目录b4_search运行。

## 当前候选与文件边界

三个新策略都从未修改的transport_compute派生，互相独立：my_drone_blue_recovery.py（boundary）、my_drone_blue_contact.py（contact）、my_drone_blue_search.py（semantic_search）。各类有独立总开关，只有显式选择该类才启用；B3旧PAYLOAD_RECOVERY仍为False，正式入口不接入。当前只保留semantic_search作为弱证据待验候选，boundary/contact拒绝，不组合。

诊断实现为devtools/blue_lifecycle.py，--lifecycle显式开启；只在本地评测包装控制器，候选运行代码不导入该模块。devtools/blue_lifecycle_report.py生成逐弹诊断与所有三类图；identities.py仅登记选择器；blue_transport_batch.py、runner.py、worker.py、evaluate_provided_blue.py只接通诊断开关，官方引擎文件没有修改。新增测试均在solutions/devtools。没有添加外部API、模型或新依赖；算法自主实现，复用本项目已有NumPy/OpenCV/SciPy，依赖版本写入每批plan。没有声称采用未核验的论文性能或第三方成绩。

计算等价补充：B0/compute在新参考批1855个匹配采样控制帧中，四项最终控制量最大差值0。只证明这些采样，不声称逐步全集验证；随机固定输入路线/转向/语义等价测试另已通过。两版得分、处置与逐弹首次事件相同。

## 本轮已执行命令（追溯；不要新源码续跑旧输出）

工作目录均为E:\桌面\gitee\swarm-eod。

    .\blue_evaluate.cmd --output output/sf13/b4_smoke --maps Map08 --variants transport_compute --profile-smoke --lifecycle
    .\blue_evaluate.cmd --output output/sf13/b4_refs --maps Map08 --variants adapted transport_compute transport --lifecycle
    .\blue_evaluate.cmd --output output/sf13/b4_single --maps Map08 --variants boundary contact --lifecycle
    .\blue_evaluate.cmd --output output/sf13/b4_search --maps Map08 --variants semantic_search --lifecycle

最新已执行测试共56项、0失败、0跳过，2.59秒：

    .\.venv\Scripts\python.exe -B -m pytest src/swarm_rescue/solutions/devtools/test_blue_search.py src/swarm_rescue/solutions/devtools/test_blue_recovery.py src/swarm_rescue/solutions/devtools/test_blue_contact.py src/swarm_rescue/solutions/devtools/test_blue_lifecycle.py src/swarm_rescue/solutions/devtools/test_blue_transport.py src/swarm_rescue/solutions/devtools/optimization/evaluation/test_evaluation.py -q --basetemp output/sf13/b4_audit/test_tmp03 --junitxml output/sf13/b4_audit/tests_search.xml

新测试涵盖路径断开、未知格不穿行、正常进展不触发恢复、窄/缺激光不盲移、恢复时限和冷却、载荷消失、中心距离抖动/短失见、真实转角而非命令步数、多人方向被挡时保留导航、释放未确认、观测禁用、通信覆盖幂等、抓取返程路线等价。没有测试通过完整负载全局可行性或多人协同无死锁，不能混称已解决。

Map09参考B0：5400步/171.08秒、原始15.92/2处置/健康19.6，但出现1次cffi碰撞回调AssertionError，退出码0仍无效；不是超时，不修改官方碰撞处理、不吞异常。蓝种子2477227837，实际NO_GPS_ZONE和NO_COM_ZONE。该局保留全部图和失败，不用于有效配对得分差。

## 八局最终结果与采用结论

全部为固定既有强布局、每图第1轮。Map08蓝种子200806601、实际NO_GPS_ZONE；Map09蓝种子2477227837、实际NO_GPS_ZONE+NO_COM_ZONE。红种子不适用。8局中7局有效、1局回调异常，0局触及180秒；没有全弹处置局。所有失败纳入结果，没有重复到胜出才保留。
|地图|版本|原始/有效蓝分|处置|发现/抓取/脱手|健康/时间分|步数/秒|有效性|
|---|---|---|---:|---|---|---|---|
|Map08|adapted|21.44/21.44|2/10|7/7/1|47.2/0|5400/173.23|有效|
|Map08|transport_compute|21.44/21.44|2/10|7/7/1|47.2/0|5400/171.60|有效|
|Map08|transport|6.56/6.56|0/10|7/8/3|32.8/0|5400/164.21|有效|
|Map08|boundary|11.36/11.36|1/10|6/8/4|26.8/0|5400/162.47|有效|
|Map08|contact|19.84/19.84|2/10|8/8/2|39.2/0|5400/153.85|有效|
|Map08|semantic_search|22.64/22.64|2/10|7/8/1|53.2/0|5400/155.65|有效|
|Map09|adapted|15.92/空|2/10|6/6/1|19.6/0|5400/171.08|回调异常|
|Map09|semantic_search|19.12/19.12|2/10|6/6/0|35.6/0|5400/171.32|有效|

最终不采用任何新版本替换B0。保留semantic_search为待复核搜索候选：Map08仅健康贡献+1.20，处置数仍2；Map09有效19.12/2处置，参考B0原始15.92/2但回调异常，不能计算有效配对增分。两张图发现数仍分别7、6，未解决漏探和返程。boundary、contact明确拒绝。当前尚未达到“针对强红布局有竞争力”的目标，也未满足稳定优于B0的采用门槛。

[完整八局逐弹图册](E:/桌面/gitee/swarm-eod/output/sf13/b4_delivery/report.md)包含每局三类图、每颗炸弹首次发现/认领/接近/所有抓取/中心接近/释放/官方处置、失败状态、源文件与布局哈希、种子/区域/资源/回放。发现是唯一弹数，抓取和脱手是事件数，不应混为同一统计口径。不能从这些数据确认精确认领过期次数或多机冲突总数；目标关联有60px歧义边界，稀疏动作帧不代表逐步日志。

[机器可读诊断](E:/桌面/gitee/swarm-eod/output/sf13/b4_delivery/analysis.json)保留事件和样本。所有原始attempt、源码ZIP、日志、录像与收据不改写；已复核原基线/正式入口哈希。

## 关键图

Map08搜索候选的蓝队探索轨迹：左离线真值、右估计；稀疏连线不是逐步路径，不是红队提交图。

![Map08语义搜索版轨迹](E:/桌面/gitee/swarm-eod/output/sf13/b4_search/semantic_search/figures/Map08_semantic_search_r01_trajectory.png)

墙体覆盖图复用历史冻结离线强布局，没有新的红队成绩。

![Map08固定强布局墙覆盖](E:/桌面/gitee/swarm-eod/output/semifinal_joint_20261113/oracle_layouts_v2/Map08.png)

本次实际加载的初始10弹位置；不是运输后的剩余位置。

![Map08本次初始弹位](E:/桌面/gitee/swarm-eod/output/sf13/b4_search/semantic_search/figures/Map08_semantic_search_r01_positions.png)

明显退步：沿障版1/10处置、健康26.8，移动范围扩大但闭环更差。

![沿障恢复版退步轨迹](E:/桌面/gitee/swarm-eod/output/sf13/b4_single/boundary/figures/Map08_boundary_r01_trajectory.png)

Map09搜索版有效19.12/2处置。参考B0回调异常图同样保留于完整图册，不能隐藏。

![Map09搜索版轨迹](E:/桌面/gitee/swarm-eod/output/sf13/b4_cross/semantic_search/figures/Map09_semantic_search_r01_trajectory.png)

## 回放与用户后续批次

本轮只运行蓝队，强布局红回放不适用。两张图候选的完整蓝回放命令如下；其余六局逐条命令在完整图册中。

    Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
    .\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\b4_search\semantic_search\runs\Map08_r01\advanced2\blue\attempt_001'
    .\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\b4_cross\semantic_search\runs\Map09_r01\advanced2\blue\attempt_001'

跨图配对的最后两局已真正执行：

    .\blue_evaluate.cmd --output output/sf13/b4_cross --maps Map09 --variants adapted semantic_search --lifecycle

用户可执行的新批（已prepare-only并验证冻结源码，未启动仿真）：

    Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
    .\blue_b4_evaluate.cmd

等价完整命令：

    .\blue_evaluate.cmd --output output/sf13/b4_pilot --maps Map08 Map09 yh_map_fs_1 yh_map_fs_2 --variants adapted semantic_search --rounds 2 --lifecycle

任务为4张指定开发图×2轮不同蓝种子×B0/搜索候选，共16个蓝阶段、0红阶段，固定同图同轮同布局；仿真上限48分钟加启动/导出。输出output/sf13/b4_pilot。这是候选验证批，不代表已成功采用；所有失败继续保留。禁止在此批运行期间改受哈希约束源码。旧bt_p1/map_p1/p2保留但不续接；100图和70/15/15划分未动，未推送、未切入口、未打包。

## 下一步优先级与明确局限

1. 优先查返程图断开、路径末端与局部导航的进度关系。已有载荷停滞可复现，简单沿障移动增加碰撞损失，不能再把它默认打开。下一版应先离线提取停滞点的当前局部栅格/激光、有效路径连通分量，再设计保留任务目标的可执行重规划。
2. 检查空载机在上部已搜索区域反复活动的原因，以及路径/地图噪声是否使其他前沿不可达。语义搜索覆盖层是独立候选，并未增加本轮最终发现数；还不能声称搜索已优化成功。
3. 中心末端保持B0作为对照，下一次用实际碰撞/相对观测核验姿态与拥堵，再决定最小改动；旋转更久已经有延迟证据，不继续整包叠加。
4. 对认领与任务成本只做上述评估，未新实现路径成本拍卖、载荷通道预约或完整SLAM。在线官方逐弹成功回执缺失，任何“消失即成功”的实现都不允许。下一步若需要更强诊断先登记预算，不绕过规则读取隐藏状态。

本阶段完成。等待用户查看、运行新批返回结果或“继续”，不自动进入下一轮优化。

交付核验：8局全部原始产物收据、计划和ZIP哈希通过；24张逐局核心图存在；用户16蓝新批冻结274个文件且0个attempt。新脚本完成后自动生成output/sf13/b4_pilot/report/report.md逐弹完整图册。采用记录：[decision.json](E:/桌面/gitee/swarm-eod/output/sf13/b4_delivery/decision.json)。

