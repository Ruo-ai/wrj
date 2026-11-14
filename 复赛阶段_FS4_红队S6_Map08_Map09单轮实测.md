# S6：Map08/Map09完整红阶段单轮实测

2026-11-13。用户明确要求真实地图、只测一次。本批Map08/Map09各A3/B/C一次，共6红、0蓝，无重试。
全部红阶段有效、10弹；无超时、崩溃、回调异常。官方难度D和完整红蓝原始/有效总分未评测，不拼接历史蓝分。
冻结计划SHA256：`bd1ae9265edf5564004c93f8c41987dd84ed0dce809992790d2f05d115b67d2e`；全部attempt收据校验通过，原始提交图重新调用原地图公式与结果一致。
Map08种子200806601，NO_GPS_ZONE；Map09种子2477227837，NO_GPS_ZONE+NO_COM_ZONE。5架红机、上限5400步/180秒、串行1CPU/4GB。
源码A3/MyDroneRedMapping；B/MyDroneRedStopTurn；C/MyDroneRedStopTurnRadar。未切换正式入口，未改策略、蓝队、共享源码或旧结果。

|图/版本|红有效|M|W|T|弹数|步数|红墙钟秒|墙/机碰撞接触|平均生命|
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
|Map08/A3 baseline|True|59.7202|38.3640|29.7090|10|4277|111.931|1/8|23.00|
|Map08/B stop-turn|True|57.8996|26.4860|22.8571|10|4536|118.333|0/7|24.40|
|Map08/C + radar|True|54.8675|37.0718|22.8571|10|4536|130.292|0/5|24.20|
|Map09/A3 baseline|True|43.6869|69.1259|22.8571|10|4536|119.463|0/12|22.00|
|Map09/B stop-turn|True|49.8115|64.5523|22.8571|10|4536|121.727|0/3|24.40|
|Map09/C + radar|True|38.9388|68.1781|34.6296|10|4091|115.924|0/7|23.40|

碰撞为接触开始事件，墙与机分列，不是伤害次数。墙钟包含同一诊断流程；官方时间分依仿真步数。

|图/候选减A3|ΔM|ΔW|ΔT|Δ步数|Δ墙钟秒|
|---|---:|---:|---:|---:|---:|
|Map08/B stop-turn|-1.8206|-11.8779|-6.8519|+259|+6.401|
|Map08/C + radar|-4.8528|-1.2922|-6.8519|+259|+18.360|
|Map09/B stop-turn|+6.1247|-4.5736|+0.0000|+0|+2.264|
|Map09/C + radar|-4.7481|-0.9478|+11.7725|-445|-3.538|

## 失效区诊断

下表是控制后、物理推进前对齐的真实位置误差；对所有有效失效机步按样本数加权，不是每机均值的平均。每组失效时段/轨迹不同，不能称相同观测输入实验。

|图/版本|失效机步|导航均值px|地图器均值px|运动私有均值px|墙命中|收费误画像素|地图信用/扣分|TURN机步|
|---|---:|---:|---:|---|---:|---:|---|---:|
|Map08/A3 baseline|8308|30.591|23.180|无|73706|150939|0.762592/0.165389|0|
|Map08/B stop-turn|9145|29.912|27.383|39.520|73102|160132|0.756342/0.177347|2267|
|Map08/C + radar|9370|34.300|22.596|37.820|69981|149962|0.724051/0.175377|2660|
|Map09/A3 baseline|3914|34.355|33.739|无|58325|87063|0.523061/0.086192|0|
|Map09/B stop-turn|7070|82.489|47.927|82.841|70493|127217|0.632185/0.134069|2266|
|Map09/C + radar|4311|54.485|49.391|53.945|57480|123416|0.515483/0.126096|1475|

Map08：B地图、墙体和时间均退步。Map09：B地图增分，但墙体下降、导航与地图器误差上升，不能证明降低漂移。C两图地图分都下降，Map09更早提交换来时间分，不能归为建图改善。
**结论：本轮不采用B/C，不调整参数、不重跑挑结果。** 单图单种子只能作本次观察；现有证据不支持把方案作为稳定降低定位漂移或提高综合表现的正式改动。
可能因素包括停转/回退改变探索和落弹路线、更长的独立角噪声累计、A3本身匹配/缓存对新观测分布的响应；目前没有完整逐帧雷达/地图证据，不能把这些假设写成已证实因果。
所有详细状态/回退/雷达接受计数、峰值Job内存、进程秒、原始布局哈希见summary.json。真值只用于离线诊断。

## 两图对比图

每类图使用同图相同视野和图例；建图差异/墙覆盖面板直接取原始官方PNG的地图区，原图未改写。炸弹图读取本轮实际落点JSON，不使用规划目标。

### Map08

![Map08地图提交与真值差异](E:/桌面/gitee/swarm-eod/output/sf13/rs6_m89_review/Map08_mapping.png)
![Map08墙体覆盖](E:/桌面/gitee/swarm-eod/output/sf13/rs6_m89_review/Map08_walls.png)
![Map08实际落弹](E:/桌面/gitee/swarm-eod/output/sf13/rs6_m89_review/Map08_bombs.png)
![Map08实际轨迹](E:/桌面/gitee/swarm-eod/output/sf13/rs6_m89_review/Map08_trajectories.png)

### Map09

![Map09地图提交与真值差异](E:/桌面/gitee/swarm-eod/output/sf13/rs6_m89_review/Map09_mapping.png)
![Map09墙体覆盖](E:/桌面/gitee/swarm-eod/output/sf13/rs6_m89_review/Map09_walls.png)
![Map09实际落弹](E:/桌面/gitee/swarm-eod/output/sf13/rs6_m89_review/Map09_bombs.png)
![Map09实际轨迹](E:/桌面/gitee/swarm-eod/output/sf13/rs6_m89_review/Map09_trajectories.png)

## 每场原始图册与回放

[六场完整原始图册](E:/桌面/gitee/swarm-eod/output/sf13/rs6_m89_once/report.md)
原图册包含每场三类图、原始分项、资源、身份与回放。所有蓝阶段均未运行，没有本轮蓝录像。

Map08 / A3 baseline / r1 / seed 200806601

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rs6_m89_once\runs\Map08_r01\advanced2_mapping\red\attempt_001'
```

Map08 / B stop-turn / r1 / seed 200806601

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rs6_m89_once\runs\Map08_r01\s6_motion\red\attempt_001'
```

Map08 / C + radar / r1 / seed 200806601

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rs6_m89_once\runs\Map08_r01\s6_radar\red\attempt_001'
```

Map09 / A3 baseline / r1 / seed 2477227837

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rs6_m89_once\runs\Map09_r01\advanced2_mapping\red\attempt_001'
```

Map09 / B stop-turn / r1 / seed 2477227837

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rs6_m89_once\runs\Map09_r01\s6_motion\red\attempt_001'
```

Map09 / C + radar / r1 / seed 2477227837

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rs6_m89_once\runs\Map09_r01\s6_radar\red\attempt_001'
```

本轮任务结束，不再启动任何评测；本报告新增离线工具仅用于绘图/核验，不改变冻结策略。

## 交付核验与保护

六场各仅attempt_001，0次重试。每份原始结果、录像、落弹JSON、地图提交、日志和被动诊断均通过收据哈希核验；6份地图分重新调用官方公式一致。32处图册图片引用可打开。1433个既有文件保持不变，冻结策略/共享源码哈希仍有效。检查产物：`output/sf13/rs6_m89_review/verification.json`。

本轮只新增离线工具`src/swarm_rescue/solutions/devtools/red_stopturn_match_report.py`、本文及两个新的输出目录；未修改上一阶段S6候选/测试/报告。工作区过程中出现了另一任务新建的B7蓝代码与测试文件，全部保留、未编辑；其可见输出为轻量测试与离线分析，未发现完整对局attempt，不能由进程可见性宣称全系统空闲。

实际运行命令（已经完成，按用户只测一次的要求不再重跑）：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\.venv\Scripts\python.exe -B -m swarm_rescue.solutions.devtools.red_stopturn_batch --output output/sf13/rs6_m89_once --maps Map08 Map09 --controllers advanced2_mapping s6_motion s6_radar --rounds 1 --run
```

仅离线绘图命令（本轮已生成review目录；若再次绘图须另选不存在目录；不触发仿真）：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\.venv\Scripts\python.exe -B -m swarm_rescue.solutions.devtools.red_stopturn_match_report --batch output/sf13/rs6_m89_once --output output/sf13/rs6_m89_review_new
```

阶段结论：B在Map09仅地图分有正收益，两个地图均有墙分回退；C两图地图分均回退。S6暂不采用，正式入口保持Advanced2/RescueExample。没有蓝队对局，D与完整官方总分未评测。没有自动开启下一批、参数搜索、重生成地图、切换入口、提交或打包。