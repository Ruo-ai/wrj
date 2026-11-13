# 红队 A5 恢复锚定专项

2026-11-13。本轮独立开发红队，先核查再实现，不修改旧文档。起始main/44f37ae；蓝队B6源码/诊断正在并行修改，两个B6测试/夹具文件未跟踪，上轮红队批测核查文档也保留。所有旧红版本、蓝队、公共定位/通信/评测/配置、入口和提交包不改。

## 实施前登记

已重读交接、总方案、图件规范及A3/A4代码与报告。ra4_pilot/ra4_pilot_r2仍0attempt；没有新的四图结果。A3/A4最近有效红对照仍是ra4_check的Map08单轮，M59.7202→61.2287、误画扣分16.5389→13.7237，命中减少1263像素、耗时增加5.49秒；D和完整总分未评测。探索运动继承A0，未被A3/A4改写。

选项与证据：A3 `SubmissionMapper.update` 在缺测后的第一个合法GPS/罗盘读数到达时，即用该帧修正整段pending，并将位姿直接置为此读数。GPS/罗盘为时间相关噪声，首帧并不等于无误差绝对真值。历史地图仍有错位/重影，但缺完整传感器，不能证明首帧锚定是历史全部误画主因。先修这一可隔离、可复现的恢复边界；不继续增加擦除，不叠加探索或多机融合。

A5以A3为父版，单开关STABLE_ANCHOR；恢复后暂存12个连续可用绝对观测，通过估计位姿的旋转/平移运动补偿，稳健估计一个局部SE(2)校正；不是直接平均运动中的GPS坐标。75%观测在10px/.12rad内部一致门槛内才使用，门槛是工程近似，不是概率置信。最多额外等待48步；不一致则在下一可用绝对读数回退A3。中途缺测清空稳定窗口，但原扫描仍缓存/近场匹配，未闭合片段不擦已锚地图。只重锚当前片段，不整体移动旧可靠地图。共享LocalMatcher和“先匹配再入参考”保留。

局限：时间相关的持续偏差可能通过内部一致性检查；不能由多帧一致宣称全局准确。48步回退保留A3首帧噪声风险，并非强制成功。长片段非均匀漂移仍用原A3插值，不做新全局SLAM。

预算：相同完整合成输入下A3/A5组件行为测试、离线历史损失审计、最多2次128步红原生冒烟。蓝队并行开发期间不跑长局/批量，四图命令交用户。A4作为独立既有参照，不组合入A5。没有新增完整红蓝成绩时所有官方分项仍未评测。

诊断：最多12条恢复样本、32条可关闭anchor事件，原pending上限1400帧保持；统计等待、拒绝、回退、最大样本/缓存。记录实际CPU/内存开销，不将合成或冒烟指标当正式比赛分。

来源：项目内A3实现及本轮自行实现的短窗口中位数刚体校正；没有导入外部代码/ROS/完整SLAM，不使用论文成绩作本项目证据。运行时只读合法传感器、原无线消息，不读地图名/真值/预设弹位。正式入口仍place=Advanced2、rescue=RescueExample，team742不变。

## 阶段结论：实现完成，地图分验收未通过，不替换基线

A5独立候选、开关、行为测试、合成同输入检查、两次128步原生红冒烟及历史像素审计均已实际完成。42项行为/回归检查通过，**另有2项地图命中不退步门槛仍失败**，显式保留为strict xfail；不能写成44项全通过。首轮原始失败XML仍在。

A5能避免本轮合成孤立首帧异常把整段缓存立即锚错，但两组异常夹具的官方地图公式得分下降。当前不推荐采用A5，不与A4组合，不切入口；保留为独立研究候选及恢复边界诊断工具。真实Map08完整红分、完整红蓝总分和多图泛化均未评测。

这一负结果说明当前地图表达也是独立瓶颈：单个3px端点格在单侧观察6px墙时，准确定位仍可能只覆盖部分评分墙厚，噪声画宽有时反而增加命中。**这只在合成夹具中已证实，不能据此宣称历史整图损失均来自墙厚。** 下一开发专项应先隔离检查端点、格子坐标与可观察墙面厚度表达，避免继续只用位姿误差或擦除量代替地图分。暂不叠加探索、墙厚推断或膨胀策略。

## 最新状态与证据优先级

|对象|本轮核实状态|限制/结论|
|---|---|---|
|main/HEAD|`44f37aeedcf59e04cf9102755779acf468e2f53b`，本轮未切分支/提交|交接/总方案中的旧HEAD不覆盖本地事实|
|A0 advanced2|仍为正式place入口；源码哈希未变|保留导航/布弹基线|
|N0 advanced5_narrow|独立窄路对照，源码未变|已有对准/通过/回退，不整包恢复、不全局缩小间距|
|A1a/A1b|面积代理、确认并集外边际布弹候选已实现|没有本轮新证据支持采用|
|A2 advanced2_semifinal|已有定位可信门控、任务保持等组合候选|旧不可信观测停写导致漏新墙问题已有A3处理，不重复开发|
|A3 advanced2_mapping|缓存、局部匹配、片段重锚、自由反证、按来源替换快照均在当前源码|旧总方案55.3326不代表最新；本地ra4_check红阶段M=59.7202|
|A4 a4_cells|只加密自由射线径向采样；保留独立开关|最新已完成完整红对照仍仅Map08/r1|
|ra4_pilot / ra4_pilot_r2|均只有准备产物，0attempt，无四图成绩|源码集合已经变化，旧计划不续接、不改哈希|
|蓝队B6|只读查到新交付、`tests_final.xml`、三组b6_ready_1113准备计划|71项行为/回归通过；0完整蓝局，未采用、未验收强度|
|本轮对手|原B0 `MyDroneBlueAdapted`保持冻结参考身份|本轮没运行蓝阶段；后续命令也不擅自把B6换进来|
|正式入口|place=`MyDroneRedAdvanced2`，rescue=`MyDroneRescueExample`|与本地A3/A4/A5评测注册不同，未修改|

本轮开始与结束间，有4个既存文件发生并行变化：B6运行器、B6测试、B6候选及B6专属报告；本任务没有写这些文件，也没有还原变化。1424项初始保护清单中其余1420项哈希一致，包含旧红候选、蓝B0、共享定位/通信/评测、正式入口和共享方案。详见[保护与冒烟核验](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_dev/verification.json>)。进程检查没有看到可见Python任务，但这不等于其他对话已停止；本轮只做秒级冒烟，未启动长局/批量。

现有100图仍为70训练/15验证/15测试；本轮只使用Map08短测及既有Map08产物。后续命令中的yh_map_fs_1、yh_map_fs_2已通过现有training_rows核实属于训练集。未生成、删除或重划任何地图。

## 地图损失：可以量化的部分与不能归因的部分

官方源码为 `tools/map_exploration_scoring.py`：墙表面距自由像素4px以内，默认只给表面精确重合像素信用；预测距真墙10px以内免扣分，但未重合不获命中信用；远处预测罚分权重0.2。墙体覆盖与时间分仍由官方独立计算，本轮没有改公式。

新离线工具对ra4_check两份冻结attempt核验收据、只读构造官方真墙并重算分数，均与原始结果一致；不是重跑比赛，也不是传感器回放。

|Map08/r1/seed 200806601|A3|A4|A4−A3|
|---|---:|---:|---:|
|地图分M|59.720212|61.228673|+1.508461|
|信用分|76.259157|74.952407|−1.306750|
|误画扣分|16.538944|13.723733|−2.815211|
|真墙命中像素|73706|72443|−1263|
|扣分预测像素|150939|125947|−24992|
|墙体分W|38.363954|38.363954|0|
|时间分T|29.708995|29.708995|0|
|放弹/红步数|10 / 4277|10 / 4277|一致|
|引擎墙钟秒|110.685068|116.177149|+5.492081|
|D / 官方原始总分 / 官方有效总分|未评测|未评测|不能比较|

两局红阶段有效，无控制器/回调异常或超时；没有蓝阶段，不能称完整有效总分。物理轨迹/炸弹状态在已有4276帧记录中一致，但没有完整雷达和地图中间状态，不能称“相同观测输入回放”。碰撞次数未直接记录，保持未知。历史有效总分不拼接。

**新增像素结论：** A4最终预测是本对照A3预测的子集，删除33627像素、新增0，其中删除真墙命中1263、删除扣分像素24992。这是最终地图集合差，包含反证、阈值、合并、后处理整个链路的净结果，不能指定是哪一帧自由射线误删了哪面墙。

|漏墙距最近预测像素|A3|A4|可支持什么|
|---|---:|---:|---|
|0–3px|4549|4773|存在近邻漏命中；不能仅凭距离确定是量化还是错位|
|3–10px|3673|4483|免罚区附近仍可能丢信用|
|10–30px|2907|3136|存在更大几何差异，原因未分离|
|30–60px|3992|3923|不能据此直接断言未探索|
|>60px|7825|7894|提交地图附近无墙预测；未见/未提交/误删尚不可区分|

|用户要求的损失分类|本轮证据|仍缺什么|
|---|---|---|
|1 未探索/未观察真墙|只有实体轨迹、稀疏位姿、最终图；不能准确计数|完整有效激光视域与遮挡、命中历史|
|2 已看见但定位画偏|A3首帧锚定机制确定；合成异常可复现；历史映射位姿稀疏误差均值约14.82px|每帧观测、匹配变换和锚前/锚后像素对应|
|3 曾画出后反证删除|A4相对A3净丢1263真墙命中是事实|每次erase前后证据；现数据不能把净差全部归因给单次擦除|
|4 补点/量化/后处理|合成准确端点仍漏墙厚已复现；历史近邻漏命中可计数|真实局各处理阶段的掩码与墙法向，才可分离贡献|
|5 多机/通信/最终合并|源码按source+sequence替换，同源转发不重复计票；最终OR可能保留某来源假墙|真实局按来源与时间的掩码；不能从总图反推某机责任|

历史新增差分图，橙色=丢失真墙命中，浅蓝=删除的其他预测，灰色=官方墙；不是本轮A5成绩：

![历史A3到A4像素差分](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_dev/historical_delta/delta.png>)

## 新增实现、开关与回退

所有新增Python都在 `src/swarm_rescue/solutions/` 内；没有编辑任何旧Python、旧报告或共享注册表。

|新增文件|职责|
|---|---|
|`my_drone_red_mapping_anchor.py`|A5短窗口锚定；`a5_anchor` / `a5_off`类；父版A3|
|`devtools/test_red_mapping_anchor.py`|恢复/转弯/异常/缺测/缓存/等价关闭测试及明确未满足的地图门槛|
|`devtools/red_anchor_fixture.py`|4组完整合成输入，记录输入SHA、像素/位姿、计时/分配、对照图|
|`devtools/red_mapping_anchor_batch.py`|红专用进程内注册、固定B0、默认prepare-only、显式run；复用公共工具但不改源码|
|`devtools/red_mapping_delta_audit.py`|只读冻结最终地图与官方真墙，输出净像素差和漏墙距离桶|

运行策略开关 `STABLE_ANCHOR=False` 即复用原A3更新路径，候选selector=`a5_off`；逐步hits/mask/pose/原stats等价测试通过。原始A3仍单独存在，不需要回滚或覆盖文件。`RECOVERY_SAMPLES=12`、`MAX_RECOVERY_TICKS=48`是本轮固定工程值，未进行参数搜索。`RECORD_ANCHORS=False`默认不保留事件；评测加`--anchor-diagnostics`才记录有界事件、累计mapper时间与每32步抽样的pending数组字节，配置写在red_scope.json。只有统计量常驻；没有逐步打印或读取离线真值。

不改全局位姿/导航、不改LocalMatcher、不改自由射线、不改通讯协议和来源融合、不改放弹/碰撞/正式入口。A5仍继承A3的片段线性插值，不能处理所有非均匀漂移；12帧AR噪声并非12个独立样本，持续偏差可通过门槛。48步后只在实际绝对观测可用时回退A3；连续缺测期间不凭空锚定，继续局部扫描与缓存。原缓存上限1400帧，恢复样本最多12、诊断事件最多32。

## 已运行测试和负结果

|检查|实际结果|时间/产物|
|---|---|---|
|A3/A4/A5首轮行为+性能假设|40通过、2失败，失败原因是真墙命中下降|pytest 2.52秒，`ra5_dev/tests_v1.xml`保留|
|分开行为正确性与未满足采用门槛后|42通过、2 strict xfail；不是地图收益通过|pytest 2.61秒，`ra5_dev/tests_v2.xml`|
|4组完整合成输入×A3/A5|全部实际计算，固定输入SHA逐组一致|进程约3.20秒，`fixtures_v1/results.json`和4张图|
|Map08 A3/A5各128步|两次status=ok、exit=0、回调=0、控制器异常=False、未墙钟触限|原始收据/录像有效；见下表|
|历史A3/A4像素审计|两份收据/离线重算分数一致|进程4.38秒，`historical_delta/audit.json`|
|保护/地图划分核验|1420既存文件未变，4个蓝任务并行变化；70/15/15未变|`verification.json`|

离线保护核验首次引用了不存在的B0文件名`my_drone_blue_adapted.py`，命令报FileNotFoundError，未执行仿真、未写核验结果；核对注册表改为实际`my_drone_eval_blue.py`后重查通过。该检索错误不是控制器运行失败。

测试命令（已经运行，下面路径表示已保留的产物；重跑请换新的XML名）：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider src/swarm_rescue/solutions/devtools/test_red_mapping.py src/swarm_rescue/solutions/devtools/test_red_mapping_cells.py src/swarm_rescue/solutions/devtools/test_red_mapping_anchor.py --junitxml=output/sf13/ra5_dev/tests_v2.xml
```

下面全是**人工墙/完整合成传感器输入**，使用官方地图公式只是检查计分语义，不是比赛地图成绩、没有墙体/时间/难度/总分。

|夹具|端点位置误差A3→A5 px|真墙命中A3→A5|扣分像素A3→A5|合成地图公式分A3→A5|
|---|---:|---:|---:|---:|
|无异常 clean|0→0|140→140|0→0|16.5485→16.5485|
|首帧异常 first_outlier|7.9188→0|405→140|454→0|44.7920→16.5485|
|运动转弯+首帧异常|7.8760→约0|345→134|424→0|37.9905→15.8392|
|持续相关偏差|15→15|825→825|225→108|96.6383→97.1773|

夹具结束于第12个恢复观测，共124步。异常夹具只构造一个首帧异常，不能代表真实传感器全部AR噪声分布。持续偏差夹具说明一致性门槛不能自证绝对准确；噪声画宽导致高命中也不能当作好定位。两项原性能门槛转为明确xfail保留，原因是实测推翻了“端点更准必然命中不退步”的假设，**没有把失败改写为通过**。

合成组件124次update：A3约23.90–26.76ms，A5约25.62–28.92ms；进程时钟分辨率较粗，不据此推断整局加速/减速。单独tracemalloc峰值1,347,409字节，包含小图mapper/NumPy已追踪分配，不是原生引擎RSS；该夹具最多23个pending扫描、12个恢复样本。整个原生进程资源另列，长局/最坏缓存未压测。

合成图统一视野。蓝色是真墙未命中，绿色是命中，红色为真墙以外预测（含免罚区，不全是扣分像素）：

![无异常同输入](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_dev/fixtures_v1/clean.png>)

![首帧异常同输入及地图分回退](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_dev/fixtures_v1/first_outlier.png>)

![转弯同输入及地图分回退](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_dev/fixtures_v1/turn_first_outlier.png>)

![持续偏差不能自证正确](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_dev/fixtures_v1/persistent_bias.png>)

## 本轮原生冒烟逐轮分项与图册

同批`ra5_smoke`、Map08/r1、seed=200806601、5红×2弹、NO_GPS_ZONE，诊断上限128步/20秒；单CPU、4,000,000,000字节Windows Job限制。不是正式5400步/180秒完整局。两局录像的drone_states/bomb_states逐元素相同；这仍不是完整观测回放。

|版本|冒烟完成|地图/墙体/时间分|难度/官方原始总分/有效总分|实际落弹|步骤/引擎墙钟秒|进程CPU秒|峰值Job提交字节|碰撞数|
|---|---|---|---|---:|---|---:|---:|---|
|A3|正常，非有效成绩|未评测|未评测|0|128 / 2.363502|6.297|576536576|未知|
|A5|正常，非有效成绩|未评测|未评测|0|128 / 2.376656|6.219|542842880|未知|

官方引擎原始短测字段M/W/T为0，是尚未提交/未布弹的占位输出，不把它当完整局0分。公共flatten中的task_failed=true/failure_reason=smoke_diagnostic表示没有有效完整成绩，不代表本次有控制器崩溃。两次均没有超时、回调异常或控制器崩溃。源ZIP内部哈希与两份attempt收据已核对。

本短程没有进入缺测恢复分支：五机anchor_windows/commits均为0；恢复算法验证来自上面的合成序列，不能声称已完成真实无GPS恢复验收。A5五机mapper累计update约0.2331秒/640次调用，未使用pending；长局CPU与内存仍待测。没有蓝阶段，没有蓝录像，十弹完成率尚未评测。

[完整逐轮图册（两组全部核心图及轨迹）](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_smoke/report.md>)。本轮只生成真实128步产物，以下代表A5；A3同尺度图见完整图册。

建图/提交差异：128步未到提交，中央为空白；不表示传感器未观察，也不证明探索改善。

![A5冒烟建图提交差异](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_smoke/runs/Map08_r01/a5_anchor/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_exploration_diff.png>)

墙体覆盖：0枚实际落弹；图中0/0等初始化字段不用于评价官方覆盖率或墙体收益。

![A5冒烟墙体覆盖](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_smoke/runs/Map08_r01/a5_anchor/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_wall_destruction.png>)

实际炸弹位置：本次没有落弹圆，灰区是实际特殊区；不补造目标点或机号。

![A5冒烟实际落弹位置为空](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_smoke/figures/Map08_a5_anchor_r01_attempt_001.png>)

稀疏离线位姿轨迹（不是雷达可见范围；直线只连接采样）：

![A5冒烟位姿轨迹](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_smoke/figures/Map08_a5_anchor_r01_red_pose.png>)

## 回放及下一批命令

本轮两次红回放，可直接运行：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\ra5_smoke\runs\Map08_r01\advanced2_mapping\red\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\ra5_smoke\runs\Map08_r01\a5_anchor\red\attempt_001'
```

本轮没有蓝回放。历史B0参照可只读回放如下，**不是A5配套蓝成绩**：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\b5_compare\adapted\runs\Map08_r01\advanced2\blue\attempt_001'
```

优先补齐尚不存在的A3/A4四图对照：Map08、Map09、训练yh_map_fs_1、yh_map_fs_2；每图1轮、A3/A4、固定B0；**8红+最多8蓝，串行16阶段**，每阶段240秒硬超时、正式引擎180秒上限。助手没有启动。等蓝队计算空闲、双方不再改源码后，由用户运行下面一个新目录命令；不使用旧ra4_pilot/r2。

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
$RedA4Out = 'output/sf13/r4_' + (Get-Date -Format 'MMddHHmmss')
if (Test-Path -LiteralPath $RedA4Out) { throw '目录已存在，请换新目录' }
.\.venv\Scripts\python.exe -B -m swarm_rescue.solutions.devtools.red_mapping_anchor_batch --output $RedA4Out --maps Map08 Map09 yh_map_fs_1 yh_map_fs_2 --controllers advanced2_mapping a4_cells --rounds 1 --full-pairs --run
```

若要收集A5真实恢复数据，先单独Map08×1轮×A3/A5，**2红+最多2蓝**，不与上批同时启动；A5仍为待验证且合成地图门槛未通过的候选。两者固定seed公式742+地图内容hash+(轮−1)×1000003，Map08/r1实际seed200806601；固定对手B0，不替换B6。诊断只加A5有限记录，时间开销必须单列。

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
$RedA5Out = 'output/sf13/r5_' + (Get-Date -Format 'MMddHHmmss')
if (Test-Path -LiteralPath $RedA5Out) { throw '目录已存在，请换新目录' }
.\.venv\Scripts\python.exe -B -m swarm_rescue.solutions.devtools.red_mapping_anchor_batch --output $RedA5Out --maps Map08 --controllers advanced2_mapping a5_anchor --rounds 1 --full-pairs --anchor-diagnostics --run
```

去掉`--run`仅准备快照，不运行仿真。准备后源码有变则换新目录；不修改已有计划、结果或收据。完整批结束后在同一PowerShell中回放（只有蓝阶段实际成功生成attempt才存在蓝录像）：

```powershell
.\replay_map.cmd "$RedA5Out/runs/Map08_r01/a5_anchor/red/attempt_001"
.\replay_map.cmd "$RedA5Out/runs/Map08_r01/a5_anchor/blue/attempt_001"
```

本轮新增离线像素工具命令已实际运行一次；重跑须新目录：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
$RedDeltaOut = 'output/sf13/rd_' + (Get-Date -Format 'MMddHHmmss')
if (Test-Path -LiteralPath $RedDeltaOut) { throw '目录已存在，请换新目录' }
.\.venv\Scripts\python.exe -B -m swarm_rescue.solutions.devtools.red_mapping_delta_audit output/sf13/ra4_check/runs/Map08_r01/advanced2_mapping/red/attempt_001 output/sf13/ra4_check/runs/Map08_r01/a4_cells/red/attempt_001 --output $RedDeltaOut
```

## 后续合并事项（未写共享文档）

建议总方案以后统一吸收：ra4_pilot/r2没有四图结果；A3/A4已完成单图只有红分，A4增M伴随真墙命中下降；A5已有实现但地图收益不满足门槛，不能由定位误差改善直接推荐；B6已交付71项测试但未有完整蓝验收；下一开发优先隔离墙面表达/量化与观测覆盖，再视数据决定探索任务。所有新候选保持独立开关与旧对照，不因版本号更新而采用。

本轮结束等待用户运行批测或发送“继续”。未自动启动大批量、参数搜索、地图生成、正式入口切换、打包、提交或推送。队伍742与既有team742_preliminary.zip保留。
