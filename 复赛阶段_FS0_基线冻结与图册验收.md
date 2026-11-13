# FS0：基线冻结、身份隔离与图册验收

2026-11-13。FS0完成，下一阶段FS1。当前只采用工具改进；没有修改红蓝策略，没有新的完整比赛成绩，不作策略强度采用决定。平台90分目标尚无达成证据。

## 核查与冻结

HEAD仍为`6557940dc18549ffc7a945db887da5b05b7ed2f1`，分支main。保留已有工作区修改：`closed_playground.py`关闭shader、统一红入口、team_info及check_submission状态；没有pull/reset/clean。未发现适用的AGENTS.md。Python新增与修改全部位于solutions目录。

|身份|实际选择器／类|SHA256|结论|
|---|---|---|---|
|A0|advanced2 / MyDroneRedAdvanced2|6e6e209a7c669fa4fb1b46a164f8f355643d07c87a94e1e8acbb31de77643c8f|与R1冻结源码一致，保留主基线|
|N0|advanced5_narrow / MyDroneRedAdvanced5Narrow|9ee893bc716e0dfbd3e684dd6f441770f30445c777518db9d910a3b769973475|与R1一致，独立对照|
|B0|adapted / MyDroneBlueAdapted|a9e448b640b79fc2ffbc11059bcbb09559cfd79c50411857d5f53d8ffdbf0082|开发参考，未完成强参考验收|
|统一提交入口|place=A0；rescue=MyDroneRescueExample|041930618d4ad2b08751e33c541d3e67588e383f47e5d5f58dc61372530d7e77|本次不切换|
|A1/N1/B1|selector=null，not_implemented|无|拒绝当作已实现候选运行|

完整[身份与文件哈希](output/semifinal_fs0_20261113/baseline/identity.json)、[冻结源码ZIP](output/semifinal_fs0_20261113/baseline/source_snapshot.zip)、[工作区状态](output/semifinal_fs0_20261113/baseline/git_status.txt)、[原有已跟踪差异](output/semifinal_fs0_20261113/baseline/tracked_changes.patch)保存在独立新目录。这里ZIP是实验追溯快照，不是提交包。现有根目录ZIP均登记哈希并复核未变；team_number仍为整数742，后续提交名仍为team742_preliminary.zip。

100图与70训练/15验证/15测试通过现有校验器的文件、分组、划分完整性核验。未生成、删除或跑100图；验证和测试集未用于调参。官方开发图的实际区域仍为Map06无通信、Map07t4无特殊区、Map08无GPS、Map09两区都有。

核对本地引擎：激光230、语义130、通信180、健康25；GPS与罗盘同时失效；GPS AR(1)=0.98/σ5，罗盘σ4°，里程计σ0.2/8°/1°，语义零均值σ2.5。官方评分配置0.5地图+0.2墙+0.2难度+0.1时间，红蓝time_best_ratio=0.3。正式预算5400步/180秒、单CPU/4,000,000,000字节，外层240秒。8步冒烟单独标为8步/20秒，不能冒充正式预算下完成。

## 实现与验证

|文件／函数|实际变化|
|---|---|
|evaluation/identities.py|统一离线红蓝注册表；B0显式身份；未来候选不可执行；baseline漂移时拒绝运行|
|evaluate_verified_revision.py、evaluate_provided_blue.py|使用注册表加载，原有策略类不改；蓝队默认仍为Adapted|
|evaluation/runner.py、__main__.py|计划冻结红蓝具体类；蓝请求明确身份；结果类不一致判无效；配置文件纳入哈希；增加--blue-variant和--prepare-only|
|evaluation/atlas.py、report.py|逐轮三图内嵌，记录红蓝种子、attempt、实际区域、原始/有效分、健康与资源；无证据的发现/抓取/运输指标明确未知；回调异常离线复核，不修改原始结果|
|evaluation/fixed_blue.py|固定红布局报告复用同一图册，补齐红蓝完整回放命令|
|report_reference_blue.py|要求新的--output；保留旧报告和原始收据，完整40组另存内嵌图册|
|evaluation/fs0.py、fs0_check.cmd|冻结并重复核验A0/N0/B0、正式/本地入口、规则、数据集、提交包；不会启动仿真|
|evaluation/test_fs0.py|4项新增回归检查：未实现候选拒绝、正式入口不变、蓝身份冻结与准备不启动、回调异常原分保留与三图/回放、当前规则及数据集（部分合并在同一测试）|

未引入第三方算法源码或新依赖，故本阶段无新增外部代码版本/许可。FS1实现算法时仍需逐项登记原始来源、版本、许可、适用边界和改写范围；不能将已有参考论文的结论算作本项目成绩。

53项既有检查通过（参考蓝队、复赛传感器适配、评测工具）；新增4项检查全部通过。工具最终复检25项通过，其中21项与前一轮重复，不能相加成78项独立检查。既有适配行为测试包含4个60步原生控制片段；另外执行下面4次8步完整工作进程冒烟。Pillow旧接口弃用警告4条，不是仿真控制器异常。

## 新冒烟：两组、每组红蓝各8步

批次`output/semifinal_fs0_20261113/smoke`。Map08，r1，红/蓝实际种子均200806601（基础seed=742经地图内容哈希派生），区域NO_GPS_ZONE，所有attempt均001。蓝队使用诊断单弹布局，不是红队8步后落弹布局，因此不作固定红布局得分对照。

|红版本 / 蓝版本|地图分|墙体分|时间分|难度分|原始完整分|有效完整分|红秒|蓝秒|蓝处置/健康原值|有效性|
|---|---:|---:|---:|---|---|---|---:|---:|---|---|
|A0 / B0|0|0|0|未产生|未产生|无|0.2399|0.3011|0/100|链路通过；仅8步|
|N0 / B0|0|0|0|未产生|未产生|无|0.2181|0.2874|0/100|链路通过；仅8步|

零分来自尚未提交的短测，不是正常完整比赛退步。两蓝原始诊断分均20，不能作强度结论。发现/抓取次数、运输成功率/耗时未知；不从静态弹位推断。

4个进程均exit=0、status=ok、回调异常0、无触限；录像元数据与实际红5/蓝10、区域一致，收据哈希通过。红/蓝进程总秒分别A0 4.297/3.906、N0 4.000/3.688；对应峰值Job内存512286720/424787968、521318400/425922560字节。单CPU和4GB Job限制均生效；这些不是平台同硬件速度测量。

完整[新冒烟图册](output/semifinal_fs0_20261113/smoke/report.md)内嵌6图，含两组的4条回放命令。以下A0/Map08/r1为本次8步诊断图：没有正式提交、没有实际红弹，不能解释为优化效果。

![本次8步诊断 建图差异](E:/桌面/gitee/swarm-eod/output/semifinal_fs0_20261113/smoke/runs/Map08_r01/advanced2/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_exploration_diff.png)

![本次8步诊断 墙体覆盖](E:/桌面/gitee/swarm-eod/output/semifinal_fs0_20261113/smoke/runs/Map08_r01/advanced2/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_wall_destruction.png)

![本次8步诊断 实际红弹为0 灰色无GPS区 青色出生区 红色处置区](E:/桌面/gitee/swarm-eod/output/semifinal_fs0_20261113/smoke/figures/Map08_advanced2_r01_attempt_001.png)

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\semifinal_fs0_20261113\smoke\runs\Map08_r01\advanced2\red\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\semifinal_fs0_20261113\smoke\runs\Map08_r01\advanced2\blue\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\semifinal_fs0_20261113\smoke\runs\Map08_r01\advanced5_narrow\red\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\semifinal_fs0_20261113\smoke\runs\Map08_r01\advanced5_narrow\blue\attempt_001'
```

## 历史重绘：完整保留退步和异常

[完整40组图册](output/semifinal_fs0_20261113/history_atlas/report.md)含5批×4图×2红布局×1轮，160个内嵌图片引用（每组3核心图+1蓝队诊断图）、80条完整回放命令。全部图路径已检查存在；关键图已打开检查。来源收据、布局、录像元数据与源码ZIP核验通过。历史全部失败照常展示，没有重跑旧批或改写成绩。

下面为历史Map07t4 / advanced2 / r1，所有红/蓝种子2396785999，无特殊区。红阶段4536步、104.2884秒，10弹。三行共用同一红布局，蓝队分别来自原R1、adapted01、enhanced01；不是FS0新成绩。

|蓝批次|地图分|墙体分|时间分|原始难度|原始总分|有效总分|蓝分|处置|健康原值|蓝秒|有效性|
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
|原R1|82.8702|39.8262|22.8571|100|71.6860|无|12.08|1|30.4|180.0000|触限且回调异常|
|adapted01|82.8702|39.8262|22.8571|100|71.6860|71.6860|29.92|4|29.6|160.0448|有效|
|enhanced01|82.8702|39.8262|22.8571|100|71.6860|71.6860|6.00|0|30.0|159.8802|有效但明显退步|

建图/提交差异（不是完整探索轨迹）：

![历史Map07t4 A0 r1 建图差异](E:/桌面/gitee/swarm-eod/output/semifinal_r1_official_pilot_20261111/runs/Map07t4_r01/advanced2/red/attempt_001/artifacts/team742_Map07t4_none_rd1_exploration_diff.png)

墙体覆盖：原始加权覆盖28.70%，归一墙体分39.8262；两者不同。

![历史Map07t4 A0 r1 墙体覆盖](E:/桌面/gitee/swarm-eod/output/semifinal_r1_official_pilot_20261111/runs/Map07t4_r01/advanced2/red/attempt_001/artifacts/team742_Map07t4_none_rd1_wall_destruction.png)

实际初始弹位；圆重叠本身不能判断墙体收益重复。

![历史Map07t4 A0 r1 实际弹位](E:/桌面/gitee/swarm-eod/output/semifinal_r1_official_pilot_20261111/figures/Map07t4_advanced2_r01_attempt_001.png)

历史明显回退：enhanced01蓝分比adapted01低23.92，少处置4颗；点云只作运动诊断，不能推定某机精确抓取事件。

![历史enhanced01零处置回退](E:/桌面/gitee/swarm-eod/output/semifinal_fs0_20261113/history_atlas/figures/enhanced01_Map07t4_advanced2_r01.png)

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\semifinal_r1_official_pilot_20261111\runs\Map07t4_r01\advanced2\red\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\semifinal_r1_official_pilot_20261111\runs\Map07t4_r01\advanced2\blue\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\semifinal_reference_blue_20261111\adapted01\runs\Map07t4_r01\advanced2\blue\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\semifinal_reference_blue_20261111\enhanced01\runs\Map07t4_r01\advanced2\blue\attempt_001'
```

## 采用结论、边界与下一步

**已证实**：身份冻结和图册工具可用；新冒烟链路通过；红策略与R1基线字节一致；原正式蓝入口仍是示例；历史增强不能据原始处置数直接采用。

**推测原因**：骨架估值与官方墙面积目标不一致可能影响落点排序；蓝队运输失败可能来自漂移、局部路径或末端动作。FS0没有新增因果证据，不据图件单独判因。

**待验证**：FS1从A0派生隔离A1，先核验可信墙面积和已确认落弹并集后的新增收益；N0独立对照，B0固定。下一次开始前核验FS0之后的工作区增量，并按新源码建新批。A1实现与小测试后再给用户可运行的正式小批命令，不让用户重复跑已完成FS0或历史批。

本阶段采用身份/图册工具，拒绝据8步短测或历史重绘宣称红蓝提升。没有待续跑任务。全局控制计算分位数、可靠抓取/释放事件统计尚待后续有针对性的诊断，本阶段未伪造这些指标。完整FS3一键矩阵仍未实现。

可选复核命令（已实际执行两次，0次仿真；检查100图划分但不跑地图；A0/N0/B0当前冻结版本；输出`output/semifinal_fs0_20261113/baseline`）：

```powershell
.\fs0_check.cmd
```

无需现在启动批量。用户发送“继续”后进入FS1。以后修改受哈希约束源码后，FS0现场复核按设计拒绝，不覆盖旧快照。已有工具可用`evaluate_maps.cmd ... --prepare-only`先冻结指定小批但不启动；本次只冻结并运行了上述4次8步冒烟，没有创建待用户运行的正式批次。
