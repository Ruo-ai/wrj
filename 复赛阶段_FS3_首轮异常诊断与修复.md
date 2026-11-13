# FS3首轮返回：候选提前崩溃，修复后重新验收

2026-11-13。用户完成批次`output/sf13/p1`。`advanced2_semifinal`是从进阶2派生的红队候选A2，`advanced2`仍是原基线A0。**这次A2跑得快，是异常提前退出，不是效率提升。A2与蓝队B2都不采用。** 本地参考B0仍为`adapted`；统一提交入口未切换，提交包未改。

## 已证实结果与完整分

四图各1轮。A0红队全部4536步、放满10弹并提交，B0四局均有效；A2四局仅241或481步、0弹且未提交。四局B2在固定A0布局上也异常退出。主矩阵实际运行8红、8蓝；由于A2失败，其对应8个蓝阶段跳过并保留失败行。p1运行已结束，没有待续跑任务。

以下完整分为固定本地B0下的红分，不是平台分；异常行的0分项来自引擎崩溃产物，不能作为完整成绩。完整原始及有效分、各自种子、实际区域、资源、每轮三类图与红蓝回放见[固定B0完整图册](output/sf13/p1/reference/report.md)及[固定B2完整图册](output/sf13/p1/candidate_blue/report.md)。两本图册覆盖全部16个组合，未运行项也列出原因。

|地图／轮次|红版本|地图分|墙体分|时间分|难度分|原始总分|有效总分|步数／秒|放弹|有效性|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|Map08／1|A0|47.7307|31.6855|22.8571|100.0000|52.4882|52.4882|4536／112.0263|10/10|有效|
|Map08／1|A2|0|0|0|—|—|—|241／5.4289|0/10|控制器异常|
|Map09／1|A0|43.8821|69.1259|22.8571|42.3611|46.5242|46.5242|4536／105.6897|10/10|有效|
|Map09／1|A2|0|0|0|—|—|—|481／11.1387|0/10|控制器异常|
|yh_map_fs_1／1|A0|77.9642|49.9234|22.8571|100.0000|71.2525|71.2525|4536／104.6207|10/10|有效|
|yh_map_fs_1／1|A2|0|0|0|—|—|—|481／10.7118|0/10|控制器异常|
|yh_map_fs_2／1|A0|81.2327|73.7211|22.8571|100.0000|77.6463|77.6463|4536／108.1692|10/10|有效|
|yh_map_fs_2／1|A2|0|0|0|—|—|—|481／10.5598|0/10|控制器异常|

每图红蓝种子相同，但分别记录在request/result：Map08=`200806601`，Map09=`2477227837`，yh_map_fs_1=`2406293846`，yh_map_fs_2=`203206726`。核对各轮conditions/result：Map08实际为NO_GPS_ZONE，其余三图均为NO_COM_ZONE＋NO_GPS_ZONE。单CPU、4GB、5400步/180秒。红版本文件哈希见下文；全依赖与地图哈希见各批plan.json，源码快照和原始失败全部保留。

|固定A0布局／轮次|蓝版本|蓝分（原始）|处置|健康（官方原值）|步数／秒|有效性|
|---|---|---:|---:|---:|---|---|
|Map08／1|B0|43.6800|6/10|38.4|5400／166.8974|有效|
|Map08／1|B2|19.4400|0/10|97.2|140／4.6504|控制器异常|
|Map09／1|B0|78.8195|10/10|39.6|3340／99.9592|有效|
|Map09／1|B2|24.5600|1/10|92.8|380／14.3333|控制器异常|
|yh_map_fs_1／1|B0|26.3200|3/10|41.6|5400／159.8547|有效|
|yh_map_fs_1／1|B2|24.1600|1/10|90.8|620／21.1978|控制器异常|
|yh_map_fs_2／1|B0|28.9600|4/10|24.8|5400／159.1013|有效|
|yh_map_fs_2／1|B2|18.0000|0/10|90.0|380／13.3594|控制器异常|

没有双方均有效的B0/B2配对，不能比较强弱。B2健康高主要对应提前结束，不能解释为生存改善；处置少同样不能独立归因为运输策略。各轮抓取、持有期间确认处置、运输耗时、资源列于完整图册；发现唯一弹数接口不足时标未知，不把语义目击次数当发现数。

|固定A0布局／轮次|蓝版本|抓取事件|持有期间确认处置／抓取|未确认脱手|成功运输平均步数|峰值Job MB|
|---|---|---:|---|---:|---:|---:|
|Map08／1|B0|9|6/9|0|2895.50|445.71|
|Map08／1|B2|0|0/0，不定义成功率|0|无样本|468.43|
|Map09／1|B0|10|10/10|0|1061.80|435.56|
|Map09／1|B2|5|1/5|0|154.00|473.12|
|yh_map_fs_1／1|B0|11|3/11|2|602.67|448.98|
|yh_map_fs_1／1|B2|4|1/4|0|569.00|473.13|
|yh_map_fs_2／1|B0|9|4/9|2|1265.25|443.91|
|yh_map_fs_2／1|B2|2|0/2|0|无样本|470.11|

抓取为事件数，允许同弹重复，不是唯一弹数；成功运输均值只基于已确认处置样本。异常B2的这些原始中途指标保留，但不计有效强度比较。

## 根因、最小修改与测试缺口

八个候选运行的堆栈均指向周期日志JSON序列化：局部扫描匹配`diagnostic.observable`保存了NumPy布尔标量，`json.dumps`报`TypeError: Object of type bool is not JSON serializable`。红队240步周期日志、蓝队周期日志首次获得匹配诊断后触发。先前64步冒烟没有走到这条路径；原先行为检查通过不能证明完整回合可运行，这是本次验证覆盖缺口。

修复只将两项诊断标志显式转为Python `bool`，不改变判据、阈值、匹配搜索预算或控制逻辑，不关闭日志、不吞异常。新增6项回归覆盖真实匹配成功、退化结构、拒绝、数据不足的JSON兼容性，以及红蓝实际周期日志入口；修复前5失败/1通过，修复后与现有相关检查合计23通过。证据：[修复前](output/sf13/fix2/logging_before.xml)、[修复后](output/sf13/fix2/logging_after.xml)。

另将默认新批次改为`output/sf13/p2`；固定布局蓝队工具支持显式选择现有候选和红布局版本，避免把失败A2布局误作蓝队输入。p1不续跑、不覆盖，旧失败结果不会被修复后的源码替换。

|文件|SHA256|
|---|---|
|A0：my_drone_red_advanced2.py|6e6e209a7c669fa4fb1b46a164f8f355643d07c87a94e1e8acbb31de77643c8f|
|B0：my_drone_eval_blue.py|a9e448b640b79fc2ffbc11059bcbb09559cfd79c50411857d5f53d8ffdbf0082|
|A2：my_drone_red_semifinal.py|15f79210f90609e3d23bb93f15e0c7aa7f5697fa51a48066c5a2a06968df2fbc|
|B2：my_drone_blue_semifinal.py|1cd5e1c1d6e76e13bd133c4814b1f3affc525cf0c0ab720166643cc49729518b|
|共享定位，p1旧版|03439bdd5e655f85c0a1a50730db3aa161c52b32e0a19a78aad566f673181749|
|共享定位，修复后|f6b7f733a4121fc457e831c5c494bd13dd81c4884425fc81f82733d7aa21fe19|

红蓝主文件未变，但共同依赖变了，所以修复后属于新源码版本，必须新批。基线A0/N0/B0、正式入口哈希均与FS0冻结一致；不修改官方规则和物理，不新增在线真值输入。

## p1关键图：Map08原基线与提前退出候选

以下是用户完成的p1原始图，种子200806601、第1轮、实际NO_GPS_ZONE；不是修复后的新结果。

A0建图／提交差异，地图分47.7307；该图不表示完整探索轨迹。

![p1 Map08 A0建图提交差异](<E:/桌面/gitee/swarm-eod/output/sf13/p1/reference/runs/Map08_r01/advanced2/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_exploration_diff.png>)

A0墙体覆盖，归一墙体分31.6855；原始覆盖率与该分数不是同一指标。

![p1 Map08 A0墙体覆盖](<E:/桌面/gitee/swarm-eod/output/sf13/p1/reference/runs/Map08_r01/advanced2/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_wall_destruction.png>)

A0实际初始10弹位置。不是计划目标，也不是蓝队搬运后的弹位；编号为产物顺序，不推测所属机号。

![p1 Map08 A0实际弹位](<E:/桌面/gitee/swarm-eod/output/sf13/p1/reference/figures/Map08_advanced2_r01_attempt_001.png>)

A2在241步崩溃、未提交，以下建图差异只是异常导出，不能称完成建图成绩。

![p1 Map08 A2异常建图图](<E:/桌面/gitee/swarm-eod/output/sf13/p1/reference/runs/Map08_r01/advanced2_semifinal/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_exploration_diff.png>)

A2未放弹，墙体覆盖为0。

![p1 Map08 A2异常墙体图](<E:/桌面/gitee/swarm-eod/output/sf13/p1/reference/runs/Map08_r01/advanced2_semifinal/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_wall_destruction.png>)

实际弹位图为空，表示尚未落弹；不据此讨论布弹优劣。

![p1 Map08 A2零弹位置图](<E:/桌面/gitee/swarm-eod/output/sf13/p1/reference/figures/Map08_advanced2_semifinal_r01_attempt_001.png>)

完整红蓝回放命令（A0/B0正常轮）：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\p1\reference\runs\Map08_r01\advanced2\red\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\p1\reference\runs\Map08_r01\advanced2\blue\attempt_001'
```

A2异常红回放、B2复用A0布局的异常蓝回放（A2对应蓝阶段未运行，没有可提供的蓝回放）：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\p1\reference\runs\Map08_r01\advanced2_semifinal\red\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\p1\candidate_blue\runs\Map08_r01\advanced2\blue\attempt_001'
```

## 修复后完整回合检查与下一步

独立新批`output/sf13/check_r2`已完成Map08一次A2红+B0蓝；两阶段均正常，无触限、回调或控制器异常。红4536步、10弹并提交，119.4818秒；蓝5400步、163.3199秒。另一次B2复用p1 A0布局在`output/sf13/check_b2`单独检查。助手只运行这3个完整阶段，不自动运行四图批量。

### 固定B0：Map08修复后红队出现退步

红蓝各自种子均200806601、第1轮、实际NO_GPS_ZONE，单CPU/4GB/5400步/180秒。A0为p1原始结果复用，A2为check_r2新评测；相同地图、种子、蓝版本和引擎条件，不是新跑的同批多种子结论。

|来源／红版本|地图分|墙体分|时间分|难度分|原始总分|有效总分|红步／秒|蓝步／秒|有效性|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|p1／A0|47.7307|31.6855|22.8571|100.0000|52.4882|52.4882|4536／112.0263|5400／166.8974|有效|
|check_r2／修复A2|46.0695|24.7064|22.8571|100.0000|50.2617|50.2617|4536／119.4818|5400／163.3199|有效|

已证实：这局A2总分下降2.2265，地图分下降1.6613，墙体分下降6.9791；同样步数耗时也没有变短。不能由一次运行做统计性能断言，但足以否定“这批已经证明更快更好”。实际加权墙覆盖并集从43684.5降至34062.5，减少9622（约22.0%）；[逐弹离线审计](output/sf13/check_r2/placement_audit.md)与官方并集完全一致。第9/10弹新增面积仅1200.5/269，末端低收益需要后续分析。新布局让B0处置6→4，但双方难度分都已饱和100，因此没有补回地图和墙体损失；不能据此声称布局难度全面提高。

推测而非定论：可信区域过滤、定位误差、候选排序和截止兜底可能共同影响覆盖。下一轮先比对逐弹估计收益、实际新增面积和落点误差，并单独消融定位/面积机制；不把整个首版当作最优基线继续叠加。

|新A2布局上的B0指标|原值|
|---|---|
|蓝分／处置／健康|29.2／4/10／26.0|
|抓取／持有期间确认处置／成功运输均步|7／4／1401.5|
|红／蓝峰值Job字节|604565504／446066688|
|红／蓝进程秒|125.0620／166.9380|

下列为修复后A2的本次新图，包含明确退步结果。建图图右部仍有漏检墙体和误画；具体原因需结合轨迹诊断，不能仅凭静态图归因。墙体图绿色为覆盖、灰色为未覆盖，大片厚墙与右侧区域仍未覆盖。弹位图显示实际初始10弹，爆炸圆相交本身不是损失证明。

![check_r2 Map08 A2建图提交差异](<E:/桌面/gitee/swarm-eod/output/sf13/check_r2/runs/Map08_r01/advanced2_semifinal/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_exploration_diff.png>)

![check_r2 Map08 A2墙体覆盖](<E:/桌面/gitee/swarm-eod/output/sf13/check_r2/runs/Map08_r01/advanced2_semifinal/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_wall_destruction.png>)

![check_r2 Map08 A2实际初始弹位](<E:/桌面/gitee/swarm-eod/output/sf13/check_r2/figures/Map08_advanced2_semifinal_r01_attempt_001.png>)

[本次完整分、资源与专项图册](output/sf13/check_r2/report.md)。完整红蓝回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\check_r2\runs\Map08_r01\advanced2_semifinal\red\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\check_r2\runs\Map08_r01\advanced2_semifinal\blue\attempt_001'
```

### 固定A0布局：修复后B2仍明显退步

`output/sf13/check_b2`已完成，无控制器/回调异常、无触限。原红队不重跑，复用p1 Map08 A0的10弹布局及红结果，布局SHA256为`b33554e0055079d7cab7c6e273ddc3eb248de1ac30dfaf4de77cd0512b625d5e`。蓝种子200806601，第1轮，实际NO_GPS_ZONE，10机、单CPU、4GB、5400步/180秒。

|蓝版本|蓝分|处置|健康|步／秒|抓取／确认处置／未确认脱手|成功运输均步|峰值Job字节／进程秒|有效性|
|---|---:|---:|---:|---|---|---|---|---|
|p1 B0|43.6800|6/10|38.4|5400／166.8974|9／6／0|2895.5|445706240／170.1090|有效|
|check_b2 修复B2|6.9600|0/10|34.8|5400／173.9562|10／0／4|暂无成功样本|478928896／177.5790|有效|

已证实：修复B2比同布局B0蓝分下降36.72、少处置6弹；存在10次抓取、4次未确认脱手，不能把脱手当作处置。日志显示部分机长时间无可信定位，但尚不能断言它是零处置的唯一原因，后续必须把返程路径、最终控制指令和释放阶段分别对齐录像。共享日志修复通过完整回合，B2强度验收仍失败。

本轮固定A0的红分项仍M=47.7307、W=31.6855、T=22.8571、D=100，原始／有效总分均52.4882；它们属于复用的A0布局，不是新A2成绩。B0/B2均使难度饱和100，故红总分相同，不能用该总分认定两蓝同样强。

[B2本次完整图册](output/sf13/check_b2/report.md)内嵌复用的红建图、墙体图、弹位图及蓝队轨迹/定位专项图。红三图来源与前文p1 A0相同，本节不把它们再标成新红图。完整对应回放：

![check_b2 Map08蓝队采样轨迹与定位误差，真值仅用于离线审计](<E:/桌面/gitee/swarm-eod/output/sf13/check_b2/figures/Map08_advanced2_r01_blue_pose.png>)

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\p1\reference\runs\Map08_r01\advanced2\red\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\check_b2\runs\Map08_r01\advanced2\blue\attempt_001'
```

## 采用结论与下一条命令

采用：仅保留诊断类型修复及其测试，修复后3个完整阶段均无原异常。**红A2与蓝B2策略均拒绝替换基线**：p1全部候选异常；修复后的Map08红墙体/地图/总分与蓝处置均退步。A0/B0继续作为开发起点；这也不构成它们在所有复赛场景全面最好的证明。下一轮优先做机制消融，避免在失败组合上继续整包叠加。

旧p1的140个收据约束文件已核验，全部一致：[完整性核对](output/sf13/fix2/p1_integrity_check.json)。[采用决定及交付检查](output/sf13/fix2/decision.json)记录拒绝结论、源码一致性与图件检查。原始失败、快照、图件、录像不覆盖；所有新源码另建批次。正式入口、100张地图及划分、既有提交包未改。

新四图诊断批`output/sf13/p2`已prepare-only冻结，尚未运行。命令与此前相同，但输出已换到p2，不会接p1：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\semifinal_evaluate.cmd
```

任务量：Map08、Map09、yh_map_fs_1、yh_map_fs_2，各1轮、主种子742；红A0/A2、蓝B0/B2，8红＋最多16蓝=最多24阶段，单CPU/4GB/5400步/180秒。每份红布局固定给两蓝复用；红失败则保留失败并跳过其蓝阶段。仿真时间上限合计72分钟，另计进程启动、导出与报告开销。p2用于判断退步是否跨图复现，不能作为已采用新版发布。

也可以先发“继续”，针对Map08证据实施独立消融；若再改受哈希约束源码，则p2保留作废、另建新批，不能继续使用p2。本阶段工作结束，等待用户查看、返回批量结果或“继续”；无助手后台评测待续跑。
