# A5完整红局复核：补齐真实地图分证据

2026-11-13。上轮只有128步原生冒烟，未提交地图、未落弹，不能评价红队地图效果。本轮补充同图同种子的完整红阶段对照，不把合成分或空白短测图当比赛结果。

预算预先登记：Map08/r1/seed200806601，原A3=`advanced2_mapping`、独立A5=`a5_anchor`各1次完整红阶段，共2红、0蓝；串行、1CPU、4GB、官方5400步/180秒，进程硬超时240秒。使用新的output/sf13/ra5_full01，另建ra5_full_review保存复核。开启有界锚定诊断，保留所有失败。不会自动扩图、参数搜索或启动大批量。

本轮不更改候选源码、蓝队或共享文件，不切入口，不提交/打包。蓝B6最新交付为71项测试通过、0完整局；源码最后改动18:56–18:58、交付文档19:04，本轮19:26检查未见可见Python进程，b6_ready_1113无attempt。CIM/tasklist查询受环境权限限制，不能声称已取得系统级完整进程命令；结合可见进程、文件稳定及蓝队完成报告判断，可推进上述两次串行小规模检查。

A3/A4历史最近完整红对照仍为ra4_check/Map08/r1，M59.7202→61.2287、W38.3640、T29.7090、10枚炸弹；不是蓝难度或官方完整总分。

## 最终实测结论

**两次完整红阶段均已完成并有效。A5地图分58.631855，比同批A3的59.720212下降1.088357，拒绝采用当前A5。** 不是只有冒烟，也不是合成分。原始输出、日志、录像、源码快照和全部失败检查保留；没有运行蓝阶段，D及官方原始/有效完整总分仍未评测。

|Map08/r1，seed200806601|A3 advanced2_mapping|A5 a5_anchor|A5−A3|
|---|---:|---:|---:|
|地图分M|59.720212|58.631855|**−1.088357**|
|墙体分W|38.363954|38.363954|0|
|时间分T|29.708995|29.708995|0|
|实际落弹/初始总库存|10/10|10/10|0|
|实际步数|4277|4277|0|
|引擎墙钟秒|108.217177|109.243496|+1.026320|
|进程CPU秒|114.000|114.782|+0.782|
|峰值Job提交字节|675483648|640094208|均低于4GB|
|退出码/回调异常|0 / 0|0 / 0|均正常|
|控制器崩溃/墙钟触限|False / False|False / False|均无|
|红分项有效|是|是|提交/shape/phase1和异常检查通过|
|D / 官方原始总分 / 官方有效总分|未评测|未评测|无蓝阶段，不拼接历史蓝分|
|独立碰撞次数|未记录|未记录|不能用健康或回调数冒充|

两局实际5架红机、每机2弹、NO_GPS_ZONE；官方上限5400步/180秒，非smoke；单CPU/4GB由公共资源限制器施加。A3本次分项精确复现ra4_check历史值，因此不是因基线悄悄更换造成A5退步。引擎墙钟不等于官方T的输入口径，实际T保持相同；本次时钟差不足以作跨机器性能承诺。

[原始逐轮结果](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_full01/results.json>) · [两版完整图册](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_full01/report.md>) · [独立收据/布局/轨迹核验](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_full_review/verification.json>)

## 地图分为什么下降

使用原官方评分代码只读重算冻结提交地图，均与比赛结果一致。不是另行定义指标：

|官方建图量|A3|A5|变化|
|---|---:|---:|---:|
|命中信用分|76.259157|76.426768|+0.167612|
|误画扣分|16.538944|17.794913|+1.255969|
|真墙命中像素|73706|73868|+162|
|扣分预测像素|150939|158862|+7923|
|漏墙像素|22946|22784|−162|
|映射位姿采样平均误差px|14.823355|15.513255|+0.689900|
|导航位姿采样平均误差px|18.090804|18.090804|0|
|引擎轨迹扩张覆盖指标%|96.519062|96.519062|0；不是雷达可见率|

净地图变化=+0.167612信用−1.255969额外扣分=−1.088357。**本真实局的直接损失是新增误画超过命中收益，不能再把合成夹具的墙厚命中损失当成A5真实回退的主因。** 本文覆盖上轮报告中基于合成反例的优先级建议；合成反例仍保留，但不代替真实局证据。

两版录制的4276帧drone_states、bomb_states逐元素相同；实际落弹JSON哈希均为`0622aaacbb2e6646458c830868ed682d8babcc44074222f8a82f17c4cccc0021`。因此本次可以把差异定位到建图链路，而非实际路径或布弹改变。但录像不含完整传感器，仍不称“同输入传感器回放”。位姿误差为控制后的稀疏采样，不冒充每次激光配准误差。

按距最近预测像素分桶，A3→A5漏墙0–10px为8222→8025、10–30px为2907→2785、30–60px为3992→4439、>60px为7825→7535。距离桶只能描述最终图缺口，不能精确分成未观察/错位/反证误删。仍缺逐扫描与处理前后证据。

实际锚定统计：61个恢复窗口、60次提交锚定、1次回退、1次中断、0次一致性拒绝；最大pending196帧、恢复样本最多12。A5机制确实在真实比赛触发。零拒绝并不表示绝对定位正确；时间相关噪声的持续偏差可能通过门槛，等待期间局部匹配及原片段插值也会改变误差传播。这些是待分离假设，不能单凭本统计认定具体一次回锁导致了哪片假墙。

A5五机mapper累计update7.739339秒/21380次调用；pending数组每32步抽样峰值2,095,792字节（不含对象/匹配器，不是进程峰值）。整体资源见上表。没有新增超时或漏弹，但地图分门槛未通过，不能据此采用。

## 本次真实图件

建图对照，上排A3、下排A5，依次为官方真墙、实际提交、评分差异；绿色命中、蓝色漏墙、橙色近墙预测、红色扣分预测。A5右侧区域仍有明显重影/误画；没有完整观测记录，不能从图片单独断言是墙后未搜或哪一帧擦除。

![真实完整红局A3与A5建图对比](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_full_review/map_audit/Map08_comparison.png>)

墙体覆盖（A5，两版布局相同）：实际10枚炸弹；原始加权覆盖约27.6%，归一化官方W=38.3640，二者不能混用。爆炸圆相交本身不能证明墙体收益重复。

![A5完整红局墙体覆盖](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_full01/runs/Map08_r01/a5_anchor/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_wall_destruction.png>)

实际初始落弹位置：10个实际点及135px爆炸半径，灰色为特殊区；编号为产物顺序，不臆测无人机归属。

![A5完整红局十枚实际炸弹](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_full01/figures/Map08_a5_anchor_r01_attempt_001.png>)

稀疏导航位姿轨迹（两版实际轨迹一致，不是雷达覆盖）：

![A5完整红局位姿轨迹](<E:/桌面/gitee/swarm-eod/output/sf13/ra5_full01/figures/Map08_a5_anchor_r01_red_pose.png>)

完整图册已包含A3/A5每组的三类核心图、轨迹和逐轮红回放；没有新蓝图，不补造。

## 诊断缺口及独立修复

发现上轮红专用评测器的`--anchor-diagnostics`仅在mapper初始化中设RECORD_ANCHORS=True，随后候选控制器构造又用类默认False覆盖。因此本局**统计量、计时和内存抽样存在，详细anchor_events为空**。最初离线汇总对空事件求min报错，未写入结果；改为明确标缺失后完成核验。这个离线诊断缺口不影响已记录的官方地图分、落弹、轨迹或运行有效性。

不改被测候选和旧运行器，新增两个红专用文件：

- `src/swarm_rescue/solutions/devtools/red_anchor_event_batch.py`：进程内设置控制器类的可选事件标志；继承原评测器，子进程也进入新入口，GO/系统资源限制完成后才导入策略；不变更算法。
- `src/swarm_rescue/solutions/devtools/test_red_anchor_event_batch.py`：执行实际A5构造/恢复链路，确认启用后写出事件、关闭后不写、地图/位姿/stats一致，异常退出恢复类标志。

修复测试与既有A5组件检查实际运行：21 passed、2 xfailed，pytest2.48秒，见`ra5_full_review/event_tests.xml`。两个xfail仍是上轮未满足的合成地图命中门槛，没有被删除或改称通过。新增诊断入口未重跑完整局，因此本局缺失的事件不能追补；后续采集使用新入口。本次真实比赛使用的原源码快照保持原样。

随后补了子进程延迟激活/退出恢复检查，最终诊断入口3项测试全部通过（1.74秒，`event_entry_tests.xml`），`--help`实际执行成功；没有因此追加比赛。实际测试命令如下，重跑应换新的输出名以保留本次XML：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider src/swarm_rescue/solutions/devtools/test_red_anchor_event_batch.py src/swarm_rescue/solutions/devtools/test_red_mapping_anchor.py --junitxml=output/sf13/ra5_full_review/event_tests.xml
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider src/swarm_rescue/solutions/devtools/test_red_anchor_event_batch.py --basetemp=output/sf13/ra5_full_review/entry_test_tmp --junitxml=output/sf13/ra5_full_review/event_entry_tests.xml
```

所有1431项既存文件在两局结束核验时无变化，冻结源文件哈希也一致；随后只继续填写本新报告，并新增上述两个红专用文件。旧A3/A4/A5、蓝代码/文档/输出、共享文件、正式入口、队伍742和提交包均未编辑。

## 命令与下一步

已实际执行的完整对照命令如下；目录现在已完成，**不要覆盖或续接**：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\.venv\Scripts\python.exe -B -m swarm_rescue.solutions.devtools.red_mapping_anchor_batch --output output/sf13/ra5_full01 --maps Map08 --controllers advanced2_mapping a5_anchor --rounds 1 --anchor-diagnostics --run
```

两次真实红回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\ra5_full01\runs\Map08_r01\advanced2_mapping\red\attempt_001'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\ra5_full01\runs\Map08_r01\a5_anchor\red\attempt_001'
```

本批未运行蓝阶段，无对应蓝回放。已有历史B0录像可看，但不能算A5对局：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\b5_compare\adapted\runs\Map08_r01\advanced2\blue\attempt_001'
```

下一用户批次优先完成A3/A4四图，而不是扩大本次已回退的A5：Map08/Map09/训练yh_map_fs_1/yh_map_fs_2×A3/A4×1轮，**8红+最多8蓝=最多16阶段**，固定开发参考B0 `MyDroneBlueAdapted`，同图同轮种子相同，70/15/15划分保持。B6尚无完整验收，不擅自换进来。蓝队计算空闲且双方源码稳定后运行；助手未启动该批。

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
$RedNextOut = 'output/sf13/r4_' + (Get-Date -Format 'MMddHHmmss')
if (Test-Path -LiteralPath $RedNextOut) { throw '目录已存在，请换新目录' }
.\.venv\Scripts\python.exe -B -m swarm_rescue.solutions.devtools.red_anchor_event_batch --output $RedNextOut --maps Map08 Map09 yh_map_fs_1 yh_map_fs_2 --controllers advanced2_mapping a4_cells --rounds 1 --full-pairs --run
```

该批成功产出后，首图首轮回放命令为（使用同一PowerShell中的变量）：

```powershell
.\replay_map.cmd "$RedNextOut/runs/Map08_r01/a4_cells/red/attempt_001"
.\replay_map.cmd "$RedNextOut/runs/Map08_r01/a4_cells/blue/attempt_001"
```

当前决策：拒绝采用A5，不与A4组合；保留所有代码和结果。A4仍只是Map08单轮M=61.2287的待跨图候选，不能称全面最优。后续开发应先定位真实锚定/反证链路中额外误画的来源，或独立评估探索，不再把合成误差降低当真实地图提升。本阶段未自动扩图、参数搜索、切入口、打包、提交或推送；等待用户批测或“继续”。
