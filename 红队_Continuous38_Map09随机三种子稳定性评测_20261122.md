# Continuous38：Map09随机三种子稳定性评测

按用户要求，使用上一轮冻结的Continuous38原代码，在Map09新抽取3个随机种子，各新执行一次完整红阶段。恢复并保持官方无GPS/无罗盘、无通信区域；沿用原生CPU雷达、1CPU、4GB、5400步/180秒上限。没有修改算法、没有重试或换种子。

种子为 **205258673, 2315063155, 4233377800**，由系统随机源 `secrets.randbits(32)` 产生，在任何飞行前写入并冻结；与地图哈希和日期无关。抽样只排除重复及上一诊断种子1492628140。

新红3场、新蓝0场。有效红阶段 **3/3**；有效地图分均值 **46.93878**，最低 **37.95112**，最高 **61.60675**，极差 **23.65563**，总体标准差 **10.45934**。上一轮75.05072是单独诊断样本，没有混入这些统计。三个样本不能证明所有种子稳定。

**本批结果的判断**

这版尚未实现随机种子下的稳定高分。实际三场跨度23.65563分，全部低于上一诊断样本75.05072。没有重跑旧版同种子配对，不能从本批直接量化相对旧算法的随机种子增益。

本批两个低分组的主要损失之一是终图未交付：第1组缺2号机终图；第3组缺2号和4号终图。保持各场实际探索观测不变，仅用实际已经封存的终图替换未交付旧版、按原渲染和补边重新汇总，离线地图分分别为62.41945、71.70273，较实际提交高24.46833、30.44426分。这是交付反事实诊断，不是新实测分数，也不是对修复通信后收益的保证。

第2组终图5/5、集合ACK5/5，但地图分仍只有61.60675；它的北侧深处到访91格、西北深处9格，明显少于上一诊断样本的145/54格。第1/3组西北深处到访44/51格，最终地图却缺北侧大片内容，进一步提示“飞过”和“交付到最终图”必须分开检查。

连续定位也未完全消除长尾：三场失定位航向P95为12.99°、4.44°、7.73°；位置P95为75.01、23.68、63.06px。继续保留Continuous38和这三场原始证据，本轮不修改算法、不替换正式ZIP、不追加测试。

**红阶段实际分项；未评测蓝队和完整总分**

|组|随机种子|有效地图分|墙体分|时间分|实际布弹|步数|耗时秒|最新源图|集合ACK|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|205258673|37.95112|70.40585|3.57143|10/10|5265|163.81|4/5|0/5|
|2|2315063155|61.60675|80.58041|19.86772|10/10|4649|152.62|5/5|5/5|
|3|4233377800|41.25847|70.70319|3.57143|10/10|5265|166.61|3/5|0/5|

红阶段原生partial总分不作为完整总分，蓝分和难度终分均未测。每组提交图来自实际提交文件；布弹为实际落点；轨迹来自原生录像。ACK是算法源图集合确认，不是另一个官方分数条件。

**探索、误画与漂移**

统一统计到原生第4644步。到访是40px格内机身中心到达，不是雷达覆盖率；北侧深处取图像y<320，西北深处另要求x<622，仅为赛后分区。漂移使用封存前每10步、GPS/罗盘失效时刻，与控制后/物理前的真值对齐。

|组|真墙命中|误画扣分|全图到访格|北侧深处格|西北深处格|航向P95|位置P95|补边新增像素|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|47,841|4.95291|596|106|44|12.99°|75.01px|0|
|2|74,016|4.77114|569|91|9|4.44°|23.68px|27,347|
|3|48,677|2.39529|600|95|51|7.73°|63.06px|0|

**交付损失的离线检查（不计入实测成绩）**

使用每架机实际在截止前封存的源图，离线替换未交付旧版，再用相同最终渲染和补边生成诊断图。先验证实收源图重放能逐像素复现实际提交；没有增加观测，没有新飞行。下表只帮助判断交付缺口的影响，不能把假设分数当实际成绩或理论上界。

|组|实际地图分|假设全部封存源图到达的离线地图分|离线变化|
|---|---:|---:|---:|
|1|37.95112|62.41945|24.46833|
|2|61.60675|61.60675|0.00000|
|3|41.25847|71.70273|30.44426|

![三组到访曲线](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/coverage.png)

![三组漂移](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/drift.png)

**逐组图示与实际交付版本**

导航图是赛末20px局部规划图，保留Clean日常清噪、没有最终提交规整和补边；粗格不是最终墙体厚度。源图只显示提交者收到的版本，不补入失联飞机尚未交付的数据。

**第1组：种子205258673**

有效红阶段：True；实际实收：0号=4643（终版4643），1号=4643（终版4643），2号=1710（终版4643），3号=4643（终版4643），4号=4643（终版4643）。未收到终图的来源：['2']。

![提交图、炸弹图、轨迹图](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r01_s205258673/submission_bombs_trajectory.png)

![地图误差与原生墙体覆盖](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r01_s205258673/errors_wall.png)

![五机导航图](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r01_s205258673/navigation_all5.png)

![实际收到的五源地图](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r01_s205258673/received_sources.png)

[本组原始数据](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r01_s205258673/data.json)

**第2组：种子2315063155**

有效红阶段：True；实际实收：0号=4643（终版4643），1号=4643（终版4643），2号=4643（终版4643），3号=4643（终版4643），4号=4643（终版4643）。未收到终图的来源：[]。

![提交图、炸弹图、轨迹图](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r02_s2315063155/submission_bombs_trajectory.png)

![地图误差与原生墙体覆盖](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r02_s2315063155/errors_wall.png)

![五机导航图](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r02_s2315063155/navigation_all5.png)

![实际收到的五源地图](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r02_s2315063155/received_sources.png)

[本组原始数据](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r02_s2315063155/data.json)

**第3组：种子4233377800**

有效红阶段：True；实际实收：0号=4643（终版4643），1号=4643（终版4643），2号=440（终版4643），3号=4643（终版4643），4号=3120（终版4643）。未收到终图的来源：['2', '4']。

![提交图、炸弹图、轨迹图](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r03_s4233377800/submission_bombs_trajectory.png)

![地图误差与原生墙体覆盖](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r03_s4233377800/errors_wall.png)

![五机导航图](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r03_s4233377800/navigation_all5.png)

![实际收到的五源地图](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r03_s4233377800/received_sources.png)

[本组原始数据](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/r03_s4233377800/data.json)

**冻结与核验**

算法及旧实测源码身份保持不变。42项产物检查通过，覆盖随机抽样冻结、全部已存文件哈希、真实类身份、资源/区域、实际提交像素、共享位姿及日志。没有额外飞行，也没有用这三场结果调代码。

[冻结计划](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/plan.json) · [随机抽样原记录](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/seed_draw.json) · [完成回执](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/completion.json) · [核验](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/audit.json) · [完整数据](E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/review/data.json)

以下仅回放已保存的红队录像，不新增仿真；本轮无蓝阶段录像。

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
$env:PYTHONPATH='src'
$env:PYGLET_SHADOW_WINDOW='false'
$env:PYTHONIOENCODING='utf-8'
.venv/Scripts/python.exe -m swarm_rescue.solutions.devtools.optimization.evaluation.replay 'E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/runs/Map09_r01/candidate_continuous38/red/attempt_001'
.venv/Scripts/python.exe -m swarm_rescue.solutions.devtools.optimization.evaluation.replay 'E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/runs/Map09_r02/candidate_continuous38/red/attempt_001'
.venv/Scripts/python.exe -m swarm_rescue.solutions.devtools.optimization.evaluation.replay 'E:/桌面/gitee/swarm-eod/output/sf13/continuous38_random3_20261122/runs/Map09_r03/candidate_continuous38/red/attempt_001'
```
