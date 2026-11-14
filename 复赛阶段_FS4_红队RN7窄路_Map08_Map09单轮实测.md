# RN7 红队窄路独立候选：Map08 / Map09 单轮实测

每个新候选每图仅运行一次完整红阶段；原版来自 rs6_m89_once，未重跑挑分。蓝队难度、官方原始/有效总分均未评测。
同地图、种子、官方参数、资源限制；引擎随机 UID 未完全受种子控制，不能声称严格相同观测输入回放。

|地图|版本|红有效|M|ΔM|W|ΔW|T|弹数|墙/机接触|步数|秒|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Map08|A3 + RN7|True|67.8746|+8.1543|39.6307|+1.2668|22.8571|10|0/5|4536|127.55|
|Map08|B + RN7|True|61.3698|+3.4702|24.3340|-2.1520|22.8571|10|0/2|4536|125.27|
|Map08|C + RN7|True|55.2036|+0.3361|23.0607|-14.0111|22.8571|10|0/2|4536|134.98|
|Map09|A3 + RN7|True|56.6783|+12.9914|71.7526|+2.6267|22.8571|10|0/5|4536|125.51|
|Map09|B + RN7|True|68.0871|+18.2755|69.7727|+5.2204|25.1058|10|0/6|4451|128.46|
|Map09|C + RN7|True|59.5931|+20.6543|71.0298|+2.8517|22.8571|10|0/5|4536|145.08|

## Map08

下列三列依次为 A3+RN7、B+RN7、C+RN7；建图差异不能冒充完整激光观测覆盖。

### 提交建图差异

![Map08 提交建图差异](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map08_mapping.png)

### 新候选实际轨迹

![Map08 新候选实际轨迹](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map08_trajectories.png)

### 官方墙体覆盖

![Map08 官方墙体覆盖](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map08_walls.png)

### 实际落弹位置

![Map08 实际落弹位置](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map08_bombs.png)

### 原版与候选轨迹对照

![Map08 原版与候选轨迹对照](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map08_trajectory_before_after.png)

## Map09

下列三列依次为 A3+RN7、B+RN7、C+RN7；建图差异不能冒充完整激光观测覆盖。

### 提交建图差异

![Map09 提交建图差异](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map09_mapping.png)

### 新候选实际轨迹

![Map09 新候选实际轨迹](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map09_trajectories.png)

### 官方墙体覆盖

![Map09 官方墙体覆盖](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map09_walls.png)

### 实际落弹位置

![Map09 实际落弹位置](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map09_bombs.png)

### 原版与候选轨迹对照

![Map09 原版与候选轨迹对照](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_review/Map09_trajectory_before_after.png)

## 关注区域的实际进入证据

下面是原始录像中机体中心位于离线框选区域的无人机帧数；不是激光覆盖率，也不是独立访问次数。未记录完整逐帧传感器，不能精准区分所有漏墙成因。

|地图|版本|原版区域帧数|RN7区域帧数|局部任务进入/完成/活跃步|
|---|---|---|---|---|
|Map08|A3 + RN7|{'bottom_lanes': 0, 'bottom_right': 0}|{'bottom_lanes': 0, 'bottom_right': 0}|40/39/1061|
|Map08|B + RN7|{'bottom_lanes': 0, 'bottom_right': 0}|{'bottom_lanes': 0, 'bottom_right': 0}|40/31/1123|
|Map08|C + RN7|{'bottom_lanes': 0, 'bottom_right': 0}|{'bottom_lanes': 0, 'bottom_right': 0}|39/33/1262|
|Map09|A3 + RN7|{'upper_left': 64, 'middle_left': 903}|{'upper_left': 2769, 'middle_left': 3130}|40/37/1188|
|Map09|B + RN7|{'upper_left': 8, 'middle_left': 65}|{'upper_left': 2538, 'middle_left': 1428}|40/35/1381|
|Map09|C + RN7|{'upper_left': 0, 'middle_left': 235}|{'upper_left': 1268, 'middle_left': 1837}|40/37/1191|

## 原始图册与回放

[全六局原始结果与完整图册](E:/桌面/gitee/swarm-eod/output/sf13/rn7_m89_once/report.md)
蓝阶段未运行，因此没有本轮蓝回放；不以历史蓝录像代替。

Map08 / A3 + RN7 / r1 / seed 200806601

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rn7_m89_once\runs\Map08_r01\rn7_a\red\attempt_001'
```

Map08 / B + RN7 / r1 / seed 200806601

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rn7_m89_once\runs\Map08_r01\rn7_b\red\attempt_001'
```

Map08 / C + RN7 / r1 / seed 200806601

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rn7_m89_once\runs\Map08_r01\rn7_c\red\attempt_001'
```

Map09 / A3 + RN7 / r1 / seed 2477227837

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rn7_m89_once\runs\Map09_r01\rn7_a\red\attempt_001'
```

Map09 / B + RN7 / r1 / seed 2477227837

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rn7_m89_once\runs\Map09_r01\rn7_b\red\attempt_001'
```

Map09 / C + RN7 / r1 / seed 2477227837

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rn7_m89_once\runs\Map09_r01\rn7_c\red\attempt_001'
```


# RN7 窄路专项开发与验证记录

本轮新增 A3+RN7、B+RN7、C+RN7 三个独立类；原 A3 / S6-B / S6-C / advanced5_narrow、蓝队、共享文件、正式入口未由本任务修改。没有提交、推送、打包或切换分支。
当前正式放弹入口仍为 Advanced2；RN7 仅由专用评测注册。新增 Python 均在 solutions 目录。

## 证据与实现

- 源码事实：原导航为 20px 栅格、十字障碍膨胀，局部避障圆半径通常25px、紧通道22px。旧 N0 局部 BODY_RADIUS=10 / TRACK_RADIUS=12，不满足当前机身外接半径16.4012px，因此未直接恢复整包。
- RN7 复用 N0 的保守雷达扇区与有界路径搜索，但采用 2px、440×440px 当前机体系局部快照；机身外接半径16.4012、跟踪18.5、规划19.9142px。未知不可通行，路径整段检查，无对共享地图的改写。
- 两侧需要实测静态墙端点，不能以未知边界充当墙。候选端点偏好低访问、新观察和前进方向。每机最多8段、每段170步，任务失败冷却，队友等待7+3×机号步后交还原控制器。
- 路点每步用合法里程计距离、相对方向、转角转换；包含加减速和侧滑，不积分指令。导航位置/提交地图算法原样保留。新扫描不通时先刹车并局部重规划，可在已观察空间内横向修正局部路点12px；不修正全局定位。
- 保留放弹/探索主流程：生命≤10、放弹尝试、距目标弹点55px、归队、提交后、进度≥68%时不接管。B/C 的 S6 运动在 RN7 接管期间让出，合法传感器累计和原建图仍运行。开关 NARROW_ENABLED=False（rn7_off）；三个基线/对照版本均保留。
- 不能声称 Map09 左上角缺探索已被证明是窄路误判：主要入口约118px，旧录像缺少每帧规划栅格。Map08 最底部则确有约50～60px口与通道。精确官方真值栅格开口见 verification.json（仅离线）。
- 当前精细观察仍较保守：46px斜通道可能被拒绝；50px带真实噪声原生场景未通。不能承诺所有物理可通通道均可通过。

## 实际漏搜诊断与采用结论

Map08 的 A3+RN7 与 B+RN7 最底部区域均未进入。B+RN7 的录像距第一处下方入口中心最近3.8188px，仍未穿入下方走廊。第2868次控制的诊断样本可确认无人机1就在入口附近：GPS/罗盘可用，原导航位置误差16.7612px、提交地图器4.0469px。该机早在第1222步已启动第8段窄路任务、第1246步结束，之后受8段总预算限制不再启用RN7。
因此，存在“到达关键入口之前，窄路总预算已经用完”的直接代码与事件证据；但仅凭现有记录还不能证明这就是漏搜的唯一原因。下一步应先按空间去重/进展刷新预算，保留每段限时和队友等待上限，再对关键入口是否实际激活做验证；不应先全面缩小避障距离。
本轮三个版本均保留为实验候选，不切换正式入口。单轮收益不代表稳健胜出；图册同时记录地图、墙体、时间、接触和区域进入结果。用户指定各图各版只跑一次，本轮不修改参数重跑挑分，不再启动额外批量。

Map08 三候选真实墙命中相对各自原版分别减少664/734/1598像素，误画扣分像素分别减少55544/27628/8696；地图分提高主要由误画减少带来，不能称为观察到更多真实墙。Map08 的 B/C 墙体分分别下降2.1520/14.0111，C组合拒绝直接采用。
Map09 三候选真实墙命中分别增加18195/20097/21128像素，左上框选区域无人机帧数由64/8/0变为2769/2538/1268；实际录像可见三版都深入左上。B 在本轮 Map09 的地图分最高68.0871，但本轮只有单种子，也没有蓝队阶段。A3+RN7、B+RN7保留继续验证，C不作为通用替换推荐。

## 实际测试及保留的失败

首轮 tests_v1：13通过、2失败（46px斜通道过于乐观的通行期望）；保留XML。没有缩小安全半径强行通过。改为明确验证55px各方向与46px斜向保守拒绝。
最终 tests_final：16项通过，真实计时见 verification.json。覆盖观测/未知、机身外接半径、整段路径、侧滑转角累计、前墙刹车、有限队友等待、任务交还、失效里程计。理想几何测试不替代真实物理。
原生 v1/v2/v3 所有日志、结果保留；v1把越过终点但绕过25px通道的行为错记passed，v2修正为必须真正进入且未绕行。v1这个标签不作为通行证据。v3为最终代码小场景结果，5/5进程正常，全部零墙接触；50px理想、80px真实噪声通过，50px真实噪声未通过；25px拒绝并绕行，死胡同未穿墙。全部仅组件检查，不是官方地图成绩。

![原生物理小场景](E:/桌面/gitee/swarm-eod/output/sf13/rn7_delivery/native_cases.png)

## 文件与复现

新增策略：src/swarm_rescue/solutions/my_drone_red_rn7.py；专用 test_red_rn7.py、red_rn7_native.py、red_rn7_batch.py、red_rn7_review.py、red_rn7_delivery.py 位于 solutions/devtools。旧文件保护校验和冻结源码审计见 verification.json。
已运行命令（下列完整对局已完成，不要当作续跑命令再次启动）：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\.venv\Scripts\python.exe -B -m pytest src/swarm_rescue/solutions/devtools/test_red_rn7.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m swarm_rescue.solutions.devtools.red_rn7_batch --output output/sf13/rn7_m89_once --maps Map08 Map09 --controllers rn7_a rn7_b rn7_c --rounds 1 --run
```

任务量6完整红局、0蓝局，Map08/09各三候选单轮；资源1CPU/4GB、5400步/180秒。当前阶段不再自动启动评测。完整蓝队难度及总分均未评测。

## 官方像素分解及资源

|地图/候选|真实墙命中像素|误画扣分像素|地图credit / penalty|Job峰值MB|进程秒|回调异常|控制器崩溃|时间触限|
|---|---:|---:|---|---:|---:|---:|---|---|
|Map08/rn7_a|73042|95395|0.755722/0.076976|683.4|136.80|0|False|False|
|Map08/rn7_b|72368|132504|0.748748/0.135050|649.7|131.11|0|False|False|
|Map08/rn7_c|68383|141266|0.707518/0.155482|656.0|140.91|0|False|False|
|Map09/rn7_a|76520|115579|0.686235/0.119452|652.1|131.66|0|False|False|
|Map09/rn7_b|90590|135229|0.812415/0.131545|650.5|134.53|0|False|False|
|Map09/rn7_c|78608|111563|0.704960/0.109030|670.0|151.05|0|False|False|

## 保护与身份核验

原已登记文件SHA256检查 1361 个；变化列表：[]。冻结源文件验证通过；正式对局attempt共6个，未重试。图册10个内嵌图链接已逐一用图片解码验证。
冻结计划SHA256：`91d64a02492adb95b50ddc90cb5d800ce18e9bedf6652a2b9ec78fa5e0d49b4a`。red_rn7_delivery.py是冻结之后新增的离线交付工具，不属于比赛策略。
原来每组的结果收据由 review_row 复核，原提交图用官方地图公式重新评分并与原分一致；没有改写旧分、旧录像或旧源文件。
正式入口、共享定位/通信/评分/物理、蓝队 B7 未由本任务修改。未运行蓝队，不给蓝队强度背书。后续若继续，先针对过早用完预算和带噪窄门拒绝修正，再由用户决定新的多地图/多种子验证与完整蓝对局；本轮结果不得替代该验收。
