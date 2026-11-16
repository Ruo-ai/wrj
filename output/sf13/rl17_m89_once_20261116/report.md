# RN7 独立窄路候选逐轮图册

冻结B0=MyDroneBlueAdapted（开发参考，未验收为强对手）。无蓝阶段时，难度及官方原始/有效总分均未评测。
轨迹覆盖、激光观察范围与地图分不同；本图册不把轨迹扩张率当作雷达观测率。
实际接触次数见 motion_diagnostics.json；局部窄路事件见 narrow_diagnostics.json。

|图/轮/红版本|红有效|M|W|T|D|官方原始总分|官方有效总分|弹数|红步/秒|
|---|---|---:|---:|---:|---|---|---|---:|---|
|Map08/r1/rl11_lines|True|63.1415|24.3340|22.8571|未评测|未评测|未评测|10|4536/145.4154|
|Map08/r1/rl17_repair|True|63.7449|24.3340|22.8571|未评测|未评测|未评测|10|4536/145.7089|
|Map09/r1/rl11_lines|True|73.3676|69.7727|25.1058|未评测|未评测|未评测|10|4451/133.8913|
|Map09/r1/rl17_repair|True|74.4220|69.7727|25.1058|未评测|未评测|未评测|10|4451/135.2517|

### Map08 / rl11_lines / unknown blue / r1

本批结果；短测不计有效成绩。

红 attempt：`E:\桌面\gitee\swarm-eod\output\sf13\rl17_m89_once_20261116\runs\Map08_r01\rl11_lines\red\attempt_001`；蓝 attempt：`None`。
红种子：200806601；蓝种子：None。
红类：MyDroneRedLines11；蓝类：未知。
实际条件：红 `{"drones": 5, "zones": ["NO_GPS_ZONE"], "max_steps": 5400, "max_walltime": 180, "instantiated_zone_types": ["NO_GPS_ZONE"]}`；蓝 `null`。
红计划 SHA256：fe8c4f4ff871910503146105f939880ce86d2457204260794e2c06f935376b92；蓝计划 SHA256：未知；布局 SHA256：未知。

|地图分|墙体分|时间分|原始难度|原始总分|有效总分|红步/秒|蓝步/秒|
|---:|---:|---:|---:|---:|---:|---|---|
|63.1415|24.3340|22.8571|未知/未产生|未知/未产生|未知/未产生|4536/145.4154|None/未知/未产生|
有效完整成绩：False；红状态=ok，蓝状态=None；触限(红/蓝)=False/None；回调异常(红/蓝)=0/未知；控制器异常(红/蓝)=False/None。

|蓝分|处置|健康指标（官方原值）|发现/抓取次数|运输成功率/耗时|红/蓝峰值Job字节|红/蓝进程秒|
|---:|---:|---:|---|---|---|---|
|未知/未产生|未知/未产生|未知/未产生|未知/未产生/未知/未产生|未知（无事件记录）|739123200/未知/未产生|153.7810/未知/未产生|

建图/提交差异（不是完整探索轨迹）

![Map08 / rl11_lines / unknown blue / r1 建图/提交差异（不是完整探索轨迹）](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/runs/Map08_r01/rl11_lines/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_exploration_diff.png>)

墙体覆盖（原始覆盖率不同于归一墙体分）

![Map08 / rl11_lines / unknown blue / r1 墙体覆盖（原始覆盖率不同于归一墙体分）](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/runs/Map08_r01/rl11_lines/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_wall_destruction.png>)

实际初始落弹位置（编号仅为产物顺序，不推测所属机号；蓝队运输后的位置另列）

![Map08 / rl11_lines / unknown blue / r1 实际炸弹位置](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/figures/Map08_rl11_lines_r01_attempt_001.png>)

红队采样轨迹及定位误差（真值仅由离线工具记录；稀疏连线不表示逐步路径）

![Map08 / rl11_lines / unknown blue / r1 红定位审计](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/figures/Map08_rl11_lines_r01_red_pose.png>)

红队完整回放命令：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rl17_m89_once_20261116\runs\Map08_r01\rl11_lines\red\attempt_001'
```

蓝阶段未运行，没有可回放录像。


### Map08 / rl17_repair / unknown blue / r1

本批结果；短测不计有效成绩。

红 attempt：`E:\桌面\gitee\swarm-eod\output\sf13\rl17_m89_once_20261116\runs\Map08_r01\rl17_repair\red\attempt_001`；蓝 attempt：`None`。
红种子：200806601；蓝种子：None。
红类：MyDroneRedRepair17；蓝类：未知。
实际条件：红 `{"drones": 5, "zones": ["NO_GPS_ZONE"], "max_steps": 5400, "max_walltime": 180, "instantiated_zone_types": ["NO_GPS_ZONE"]}`；蓝 `null`。
红计划 SHA256：fe8c4f4ff871910503146105f939880ce86d2457204260794e2c06f935376b92；蓝计划 SHA256：未知；布局 SHA256：未知。

|地图分|墙体分|时间分|原始难度|原始总分|有效总分|红步/秒|蓝步/秒|
|---:|---:|---:|---:|---:|---:|---|---|
|63.7449|24.3340|22.8571|未知/未产生|未知/未产生|未知/未产生|4536/145.7089|None/未知/未产生|
有效完整成绩：False；红状态=ok，蓝状态=None；触限(红/蓝)=False/None；回调异常(红/蓝)=0/未知；控制器异常(红/蓝)=False/None。

|蓝分|处置|健康指标（官方原值）|发现/抓取次数|运输成功率/耗时|红/蓝峰值Job字节|红/蓝进程秒|
|---:|---:|---:|---|---|---|---|
|未知/未产生|未知/未产生|未知/未产生|未知/未产生/未知/未产生|未知（无事件记录）|741343232/未知/未产生|153.4220/未知/未产生|

建图/提交差异（不是完整探索轨迹）

![Map08 / rl17_repair / unknown blue / r1 建图/提交差异（不是完整探索轨迹）](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/runs/Map08_r01/rl17_repair/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_exploration_diff.png>)

墙体覆盖（原始覆盖率不同于归一墙体分）

![Map08 / rl17_repair / unknown blue / r1 墙体覆盖（原始覆盖率不同于归一墙体分）](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/runs/Map08_r01/rl17_repair/red/attempt_001/artifacts/team742_Map08_no_gps_zone_rd1_wall_destruction.png>)

实际初始落弹位置（编号仅为产物顺序，不推测所属机号；蓝队运输后的位置另列）

![Map08 / rl17_repair / unknown blue / r1 实际炸弹位置](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/figures/Map08_rl17_repair_r01_attempt_001.png>)

红队采样轨迹及定位误差（真值仅由离线工具记录；稀疏连线不表示逐步路径）

![Map08 / rl17_repair / unknown blue / r1 红定位审计](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/figures/Map08_rl17_repair_r01_red_pose.png>)

红队完整回放命令：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rl17_m89_once_20261116\runs\Map08_r01\rl17_repair\red\attempt_001'
```

蓝阶段未运行，没有可回放录像。


### Map09 / rl11_lines / unknown blue / r1

本批结果；短测不计有效成绩。

红 attempt：`E:\桌面\gitee\swarm-eod\output\sf13\rl17_m89_once_20261116\runs\Map09_r01\rl11_lines\red\attempt_001`；蓝 attempt：`None`。
红种子：2477227837；蓝种子：None。
红类：MyDroneRedLines11；蓝类：未知。
实际条件：红 `{"drones": 5, "zones": ["NO_COM_ZONE", "NO_GPS_ZONE"], "max_steps": 5400, "max_walltime": 180, "instantiated_zone_types": ["NO_COM_ZONE", "NO_GPS_ZONE"]}`；蓝 `null`。
红计划 SHA256：fe8c4f4ff871910503146105f939880ce86d2457204260794e2c06f935376b92；蓝计划 SHA256：未知；布局 SHA256：未知。

|地图分|墙体分|时间分|原始难度|原始总分|有效总分|红步/秒|蓝步/秒|
|---:|---:|---:|---:|---:|---:|---|---|
|73.3676|69.7727|25.1058|未知/未产生|未知/未产生|未知/未产生|4451/133.8913|None/未知/未产生|
有效完整成绩：False；红状态=ok，蓝状态=None；触限(红/蓝)=False/None；回调异常(红/蓝)=0/未知；控制器异常(红/蓝)=False/None。

|蓝分|处置|健康指标（官方原值）|发现/抓取次数|运输成功率/耗时|红/蓝峰值Job字节|红/蓝进程秒|
|---:|---:|---:|---|---|---|---|
|未知/未产生|未知/未产生|未知/未产生|未知/未产生/未知/未产生|未知（无事件记录）|740306944/未知/未产生|141.4070/未知/未产生|

建图/提交差异（不是完整探索轨迹）

![Map09 / rl11_lines / unknown blue / r1 建图/提交差异（不是完整探索轨迹）](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/runs/Map09_r01/rl11_lines/red/attempt_001/artifacts/team742_Map09_no_com_zone__no_gps_zone_rd1_exploration_diff.png>)

墙体覆盖（原始覆盖率不同于归一墙体分）

![Map09 / rl11_lines / unknown blue / r1 墙体覆盖（原始覆盖率不同于归一墙体分）](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/runs/Map09_r01/rl11_lines/red/attempt_001/artifacts/team742_Map09_no_com_zone__no_gps_zone_rd1_wall_destruction.png>)

实际初始落弹位置（编号仅为产物顺序，不推测所属机号；蓝队运输后的位置另列）

![Map09 / rl11_lines / unknown blue / r1 实际炸弹位置](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/figures/Map09_rl11_lines_r01_attempt_001.png>)

红队采样轨迹及定位误差（真值仅由离线工具记录；稀疏连线不表示逐步路径）

![Map09 / rl11_lines / unknown blue / r1 红定位审计](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/figures/Map09_rl11_lines_r01_red_pose.png>)

红队完整回放命令：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rl17_m89_once_20261116\runs\Map09_r01\rl11_lines\red\attempt_001'
```

蓝阶段未运行，没有可回放录像。


### Map09 / rl17_repair / unknown blue / r1

本批结果；短测不计有效成绩。

红 attempt：`E:\桌面\gitee\swarm-eod\output\sf13\rl17_m89_once_20261116\runs\Map09_r01\rl17_repair\red\attempt_001`；蓝 attempt：`None`。
红种子：2477227837；蓝种子：None。
红类：MyDroneRedRepair17；蓝类：未知。
实际条件：红 `{"drones": 5, "zones": ["NO_COM_ZONE", "NO_GPS_ZONE"], "max_steps": 5400, "max_walltime": 180, "instantiated_zone_types": ["NO_COM_ZONE", "NO_GPS_ZONE"]}`；蓝 `null`。
红计划 SHA256：fe8c4f4ff871910503146105f939880ce86d2457204260794e2c06f935376b92；蓝计划 SHA256：未知；布局 SHA256：未知。

|地图分|墙体分|时间分|原始难度|原始总分|有效总分|红步/秒|蓝步/秒|
|---:|---:|---:|---:|---:|---:|---|---|
|74.4220|69.7727|25.1058|未知/未产生|未知/未产生|未知/未产生|4451/135.2517|None/未知/未产生|
有效完整成绩：False；红状态=ok，蓝状态=None；触限(红/蓝)=False/None；回调异常(红/蓝)=0/未知；控制器异常(红/蓝)=False/None。

|蓝分|处置|健康指标（官方原值）|发现/抓取次数|运输成功率/耗时|红/蓝峰值Job字节|红/蓝进程秒|
|---:|---:|---:|---|---|---|---|
|未知/未产生|未知/未产生|未知/未产生|未知/未产生/未知/未产生|未知（无事件记录）|738156544/未知/未产生|142.7340/未知/未产生|

建图/提交差异（不是完整探索轨迹）

![Map09 / rl17_repair / unknown blue / r1 建图/提交差异（不是完整探索轨迹）](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/runs/Map09_r01/rl17_repair/red/attempt_001/artifacts/team742_Map09_no_com_zone__no_gps_zone_rd1_exploration_diff.png>)

墙体覆盖（原始覆盖率不同于归一墙体分）

![Map09 / rl17_repair / unknown blue / r1 墙体覆盖（原始覆盖率不同于归一墙体分）](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/runs/Map09_r01/rl17_repair/red/attempt_001/artifacts/team742_Map09_no_com_zone__no_gps_zone_rd1_wall_destruction.png>)

实际初始落弹位置（编号仅为产物顺序，不推测所属机号；蓝队运输后的位置另列）

![Map09 / rl17_repair / unknown blue / r1 实际炸弹位置](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/figures/Map09_rl17_repair_r01_attempt_001.png>)

红队采样轨迹及定位误差（真值仅由离线工具记录；稀疏连线不表示逐步路径）

![Map09 / rl17_repair / unknown blue / r1 红定位审计](<E:/桌面/gitee/swarm-eod/output/sf13/rl17_m89_once_20261116/figures/Map09_rl17_repair_r01_red_pose.png>)

红队完整回放命令：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\rl17_m89_once_20261116\runs\Map09_r01\rl17_repair\red\attempt_001'
```

蓝阶段未运行，没有可回放录像。

