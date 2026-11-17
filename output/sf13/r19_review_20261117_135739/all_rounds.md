# R19 A 全轮次配对图册

来源 `E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636`；[总报告](E:/桌面/gitee/swarm-eod/复赛阶段_R19_配对红队与恢复权重结果核验_20261117.md)。24/24有效，12对全部展示，左RL11、右RL17-R。
每对实际区域均为NO_COM_ZONE+NO_GPS_ZONE，所有D及完整总分未评测。图中“配对记录相同”指探索采样/轨迹状态/弹位；墙钟时间戳不同。
图像视野和比例相同。离线真值不参与策略。轨迹按录像槽位标色，炸弹编号只是产物列表顺序。

## yh_map_fs_24 / r1 / seed 2026111701

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|77.0690|53.6058|30.0265|75508/5138/120472|10|4265/128.084|未评测/未评测|
|RL17-R|78.7656|53.6058|30.0265|75411/5235/113247|10|4265/127.097|未评测/未评测|

![yh_map_fs_24 r1 seed2026111701 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r1_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_24 r1 seed2026111701 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r1_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_24 r1 seed2026111701 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r1_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_24 r1 seed2026111701 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r1_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_24 r1 seed2026111701 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r1_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_24_r01\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_24_r01\rl17_repair\red\attempt_001'
```

## yh_map_fs_24 / r2 / seed 2026111702

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|83.4352|57.1975|22.8571|74958/5688/80273|10|4536/129.124|未评测/未评测|
|RL17-R|83.9695|57.1975|22.8571|74937/5709/77054|10|4536/128.469|未评测/未评测|

![yh_map_fs_24 r2 seed2026111702 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r2_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_24 r2 seed2026111702 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r2_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_24 r2 seed2026111702 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r2_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_24 r2 seed2026111702 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r2_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_24 r2 seed2026111702 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_24_r2_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_24_r02\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_24_r02\rl17_repair\red\attempt_001'
```

## yh_map_fs_37 / r1 / seed 2026111701

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|57.3600|51.2577|22.8571|62928/25844/111240|10|4536/131.680|未评测/未评测|
|RL17-R|57.1753|51.2577|22.8571|62014/26758/109159|10|4536/132.924|未评测/未评测|

![yh_map_fs_37 r1 seed2026111701 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r1_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_37 r1 seed2026111701 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r1_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_37 r1 seed2026111701 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r1_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_37 r1 seed2026111701 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r1_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_37 r1 seed2026111701 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r1_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_37_r01\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_37_r01\rl17_repair\red\attempt_001'
```

## yh_map_fs_37 / r2 / seed 2026111702

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|59.7194|59.2088|22.8571|65389/23383/113762|10|4536/131.896|未评测/未评测|
|RL17-R|59.8362|59.2088|22.8571|65841/22931/115739|10|4536/132.068|未评测/未评测|

![yh_map_fs_37 r2 seed2026111702 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r2_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_37 r2 seed2026111702 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r2_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_37 r2 seed2026111702 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r2_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_37 r2 seed2026111702 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r2_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_37 r2 seed2026111702 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_37_r2_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_37_r02\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_37_r02\rl17_repair\red\attempt_001'
```

## yh_map_fs_27 / r1 / seed 2026111701

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|25.1309|34.4693|37.8042|28867/70296/49005|10|3971/118.161|未评测/未评测|
|RL17-R|25.1094|34.4693|37.8042|28842/70321/48612|10|3971/121.432|未评测/未评测|

![yh_map_fs_27 r1 seed2026111701 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r1_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_27 r1 seed2026111701 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r1_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_27 r1 seed2026111701 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r1_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_27 r1 seed2026111701 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r1_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_27 r1 seed2026111701 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r1_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_27_r01\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_27_r01\rl17_repair\red\attempt_001'
```

## yh_map_fs_27 / r2 / seed 2026111702

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|18.6498|30.6025|39.7090|23772/75391/56291|10|3899/115.946|未评测/未评测|
|RL17-R|18.6784|30.6025|39.7090|23763/75400/55889|10|3899/116.403|未评测/未评测|

![yh_map_fs_27 r2 seed2026111702 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r2_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_27 r2 seed2026111702 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r2_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_27 r2 seed2026111702 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r2_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_27 r2 seed2026111702 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r2_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_27 r2 seed2026111702 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_27_r2_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_27_r02\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_27_r02\rl17_repair\red\attempt_001'
```

## yh_map_fs_72 / r1 / seed 2026111701

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|9.6836|29.9886|22.8571|13016/77700/35326|10|4536/118.276|未评测/未评测|
|RL17-R|9.6836|29.9886|22.8571|13016/77700/35326|10|4536/123.965|未评测/未评测|

![yh_map_fs_72 r1 seed2026111701 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r1_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_72 r1 seed2026111701 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r1_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_72 r1 seed2026111701 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r1_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_72 r1 seed2026111701 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r1_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_72 r1 seed2026111701 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r1_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_72_r01\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_72_r01\rl17_repair\red\attempt_001'
```

## yh_map_fs_72 / r2 / seed 2026111702

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|10.6640|24.8142|40.0000|13049/77667/31527|10|3888/109.375|未评测/未评测|
|RL17-R|10.6640|24.8142|40.0000|13049/77667/31527|10|3888/106.614|未评测/未评测|

![yh_map_fs_72 r2 seed2026111702 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r2_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_72 r2 seed2026111702 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r2_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_72 r2 seed2026111702 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r2_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_72 r2 seed2026111702 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r2_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_72 r2 seed2026111702 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_72_r2_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_72_r02\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_72_r02\rl17_repair\red\attempt_001'
```

## yh_map_fs_55 / r1 / seed 2026111701

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|15.9248|31.3233|22.8571|24935/83177/67246|10|4536/126.726|未评测/未评测|
|RL17-R|16.4214|31.3233|22.8571|24777/83335/62979|10|4536/131.143|未评测/未评测|

![yh_map_fs_55 r1 seed2026111701 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r1_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_55 r1 seed2026111701 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r1_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_55 r1 seed2026111701 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r1_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_55 r1 seed2026111701 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r1_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_55 r1 seed2026111701 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r1_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_55_r01\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_55_r01\rl17_repair\red\attempt_001'
```

## yh_map_fs_55 / r2 / seed 2026111702

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|16.8154|39.8499|22.8571|22572/85540/42598|10|4536/124.923|未评测/未评测|
|RL17-R|16.8154|39.8499|22.8571|22572/85540/42598|10|4536/124.002|未评测/未评测|

![yh_map_fs_55 r2 seed2026111702 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r2_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_55 r2 seed2026111702 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r2_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_55 r2 seed2026111702 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r2_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_55 r2 seed2026111702 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r2_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_55 r2 seed2026111702 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_55_r2_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_55_r02\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_55_r02\rl17_repair\red\attempt_001'
```

## yh_map_fs_90 / r1 / seed 2026111701

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|26.7826|39.7852|22.8571|31319/79224/29676|10|4536/120.867|未评测/未评测|
|RL17-R|26.7826|39.7852|22.8571|31319/79224/29676|10|4536/120.949|未评测/未评测|

![yh_map_fs_90 r1 seed2026111701 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r1_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_90 r1 seed2026111701 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r1_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_90 r1 seed2026111701 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r1_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_90 r1 seed2026111701 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r1_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_90 r1 seed2026111701 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r1_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_90_r01\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_90_r01\rl17_repair\red\attempt_001'
```

## yh_map_fs_90 / r2 / seed 2026111702

完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。

|版本|M|W|T|命中/漏画/受罚|弹|步/引擎秒|D/完整总分|
|---|---|---|---|---|---|---|---|
|RL11|20.3127|35.3667|22.8571|24344/86199/26368|10|4536/120.052|未评测/未评测|
|RL17-R|20.3127|35.3667|22.8571|24344/86199/26368|10|4536/120.050|未评测/未评测|

![yh_map_fs_90 r2 seed2026111702 RL11 vs RL17-R mapping](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r2_mapping.png)

绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。

![yh_map_fs_90 r2 seed2026111702 RL11 vs RL17-R exploration](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r2_exploration.png)

浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。

![yh_map_fs_90 r2 seed2026111702 RL11 vs RL17-R trajectories](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r2_trajectories.png)

颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。

![yh_map_fs_90 r2 seed2026111702 RL11 vs RL17-R walls](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r2_walls.png)

官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。

![yh_map_fs_90 r2 seed2026111702 RL11 vs RL17-R bombs](E:/桌面/gitee/swarm-eod/output/sf13/r19_review_20261117_135739/figures/yh_map_fs_90_r2_bombs.png)

蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。

RL11 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_90_r02\rl11_lines\red\attempt_001'
```

RL17-R 原始红阶段回放：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r19a_20261117_123636\runs\yh_map_fs_90_r02\rl17_repair\red\attempt_001'
```
