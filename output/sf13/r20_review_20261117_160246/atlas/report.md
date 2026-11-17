# R20 三版本红阶段图册

实际完整红阶段；逐局失败原样保留。
蓝队难度分及完整总分均未评测。实际雷达覆盖为每8步/4px栅格的物理射线采样近似，非精确覆盖率。
真值仅用于离线评分和绘图；候选轨迹改变时不假设同种子等于同输入。

|地图/轮/版本|种子|实际区|M|H/L/C|W|T|弹|步/秒|雷达采样%|卡住步计数|状态/有效/原因|
|---|---|---|---|---|---|---|---|---|---|---|---|
|yh_map_fs_72/r1/rl11_lines|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|9.6836|13016/77700/35326|29.9886|22.8571|10|4536/110.0978|27.3773|240|ok/True/[]|
|yh_map_fs_72/r1/rl17_repair|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|9.6836|13016/77700/35326|29.9886|22.8571|10|4536/104.5283|27.3773|240|ok/True/[]|
|yh_map_fs_72/r1/r20_budget|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|11.1881|13457/77259/30501|24.7111|40.0000|10|3888/107.7528|16.8807|180|ok/True/[]|
|yh_map_fs_72/r2/r20_budget|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|10.9641|13082/77634/29449|25.3254|39.9206|10|3891/103.2775|14.5573|780|ok/True/[]|
|yh_map_fs_72/r2/rl17_repair|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|10.6640|13049/77667/31527|24.8142|40.0000|10|3888/94.4107|14.4268|180|ok/True/[]|
|yh_map_fs_72/r2/rl11_lines|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|10.6640|13049/77667/31527|24.8142|40.0000|10|3888/97.5903|14.4268|180|ok/True/[]|
|yh_map_fs_55/r1/rl11_lines|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|15.9248|24935/83177/67246|31.3233|22.8571|10|4536/111.3547|26.2805|1440|ok/True/[]|
|yh_map_fs_55/r1/rl17_repair|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|16.4214|24777/83335/62979|31.3233|22.8571|10|4536/116.7986|26.2805|1440|ok/True/[]|
|yh_map_fs_55/r1/r20_budget|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|13.6505|24068/84044/76226|38.1047|22.8571|10|4536/119.0135|28.1508|1020|ok/True/[]|
|yh_map_fs_55/r2/r20_budget|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|17.8918|22557/85555/34113|37.5078|37.5926|10|3979/105.8480|20.9370|840|ok/True/[]|
|yh_map_fs_55/r2/rl17_repair|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|16.8154|22572/85540/42598|39.8499|22.8571|10|4536/104.8749|32.4561|720|ok/True/[]|
|yh_map_fs_55/r2/rl11_lines|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|16.8154|22572/85540/42598|39.8499|22.8571|10|4536/107.7699|32.4561|720|ok/True/[]|
|yh_map_fs_37/r1/rl11_lines|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|57.3600|62928/25844/111240|51.2577|22.8571|10|4536/117.1630|92.9433|660|ok/True/[]|
|yh_map_fs_37/r1/rl17_repair|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|57.1753|62014/26758/109159|51.2577|22.8571|10|4536/117.6382|92.9433|660|ok/True/[]|
|yh_map_fs_37/r1/r20_budget|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|50.5975|58513/30259/119812|50.7364|32.9365|10|4155/117.7000|67.5278|840|ok/True/[]|
|yh_map_fs_37/r2/r20_budget|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|49.2222|54715/34057/107077|63.3007|22.8571|10|4536/124.1910|80.8916|300|ok/True/[]|
|yh_map_fs_37/r2/rl17_repair|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|59.8362|65841/22931/115739|59.2088|22.8571|10|4536/122.5850|94.1462|240|ok/True/[]|
|yh_map_fs_37/r2/rl11_lines|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|59.7194|65389/23383/113762|59.2088|22.8571|10|4536/121.3991|94.1462|240|ok/True/[]|
|yh_map_fs_24/r1/rl11_lines|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|77.0690|75508/5138/120472|53.6058|30.0265|10|4265/110.7389|93.9550|480|ok/True/[]|
|yh_map_fs_24/r1/rl17_repair|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|78.7656|75411/5235/113247|53.6058|30.0265|10|4265/111.4333|93.9550|480|ok/True/[]|
|yh_map_fs_24/r1/r20_budget|2026111701|['NO_COM_ZONE', 'NO_GPS_ZONE']|88.0653|79124/1522/90055|54.7038|22.8571|10|4536/123.5924|99.1692|600|ok/True/[]|
|yh_map_fs_24/r2/r20_budget|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|79.3941|71879/8767/81335|49.7200|32.5132|10|4171/118.6883|95.2688|540|ok/True/[]|
|yh_map_fs_24/r2/rl17_repair|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|83.6695|74911/5735/78473|57.1975|22.8571|10|4536/118.4832|99.1385|600|ok/True/[]|
|yh_map_fs_24/r2/rl11_lines|2026111702|['NO_COM_ZONE', 'NO_GPS_ZONE']|83.4352|74958/5688/80273|57.1975|22.8571|10|4536/115.2412|99.1385|600|ok/True/[]|

## 配对差值及退步（候选减对照）

- yh_map_fs_24 r1 seed2026111701 / r20_budget−rl17_repair / 有效=True：{'map_score': 9.299755063797335, 'wall_hits': 3713, 'wall_missed': -3713, 'charged_pixels': -23192, 'wall_score': 1.0979407456642463, 'time_score': -7.169312169312171, 'bombs': 0, 'wall_seconds': 12.15908432006836, 'sampled_lidar_free_percent': 5.214222041682191, 'stalled_steps': 120, 'yield_steps': -36, 'lifetime_passage_episodes': 55}
- yh_map_fs_24 r1 seed2026111701 / rl17_repair−rl11_lines / 有效=True：{'map_score': 1.6965997616124753, 'wall_hits': -97, 'wall_missed': 97, 'charged_pixels': -7225, 'wall_score': 0.0, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': 0.6944215297698975, 'sampled_lidar_free_percent': 0.0, 'stalled_steps': 0, 'yield_steps': 0, 'lifetime_passage_episodes': 0}
- yh_map_fs_24 r2 seed2026111702 / r20_budget−rl17_repair / 有效=True：{'map_score': -4.2754522852962396, 'wall_hits': -3032, 'wall_missed': 3032, 'charged_pixels': 2862, 'wall_score': -7.477567298105683, 'time_score': 9.656084656084655, 'bombs': 0, 'wall_seconds': 0.20511555671691895, 'sampled_lidar_free_percent': -3.869650090834213, 'stalled_steps': -60, 'yield_steps': 322, 'lifetime_passage_episodes': 52}
- yh_map_fs_24 r2 seed2026111702 / rl17_repair−rl11_lines / 有效=True：{'map_score': 0.23435950016120444, 'wall_hits': -47, 'wall_missed': 47, 'charged_pixels': -1800, 'wall_score': 0.0, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': 3.2420573234558105, 'sampled_lidar_free_percent': 0.0, 'stalled_steps': 0, 'yield_steps': 0, 'lifetime_passage_episodes': 0}
- yh_map_fs_37 r1 seed2026111701 / r20_budget−rl17_repair / 有效=True：{'map_score': -6.577726859257432, 'wall_hits': -3501, 'wall_missed': 3501, 'charged_pixels': 10653, 'wall_score': -0.5212713936430262, 'time_score': 10.079365079365079, 'bombs': 0, 'wall_seconds': 0.06174421310424805, 'sampled_lidar_free_percent': -25.415525295245757, 'stalled_steps': 180, 'yield_steps': 93, 'lifetime_passage_episodes': 49}
- yh_map_fs_37 r1 seed2026111701 / rl17_repair−rl11_lines / 有效=True：{'map_score': -0.18478518001172262, 'wall_hits': -914, 'wall_missed': 914, 'charged_pixels': -2081, 'wall_score': 0.0, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': 0.4752929210662842, 'sampled_lidar_free_percent': 0.0, 'stalled_steps': 0, 'yield_steps': 0, 'lifetime_passage_episodes': 0}
- yh_map_fs_37 r2 seed2026111702 / r20_budget−rl17_repair / 有效=True：{'map_score': -10.614069610913347, 'wall_hits': -11126, 'wall_missed': 11126, 'charged_pixels': -8662, 'wall_score': 4.091931540342301, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': 1.606074571609497, 'sampled_lidar_free_percent': -13.254601131409515, 'stalled_steps': 60, 'yield_steps': 313, 'lifetime_passage_episodes': 50}
- yh_map_fs_37 r2 seed2026111702 / rl17_repair−rl11_lines / 有效=True：{'map_score': 0.11681875337944092, 'wall_hits': 452, 'wall_missed': -452, 'charged_pixels': 1977, 'wall_score': 0.0, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': 1.1858940124511719, 'sampled_lidar_free_percent': 0.0, 'stalled_steps': 0, 'yield_steps': 0, 'lifetime_passage_episodes': 0}
- yh_map_fs_55 r1 seed2026111701 / r20_budget−rl17_repair / 有效=True：{'map_score': -2.7708671042529947, 'wall_hits': -709, 'wall_missed': 709, 'charged_pixels': 13247, 'wall_score': 6.7814009661835755, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': 2.2149550914764404, 'sampled_lidar_free_percent': 1.8702380590085923, 'stalled_steps': -420, 'yield_steps': 162, 'lifetime_passage_episodes': 72}
- yh_map_fs_55 r1 seed2026111701 / rl17_repair−rl11_lines / 有效=True：{'map_score': 0.4965394683291393, 'wall_hits': -158, 'wall_missed': 158, 'charged_pixels': -4267, 'wall_score': 0.0, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': 5.443808317184448, 'sampled_lidar_free_percent': 0.0, 'stalled_steps': 0, 'yield_steps': 0, 'lifetime_passage_episodes': 0}
- yh_map_fs_55 r2 seed2026111702 / r20_budget−rl17_repair / 有效=True：{'map_score': 1.0763835695274366, 'wall_hits': -15, 'wall_missed': 15, 'charged_pixels': -8485, 'wall_score': -2.3421325051759823, 'time_score': 14.735449735449738, 'bombs': 0, 'wall_seconds': 0.9731290340423584, 'sampled_lidar_free_percent': -11.519152843197098, 'stalled_steps': 120, 'yield_steps': 43, 'lifetime_passage_episodes': 70}
- yh_map_fs_55 r2 seed2026111702 / rl17_repair−rl11_lines / 有效=True：{'map_score': 0.0, 'wall_hits': 0, 'wall_missed': 0, 'charged_pixels': 0, 'wall_score': 0.0, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': -2.894960880279541, 'sampled_lidar_free_percent': 0.0, 'stalled_steps': 0, 'yield_steps': 0, 'lifetime_passage_episodes': 0}
- yh_map_fs_72 r1 seed2026111701 / r20_budget−rl17_repair / 有效=True：{'map_score': 1.504562918200758, 'wall_hits': 441, 'wall_missed': -441, 'charged_pixels': -4825, 'wall_score': -5.277534965034967, 'time_score': 17.142857142857142, 'bombs': 0, 'wall_seconds': 3.224501848220825, 'sampled_lidar_free_percent': -10.496527394370784, 'stalled_steps': -60, 'yield_steps': 216, 'lifetime_passage_episodes': 49}
- yh_map_fs_72 r1 seed2026111701 / rl17_repair−rl11_lines / 有效=True：{'map_score': 0.0, 'wall_hits': 0, 'wall_missed': 0, 'charged_pixels': 0, 'wall_score': 0.0, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': -5.569565534591675, 'sampled_lidar_free_percent': 0.0, 'stalled_steps': 0, 'yield_steps': 0, 'lifetime_passage_episodes': 0}
- yh_map_fs_72 r2 seed2026111702 / r20_budget−rl17_repair / 有效=True：{'map_score': 0.3000792824584426, 'wall_hits': 33, 'wall_missed': -33, 'charged_pixels': -2078, 'wall_score': 0.5111208236208213, 'time_score': -0.07936507936507553, 'bombs': 0, 'wall_seconds': 8.866750717163086, 'sampled_lidar_free_percent': 0.13046691185838988, 'stalled_steps': 600, 'yield_steps': -539, 'lifetime_passage_episodes': 47}
- yh_map_fs_72 r2 seed2026111702 / rl17_repair−rl11_lines / 有效=True：{'map_score': 0.0, 'wall_hits': 0, 'wall_missed': 0, 'charged_pixels': 0, 'wall_score': 0.0, 'time_score': 0.0, 'bombs': 0, 'wall_seconds': -3.1795802116394043, 'sampled_lidar_free_percent': 0.0, 'stalled_steps': 0, 'yield_steps': 0, 'lifetime_passage_episodes': 0}

## yh_map_fs_24 / r1 / seed 2026111701


![yh_map_fs_24 / r1 / seed 2026111701 / mapping / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r1_mapping.png)


![yh_map_fs_24 / r1 / seed 2026111701 / exploration / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r1_exploration.png)


![yh_map_fs_24 / r1 / seed 2026111701 / trajectories / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r1_trajectories.png)


![yh_map_fs_24 / r1 / seed 2026111701 / walls / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r1_walls.png)


![yh_map_fs_24 / r1 / seed 2026111701 / bombs / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r1_bombs.png)


rl11_lines 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_24_r01\rl11_lines\red\attempt_001'
```

rl17_repair 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_24_r01\rl17_repair\red\attempt_001'
```

r20_budget 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_24_r01\r20_budget\red\attempt_001'
```

## yh_map_fs_24 / r2 / seed 2026111702


![yh_map_fs_24 / r2 / seed 2026111702 / mapping / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r2_mapping.png)


![yh_map_fs_24 / r2 / seed 2026111702 / exploration / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r2_exploration.png)


![yh_map_fs_24 / r2 / seed 2026111702 / trajectories / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r2_trajectories.png)


![yh_map_fs_24 / r2 / seed 2026111702 / walls / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r2_walls.png)


![yh_map_fs_24 / r2 / seed 2026111702 / bombs / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_24_r2_bombs.png)


rl11_lines 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_24_r02\rl11_lines\red\attempt_001'
```

rl17_repair 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_24_r02\rl17_repair\red\attempt_001'
```

r20_budget 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_24_r02\r20_budget\red\attempt_001'
```

## yh_map_fs_37 / r1 / seed 2026111701


![yh_map_fs_37 / r1 / seed 2026111701 / mapping / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r1_mapping.png)


![yh_map_fs_37 / r1 / seed 2026111701 / exploration / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r1_exploration.png)


![yh_map_fs_37 / r1 / seed 2026111701 / trajectories / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r1_trajectories.png)


![yh_map_fs_37 / r1 / seed 2026111701 / walls / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r1_walls.png)


![yh_map_fs_37 / r1 / seed 2026111701 / bombs / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r1_bombs.png)


rl11_lines 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_37_r01\rl11_lines\red\attempt_001'
```

rl17_repair 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_37_r01\rl17_repair\red\attempt_001'
```

r20_budget 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_37_r01\r20_budget\red\attempt_001'
```

## yh_map_fs_37 / r2 / seed 2026111702


![yh_map_fs_37 / r2 / seed 2026111702 / mapping / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r2_mapping.png)


![yh_map_fs_37 / r2 / seed 2026111702 / exploration / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r2_exploration.png)


![yh_map_fs_37 / r2 / seed 2026111702 / trajectories / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r2_trajectories.png)


![yh_map_fs_37 / r2 / seed 2026111702 / walls / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r2_walls.png)


![yh_map_fs_37 / r2 / seed 2026111702 / bombs / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_37_r2_bombs.png)


rl11_lines 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_37_r02\rl11_lines\red\attempt_001'
```

rl17_repair 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_37_r02\rl17_repair\red\attempt_001'
```

r20_budget 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_37_r02\r20_budget\red\attempt_001'
```

## yh_map_fs_55 / r1 / seed 2026111701


![yh_map_fs_55 / r1 / seed 2026111701 / mapping / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r1_mapping.png)


![yh_map_fs_55 / r1 / seed 2026111701 / exploration / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r1_exploration.png)


![yh_map_fs_55 / r1 / seed 2026111701 / trajectories / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r1_trajectories.png)


![yh_map_fs_55 / r1 / seed 2026111701 / walls / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r1_walls.png)


![yh_map_fs_55 / r1 / seed 2026111701 / bombs / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r1_bombs.png)


rl11_lines 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_55_r01\rl11_lines\red\attempt_001'
```

rl17_repair 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_55_r01\rl17_repair\red\attempt_001'
```

r20_budget 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_55_r01\r20_budget\red\attempt_001'
```

## yh_map_fs_55 / r2 / seed 2026111702


![yh_map_fs_55 / r2 / seed 2026111702 / mapping / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r2_mapping.png)


![yh_map_fs_55 / r2 / seed 2026111702 / exploration / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r2_exploration.png)


![yh_map_fs_55 / r2 / seed 2026111702 / trajectories / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r2_trajectories.png)


![yh_map_fs_55 / r2 / seed 2026111702 / walls / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r2_walls.png)


![yh_map_fs_55 / r2 / seed 2026111702 / bombs / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_55_r2_bombs.png)


rl11_lines 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_55_r02\rl11_lines\red\attempt_001'
```

rl17_repair 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_55_r02\rl17_repair\red\attempt_001'
```

r20_budget 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_55_r02\r20_budget\red\attempt_001'
```

## yh_map_fs_72 / r1 / seed 2026111701


![yh_map_fs_72 / r1 / seed 2026111701 / mapping / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r1_mapping.png)


![yh_map_fs_72 / r1 / seed 2026111701 / exploration / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r1_exploration.png)


![yh_map_fs_72 / r1 / seed 2026111701 / trajectories / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r1_trajectories.png)


![yh_map_fs_72 / r1 / seed 2026111701 / walls / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r1_walls.png)


![yh_map_fs_72 / r1 / seed 2026111701 / bombs / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r1_bombs.png)


rl11_lines 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_72_r01\rl11_lines\red\attempt_001'
```

rl17_repair 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_72_r01\rl17_repair\red\attempt_001'
```

r20_budget 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_72_r01\r20_budget\red\attempt_001'
```

## yh_map_fs_72 / r2 / seed 2026111702


![yh_map_fs_72 / r2 / seed 2026111702 / mapping / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r2_mapping.png)


![yh_map_fs_72 / r2 / seed 2026111702 / exploration / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r2_exploration.png)


![yh_map_fs_72 / r2 / seed 2026111702 / trajectories / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r2_trajectories.png)


![yh_map_fs_72 / r2 / seed 2026111702 / walls / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r2_walls.png)


![yh_map_fs_72 / r2 / seed 2026111702 / bombs / RED STAGE ONLY](E:/桌面/gitee/swarm-eod/output/sf13/r20_review_20261117_160246/atlas/yh_map_fs_72_r2_bombs.png)


rl11_lines 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_72_r02\rl11_lines\red\attempt_001'
```

rl17_repair 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_72_r02\rl17_repair\red\attempt_001'
```

r20_budget 原始录像：

```powershell
Set-Location -LiteralPath 'E:\桌面\gitee\swarm-eod'
.\replay_map.cmd 'E:\桌面\gitee\swarm-eod\output\sf13\r20a_20261117_150615\runs\yh_map_fs_72_r02\r20_budget\red\attempt_001'
```
