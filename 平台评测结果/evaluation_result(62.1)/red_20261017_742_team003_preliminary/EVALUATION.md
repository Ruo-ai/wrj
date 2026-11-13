# 评测结果：20261017_742_team003_preliminary

- 赛道：红队
- 名次：36
- 最终总分：62.1
- 提交：`20261017_742_team003_preliminary.zip`
- 轮数：2

## 结果文件

- `items.csv`：本队逐局/逐轮结果。“本局分(校准)”为逐轮最终分；“本局分(未校准)”是 place 阶段中间分（当时炸弹分尚未计入）。
- `result.json`：本队结果的机器可读版本。
- `manifest.json`：本队信息及附件数量。
- `final_score.json`：本队最终总分，等于各轮最终分中的最高分。

## 附件

- `*_bombs.json`：本局炸弹位置。
- `*_exploration_diff.png`：提交探索图与地图真值的对照图，包含评分可视化。
- `*_wall_destruction.png`：炸弹炸墙结果与计分可视化。
- `run.npz`：可用回放共 2 份；缺失的回放不在包内。

## 本地回放

在选手项目的图形界面环境中运行，替换为实际文件路径：

```bash
.venv/bin/python -m swarm_rescue.launcher --replay <run.npz路径>
```
