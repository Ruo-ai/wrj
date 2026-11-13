"""Plot the two archived platform submissions. The map ground truth is offline only."""
import io
import json
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch
import numpy as np

from swarm_rescue.solutions.devtools import evaluate_red as red
from swarm_rescue.solutions.devtools.blind_layout_attribution import trace_audit


def main():
    red.configure_ray_backend('cpu')
    from swarm_rescue.maps.map_07 import Map07
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreManager, walls_and_boundary_grid
    world = Map07(drone_type=DroneMotionless, number_drones=5)
    walls, boundary = walls_and_boundary_grid(world._playground)
    world._playground.window.close()
    h, w = walls.shape
    out = red.ROOT/'output/platform_blind_compare'
    out.mkdir(exist_ok=True, parents=True)
    scorer = PlaceScoreManager()
    stats = []
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    motion, paths = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#ffffff', '#596674', '#1d976a', '#e68a00']
    for col, (key, label, pattern) in enumerate([
        ('advanced2','进阶2 · 发布4.0','*进阶2*.zip'),
        ('advanced5_narrow','进阶5 · 板块二窄路版','*板块二*.zip')]):
        source, = (red.ROOT/'平台评测结果').glob(pattern)
        with ZipFile(source) as archive:
            result, = [n for n in archive.namelist() if n.endswith('/result.json')]
            for detail in json.loads(archive.read(result))['detail']:
                member, = [n for n in archive.namelist() if n.endswith('/'+detail['炸弹文件'])]
                layout = json.loads(archive.read(member))
                xy = np.array([(b['x'], b['y']) for b in layout['bombs']])
                score = scorer.compute_wall_score(xy.tolist(), walls, boundary_mask=boundary,
                                                  expected_n_bombs=10)
                if layout['map_name'] != 'Map07' or abs(100*score.score-detail['炸墙分']) >= .006:
                    raise ValueError(f'Geometry mismatch: {member}')
                info = score.breakdown
                radius = info['blast_radius']
                masks = np.array([scorer.wall_destruction_masks(walls, [p.tolist()], radius,
                                  boundary_mask=boundary)[2].astype(bool) for p in xy])
                count = masks.sum(axis=0)
                weight = np.where(boundary, info['boundary_wall_weight'], 1.)
                total = float(weight[count > 0].sum())
                if total != info['weighted_destroyed']:
                    raise ValueError('Wall union mismatch')
                gross = float((count*weight).sum())
                exclusive = [float(weight[m & (count == 1)].sum()) for m in masks]
                with np.load(io.BytesIO(archive.read(member.rsplit('/', 1)[0]+'/run.npz')),
                             allow_pickle=False) as z:
                    active = z['bomb_states']['active']
                    first = np.flatnonzero(active.any(axis=1))[0]+1
                    last = np.flatnonzero(active.sum(axis=1)==10)[0]+1
                    audit = trace_audit(z, xy, masks, weight)
                    track = z['drone_states']
                number = int(detail['轮次'])
                panel = paths[number-1,col]
                panel.imshow(walls, cmap=ListedColormap(['#ffffff','#dddddd']),
                             extent=(-w/2,w/2,-h/2,h/2), origin='upper', interpolation='nearest')
                for drone in range(track.shape[1]):
                    tail = track[last-1:,drone]
                    line, = panel.plot(tail['x'], tail['y'], lw=.8, alpha=.8, label=f'{drone}号')
                    panel.scatter(tail['x'][0], tail['y'][0], c=line.get_color(), s=18)
                    panel.scatter(tail['x'][-1], tail['y'][-1], c=line.get_color(), marker='x', s=25)
                panel.set(xlim=(-w/2,w/2), ylim=(-h/2,h/2), xticks=[], yticks=[],
                          title=f'{label} 第{number}轮 | 第{last}至{len(track)}帧')
                overlap = 100*(1-total/gross)
                stats.append(dict(version=key, round=number, **detail, union=total,
                                  overlap=overlap, per_bomb_exclusive=exclusive,
                                  first_frame=int(first), all_ten_frame=int(last), audit=audit))
                ax = axes[number-1,col]
                pixels = np.where(count > 1, 3, np.where(count == 1, 2, walls.astype(bool)))
                ax.imshow(pixels, cmap=ListedColormap(colors), vmin=0,vmax=3,
                          extent=(-w/2,w/2,-h/2,h/2),origin='upper',interpolation='nearest')
                for i, (x,y) in enumerate(xy, 1):
                    ax.add_patch(Circle((x,y),radius, facecolor='none',edgecolor='#387eb3',lw=.9))
                    ax.text(x,y,str(i),ha='center',va='center',color='#172635',fontsize=9,
                            bbox=dict(boxstyle='round,pad=.14',fc='white',ec='#387eb3',lw=.6))
                ax.set(xlim=(-w/2,w/2),ylim=(-h/2,h/2),xticks=[],yticks=[],
                       title=f'{label}   第{number}轮   |   墙体 {detail["炸墙分"]:.2f}   难度 {detail["炸弹分(校准)"]:.2f}')
    fig.legend(handles=[Patch(color=colors[i],label=l) for i,l in enumerate(
        ['空地','未覆盖墙体','单弹覆盖墙体','重复覆盖墙体']) if i],
        loc='lower center', ncol=3, frameon=False)
    fig.suptitle('平台盲测实际布弹布局   ·   编号为导出顺序，蓝圈为爆炸半径', fontsize=16)
    fig.tight_layout(rect=(0,.035,1,.965))
    fig.savefig(out/'platform_layouts.png',dpi=140)
    plt.close(fig)
    motion.suptitle('十枚放满后的五机轨迹 · 实心点为起点，叉号为末帧', fontsize=16)
    motion.legend(*paths[0,0].get_legend_handles_labels(), loc='lower center', ncol=5)
    motion.tight_layout(rect=(0,.04,1,.96), h_pad=1.5)
    motion.savefig(out/'post_placement_paths.png', dpi=130)
    plt.close(motion)
    (out/'comparison.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
