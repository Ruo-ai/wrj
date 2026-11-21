"""Draw comparable map panels and visit curves from verified recordings."""

import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from . import red_exploration22_batch as prior
from .map23_replay import visits, CHECKPOINTS
from .map23_trial import HISTORY
from .optimization.evaluation import runner
from .offline_platform_map_regions import platform_rows
from .offline_platform_map_regions import region_bounds
from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer


VERSIONS = ('RL11', 'RL17-R', 'RL23')
COLORS = {'hit': '#14765b', 'miss': '#38559b', 'charge': '#ba3838',
          'free': '#c8ae66', 'inside': '#aab3b8', 'empty': '#fafaf7'}
plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'figure.facecolor': 'white'})


def error_rgb(grid, truth, surface, far):
    from matplotlib.colors import to_rgb
    result = np.zeros((*truth.shape, 3), np.uint8)
    for mask, key in ((np.ones_like(grid, bool), 'empty'), (truth, 'inside'),
                      (grid & ~truth, 'free'), (surface, 'miss'),
                      (surface & grid, 'hit'), (grid & far, 'charge')):
        result[mask] = np.rint(255*np.array(to_rgb(COLORS[key]))).astype(np.uint8)
    return result


def tile(mask, size=96):
    """Centred on the densest 96 x 96 block, with half-block strides."""
    h, w = mask.shape
    best = None
    for y in list(range(0, h - size, size//2)) + [h-size]:
        for x in list(range(0, w - size, size//2)) + [w-size]:
            score = int(mask[y:y+size, x:x+size].sum())
            if best is None or score > best[0]:
                best = score, x, y
    _, x, y = best
    return x, y, x+size, y+size


def regional_evidence(folder, report, surface, seen, base, candidate, cost):
    attempt = Path(report['attempt'])
    (recording,) = (attempt / 'artifacts/recordings').glob('*.npz')
    with np.load(recording) as z:
        states = z['drone_states']
    h, w = surface.shape
    x = states['x'].astype(float) + w / 2
    y = h / 2 - states['y'].astype(float)
    rows = {}
    for name, (x0, y0, x1, y1) in region_bounds(w, h, None, ()):
        if not name.endswith('9'):
            continue
        inside = states['active'] & (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        loc = np.s_[int(np.ceil(y0)):int(np.ceil(y1)), int(np.ceil(x0)):int(np.ceil(x1))]
        frames = np.flatnonzero(inside.any(axis=1))
        rows[name] = dict(first_gui_step=int(frames[0])+2 if len(frames) else None,
            team_frames=int(len(frames)), unique_center_cells=len(set(zip((x[inside]//40).astype(int),
                                                                          (y[inside]//40).astype(int)))),
            wall_pixels=int(surface[loc].sum()), directly_observed=int(seen[loc].sum()),
            observed_but_missed=int((seen[loc] & ~candidate[loc]).sum()),
            submitted_hits=int((surface[loc] & candidate[loc]).sum()),
            missed=int((surface[loc] & ~candidate[loc]).sum()),
            added_hits=int((surface[loc] & candidate[loc] & ~base[loc]).sum()),
            lost_hits=int((surface[loc] & base[loc] & ~candidate[loc]).sum()),
            penalty=float(cost[loc][candidate[loc]].sum(dtype=float)))
    assert sum(r['wall_pixels'] for r in rows.values()) == int(surface.sum())
    runner.write(folder / 'regions.json', rows)


def map_figures(folder):
    data = runner.read(folder / 'comparison.json')
    with np.load(folder / 'maps.npz') as saved:
        truth = saved['truth'].astype(bool)
        maps = {key: saved[key].astype(bool) for key in VERSIONS}
    scorer = MapExplorationScorer()
    surface = scorer.split_regions(truth.astype(np.uint8))[0].astype(bool)
    distance = scorer.distance_field(truth.astype(np.uint8))
    far = distance > scorer.config.dead_zone
    cost = 100 * scorer.config.penalty_weight * scorer.penalty_curve(distance) / int(surface.sum())
    error = {key: error_rgb(grid, truth, surface, far) for key, grid in maps.items()}
    with np.load(Path(data['attempt']) / 'wall_returns.npz') as z:
        seen = (z['first_tick'] > 0) & surface
    regional_evidence(folder, data, surface, seen, maps['RL17-R'], maps['RL23'], cost)
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.4))
    fig.subplots_adjust(left=.01, right=.99, top=.84, bottom=.09, wspace=.035, hspace=.20)
    fig.suptitle(f"{data['map']}  |  同一份合法观测与实收版本", fontsize=23, y=.98)
    fig.text(.5, .91, f"官方地图 · 第 1 轮 · 种子 {data['seed']} · 第 {data['gui_end_step']:,} 步终止", ha='center', fontsize=12)
    axes[0,0].imshow(~truth, cmap='gray', vmin=0, vmax=1)
    axes[0,0].set_title('评分真值（含墙体内部）')
    axes[1,0].imshow(~seen, cmap='gray', vmin=0, vmax=1)
    axes[1,0].set_title(f"雷达端点直接命中的计分墙面  {seen.sum():,} px")
    for col, key in enumerate(VERSIONS, 1):
        axes[0,col].imshow(~maps[key], cmap='gray', vmin=0, vmax=1)
        m = data['metrics'][key]
        axes[0,col].set_title(f"{key}  |  提交图  {m['map_score']:.5f}")
        axes[1,col].imshow(error[key])
        axes[1,col].set_title(f"真墙命中 {m['hits']:,} px  |  误画扣 {m['penalty']:.3f} 分")
    for ax in axes.ravel():
        ax.set_axis_off()
    legend = [('hit','命中计分墙面'),('miss','漏画计分墙面'),('charge','受罚误画'),
              ('free','容差内免罚'),('inside','墙体内部')]
    fig.legend([Patch(color=COLORS[k]) for k,_ in legend], [s for _,s in legend],
               loc='lower center', ncol=5, frameon=False, fontsize=12)
    fig.savefig(folder / 'comparison.png', dpi=140)
    plt.close(fig)

    base, candidate = maps['RL17-R'], maps['RL23']
    removed, added = base & ~candidate, candidate & ~base
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    fig.subplots_adjust(top=.87, bottom=.08, left=.11, right=.97, wspace=.08, hspace=.32)
    fig.suptitle(f"{data['map']}  |  最终出图改动的两个局部", fontsize=19)
    examples = []
    for row, (label, selection) in enumerate([('去除误画', removed & far), ('损失真墙', removed & surface)]):
        bounds = tile(selection)
        x, y, right, bottom = bounds
        region = np.s_[y:bottom,x:right]
        n, l = int((added & surface)[region].sum()), int((removed & surface)[region].sum())
        penalty = float(cost[region][added[region]].sum(dtype=float) - cost[region][removed[region]].sum(dtype=float))
        examples.append(dict(bounds_ltrb=list(bounds), added_hits=n, lost_hits=l,
            charged_removed=int((removed & far)[region].sum()), charged_added=int((added & far)[region].sum()),
            delta_penalty=penalty))
        for col, (title, img) in enumerate((('评分真值', ~truth), ('RL17-R 误差',error['RL17-R']),
                                            ('RL23 误差',error['RL23']))):
            ax = axes[row,col]
            ax.imshow(img[region], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row,0].set_ylabel(f'{label}\n({x}, {y})', fontsize=11)
        axes[row,1].set_xlabel(f'真墙新增 {n} px，损失 {l} px；误画扣分变化 {penalty:+.4f}', fontsize=11)
    fig.savefig(folder / 'details.png', dpi=130)
    plt.close(fig)
    runner.write(folder / 'examples.json', examples)


def history_audit(review):
    region_file = runner.REPO / 'output/sf13/platform_4626_audit_20261120/map09_regions.json'
    platform = next((runner.REPO / '平台评测结果/复赛').glob('*46.26*'))
    if prior.digest(platform) != runner.read(region_file)['platform_sha256']:
        raise ValueError('Different platform ZIP')
    regions = list(platform_rows(platform, None))
    panels = []
    with ZipFile(platform) as z:
        for r in regions:
            for i in range(1, 6):
                path = r['source'] + f'/heatmap_panel-{i}.png'
                rgb = np.asarray(Image.open(io.BytesIO(z.read(path))).convert('RGB'))
                small = rgb[::4, ::4]
                if not np.array_equal(rgb, small.repeat(4, axis=0).repeat(4, axis=1)):
                    raise ValueError('Unexpected display grid')
                grey = (small[:,:,0] == small[:,:,1]) & (small[:,:,1] == small[:,:,2])
                mask = grey & (small[:,:,0] > 0)
                h,w = mask.shape
                padded = np.pad(mask, ((0, -h % 4), (0, -w % 4)))
                coarse = padded.reshape(padded.shape[0]//4, 4, padded.shape[1]//4, 4).any(axis=(1,3))
                panels.append(dict(round=r['round'], panel=i, fine_fraction=float(mask.mean()),
                                   coarse_count=int(coarse.sum()), coarse_total=coarse.size))
    old = runner.read(HISTORY / 'paired_summary.json')['rows']
    checked = 0
    for f in HISTORY.glob('runs/*/*/*/done.json'):
        _, run = runner.load_done(f.parent, prior.digest(HISTORY / 'plan.json'))
        checked += len(runner.read(run / 'integrity.json')['files'])
    record = dict(old_rl22_files_checked=checked, historical_plan_sha256=prior.digest(HISTORY / 'plan.json'),
        archive_sha256=prior.digest(prior.ARCHIVE), rows=[{k: r[k] for k in
        ('map','round','controller','map_score','red_valid','blue_valid','failure_reason','total_score')} for r in old],
        platform_sha256=prior.digest(platform), blue_panels=panels,
        platform_map09=[dict(round=r['round'], score=r['score_map'], all=r['regions']['all'],
                             NW9=r['regions']['NW9']) for r in regions])
    runner.write(review / 'history.json', record)


def visit_figure(review):
    fig, axes = plt.subplots(1,3,figsize=(16,4.8), sharey=True)
    fig.subplots_adjust(top=.77,bottom=.24,left=.055,right=.99,wspace=.10)
    fig.suptitle('红队机身中心首次到访的粗格', fontsize=21, y=.98)
    fig.text(.5,.875,'40 × 40 像素一格；折线终点为最后实测帧。灰色虚线为官方 5,400 步预算的四分位点。',ha='center', fontsize=12)
    curves = {}
    for ax, name in zip(axes, prior.MAPS):
        folder = review / name
        report = runner.read(folder / 'comparison.json')
        with np.load(folder / 'maps.npz') as z:
            shape = z['truth'].shape
        curves[name] = []
        lines = [('RL23 本轮', folder, report['visits'], '#132c4e', '-')]
        for selector,label,color in [('rl11_lines','RL11 历史','#b46623'),('archived_rl17r','RL17-R 历史','#028073'),
                                      ('candidate_rl22','RL22 历史','#9364a2')]:
            phase = HISTORY / 'runs' / (name + '_r01') / selector / 'red'
            if not (phase / 'done.json').is_file():
                continue
            r, attempt = runner.load_done(phase, prior.digest(HISTORY / 'plan.json'))
            if not runner.phase_valid(r, 'red'):
                continue
            dest = folder / 'historical_visits' / selector
            dest.mkdir(parents=True, exist_ok=True)
            v = visits(attempt, shape, r['elapsed_steps'], dest)
            lines.append((label,dest,v,color,'--'))
        for label,dest,summary,color,style in lines:
            with np.load(dest / 'visits.npz') as z:
                steps = z['steps']; pct = 100*z['counts']/summary['total_cells']
            ax.plot(steps,pct,style,color=color,label=label,linewidth=2.5 if style=='-' else 1.3, zorder=4 if style=='-' else 2)
            ax.scatter(steps[-1],pct[-1],s=20,color=color,zorder=5)
            curves[name].append(dict(version=label, **summary))
        for cutoff in CHECKPOINTS:
            ax.axvline(cutoff,color='#bac0c5',ls=':',lw=1)
        ax.set_title(name,fontsize=15)
        ax.set_xlim(0,5500); ax.set_ylim(0,100); ax.set_xticks([0,*CHECKPOINTS])
        ax.set_xlabel('官方步数'); ax.grid(axis='y',alpha=.18)
        ax.spines[['right','top']].set_visible(False)
    axes[0].set_ylabel('已到访粗格 / 全部粗格（%）')
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc='lower center',frameon=False)
    fig.savefig(review / 'visits.png',dpi=140)
    plt.close(fig)
    runner.write(review / 'visit_comparison.json',curves)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('review', type=Path)
    p.add_argument('--map', choices=prior.MAPS)
    p.add_argument('--history', action='store_true')
    args = p.parse_args()
    if args.map:
        map_figures(args.review / args.map)
    elif args.history:
        history_audit(args.review)
    else:
        visit_figure(args.review)
