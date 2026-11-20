"""Read-only official three-map paired review; never starts simulations.

All scheduled rows, incomplete attempts, regressions and invalid blue stages
remain visible. Smoke input is rejected rather than rendered as formal data.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from swarm_rescue.solutions.devtools import red_exploration22_batch as batch
from swarm_rescue.solutions.devtools.optimization.evaluation import runner
from swarm_rescue.solutions.devtools.red_pair19_results_review import truth_metric, only
from swarm_rescue.solutions.devtools import red_stopturn_match_report as plot
from swarm_rescue.solutions.devtools.red_lines11_review import FONT


ORDER = ('candidate_rl22', 'rl11_lines', 'archived_rl17r')
LABELS = dict(candidate_rl22='RL22 candidate', rl11_lines='RL11 retained', archived_rl17r='RL17-R actual ZIP')
KINDS = ('exploration', 'mapping', 'trajectories', 'bombs', 'walls')
NOTES = {
    'exploration': '浅绿=每8步/4px栅格采样的物理雷达可见近似；黑=真墙。这不是精确探索率，也不是提交图。',
    'mapping': '绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。',
    'trajectories': '颜色=录像槽位0–4；灰=无GPS/罗盘；黄=无通信；青=出生区。轨迹不等于雷达覆盖。',
    'bombs': '蓝点/圆=仿真实际落点及作用半径；编号=产物顺序，非机号或放弹时序；灰/黄=特殊区。',
    'walls': '官方墙体覆盖面板：绿=覆盖内墙；蓝=覆盖边界；灰=未覆盖墙；红圆=作用范围。',
}
RED_METRICS = ('map_score', 'wall_missed', 'charged_pixels', 'contacts_wall',
               'contacts_drone', 'sampled_lidar_free_percent', 'wall_score',
               'time_score', 'bombs', 'red_walltime')
read = batch.read


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def fmt(value):
    return '暂无' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)


def _raw_phase(phase_dir, plan_sha):
    """Keep unreceipted failures readable, but never promote them to valid."""
    attempts = sorted(phase_dir.glob('attempt_*')) if phase_dir.exists() else []
    issues = []
    try:
        result, attempt = runner.load_done(phase_dir, plan_sha)
    except Exception as exc:
        result, attempt = None, None
        issues.append('receipt:' + repr(exc))
    if result is None:
        issues.append('not_run' if not attempts else 'missing_verified_done')
        attempt = attempts[-1] if attempts else phase_dir / 'attempt_001'
        result = read(attempt / 'result.json') if (attempt / 'result.json').is_file() else {}
    if len(attempts) > 1:
        issues.append('multiple_attempts_preserved')
    return result, attempt, issues, [str(p) for p in attempts]


def inspect_phase(folder, plan, item, phase):
    phase_dir = folder / 'runs' / f"{item['name']}_r{item['round']:02d}" / item['controller'] / phase
    raw, attempt, issues, attempts = _raw_phase(phase_dir, batch.digest(folder / 'plan.json'))
    if raw:
        try:
            issues += batch.phase_issues(raw, attempt, phase, False)
            request = read(attempt / 'request.json')
            if any(raw.get(k) != item[k] or request.get(k) != item[k] for k in ('map', 'round', 'seed')):
                issues.append('map_round_seed_mismatch')
            identity = read(attempt / 'loaded_identity.json')
            for name, sha in identity.get('loaded_sources', {}).items():
                if plan['source_sha256'].get(name) != sha:
                    issues.append('loaded_source_mismatch:' + name)
            actual = raw.get('actual_conditions') or {}
            if actual.get('max_steps') != 5400 or actual.get('max_walltime') != 180:
                issues.append('engine_limits')
            if actual.get('drones') != (5 if phase == 'red' else 10):
                issues.append('actual_team_size')
            if actual.get('instantiated_zone_types') != item['scenario']['zones']:
                issues.append('actual_special_zones')
            resource = read(attempt / 'resources.json')
            if resource.get('cpu_count') != 1 or resource.get('memory_limit_bytes') != 4_000_000_000:
                issues.append('resource_contract')
            observed = identity.get('observed_instances', {})
            if phase == 'red':
                expected = plan['configuration']['controller_identities'][item['controller']]
                module = expected['controller_module']
                if item['controller'] == 'archived_rl17r':
                    runtime = identity.get('archive_runtime') or {}
                    if runtime.get('zip_sha256') != plan['configuration']['archive_sha256']:
                        issues.append('archive_sha256')
                    if runtime.get('workspace_solution_fallback_allowed') is not False:
                        issues.append('archive_fallback_not_forbidden')
                    for name, sha in plan['archive_audit']['red_dependency_sha256'].items():
                        if runtime.get('loaded_member_sha256', {}).get(name) != sha:
                            issues.append('archive_member:' + name)
                    module = runtime.get('isolated_module', '')
                expected_class = module + '.' + expected['controller_class']
                if len(observed) != 5 or any(v.get('controller') != expected_class for v in observed.values()):
                    issues.append('actual_red_controller_instances')
            else:
                config = plan['configuration']
                expected_class = config['blue_controller_module'] + '.' + config['blue_controller_class']
                if len(observed) != 10 or any(v.get('controller') != expected_class for v in observed.values()):
                    issues.append('actual_blue_controller_instances')
                if raw.get('blue_variant') != config['blue_variant']:
                    issues.append('blue_variant')
                for field, request_key in (('layout_sha256', 'layout'), ('red_result_sha256', 'red_result')):
                    if not request.get(request_key) or raw.get(field) != batch.digest(request[request_key]):
                        issues.append('paired_binding:' + field)
            log = (attempt / 'run.log').read_text('utf-8', errors='replace')
            if any(token in log for token in ('Traceback (most recent call last)', 'Exception ignored', 'AssertionError')):
                issues.append('exception_in_log')
        except Exception as exc:
            issues.append('audit:' + repr(exc))
    return raw, attempt, list(dict.fromkeys(issues)), attempts


def score_row(item, red, blue, red_issues, blue_issues):
    """Separate valid red evidence from a possibly invalid paired blue score."""
    phase = red.get('phase1') or {}
    full = (blue or {}).get('phase1') or {}
    breakdown = phase.get('map_breakdown') or {}
    red_valid = bool(not red_issues and runner.phase_valid(red, 'red') and not red.get('smoke'))
    blue_valid = bool(blue and not blue_issues and runner.phase_valid(blue, 'blue') and not blue.get('smoke'))
    full_valid = red_valid and blue_valid
    return dict(map=item['name'], round=item['round'], seed=item['seed'], controller=item['controller'],
        red_valid=red_valid, blue_valid=blue_valid, full_valid=full_valid,
        red_status=red.get('status', 'not_run'), blue_status=(blue or {}).get('status', 'not_run'),
        red_issues=red_issues, blue_issues=blue_issues,
        map_score=phase.get('score_map'), wall_score=phase.get('score_wall'), time_score=phase.get('score_time'),
        wall_hits=breakdown.get('wall_hits'),
        wall_missed=breakdown['wall_pixels'] - breakdown['wall_hits'] if 'wall_pixels' in breakdown and 'wall_hits' in breakdown else None,
        charged_pixels=breakdown.get('charged_pixels'), bombs=red.get('placed_count'),
        red_steps=red.get('elapsed_steps'), red_walltime=red.get('elapsed_walltime'),
        blue_steps=(blue or {}).get('elapsed_steps'), blue_walltime=(blue or {}).get('elapsed_walltime'),
        raw_total=full.get('round_score'), effective_total=full.get('round_score') if full_valid else None,
        raw_difficulty=full.get('score_bomb'), difficulty=full.get('score_bomb') if full_valid else None,
        actual_zones=(red.get('actual_conditions') or {}).get('instantiated_zone_types'),
        health=red.get('mean_health'), contacts_wall=None, contacts_drone=None,
        sampled_lidar_free_pixels=None, sampled_lidar_free_percent=None, artifact_issues=[],
        timeouts=[p for p, r in (('red', red), ('blue', blue or {}))
                  if r.get('walltime_limited') or r.get('status') == 'process_timeout'],
        callback_errors={p: r.get('engine_callback_errors', 0) for p, r in (('red', red), ('blue', blue or {}))})


def load_assets(row):
    """Load each asset independently, keeping other panels if one is missing."""
    run = Path(row['red_attempt'])
    assets = {}
    def capture(key, function):
        try:
            assets[key] = function()
        except Exception as exc:
            row['artifact_issues'].append(key + ':' + repr(exc))
    capture('truth_bundle', lambda: truth_metric(run))
    capture('pred', lambda: np.asarray(read(only(run / 'artifacts', '*_exploration.json'))['grid'], np.uint8))
    capture('bombs', lambda: read(only(run / 'artifacts', '*_bombs.json'))['bombs'])
    def map_spec():
        path = Path(read(run / 'request.json')['map_file'])
        if batch.digest(path) != row['map_file_sha256']:
            raise ValueError('Map JSON differs from frozen source; omit live zone overlay')
        return read(path)
    capture('spec', map_spec)
    def array(path, key):
        with np.load(path, allow_pickle=False) as data:
            return data[key].copy()
    capture('trajectories', lambda: array(only(run / 'artifacts/recordings', '*.npz'), 'drone_states'))
    def visibility():
        with np.load(run / 'local13_diagnostics.npz', allow_pickle=False) as data:
            grids = [data[key].astype(bool) for key in data.files if key.startswith('visible_')]
            if not grids:
                raise ValueError('No visibility samples')
            return np.repeat(np.repeat(np.logical_or.reduce(grids), 4, 0), 4, 1)
    capture('visible', visibility)
    def walls():
        with Image.open(only(run / 'artifacts', '*_wall_destruction.png')) as image:
            return np.asarray(image.convert('RGB'))
    capture('walls', walls)
    capture('motion', lambda: read(run / 'motion_diagnostics.json'))
    if 'motion' in assets:
        contacts = assets['motion'].get('collision_contacts', {})
        row['contacts_wall'], row['contacts_drone'] = contacts.get('wall', 0), contacts.get('drone', 0)
    if 'truth_bundle' in assets:
        raw, truth, metric = assets['truth_bundle']
        assets.update(raw=raw, truth=truth, metric=metric)
        h, w = truth.shape
        if 'visible' in assets and assets['visible'].shape[0] >= h and assets['visible'].shape[1] >= w:
            assets['visible'] = assets['visible'][:h, :w]
            free = ~truth.astype(bool)
            row['sampled_lidar_free_pixels'] = int((assets['visible'] & free).sum())
            row['sampled_lidar_free_percent'] = 100 * row['sampled_lidar_free_pixels'] / int(free.sum())
        if 'walls' in assets:
            assets['walls'] = assets['walls'][plot.COMPARISON_HEADER_HEIGHT:plot.COMPARISON_HEADER_HEIGHT+h, :w]
        if 'pred' in assets and assets['pred'].shape == truth.shape:
            score = metric.count(assets['pred'])
            if row['map_score'] is not None and abs(score['map_score'] - row['map_score']) > 1e-6:
                row['red_issues'].append('official_map_score_not_reproduced')
                row['red_valid'] = row['full_valid'] = False
                row['effective_total'] = row['difficulty'] = None
    return assets


def pairwise(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row['map'], row['round'], row['seed']][row['controller']] = row
    pairs = []
    for (name, number, seed), arms in sorted(groups.items()):
        for candidate_name, reference in (('candidate_rl22', 'rl11_lines'),
                                          ('candidate_rl22', 'archived_rl17r'),
                                          ('archived_rl17r', 'rl11_lines')):
            candidate, base = arms[candidate_name], arms[reference]
            valid = candidate['red_valid'] and base['red_valid'] and candidate['actual_zones'] == base['actual_zones']
            full = valid and candidate['full_valid'] and base['full_valid']
            raw = {key: candidate[key] - base[key] if candidate.get(key) is not None and base.get(key) is not None else None
                   for key in (*RED_METRICS, 'effective_total', 'blue_walltime')}
            regressions = []
            if base['red_valid'] and not candidate['red_valid']:
                regressions.append('valid_red_completion')
            if valid:
                regressions += [key for key in ('map_score', 'wall_score', 'time_score', 'bombs', 'sampled_lidar_free_percent')
                                if raw[key] is not None and raw[key] < -1e-6]
                regressions += [key for key in ('wall_missed', 'charged_pixels', 'contacts_wall', 'contacts_drone')
                                if raw[key] is not None and raw[key] > 1e-6]
            if base['full_valid'] and not candidate['full_valid']:
                regressions.append('valid_full_completion')
            if full and raw['effective_total'] is not None and raw['effective_total'] < -1e-6:
                regressions.append('effective_total')
            pairs.append(dict(map=name, round=number, seed=seed, candidate=candidate_name, baseline=reference,
                red_pair_valid=valid, full_pair_valid=full, raw_delta=raw,
                red_delta={k: raw[k] for k in RED_METRICS} if valid else None,
                effective_total_delta=raw['effective_total'] if full else None, regressions=regressions))
    return pairs


def summarize(folder):
    folder = Path(folder).resolve()
    plan = read(folder / 'plan.json')
    if plan['configuration']['smoke']:
        raise ValueError('SMOKE INPUT: no formal data; formal atlas refuses smoke records')
    batch.validate_plan(folder, plan, live=False)
    rows, assets = [], {}
    for number, item in enumerate(plan['schedule'], 1):
        red, red_run, red_issues, red_attempts = inspect_phase(folder, plan, item, 'red')
        blue, blue_run, blue_issues, blue_attempts = inspect_phase(folder, plan, item, 'blue')
        wrapper = folder / f'match_{number:02d}.json'
        if wrapper.is_file():
            recorded = read(wrapper)
            if recorded.get('red') != red:
                red_issues.append('match_wrapper_red_mismatch')
            if (recorded.get('blue') or {}) != blue:
                blue_issues.append('match_wrapper_blue_mismatch')
        row = score_row(item, red, blue, red_issues, blue_issues)
        row.update(red_attempt=str(red_run), blue_attempt=str(blue_run),
                   map_file_sha256=plan['source_sha256'][item['map_file']],
                   all_red_attempts=red_attempts, all_blue_attempts=blue_attempts)
        try:
            arm_assets = load_assets(row)
        except Exception as exc:
            row['artifact_issues'].append('asset_processing:' + repr(exc))
            arm_assets = {}
        assets[item['name'], item['round'], item['seed'], item['controller']] = arm_assets
        rows.append(row)
    pairs = pairwise(rows)
    result = dict(scope='official local paired matches; all rows retained; no cross-map mean decides adoption',
        source_batch=str(folder), plan_sha256=batch.digest(folder / 'plan.json'), rows=rows, pairs=pairs,
        expected_matches=len(plan['schedule']), valid_red=sum(r['red_valid'] for r in rows),
        valid_full=sum(r['full_valid'] for r in rows),
        invalid=[r for r in rows if not r['red_valid'] or not r['full_valid']],
        regressions=[p for p in pairs if p['regressions']],
        formal_data_available=any(r['red_status'] != 'not_run' for r in rows),
        artifact_gaps=[dict(map=r['map'], round=r['round'], seed=r['seed'], controller=r['controller'],
                            issues=r['artifact_issues']) for r in rows if r['artifact_issues']],
        visibility_definition='physical lidar sampled every 8 ticks; 4px cells; between-ray polygon approximation',
        collision_definition='begin-contact events; not damage count or contact duration',
        flights_started=0)
    return plan, result, assets


def draw_panel(ax, row, assets, kind, regressions):
    needed = {'mapping': ('truth', 'metric', 'pred'), 'exploration': ('truth', 'visible', 'spec'),
              'trajectories': ('truth', 'trajectories', 'spec'), 'bombs': ('truth', 'bombs', 'raw', 'spec'),
              'walls': ('truth', 'walls')}[kind]
    missing = [key for key in needed if key not in assets]
    status = 'RED VALID' if row['red_valid'] else 'RED INVALID / NOT RUN'
    if not row['full_valid']:
        status += ' | FULL INVALID / MISSING'
    if any(p['candidate'] == row['controller'] for p in regressions):
        status += ' | REGRESSION'
    ax.set_title(f"{LABELS[row['controller']]}\n{status}\nM={fmt(row['map_score'])}; missed={fmt(row['wall_missed'])}; charged={fmt(row['charged_pixels'])}", fontsize=9, fontproperties=FONT)
    if missing:
        ax.text(.5, .5, 'MISSING / NOT RUN\n' + ', '.join(missing), transform=ax.transAxes,
                ha='center', va='center', color='firebrick')
        ax.set_axis_off()
        return
    truth = assets['truth']; h, w = truth.shape
    if kind == 'mapping':
        ax.imshow(assets['metric'].rgb(assets['pred']))
    elif kind == 'walls':
        ax.imshow(assets['walls'])
    elif kind == 'exploration':
        image = np.full((*truth.shape, 3), .97)
        image[assets['visible']] = [.55, .84, .69]
        image[truth.astype(bool)] = .12
        ax.imshow(image); plot.areas(ax, assets['spec'])
    else:
        ax.imshow(truth, cmap='Greys', vmin=0, vmax=1); plot.areas(ax, assets['spec'])
        if kind == 'bombs':
            radius = assets['raw']['phase1']['wall_breakdown']['blast_radius']
            for index, bomb in enumerate(assets['bombs'], 1):
                x, y = bomb['x'] + w/2, h/2 - bomb['y']
                ax.plot(x, y, 'bo', ms=3)
                ax.add_patch(plot.Circle((x, y), radius, fill=False, color='royalblue', lw=.8))
                ax.annotate(str(index), (x, y), fontsize=8)
        else:
            for slot in range(assets['trajectories'].shape[1]):
                state = assets['trajectories'][:, slot]
                good = state['active'] & np.isfinite(state['x']) & np.isfinite(state['y'])
                x = np.where(good, state['x'] + w/2, np.nan)
                y = np.where(good, h/2 - state['y'], np.nan)
                jump = np.r_[False, np.hypot(np.diff(x), np.diff(y)) > 40]
                x[jump] = y[jump] = np.nan
                ax.plot(x, y, lw=.6, label=f'slot {slot}')
            ax.legend(fontsize=7)
    ax.set(xlim=(0, w), ylim=(h, 0), aspect='equal'); ax.set_xticks([]); ax.set_yticks([])


def atlas(folder, output):
    plan, summary, assets = summarize(folder)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lines = ['# 官方三图：RL22候选、RL11、实际旧ZIP配对图册', '',
        '仅读取已完成/失败记录，新增飞行0局。每图每种子单列；不使用跨图平均分掩盖退步。',
        '有效红阶段与有效完整红蓝总分分开。蓝阶段无效时保留原始总分，有效完整分为空。',
        '暂无正式数据；下方是未运行占位，不代表结果。' if not summary['formal_data_available'] else '以下来自本批实际记录；未完成、超时、回调异常与退步同列。', '',
        '|地图/轮/版本|种子|红/蓝/完整有效|M|漏墙/受罚|墙/机碰撞|采样可见%|W/T|实际弹数|有效总分/原始总分|红/蓝秒|',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in summary['rows']:
        lines.append(f"|{r['map']}/r{r['round']}/{r['controller']}|{r['seed']}|{r['red_valid']}/{r['blue_valid']}/{r['full_valid']}|{fmt(r['map_score'])}|{fmt(r['wall_missed'])}/{fmt(r['charged_pixels'])}|{fmt(r['contacts_wall'])}/{fmt(r['contacts_drone'])}|{fmt(r['sampled_lidar_free_percent'])}|{fmt(r['wall_score'])}/{fmt(r['time_score'])}|{fmt(r['bombs'])}|{fmt(r['effective_total'])}/{fmt(r['raw_total'])}|{fmt(r['red_walltime'])}/{fmt(r['blue_walltime'])}|")
    lines += ['', '## 每组配对差值与退步', '', '差值均为候选减所列对照；有效性不足时不提供有效配对差值。', '']
    for p in summary['pairs']:
        lines += [f"- {p['map']} / r{p['round']} / seed {p['seed']} / {p['candidate']} 减 {p['baseline']}：红配对有效={p['red_pair_valid']}；完整配对有效={p['full_pair_valid']}；Δ={p['red_delta']}；Δ有效总分={p['effective_total_delta']}；退步={p['regressions'] or '本对未检出'}。"]
    lines += ['', '## 异常、超时和未运行局', '']
    for r in summary['invalid']:
        lines += [f"- {r['map']} / r{r['round']} / seed {r['seed']} / {r['controller']}：红={r['red_issues']}；蓝={r['blue_issues']}；超时={r['timeouts']}；回调={r['callback_errors']}；原始总分={r['raw_total']}；有效总分={r['effective_total']}。"]
    if not summary['invalid']:
        lines += ['无。']
    figure_errors, manifest = [], {}
    keys = sorted({(r['map'], r['round'], r['seed']) for r in summary['rows']})
    for name, number, seed in keys:
        rows = [next(r for r in summary['rows'] if (r['map'], r['round'], r['seed'], r['controller']) == (name, number, seed, controller)) for controller in ORDER]
        regressions = [p for p in summary['regressions'] if (p['map'], p['round'], p['seed']) == (name, number, seed)]
        lines += ['', f'## {name} / 第{number}轮 / 红蓝种子 {seed}', '',
                  f"退步：{[p['candidate'] + ' - ' + p['baseline'] + ':' + ','.join(p['regressions']) for p in regressions] or '本组未检出/尚无有效配对'}。"]
        for kind in KINDS:
            fig, axes = plot.plt.subplots(1, 3, figsize=(18, 5.7))
            for ax, row in zip(axes, rows):
                try:
                    draw_panel(ax, row, assets[name, number, seed, row['controller']], kind, regressions)
                except Exception as exc:
                    ax.clear(); ax.text(.5, .5, 'ARTIFACT ERROR\n' + repr(exc), ha='center', transform=ax.transAxes, wrap=True)
                    ax.set_title(LABELS[row['controller']] + ' / INVALID FIGURE')
                    figure_errors.append(dict(map=name, round=number, seed=seed, controller=row['controller'], kind=kind, error=repr(exc)))
            title = f'{name} | round {number} | red/blue seed {seed} | {kind}'
            fig.suptitle(title, fontsize=13)
            fig.text(.5, .015, NOTES[kind], fontproperties=FONT, ha='center', fontsize=10)
            fig.tight_layout(rect=(0, .05, 1, .93))
            path = output / f'{name}_r{number}_seed{seed}_{kind}.png'
            plot.save(fig, path)
            manifest[path.name] = batch.digest(path)
            lines += ['', f'![{title}]({path.as_posix()})', NOTES[kind]]
        for row in rows:
            if row['artifact_issues']:
                lines += ['', f"图件缺项 {row['controller']}：{row['artifact_issues']}"]
            for phase in ('red', 'blue'):
                attempt = Path(row[phase + '_attempt'])
                if list((attempt / 'artifacts/recordings').glob('*.npz')):
                    lines += ['', f"{row['controller']} 原始{phase}阶段回放：", '', '```powershell',
                              f"Set-Location -LiteralPath '{batch.REPO}'", f".\\replay_map.cmd '{attempt}'", '```']
                else:
                    lines += ['', f"{row['controller']} {phase}阶段回放：暂无录像（未运行或产物缺失）。"]
    summary['figure_errors'] = figure_errors
    write(output / 'summary.json', summary)
    write(output / 'figure_manifest.json', manifest)
    with (output / 'report.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return output / 'report.md'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(atlas(args.batch.resolve(), args.output.resolve()))


if __name__ == '__main__':
    main()
