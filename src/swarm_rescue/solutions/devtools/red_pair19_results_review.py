"""Read-only R19 result audit, paired scientific figures and review.

This module never starts a flight or registers a candidate. All writes are new
artifacts in the supplied review directory; the frozen R19 tools stay unchanged.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import statistics
import zipfile

import numpy as np
from PIL import Image

from swarm_rescue.solutions.devtools.red_pair19_batch import REPO, read, write, digest
from swarm_rescue.solutions.devtools.red_recovery19_weights import metric_count
from swarm_rescue.solutions.devtools.red_recovery18_audit import Metrics
from swarm_rescue.solutions.devtools import red_stopturn_match_report as plot
from swarm_rescue.solutions.devtools.red_lines11_review import FONT
from swarm_rescue.solutions.my_drone_red_repair17 import RepairMapper17

LABELS = {'rl11_lines': 'RL11', 'rl17_repair': 'RL17-R'}
KINDS = ('mapping', 'exploration', 'trajectories', 'walls', 'bombs')
NOTES = {
    'mapping': '绿=真墙命中；蓝=漏画；红=受罚误画；黄=免罚预测；灰=墙内部。',
    'exploration': '浅绿=8步/4px采样雷达可见范围近似；黑=真墙。不是精确覆盖率，也不是提交地图。',
    'trajectories': '颜色=录像槽位0–4；灰=无GPS/罗盘区；黄=无通信区；青=出生区；红框=处置区。',
    'walls': '官方墙体覆盖图：绿=覆盖内墙；蓝=覆盖边界墙；灰=未覆盖墙；红圆=爆炸范围。W不是原始覆盖率。',
    'bombs': '蓝圆=实际落弹及爆炸半径；编号=产物列表顺序，不推测机号或放弹时间；灰/黄=特殊区。',
}


def same(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.names:
        return all(same(a[k], b[k]) for k in a.dtype.names)
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)


def compare_npz(a, b, keys=None):
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        selected = sorted(keys or x.files)
        if keys is None and set(x.files) != set(y.files):
            return dict(equal=False, mismatched=['member_names'], arrays=0)
        bad = [k for k in selected if k not in y or not same(x[k], y[k])]
        return dict(equal=not bad, mismatched=bad, arrays=len(selected))


def only(directory, pattern):
    files = list(Path(directory).glob(pattern))
    if len(files) != 1:
        raise ValueError(f'Expected one {pattern}: {files}')
    return files[0]


def truth_metric(run):
    raw = read(run/'result.json')
    h, w = raw['expected_shape']
    with Image.open(only(run/'artifacts', '*_exploration_diff.png')) as image:
        rgb = np.asarray(image.convert('RGB'))
    truth = np.all(rgb[plot.COMPARISON_HEADER_HEIGHT:plot.COMPARISON_HEADER_HEIGHT+h, :w] == 40, axis=2).astype(np.uint8)
    if truth.shape != (h, w) or not truth.any():
        raise ValueError('Missing truth panel')
    return raw, truth, Metrics(truth, (0, 0, w-1, h-1))


def groups(summary):
    rows = summary['rows']
    for pair in summary['pairs']:
        arms = [next(r for r in rows if r['map'] == pair['map'] and r['round'] == pair['round'] and
                     r['seed'] == pair['seed'] and r['controller'] == c) for c in LABELS]
        yield pair, arms


def audit(a, b, out, recheck):
    summary = read(out/'summary.json')
    plan = read(a/'plan.json')
    bp = read(b/'input_plan.json')
    failures, warnings, pair_checks, intervals = [], [], [], []
    crc = 0
    for root in (a, b):
        for path in root.rglob('*.npz'):
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                if bad is not None:
                    failures.append(f'CRC {path}:{bad}')
            crc += 1
        for path in root.rglob('*.png'):
            with Image.open(path) as image:
                image.verify()
    for index, row in enumerate(summary['rows'], 1):
        run = Path(row['attempt'])
        raw, truth, metric = truth_metric(run)
        pred = np.asarray(read(only(run/'artifacts', '*_exploration.json'))['grid'], np.uint8)
        got = metric_count(metric, pred)
        for k, rk in [('hits', 'wall_hits'), ('charged', 'charged_pixels'), ('missed', 'wall_missed'), ('map_score', 'map_score')]:
            if abs(got[k]-row[rk]) > 1e-6:
                failures.append(f'{row["map"]}/{row["round"]}/{row["controller"]}:score:{k}')
        rec = read(a/f'run_{index:02d}.json')
        if rec['item'] != plan['schedule'][index-1] or rec['result'] != raw or rec['changed_during_run']:
            failures.append(f'Wrapper {index} mismatch')
        req = read(run/'request.json')
        if digest(req['map_file']) != row['map_sha256']:
            failures.append(f'Map hash {index}')
        text = (run/'run.log').read_text('utf-8', errors='replace')
        matches = [line for line in text.splitlines() if re.search(r'Warning:|WARNING:|Traceback|Exception ignored|AssertionError|ERROR:|RuntimeError:', line)]
        if matches:
            warnings.append(dict(map=row['map'], round=row['round'], controller=row['controller'], lines=matches))
        intervals.append(dict(index=index, request_mtime=(run/'request.json').stat().st_mtime,
                              receipt_mtime=(run/'integrity.json').stat().st_mtime))
    for pair, arms in groups(summary):
        x, y = [Path(r['attempt']) for r in arms]
        legal = compare_npz(x/'mapper_legal_inputs.npz', y/'mapper_legal_inputs.npz')
        states = compare_npz(only(x/'artifacts/recordings', '*.npz'), only(y/'artifacts/recordings', '*.npz'),
                             ['drone_states', 'bomb_states', 'timestamps'])
        coverage = compare_npz(x/'local13_diagnostics.npz', y/'local13_diagnostics.npz')
        bombs = read(only(x/'artifacts', '*_bombs.json')) == read(only(y/'artifacts', '*_bombs.json'))
        conditions = all(read(x/'result.json')[k] == read(y/'result.json')[k] for k in
                         ('actual_conditions', 'engine_profile', 'score_configuration'))
        pair_checks.append(dict(map=pair['map'], round=pair['round'], seed=pair['seed'],
                                legal_inputs=legal, recording=states, coverage=coverage,
                                bombs_equal=bombs, conditions_equal=conditions))
        if not conditions:
            failures.append(f'Pair conditions: {pair["map"]}/{pair["round"]}')
    b_checks = []
    for item in bp['rows']:
        name = item['map']
        original, repeated = read(b/name/'results.json'), read(recheck/name/'results.json')
        original.pop('wall_seconds'); repeated.pop('wall_seconds')
        arrays = compare_npz(b/name/'maps.npz', recheck/name/'maps.npz')
        events = [(e['owner'], e['step']) for e in original['events']]
        expected = [(e['owner'], e['step']) for e in item['events']]
        check = dict(map=name, json_equal_except_duration=original == repeated, maps=arrays,
                     all_events_once=Counter(events) == Counter(expected) and len(events) == len(set(events)),
                     full=original['valid_full_replay'], checks=original['checks'])
        if not all((check['json_equal_except_duration'], arrays['equal'], check['all_events_once'], check['full'])):
            failures.append('Offline recheck mismatch '+name)
        b_checks.append(check)
    bs = read(b/'summary.json')
    if bs['results'] != read(recheck/'summary.json')['results'] or not bs['full'] or bs['completed_events'] != bp['event_count']:
        failures.append('Offline summary mismatch')
    serial = all(x['receipt_mtime'] <= y['request_mtime'] for x, y in zip(intervals, intervals[1:]))
    if not serial:
        failures.append('Recorded file intervals overlap')
    resources = [r['resources'] for r in summary['rows']]
    result = dict(valid=not failures, failures=failures, log_warnings=warnings, checked_npz_crc=crc,
                  pair_checks=pair_checks, offline_reproduction=b_checks, observed_file_intervals_serial=serial,
                  intervals=intervals, other_windows_cpu_contention='not retrospectively measurable; CIM unavailable',
                  resources=dict(total_engine_seconds=sum(r['wall_seconds'] for r in summary['rows']),
                                 total_process_seconds=sum(r['process_seconds'] for r in summary['rows']),
                                 max_engine_seconds=max(r['wall_seconds'] for r in summary['rows']),
                                 max_process_seconds=max(r['process_seconds'] for r in summary['rows']),
                                 min_peak_commit=min(r['peak_job_commit_bytes'] for r in resources),
                                 max_peak_commit=max(r['peak_job_commit_bytes'] for r in resources)),
                  b_output_receipt_limitation='No original completion receipt; current hashes and independent deterministic rerun verified.',
                  reviewed_utc=datetime.now(timezone.utc).isoformat())
    write(out/'detailed_audit.json', result)
    print('AUDIT', result['valid'], 'CRC', crc, 'pair checks', len(pair_checks), 'warnings', len(warnings), flush=True)
    if failures:
        raise ValueError(failures)


def load_arm(row):
    run = Path(row['attempt']); raw, truth, metric = truth_metric(run)
    with np.load(only(run/'artifacts/recordings', '*.npz'), allow_pickle=False) as z:
        trajectories = z['drone_states']
    with np.load(run/'local13_diagnostics.npz', allow_pickle=False) as z:
        visible = np.logical_or.reduce([z[k].astype(bool) for k in z.files])
    h, w = truth.shape
    with Image.open(only(run/'artifacts', '*_wall_destruction.png')) as image:
        walls = np.asarray(image.convert('RGB'))[plot.COMPARISON_HEADER_HEIGHT:plot.COMPARISON_HEADER_HEIGHT+h, :w]
    return dict(row=row, raw=raw, truth=truth, metric=metric,
                spec=read(read(run/'request.json')['map_file']), trajectories=trajectories,
                visible=np.repeat(np.repeat(visible, 4, 0), 4, 1)[:h, :w],
                pred=np.asarray(read(only(run/'artifacts', '*_exploration.json'))['grid'], np.uint8),
                bombs=read(only(run/'artifacts', '*_bombs.json'))['bombs'], walls=walls)


def draw_arm(ax, arm, kind):
    truth, row = arm['truth'], arm['row']; h, w = truth.shape
    if kind == 'mapping':
        ax.imshow(arm['metric'].rgb(arm['pred']))
    elif kind == 'walls':
        ax.imshow(arm['walls'])
    elif kind == 'exploration':
        rgb = np.full((*truth.shape, 3), .97); rgb[arm['visible']] = [.55, .84, .69]; rgb[truth.astype(bool)] = .12
        ax.imshow(rgb); plot.areas(ax, arm['spec'])
    else:
        ax.imshow(truth, cmap='Greys', vmin=0, vmax=1); plot.areas(ax, arm['spec'])
        if kind == 'bombs':
            for i, bomb in enumerate(arm['bombs'], 1):
                x, y = bomb['x']+w/2, h/2-bomb['y']
                ax.plot(x, y, 'bo', ms=3)
                ax.add_patch(plot.Circle((x, y), arm['raw']['phase1']['wall_breakdown']['blast_radius'], fill=False, color='royalblue', lw=1))
                ax.annotate(str(i), (x, y), fontsize=8, color='navy')
        else:
            for slot in range(arm['trajectories'].shape[1]):
                t = arm['trajectories'][:, slot]; good = t['active'] & np.isfinite(t['x']) & np.isfinite(t['y'])
                x = np.where(good, t['x']+w/2, np.nan); y = np.where(good, h/2-t['y'], np.nan)
                jump = np.r_[False, np.hypot(np.diff(x), np.diff(y)) > 40]; x[jump] = y[jump] = np.nan
                ax.plot(x, y, lw=.7, label=f'slot {slot}')
            ax.legend(fontsize=7, loc='upper right')
    ax.set(xlim=(0, w), ylim=(h, 0), aspect='equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{LABELS[row['controller']]} | M {row['map_score']:.4f} | W {row['wall_score']:.4f} | T {row['time_score']:.4f}\n"
                 f"hit {row['wall_hits']} / missed {row['wall_missed']} / charged {row['charged_pixels']} / bombs {row['bombs']}", fontsize=10)


def figures(a, b, out):
    summary = read(out/'summary.json'); dest = out/'figures'; dest.mkdir(exist_ok=False)
    paths = []
    for pair, rows in groups(summary):
        arms = [load_arm(r) for r in rows]
        for kind in KINDS:
            fig, axes = plot.plt.subplots(1, 2, figsize=(16, 6.2))
            for ax, arm in zip(axes, arms):
                draw_arm(ax, arm, kind)
            label = f"{pair['map']} / r{pair['round']} / seed {pair['seed']} / {kind} / valid={pair['valid']}"
            fig.suptitle(label+f"\nRL17-R - RL11: M {pair['delta']['map_score']:+.4f}; NO_COM + NO_GPS (both arms)", fontsize=13)
            note = NOTES[kind]
            if kind in ('exploration', 'trajectories', 'bombs'):
                note += ' 本批配对记录相同；提交地图可以不同。'
            fig.text(.5, .016, note, fontproperties=FONT, ha='center', fontsize=10)
            fig.tight_layout(rect=(0, .045, 1, .91))
            path = dest/f"{pair['map']}_r{pair['round']}_{kind}.png"; plot.save(fig, path); paths.append(path)
        print('FIGURES', pair['map'], pair['round'], flush=True)
    for item in read(b/'input_plan.json')['rows']:
        name = item['map']; result = read(b/name/'results.json')
        raw, truth, metric = truth_metric(Path(result['source']))
        mapper = RepairMapper17((truth.shape[1], truth.shape[0]))
        with np.load(b/name/'maps.npz', allow_pickle=False) as maps:
            fig, axes = plot.plt.subplots(1, 2, figsize=(16, 6.4))
            for ax, key, title in zip(axes, ('baseline', 'candidate'), ('uniform recovery 0.8', 'preserve frame weights')):
                score = result['final'][key]; ax.imshow(metric.rgb(maps['submitted_'+key])); ax.axis('off')
                ax.set_title(f"{title}\nM {score['map_score']:.5f} | hit {score['hits']} | missed {score['missed']} | charged {score['charged']}", fontsize=11)
            fig.suptitle(f"{name} / historical RL17-R / r{item['round']} / seed {item['seed']}\nOFFLINE conditional submission; actual received versions; NOT new flight", fontsize=13)
            fig.text(.5, .02, NOTES['mapping']+' 轨迹、匹配与收发时序固定。', fontproperties=FONT, ha='center')
            fig.tight_layout(rect=(0, .05, 1, .91)); path = dest/f'{name}_offline_final.png'; plot.save(fig, path); paths.append(path)
            wanted = (3, 1755) if name == 'Map08' else (3, 2793)
            event = next(e for e in result['events'] if (e['owner'], e['step']) == wanted)
            # Separate position correction and weight comparisons with equal views.
            fig, axes = plot.plt.subplots(2, 2, figsize=(16, 11))
            for ax, key in zip(axes.flat, ('baseline_unanchored_promoted', 'baseline_anchored', 'baseline_after', 'candidate_after')):
                score = event['scores'][key]
                ax.imshow(metric.rgb(mapper.to_pixels(maps[f'd{wanted[0]}_t{wanted[1]}_{key}']))); ax.axis('off')
                ax.set_title(f"{key}\nM {score['map_score']:.4f} | hit {score['hits']} | charged {score['charged']}", fontsize=10)
            fig.suptitle(f"{name} / historical RL17-R / r{item['round']} / seed {item['seed']} / d{wanted[0]} t{wanted[1]}\n"
                         f"Single drone, NOT team score | projection charged {event['baseline_projection_delta']['charged']:+d}; weight delta M {event['weight_delta']['map_score']:+.4f}", fontsize=12)
            fig.text(.5, .012, NOTES['mapping']+' 上排固定0.8只校正位姿；下排比较恢复权重（含之前事件累计效果）。', fontproperties=FONT, ha='center')
            fig.tight_layout(rect=(0, .035, 1, .93)); path = dest/f'{name}_offline_event.png'; plot.save(fig, path); paths.append(path)
    write(out/'figure_manifest.json', {str(p): digest(p) for p in paths})
    print('FIGURE TOTAL', len(paths), flush=True)


def fmt(value, signed=False):
    if value is None:
        return '未评测'
    if isinstance(value, float):
        return format(value, '+.4f' if signed else '.4f')
    return format(value, '+d') if signed and isinstance(value, int) else str(value)


def table(headers, rows):
    return ['|'+'|'.join(headers)+'|', '|'+'|'.join('---' for _ in headers)+'|'] + [
        '|'+'|'.join(str(x).replace('|', '/') for x in row)+'|' for row in rows]


def embed(path, label):
    if not path.is_file():
        raise FileNotFoundError(path)
    return ['', f'![{label}]({path.as_posix()})', '']


def put_text(path, lines):
    with path.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')


def report(a, b, out):
    s = read(out/'summary.json'); audit_doc = read(out/'detailed_audit.json'); initial = read(out/'initial_audit.json')
    assert audit_doc['valid'] and s['valid_count'] == 24 and s['invalid_count'] == 0
    plan = read(a/'plan.json'); bp = read(b/'input_plan.json'); bs = read(b/'summary.json')
    figures_dir = out/'figures'; atlas_path = out/'all_rounds.md'; details_path = out/'offline_all_events.md'
    root_report = REPO/'复赛阶段_R19_配对红队与恢复权重结果核验_20261117.md'
    version_table = []
    for c, stats in s['unpaired_valid_aggregate'].items():
        version_table.append([LABELS[c], fmt(stats['map_score']['mean']), fmt(stats['map_score']['median']),
                              fmt(stats['map_score']['min']), fmt(stats['wall_score']['mean']),
                              fmt(stats['time_score']['mean']), fmt(stats['wall_hits']['mean']),
                              fmt(stats['charged_pixels']['mean']), fmt(stats['wall_seconds']['mean'])])
    lines = ['# R19：RL11 / RL17-R 配对红队与恢复权重离线核验', '', '2026-11-17；本轮只分析用户已完成输出，另做一次固定输入离线复算，新增飞行 0、蓝队阶段 0。', '',
             '**结论：本批不能证明 RL17-R 稳定优于 RL11。RL11 继续作为稳定对照，RL17-R 保留研究候选，正式入口不切换。保留帧权重减少误画，但出现一致的真墙损失和跨图收益反向，暂不推进飞行或接入；可继续离线研究权重与位姿的原因。**', '',
             f'A 输入：`{a}`；B 输入：`{b}`。按本地最新 r19a_/r19b_ 目录读取，未覆盖原始批次。',
             f'当前 HEAD=`{initial["head"]}`，来源为本次 git 核对，未使用旧文档 HEAD。接手时仅有准备阶段四个独立工具及准备报告未跟踪，完整保留。',
             f'[全部12对/24局图册与回放]({atlas_path.as_posix()}) · [全部64恢复事件与每机/提交图表]({details_path.as_posix()}) · [机器可读原始核验]({(out/"detailed_audit.json").as_posix()})', '',
             '## 核验与边界', '',
             f'- A：24/24 有效，0 缺失、0 多余 attempt、0 超时、0 控制器/回调异常、0 日志警告。24 份完成收据逐文件 SHA256 重算一致；24 次真实五机类/地图器、请求/结果身份和实际区域均与计划一致。{initial["frozen_sources"]} 个冻结文件的解码字节、现有源码与依赖环境一致；重新计算 summary 与原文件相同。',
             '- 地图：100 张原图和冻结清单重验通过，70 训练 / 15 验证 / 15 测试不变；本批6图都属于训练集且分属6个冻结组。全部来自混合分区同一生成族，全部有静态 NO_COM_ZONE 和 NO_GPS_ZONE；无无特殊区样本，也不是独立未见测试。未调验证/测试集。',
             '- A 12对：11个合法输入数组、5个可见范围数组均逐项相同（含 NaN）；录像逐帧无人机位置/角度/健康/活动状态、炸弹状态及实际落弹列表相同。录像 timestamps 是墙钟时间，各局不同；没有声称整份录像字节相同。相同步数和输入排除了本批轨迹改变带来的收益，不能据此承诺别的负载下仍完全一致。',
             '- B：完整覆盖 Map08 的50事件、Map09的14事件，无遗漏或重复。原输入计划等于冻结计划、历史输入哈希一致。独立重放后408个数组逐值一致，结果JSON除耗时外完全一致。44925个机步的基线位姿误差为0，145个保存发布快照、10个本机终图、实际提交源图和2张提交图复现。',
             '- B 原工具没有离线输出的完成时哈希收据。本次保存了读取时哈希并用独立重放验证；不能把新清单说成原有收据。中间包保存了发布/接收版本及哈希，没有每包完整字节；中间快照来自确定性重建。候选没有实际发送无线包，结论是固定真实收发日程的条件离线提交图。',
             f'- 全部122个原始NPZ通过CRC，原始PNG可解码。A 24个录像通过回放身份校验、时间单调及帧数检查（均为引擎步数减1，符合记录位置）；replay_map.cmd --help 实测可运行，未逐一打开GUI。',
             '- D（蓝队难度）、完整原始总分、完整有效总分全部未评测。result.json 内的 partial round_score 和 score_bomb=0 占位值不充当完整分或难度成绩。真值只用于离线评分/绘图。', '',
             '## 资源、源码与实际入口', '',
             'A 原生5400步/180秒，外层240秒；每局5机×2弹、单CPU亲和、Windows Job提交内存限制4,000,000,000字节，按原计划串行。请求/收据文件时间序列无重叠。CIM在本机不可用，无法事后证明其他窗口当时没有争抢CPU；本地约束也不等于远端平台硬件性能。',
             f"A 引擎累计 {audit_doc['resources']['total_engine_seconds']:.2f} 秒（{audit_doc['resources']['total_engine_seconds']/60:.2f}分钟）；进程累计 {audit_doc['resources']['total_process_seconds']:.2f} 秒（{audit_doc['resources']['total_process_seconds']/60:.2f}分钟），两者均非CPU时间。最长引擎 {audit_doc['resources']['max_engine_seconds']:.2f} 秒，最长进程 {audit_doc['resources']['max_process_seconds']:.2f} 秒；峰值Job提交内存范围 {audit_doc['resources']['min_peak_commit']:,}–{audit_doc['resources']['max_peak_commit']:,} 字节。", 
             f"B 原离线墙钟 {bs['resources']['wall_seconds']:.2f} 秒、CPU {bs['resources']['cpu_seconds']:.2f} 秒、RSS峰值 {bs['resources']['peak_rss_bytes']:,} 字节；Job峰值 {read(b.with_name(b.name+'_resources.json'))['peak_job_commit']:,} 字节。一次复算亦受单CPU/4GB限制，未启动仿真。", '',
             '|文件/计划|SHA256|', '|---|---|']
    for name in ('my_drone_red_lines11.py', 'my_drone_red_repair17.py', 'my_drone_red_p2l17.py', 'my_drone_red_mapping.py', 'my_drone_eval.py'):
        key = 'src/swarm_rescue/solutions/'+name
        lines.append(f'|{name}|`{plan["source_sha256"][key]}`|')
    lines += [f'|A plan.json|`{initial["a_plan_sha256"]}`|', f'|B input_plan.json|`{initial["b_plan_sha256"]}`|', '',
              'RL11实际类 MyDroneRedLines11 / WallLineMapper；RL17-R实际类 MyDroneRedRepair17 / RepairMapper17。正式 my_drone_eval.py 仍为 place=MyDroneRedAdvanced2、rescue=MyDroneRescueExample。Python/第三方版本及RECORD哈希见原计划 environment；本次现环境完全匹配。未引入模型/API。', '',
              '## A：地图与配对清单', '', '每图r1种子2026111701（先RL11），r2种子2026111702（先RL17-R）；24红/0蓝。尺寸、墙、区域在两版间冻结。', '']
    lines += table(['训练地图', '选择理由', '地图JSON SHA256'], [[r['name'], r['reason'], '`'+r['map_sha256']+'`'] for r in plan['selection']])
    lines += ['', '## A：24局实际分项', '', '区域“COM+GPS”均表示实测 NO_COM_ZONE + NO_GPS_ZONE；NoGps同时关闭GPS/罗盘。命中/漏画为官方墙面像素，受罚为超出免罚区的预测像素，不等于扣分百分点。', '']
    lines += table(['地图/轮', '版本', '红种子', '实际区', 'M', '命中/漏画/受罚', 'W', 'T', '弹', '步', '引擎秒/进程秒', '峰值Job字节', '完成/有效', 'D/完整总分'],
                   [[f"{r['map']}/r{r['round']}", LABELS[r['controller']], r['seed'], 'COM+GPS', fmt(r['map_score']),
                     f"{r['wall_hits']}/{r['wall_missed']}/{r['charged_pixels']}", fmt(r['wall_score']), fmt(r['time_score']), r['bombs'], r['steps'],
                     f"{r['wall_seconds']:.3f}/{r['process_seconds']:.3f}", r['resources']['peak_job_commit_bytes'], f"{r['status']}/是", '未评测/未评测'] for r in s['rows']])
    lines += ['', '## A：逐对差值与所有退步', '', '差值均为RL17-R减RL11；负ΔM/命中、正Δ漏画/受罚为对应指标退步。每一项都保留，不以净增分掩盖真墙丢失。', '']
    differences = []
    for pair in s['pairs']:
        d = pair['delta']; flags = []
        if d['map_score'] < 0: flags.append('地图分下降')
        if d['wall_hits'] < 0: flags.append('命中减少/漏画增加')
        if d['charged_pixels'] > 0: flags.append('受罚误画增加')
        if d['wall_seconds'] > 0: flags.append('耗时增加')
        differences.append([f"{pair['map']}/r{pair['round']}", pair['seed']] + [fmt(d[k], True) for k in
                           ('map_score', 'wall_hits', 'wall_missed', 'charged_pixels', 'wall_score', 'time_score', 'bombs', 'wall_seconds')] + ['；'.join(flags) or '无退步'])
    lines += table(['地图/轮', '种子', 'ΔM', 'Δ命中', 'Δ漏画', 'Δ受罚', 'ΔW', 'ΔT', 'Δ弹', 'Δ秒', '退步项'], differences)
    lines += ['', '**分数退步2对**：37/r1 −0.1848、27/r1 −0.0216；**命中下降6对**：24/r1、24/r2、37/r1、27/r1、27/r2、55/r1；**误画增加1对**：37/r2 +1977。合计7对有至少一项地图质量退步。耗时增加7对已全部列在上表，最大72/r1 +5.6883秒，不将单次计时差直接归因算法成本。', '',
              '## A：均值、中位数与最差表现', '']
    lines += table(['版本', 'M均值', 'M中位数', 'M最低', 'W均值', 'T均值', '命中均值', '受罚均值', '引擎秒均值'], version_table)
    lines += [''] + table(['配对指标', 'n', '差值均值', '差值中位数', '最小差值', '最大差值'],
                          [[k, v['n']] + [fmt(v[t], True) for t in ('mean', 'median', 'min', 'max')] for k, v in s['paired_aggregate'].items()])
    lines += [''] + table(['地图', '2轮平均ΔM', '中位ΔM', '最差ΔM', '最好ΔM'],
                          [[name] + [fmt(v['map_score'][k], True) for k in ('mean', 'median', 'min', 'max')] for name, v in s['per_map_paired_aggregate'].items()])
    lines += ['', '事实：地图分5胜/5平/2负，配对平均+0.2222、中位0；绝对版本中位数RL17-R略低。24两轮贡献净总差值的大部分，37/27内部出现正负反向，72/90完全无增益。所有W/T/弹数逐对相同，均10弹。',
              '最低地图分两版同为72/r1：9.6836，命中13016、漏画77700；最低W为72/r2的24.8142。55/27/90也有低图分局，不能因“有效完成/放满10弹”忽略质量问题。大范围漏画存在，但本次未做其导航/可达性/定位的原因归因。',
              '判断：本批支持“RL17-R在部分样本净减误画，平均有小幅地图收益”，不支持“稳定全面优于”。6张相关训练图、每图2种子不能作统计显著性或跨地图族泛化声明。后续保留RL11对照和RL17-R研究候选，不根据本批自动替换基线或正式入口。', '',
              '## A：关键退步与最低分图示', '']
    for name, number, kinds, note in [
        ('yh_map_fs_37', 1, KINDS, '本批最大地图分退步：少2081受罚像素，但丢914真墙命中，净−0.1848。两版实际探索、轨迹和弹位相同。'),
        ('yh_map_fs_27', 1, ('mapping',), '第二个地图分退步：−0.0216，虽幅度小仍保留。'),
        ('yh_map_fs_72', 1, ('mapping', 'exploration', 'trajectories', 'walls', 'bombs'), '两版最低分并列9.6836；这是低质量有效局，不是超时或损坏记录。')]:
        lines += ['', note]
        for kind in kinds:
            lines += embed(figures_dir/f'{name}_r{number}_{kind}.png', f'{name} r{number} seed2026111701 RL11 vs RL17-R {kind}') + [NOTES[kind]]
    lines += ['', '## B：单因素离线结果与判断', '',
              '使用历史RL17-R Map08/r1/seed200806601、Map09/r1/seed2477227837，来源 rl17_m89_once_20261116，并核对RL18审计输入；二者是已诊断样本。**B暂无新飞行图**，以下为原始输入带的条件离线重建，不是新权重飞行成绩。',
              '实际流程：正常帧权重1.0；缺测缓存依最近匹配状态为0.45或0.2；恢复流程原来统一0.8。候选只在恢复时保留frame[4]，墙命中与自由擦除同步改变（擦除强度仍0.35×权重，阈值2.4、上限30不变）。本次恢复帧实测3782个0.45、105个0.2，共3887；这不代表整个算法只有两种权重。',
              '固定历史轨迹、扫描匹配实际返回值、位姿/锚定校正、正常帧、通信发布/接受版本和提交时刻。每个接收版本来自当时已经发布的来源；最终并集只用提交者实际持有版本，无未来信息或理想化全机终图合并。候选没有反哺匹配或导航。', '']
    final_rows = []
    results = [read(b/item['map']/'results.json') for item in bp['rows']]
    for r in results:
        f = r['final']; delta = f['delta']
        final_rows.append([r['map'], r['seed'], fmt(f['baseline']['map_score']), fmt(f['candidate']['map_score']),
                           fmt(delta['map_score'], True), delta['hits'], delta['missed'], delta['charged'],
                           fmt(delta['hits']/f['baseline']['hits']*100, True)+'%', fmt(delta['charged']/f['baseline']['charged']*100, True)+'%'])
    lines += table(['历史图', '种子', '统一权重M', '保留权重M', 'ΔM', 'Δ命中', 'Δ漏画', 'Δ受罚', '命中相对变化', '受罚相对变化'], final_rows)
    lines += ['', '两图等权平均ΔM +0.1922、中位+0.1922、最差−0.2855；只有两张已诊断历史图，不能把微小正均值当作可靠收益。2/2提交图真墙命中减少，10/10单机终图地图分下降，64/64恢复后单机候选地图分下降。这64项含之前恢复的累计影响、彼此相关，不能当64个独立实验，也不是每次孤立改权重的因果量。',
              'Map08单机最差终图为d2：−4.1404；Map09为d2：−1.6198。事件最差分别Map08/d2/t3970 −4.5917、Map09/d2/t822 −1.7828。全部事件、每机绝对命中/漏画/受罚与分数见专表，没有筛掉负面结果。',
              '可从固定来源并集机制解释为什么单机分均降但Map08团队分仍升：单机丢失的部分真墙可由其他来源覆盖，误画减少则仍有收益；这是与本结果相容的机制解释，未作逐像素唯一因果分解。',
              '判断：这个“全部恢复帧保留原权重”的候选不宜进入下一轮飞行或接入正式建图。它展示了权重会同时改变误画与有效墙证据，值得继续离线归因；当前没有支持自动叠加阈值、门控或其他补丁。保留原0.8对照和全部失败证据。', '']
    for r in results:
        lines += embed(figures_dir/f"{r['map']}_offline_final.png", f"{r['map']} 历史RL17-R r1 seed{r['seed']} 条件提交图权重对照")
    lines += ['', '## B：恢复重投的正反案例', '',
              'Map08/d3/t1755：固定0.8时位姿重投使受罚+6082；Map09/d3/t2793：同样比较使受罚−15656。它们支持“不能统一取消恢复重投”。新权重候选在这两个恢复后单机图的ΔM却分别−2.4098、−0.3927，说明改变位姿与改变权重不能混为一谈。',
              '图中上排只比较统一0.8下校正位姿前后；下排比较实际恢复后统一权重与保留帧权重。都是单机中间图，不能冒充最终团队得分。', '']
    for r in results:
        lines += embed(figures_dir/f"{r['map']}_offline_event.png", f"{r['map']} 历史RL17-R r1 seed{r['seed']} 指定恢复事件")
    lines += ['', '## 异常、限制与后续', '',
              '已检出的运行失败/超时/回调异常：0。性能/质量退步：全部12对差值、7对质量退步、7对计时增加、64个恢复后单机差值、10张单机终图及Map09最终提交退步均已保留。数据限制包括B缺原始结果收据/完整中间包、A无无特殊区样本、同族训练样本、没有蓝队、无法事后确认外部CPU负载；这些限制没有改写为运行失败。',
              '建议下一步先离线检查本批两个分数退步样本、低图分样本及权重导致的真墙损失。尚未运行进一步飞行、验证集/测试集、固定蓝队对抗；未切换入口、未接入RL18影子方案、未组合新补丁。',
              f'所有新增产物位于 `{out}`；独立工具 `src/swarm_rescue/solutions/devtools/red_pair19_results_review.py`。已有四个准备工具、源码、地图、旧文档、批次与提交包均受前后哈希保护，最终核验见 [final_audit.json]({(out/"final_audit.json").as_posix()})。',
              '回放只打开已有录像，不重跑策略；每组具体命令已写入全轮次图册。蓝队无运行，因此无可提供的本批蓝队回放。']
    put_text(root_report, lines)

    atlas = ['# R19 A 全轮次配对图册', '', f'来源 `{a}`；[总报告]({root_report.as_posix()})。24/24有效，12对全部展示，左RL11、右RL17-R。',
             '每对实际区域均为NO_COM_ZONE+NO_GPS_ZONE，所有D及完整总分未评测。图中“配对记录相同”指探索采样/轨迹状态/弹位；墙钟时间戳不同。',
             '图像视野和比例相同。离线真值不参与策略。轨迹按录像槽位标色，炸弹编号只是产物列表顺序。']
    for pair, arms in groups(s):
        name, number = pair['map'], pair['round']
        atlas += ['', f"## {name} / r{number} / seed {pair['seed']}", '', '完成ok/有效；实际特殊区COM+GPS；源码与地图哈希见总报告。', '']
        atlas += table(['版本', 'M', 'W', 'T', '命中/漏画/受罚', '弹', '步/引擎秒', 'D/完整总分'],
                       [[LABELS[r['controller']], fmt(r['map_score']), fmt(r['wall_score']), fmt(r['time_score']),
                         f"{r['wall_hits']}/{r['wall_missed']}/{r['charged_pixels']}", r['bombs'], f"{r['steps']}/{r['wall_seconds']:.3f}", '未评测/未评测'] for r in arms])
        for kind in KINDS:
            atlas += embed(figures_dir/f'{name}_r{number}_{kind}.png', f"{name} r{number} seed{pair['seed']} RL11 vs RL17-R {kind}") + [NOTES[kind]]
        for r in arms:
            atlas += ['', LABELS[r['controller']]+' 原始红阶段回放：', '', '```powershell', f"Set-Location -LiteralPath '{REPO}'", fr".\replay_map.cmd '{r['attempt']}'", '```']
    put_text(atlas_path, atlas)

    details = ['# R19 B：全部恢复事件、每机与提交者版本', '', f'来源 `{b}`；[总报告]({root_report.as_posix()})。',
               '暂无新飞行图。历史RL17-R两图五机全输入，64个事件完整列出；恢复后的候选差值包含之前事件累计效果，不是孤立事件因果量。',
               '基线=统一恢复0.8，候选=每帧原权重；轨迹/匹配/校正/收发时序固定。M为地图分，H/L/C为真墙命中/漏画/受罚像素。蓝队难度及完整总分未评测。']
    for r in results:
        details += ['', f"## {r['map']} / 历史RL17-R / r{r['round']} / seed {r['seed']}", '', f"源 `{r['source']}`；{r['event_count']}/{r['expected_events']}恢复事件；完整条件离线重建。", '']
        items = [('条件最终提交图', r['final'])] + [(f"本机d{x['owner']}终图（非提交）", x) for x in r['per_drone_final']]
        details += table(['范围', '基线M', '候选M', '基线H/L/C', '候选H/L/C', 'ΔM', 'ΔH', 'ΔL', 'ΔC'],
                         [[name, fmt(x['baseline']['map_score']), fmt(x['candidate']['map_score']),
                           '/'.join(str(x['baseline'][k]) for k in ('hits', 'missed', 'charged')),
                           '/'.join(str(x['candidate'][k]) for k in ('hits', 'missed', 'charged'))] +
                          [fmt(x['delta'][k], True) for k in ('map_score', 'hits', 'missed', 'charged')] for name, x in items])
        details += ['', f"最终提交者步={r['final']['step']}，实际已接受来源版本={r['final']['actual_sources']}。这不是各机终图的全知并集。", '']
        details += embed(figures_dir/f"{r['map']}_offline_final.png", f"{r['map']} r1 seed{r['seed']} 条件最终图")
        erows = []
        for e in r['events']:
            x, y = e['scores']['baseline_after'], e['scores']['candidate_after']
            erows.append([f"d{e['owner']}/t{e['step']}", str(e['frame_weights']), e['pending_count'],
                          fmt(x['map_score']), fmt(y['map_score']), '/'.join(str(x[k]) for k in ('hits', 'missed', 'charged')),
                          '/'.join(str(y[k]) for k in ('hits', 'missed', 'charged')),
                          fmt(e['weight_delta']['map_score'], True), e['weight_delta']['hits'], e['weight_delta']['charged'],
                          e['baseline_projection_delta']['charged'], e['first_holder_receipt'] if e['first_holder_receipt'] is not None else '提交前未接受'])
        details += ['', '### 全部恢复事件（无筛选）', ''] + table(
            ['机/恢复步', '帧权重计数', '帧数', '基线M', '候选M', '基线H/L/C', '候选H/L/C', 'ΔM', 'ΔH（ΔL取反）', 'ΔC', '基线仅重投ΔC', '首次相关接收步'], erows)
        details += ['', '首次相关接收表示提交者首次接受该来源且版本不早于恢复步，之后的版本可能包含其他扫描；不是该事件的唯一贡献时刻。', '',
                    '### 提交者真实来源版本时序', '']
        details += table(['提交者步', '当时实际来源版本', '基线M', '候选M', 'ΔM', 'ΔH', 'ΔL', 'ΔC'],
                         [[x['step'], x['actual_sources'], fmt(x['baseline']['map_score']), fmt(x['candidate']['map_score'])] +
                          [fmt(x['delta'][k], True) for k in ('map_score', 'hits', 'missed', 'charged')] for x in r['holder_timeline']])
        details += embed(figures_dir/f"{r['map']}_offline_event.png", f"{r['map']} r1 seed{r['seed']} 正反恢复诊断")
        details += ['', '历史基线红阶段回放（不会播放未运行的候选）：', '', '```powershell', f"Set-Location -LiteralPath '{REPO}'", fr".\replay_map.cmd '{r['source']}'", '```']
    put_text(details_path, details)
    print('REPORT', root_report, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a', type=Path, required=True); p.add_argument('--b', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--recheck', type=Path)
    p.add_argument('--phase', choices=('audit', 'figures', 'report'), required=True)
    args = p.parse_args(); a, b, out = args.a.resolve(), args.b.resolve(), args.output.resolve()
    if args.phase == 'audit':
        audit(a, b, out, args.recheck.resolve())
    elif args.phase == 'figures':
        figures(a, b, out)
    else:
        report(a, b, out)


if __name__ == '__main__':
    main()
