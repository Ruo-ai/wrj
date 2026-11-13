"""Offline second-pass report. No simulator or online policy changes."""
import argparse
import multiprocessing
import statistics
from pathlib import Path
from collections import Counter
from swarm_rescue.solutions.devtools import report_advanced5 as old
from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev

LABELS = {'advanced5_narrow': '窄路基线', 'advanced5_progress': '可信进展候选'}


def draw_one_map(base, name, records):
    old.graphics(base, [name], records)
    return ev.read(base/'report/wall_overlap.json')


def report(base, document, figures=False, partial=False):
    base = Path(base).resolve()
    plan = ev.read(base/'plan.json')
    completion = ev.read(base/'completion.json')
    if not completion['placement_complete'] or not completion['sources_unchanged']:
        raise ValueError('Incomplete red placement or source verification')
    if not partial and not completion['complete']:
        raise ValueError('Full comparison requires all red and blue phases to pass')
    if completion['completed'] != completion['expected']:
        raise ValueError('Batch did not reach the end of its schedule')
    assert ev.digest(base/'source_snapshot.zip') == plan['source_snapshot_sha256']
    old.LABELS = {s: LABELS[s] for s in plan['controllers']}
    records = old.audit(base)
    expected = {(m, r) for m in plan['maps'] for r in range(1, plan['rounds']+1)}
    scores = {}
    audits = {}
    for selector in plan['controllers']:
        red_rows = ev.read(base/selector/'red/results.json')
        if (len(red_rows) != len(expected) or
                {(r['map'], r['round']) for r in red_rows} != expected or
                any(r['status'] != 'ok' or r['walltime_limited'] or not r['submitted']
                    for r in red_rows)):
            raise ValueError(f'Incomplete valid red coverage: {selector}')
        for phase in (('red',) if partial else ('red', 'blue')):
            manifest = base/selector/phase/'manifest.json'
            if not partial:
                ev.verify_sources(ev.read(manifest))
            for row in ev.read(base/selector/phase/'results.json'):
                ev.validate_receipt(base/selector/phase/ev.round_name(row), ev.digest(manifest))
        if not partial:
            scores[selector] = old.paired(base, selector, expected)
        audits[selector] = []
        for row in records[selector]:
            events = old.events(Path(row['run']))
            transitions = [e for e in events if e['event'] == 'progress_goal_transition']
            confirmed = [e['step'] for e in events if e['event'] == 'placement_confirmed']
            audits[selector].append(dict(map=row['map'], round=row['round'],
                reasons=dict(Counter(c['reason'] for e in transitions for c in e['causes'])),
                information_only_success=sum(e['event']=='loop_recovery_verified' and
                    e.get('successful', False) and e.get('distance', 0)<45 for e in events),
                first_confirmation=min(confirmed, default=None),
                last_confirmation=max(confirmed, default=None), confirmations=len(confirmed),
                loop_frames=row['combined_frames'], mean_health=row['mean_health']))
    ev.atomic_json(base/'report/progress_audit.json', audits)
    means = {s: statistics.mean(r['round_score'] for r in rows) for s, rows in scores.items()}
    ev.atomic_json(base/'report/full_scores.json', dict(rounds=scores, means=means, partial=partial))
    if figures:
        overlap = []
        # Arcade retains process-global OpenGL state even after a window closes.
        for name in plan['maps']:
            with multiprocessing.get_context('spawn').Pool(1) as pool:
                overlap.extend(pool.apply(draw_one_map, (base, name, records)))
        ev.atomic_json(base/'report/wall_overlap.json', overlap)
    lines = ['# 第二轮P0＋P1a：评测结果', '',
        f'批次 {base.as_posix()}。已执行 {completion["completed"]}/{completion["expected"]} 项，'
        f'通过有效性核验 {sum(row["status"] == "ok" for row in completion["history"])} 项。', '',
        '方法：等待期间保留历史，累计等待预算；恢复成功要求空间离开。修订版保留集合、布弹及确认阶段的原弱恢复门控。任务终点变化关联原因；尚未建立稳定分支任务或启用协同接替。', '',
        '正常原地观测仍受保护；短期离开不等于持续脱困。所有几何循环指标沿用旧版，只是疑似，含必要返回。', '',
        '## 采用判断', '']
    if partial:
        lines += ['本表只比较有效红队的地图、墙体和时间三项。保留窄路基线。']
        blue_history = [r for r in completion['history'] if r['phase'] == 'blue']
        if blue_history:
            lines += ['', '|蓝队使用的红队布局|轮次|步数|处置枚数|墙钟秒|状态|',
                      '|---|---|---:|---:|---:|---|']
            ledger = ev.read(base/'execution_receipts.json')
            for item in blue_history:
                key = f'{item["controller"]}/blue/{ev.round_name(item)}'
                run = base/key
                manifest = run.parent/'manifest.json'
                if ev.integrity(run, ev.digest(manifest)) != ledger[key]:
                    raise ValueError(f'Blue artifacts changed: {key}')
                row = ev.read(run/'result.json')
                status = '触限，仅作诊断' if row['walltime_limited'] else row['status']
                lines.append(f'|{LABELS[item["controller"]]}|{run.name}|{row["elapsed_steps"]}|'
                             f'{row["disposed_count"]}|{row["elapsed_walltime"]:.2f}|{status}|')
    else:
        a, b = means['advanced5_narrow'], means['advanced5_progress']
        full = len(expected) == 18 and len(plan['schedule']) == 72
        safe = all(not r['red_walltime_limited'] and not r['blue_walltime_limited']
                   for rows in scores.values() for r in rows)
        adopt = full and safe and b > a
        ev.atomic_json(base/'selection.json', dict(baseline_mean=a, candidate_mean=b,
            delta=b-a, adopt_candidate=adopt, best='advanced5_progress' if adopt else 'advanced5_narrow',
            package_changed=False, independent_blue_repeat=False))
        lines += [f'完整均分{a:.4f}→{b:.4f}（{b-a:+.4f}）。' +
                  ('达到本批完整均分采用条件，仍需结合损伤和专项审计说明局限。' if adopt else '未达到采用条件，保留窄路基线。'),
                  '原比赛评分器重算与红蓝布局关联已核验；本地小样本不证明统计显著性或平台成绩。']
    lines += ['', '## 逐轮结果', '',
              '|版本|地图/轮|地图|墙体|时间|难度|蓝队原分|完整分/小批前三项|疑似循环步|平均生命|结束步|',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for selector, rows in records.items():
        for r in rows:
            score = r['score'] if partial else next(x for x in scores[selector]
                if (x['map'],x['round']) == (r['map'],r['round']))
            values = [f'{score[k]:.4f}' for k in ('score_map','score_wall','score_time')]
            values += ['待评', '待评'] if partial else [f'{score[k]:.4f}' for k in ('score_bomb','blue_score')]
            lines.append(f'|{LABELS[selector]}|{r["map"]}/{r["round"]}|'+'|'.join(values)+
                         f'|{score["round_score"]:.4f}|{r["combined_frames"]}|{r["mean_health"]:.2f}|{r["elapsed_steps"]}|')
    if figures:
        lines += ['', '## 探索与布弹衔接', '',
                  '确认步来自单机日志。重叠率为多枚炸弹重复计入的加权墙体像素占各枚覆盖总和的比例。', '',
                  '|版本|地图/轮|首次确认步|末次确认步|确认次数|覆盖重叠率|',
                  '|---|---|---:|---:|---:|---:|']
        for item in ev.read(base/'report/wall_overlap.json'):
            selector = item['controller']
            details = next(r for r in audits[selector]
                           if (r['map'], r['round']) == (item['map'], item['round']))
            lines.append(f'|{LABELS[selector]}|{item["map"]}/{item["round"]}|'
                         f'{details["first_confirmation"]}|{details["last_confirmation"]}|'
                         f'{details["confirmations"]}|{item["overlap_percent"]:.2f}%|')
    lines += ['', '## 每图每轮图件与独立演示命令', '']
    for name in plan['maps']:
        for number in range(1,plan['rounds']+1):
            stem=f'{name}_r{number:02}'
            lines += [f'### {name} 第{number}轮', '']
            for selector in plan['controllers']:
                run=base/selector/'red'/stem
                lines += [f'**{LABELS[selector]}**', '']
                for label, pattern in [('CPU提交','*_exploration_diff.png'),('墙体覆盖','*_wall_destruction.png')]:
                    files=list((run/'artifacts').glob(pattern))
                    if files: lines += [f'![{label}]({files[0].as_posix()})', '']
                for suffix in ('bomb_layout','motion'):
                    image=base/'report'/selector/(stem+'_'+suffix+'.png')
                    if image.exists(): lines += [f'![{suffix}]({image.as_posix()})', '']
                for phase in ('red','blue'):
                    if not (base/selector/phase/stem).is_dir():
                        continue
                    recording=next((base/selector/phase/stem/'artifacts/recordings').glob('*.npz'))
                    lines += [f'{phase}回放：', '', '```powershell',
                        f'.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.evaluate_red --ray-backend cpu --replay "{recording.as_posix()}"', '```', '']
    lines += ['## 每图实时演示', '']
    for name in plan['maps']:
        lines += ['```powershell', f'.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.evaluate_retained --live --controllers advanced5_progress --maps {name}', '```', '']
    Path(document).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('REPORT', document, means, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('base'); parser.add_argument('document')
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--partial', action='store_true')
    args=parser.parse_args()
    report(args.base,args.document,args.figures,args.partial)
