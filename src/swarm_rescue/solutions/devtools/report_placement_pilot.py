"""Report the paired fixed-layout placement pilot from its original artifacts."""
import argparse
import multiprocessing
from pathlib import Path

from swarm_rescue.solutions.devtools import report_advanced5 as old
from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev
from swarm_rescue.solutions.devtools.report_advanced5_progress import draw_one_map


def report(base, show_invalid=False):
    plan = ev.read(base / 'plan.json')
    old.LABELS = {s: s for s in plan['controllers']}
    records = old.audit(base)
    lines = ['## 同期红蓝试验逐轮明细', '']
    for name in plan['maps']:
        with multiprocessing.get_context('spawn').Pool(1) as pool:
            overlap = pool.apply(draw_one_map, (base, name, records))
        for item in overlap:
            selector = item['controller']
            stem = ev.round_name(item)
            folder = base / selector
            red = folder / 'red' / stem
            blue = folder / 'blue' / stem
            result = ev.read(blue / 'result.json')
            ev.validate_receipt(red, ev.digest(red.parent / 'manifest.json'))
            if (result['layout_sha256'] != ev.digest(ev.one_layout(red)) or
                    result['red_result_sha256'] != ev.digest(red / 'result.json')):
                raise ValueError(f'Invalid blue pair {blue}')
            valid = result['status'] == 'ok' and not result['controller_crashed'] and not result['walltime_limited']
            if not valid and not show_invalid:
                raise ValueError(f'Invalid blue result {blue}')
            score = result['phase1']
            lines += [f'### {selector} · {stem}', '',
                      '有效配对。' if valid else f'本轮蓝队无效（{result["status"]}）；难度、蓝队原分和完整分仅作诊断。', '',
                      '|地图分|墙体分|时间分|布局难度|蓝队原分|完整分|覆盖重叠率|',
                      '|---:|---:|---:|---:|---:|---:|---:|',
                      '|' + '|'.join(f'{v:.4f}' for v in (
                          score['score_map'], score['score_wall'], score['score_time'],
                          score['score_bomb'], result['blue_score'], score['round_score'])) +
                      f'|{item["overlap_percent"]:.2f}%|', '']
            for label, image in [
                ('CPU提交地图', next((red/'artifacts').glob('*_exploration_diff.png'))),
                ('墙体覆盖', next((red/'artifacts').glob('*_wall_destruction.png'))),
                ('炸弹位置', base/'report'/selector/(stem+'_bomb_layout.png')),
                ('运动轨迹', base/'report'/selector/(stem+'_motion.png')),
            ]:
                lines += [f'![{label}]({image.relative_to(ev.ROOT).as_posix()})', '']
            for label, run in [('红队', red), ('蓝队', blue)]:
                recording = next((run/'artifacts/recordings').glob('*.npz'))
                lines += [f'{label}回放', '', '```powershell',
                          f'& "{ev.ROOT}/.venv/Scripts/python.exe" -B -m swarm_rescue.solutions.devtools.evaluate_red --ray-backend cpu --replay "{recording}"',
                          '```', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('document', type=Path)
    parser.add_argument('--show-invalid', action='store_true')
    args = parser.parse_args()
    content = report(args.batch.resolve(), args.show_invalid)
    intro = args.document.read_text(encoding='utf-8').split('## 同期红蓝试验逐轮明细')[0]
    args.document.write_text(intro.rstrip() + '\n\n' + content, encoding='utf-8')
