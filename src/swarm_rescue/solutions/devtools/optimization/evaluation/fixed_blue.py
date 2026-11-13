"""Run an explicitly selected blue against immutable R1 layouts, never rerun red."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import zipfile

from .runner import (REPO, read, write, file_hash, source_receipt, validate_sources,
                     execute_phase, receipt, phase_valid)
from .scenarios import official_rows, engine_profile
from .replay import validate_replay_sources
from .identities import RED_CONTROLLERS, BLUE_CONTROLLERS
from ..map_generator.append_maps import append_lock


def source_pairs(source, maps=None, controllers=None):
    plan = read(source/'plan.json')
    validate_replay_sources(plan['source_sha256'])
    if file_hash(source/'source_snapshot.zip') != plan['source_snapshot_sha256']:
        raise ValueError('Original source archive changed')
    pairs = []
    for row in read(source/'results.json'):
        if maps and row['map'] not in maps:
            continue
        if controllers and row['controller'] not in controllers:
            continue
        run = Path(row['red_run'])
        if read(run/'integrity.json') != receipt(run, file_hash(source/'plan.json')):
            raise ValueError('Original red artifacts changed')
        result = read(run/'result.json')
        if not phase_valid(result, 'red') or result['placed_count'] != 10:
            raise ValueError('Expected completed ten-bomb red layout')
        item = next(s for s in plan['schedule'] if s['name']==row['map']
                    and s['controller']==row['controller'] and s['round']==row['round'])
        # The actual red controller itself must remain the R1 controller.
        request=read(run/'request.json')
        red_file = 'src/'+request['controller_module'].replace('.','/')+'.py'
        validate_sources({red_file: plan['source_sha256'][red_file]})
        layout = next((run/'artifacts').glob('*_bombs.json'))
        pairs.append(dict(item=item, red_run=str(run), original_blue_run=row['blue_run'],
                          red_result_sha256=file_hash(run/'result.json'), layout_sha256=file_hash(layout)))
    if not pairs:
        raise ValueError('No selected layouts')
    return pairs


def summarize(output, plan, results):
    lines = ['# 固定弹图参考蓝队对照', '',
             '红队没有重跑。四项红分中地图/墙体/时间沿用原始红结果，仅难度按本轮蓝队重算；触限行不计有效完整分。', '',
             '|地图/红布局|蓝版本|原蓝处置|本次处置|蓝分|红地图分|红墙体分|红时间分|原始难度|原始红总分|秒|有效|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in results:
        r = row['result']; old = read(Path(row['original_blue_run'])/'result.json')
        p = r.get('phase1') or {}
        fmt = lambda v: '—' if v is None else f'{v:.4f}'
        lines.append('|'+ '|'.join([row['map']+'/'+row['controller'],r.get('blue_variant','?'),
            str(old.get('disposed_count')),str(r.get('disposed_count')),fmt(r.get('blue_score')),
            *[fmt(p.get(k)) for k in ('score_map','score_wall','score_time','score_bomb','round_score')],
            fmt(r.get('elapsed_walltime')),str(phase_valid(r,'blue'))])+'|')
    for row in results:
        run = Path(row['run']); red = Path(row['red_run'])
        lines += ['',f"## {row['map']} / {row['controller']} / 第{row['round']}轮",'',
            f"固定弹图SHA256：{row['layout_sha256']}。蓝种子：{row['seed']}。",'']
        from .atlas import round_section
        item=next(p['item'] for p in plan['fixed_layouts'] if p['red_run']==row['red_run'])
        pair=dict(item,red=read(red/'result.json'),blue=row['result'],red_run=str(red),blue_run=str(run))
        lines+=['红队图件复用固定来源，本批仅重跑蓝队。']+round_section(pair,output)
    write(output/'results.json', results)
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    pairs = source_pairs(source,args.maps,getattr(args,'controllers',None))
    sources = source_receipt(official_rows(sorted({p['item']['name'] for p in pairs})))
    config = dict(blue_variant=args.variant, memory_bytes=4_000_000_000, timeout=240,
                  smoke=False, engine_profile=engine_profile(), source=str(source),
                  source_plan_sha256=file_hash(source/'plan.json'), profile=args.profile)
    # Profiling is a separate diagnostic run, never included in timed acceptance.
    for pair in pairs:
        pair['item'] = dict(pair['item'], blue_variant=args.variant)
    candidate = dict(configuration=config, source_sha256=sources, fixed_layouts=pairs)
    if output.exists():
        plan = read(output/'plan.json')
        if any(plan[k] != v for k,v in candidate.items()):
            raise ValueError('Frozen fixed-layout batch changed; use a new output')
        if file_hash(output/'source_snapshot.zip') != plan['source_snapshot_sha256']:
            raise ValueError('Source snapshot changed')
    else:
        output.mkdir(parents=True)
        with zipfile.ZipFile(output/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(REPO/name,name)
        plan = dict(candidate,created_utc=datetime.now(timezone.utc).isoformat(),
                    source_snapshot_sha256=file_hash(output/'source_snapshot.zip'))
        write(output/'plan.json',plan)
    with append_lock(output):
        results=[]
        for pair in pairs:
            validate_sources(plan['source_sha256'])
            item = pair['item']
            print(f"[{len(results)+1}/{len(pairs)}] {item['name']} / {item['controller']} / {args.variant}",flush=True)
            write(output/'progress.json',dict(completed=len(results),expected=len(pairs),active=item))
            result, attempt = execute_phase(output,plan,item,'blue',Path(pair['red_run']))
            results.append(dict(map=item['name'],controller=item['controller'],round=item['round'],seed=item['seed'],
                                run=str(attempt),result=result,**{k:v for k,v in pair.items() if k!='item'}))
            summarize(output,plan,results)
            print(f"  disposed={result.get('disposed_count')}, score={result.get('blue_score')}, valid={phase_valid(result,'blue')}",flush=True)
        write(output/'progress.json',dict(completed=len(results),expected=len(pairs),active=None,complete=True))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=REPO/'output/semifinal_r1_official_pilot_20261111')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=BLUE_CONTROLLERS,required=True)
    parser.add_argument('--maps',nargs='+')
    parser.add_argument('--controllers',nargs='+',choices=RED_CONTROLLERS)
    parser.add_argument('--profile',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.profile:
        parser.error('Use a separate profiler; timed batches never enable profiling')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
