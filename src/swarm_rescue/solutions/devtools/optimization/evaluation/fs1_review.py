"""Offline FS1 screening report: preserve all failures and separate red/blue validity."""
import argparse
import json
from pathlib import Path
import statistics

from .runner import read, write, receipt, file_hash, phase_valid


def comparison(rows, reference, candidate):
    base={(r['map'],r['round'],r['red_seed'],r['blue_seed']):r for r in rows if r['controller']==reference}
    output=[]
    for r in rows:
        if r['controller']!=candidate:
            continue
        key=r['map'],r['round'],r['red_seed'],r['blue_seed']
        b=base.get(key)
        if b is None:
            continue
        red_valid=r['red_valid'] and b['red_valid']
        full_valid=r['full_valid'] and b['full_valid']
        output.append(dict(map=r['map'],round=r['round'],reference=reference,candidate=candidate,
            red_pair_valid=red_valid,full_pair_valid=full_valid,
            delta_map=r['M']-b['M'] if red_valid else None,
            delta_wall=r['W']-b['W'] if red_valid else None,
            delta_red_objective=(.5*(r['M']-b['M'])+.2*(r['W']-b['W'])+.1*(r['T']-b['T'])) if red_valid else None,
            delta_full=r['total']-b['total'] if full_valid else None,
            placed=r['placed'],reference_placed=b['placed']))
    return output


def events(log):
    output=[]
    for line in log.splitlines():
        if not line.startswith('V3_EVENT '):
            continue
        row=json.loads(line[len('V3_EVENT '):])
        if row['event'].startswith('fs1_') or 'fs1' in row:
            output.append(row)
    return output


def review(batch):
    plan=read(batch/'plan.json');plan_sha=file_hash(batch/'plan.json')
    rows=[]
    for row in read(batch/'results.json'):
        rpath=Path(row['red_run']);bpath=Path(row['blue_run']) if row['blue_run'] else None
        for path in (rpath,bpath):
            if path and read(path/'integrity.json')!=receipt(path,plan_sha):
                raise ValueError('Changed attempt: '+str(path))
        r=read(rpath/'result.json');b=read(bpath/'result.json') if bpath else {}
        rq=read(rpath/'request.json');bq=read(bpath/'request.json') if bpath else {}
        rs=r.get('phase1') or {};bs=b.get('phase1') or {}
        rows.append(dict(map=row['map'],round=row['round'],controller=row['controller'],
            red_seed=rq['seed'],blue_seed=bq.get('seed'),red_valid=bool(phase_valid(r,'red') and not r.get('smoke')),
            full_valid=row['valid_score'],M=rs.get('score_map'),W=rs.get('score_wall'),T=rs.get('score_time'),
            raw_total=bs.get('round_score'),total=row['total_score'],placed=r.get('placed_count'),
            failure=row['failure_reason'],red_run=str(rpath),blue_run=str(bpath) if bpath else None,
            deployment_events=events((rpath/'run.log').read_text(encoding='utf-8',errors='replace'))))
    contrasts=[]
    for reference,candidate in [('advanced2','advanced2_fs1_area'),('advanced2','advanced2_fs1_gain'),
                                ('advanced2_fs1_area','advanced2_fs1_gain'),('advanced2','advanced5_narrow')]:
        contrasts.extend(comparison(rows,reference,candidate))
    complete=read(batch/'progress.json').get('complete',False)
    write(batch/'fs1_review.json',dict(complete=complete,adopted=False,rows=rows,contrasts=contrasts,
        warning='One-seed development screening only. No adoption; all invalid rows retained.'))
    lines=['# FS1开发筛查（不作正式采用）','',
           f'已记录 {len(rows)}/{len(plan["schedule"])} 组；全部完成标记={complete}。',
           '红专项仅取双方红阶段有效交集；完整分另取双方红蓝均有效交集。无效行全部保留。',
           '在线面积和5点扰动为未校准代理；本报告不将其当作官方真实墙收益。', '',
           '|地图/轮次|红版本|红有效|完整有效|M|W|T|原始总分|有效总分|放弹|失败原因|',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('|'+ '|'.join(str(v) for v in [f"{r['map']}/r{r['round']}",r['controller'],r['red_valid'],
            r['full_valid'],r['M'],r['W'],r['T'],r['raw_total'],r['total'],r['placed'],r['failure']])+'|')
    for candidate in sorted({(x['reference'],x['candidate']) for x in contrasts}):
        selected=[x for x in contrasts if (x['reference'],x['candidate'])==candidate]
        red=[x for x in selected if x['red_pair_valid']];full=[x for x in selected if x['full_pair_valid']]
        mean=lambda items,key:statistics.mean(x[key] for x in items) if items else None
        lines+=['',f'## {candidate[1]} 对 {candidate[0]}','',
            f"红有效配对{len(red)}/{len(selected)}；ΔW={mean(red,'delta_wall')}；红加权分项Δ={mean(red,'delta_red_objective')}。",
            f"完整有效配对{len(full)}/{len(selected)}；Δ完整分={mean(full,'delta_full')}。",
            '逐图全部差值（包括退步、无效）：','', '```json',json.dumps(selected,ensure_ascii=False,indent=2),'```']
    lines+=['','[完整逐轮三图、种子、资源与红蓝回放](report.md)',
             '当前仅提供值得人工检查的候选证据；采用须遵守多种子、完成率与留出门槛。','']
    (batch/'fs1_review.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    review(parser.parse_args().batch.resolve())
