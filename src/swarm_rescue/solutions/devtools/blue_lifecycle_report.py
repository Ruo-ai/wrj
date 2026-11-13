"""Read-only per-bomb diagnosis and complete atlas from frozen blue attempts."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def summarize(attempt):
    r=read(attempt/'result.json');t=read(attempt/'telemetry.json');l=t.get('lifecycle',{})
    valid=r.get('status')=='ok' and not any(r.get(k) for k in ('walltime_limited','controller_crashed','engine_callback_errors','smoke'))
    # Removed drones retain their last controller frame in early diagnostics.
    # Count each actual control step once; never call stale frames new actions.
    frames=list({(f['drone'],f['step']):f for f in l.get('frames',[])}.values());dr=[]
    for ident in sorted({f['drone'] for f in frames}):
        f=[x for x in frames if x['drone']==ident];carry=[x for x in f if x['carrying']]
        tail=carry[-20:];points=np.array([x['truth'] for x in tail])
        dr.append(dict(drone=ident,carrying_samples=len(carry),last=f[-1],
            tail_carry_span=float(np.linalg.norm(np.ptp(points,axis=0))) if len(points) else None,
            carry_source_counts=dict(Counter(x['source'] for x in carry)),
            stopped_carry_samples=sum(abs(x['command']['forward'])+abs(x['command']['lateral'])<.01 for x in carry)))
    bombs=[]
    for b in l.get('bombs',[]):
        pickup=b['pickups'][0]['step'] if b['pickups'] else None
        if b['disposed'] is not None:reason='official_disposed'
        elif b['first_seen'] is None:reason='not_observed'
        elif not b['pickups']:reason='no_pickup'
        elif b['losses'] and (not b['pickups'] or b['losses'][-1]['step']>b['pickups'][-1]['step']):reason='unconfirmed_loss'
        elif b['first_center_near'] is not None:reason='center_not_disposed'
        else:reason='return_not_completed'
        bombs.append(dict(b,reason=reason,discovery_to_pickup=None if pickup is None or b['first_seen'] is None else pickup-b['first_seen'],
            pickup_to_dispose=None if pickup is None or b['disposed'] is None else b['disposed']-pickup))
    times=Counter()
    for values in l.get('function_seconds',{}).values():times.update(values)
    return dict(attempt=str(attempt.resolve()),result=r,valid=valid,bombs=bombs,drones=dr,
        function_seconds=dict(times),observer_seconds=l.get('observer_seconds'),
        discovered=l.get('discovered'),phase_failures=dict(Counter(b['reason'] for b in bombs)),
        transport=t.get('transport'),resources=read(attempt/'resources.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('sources',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    attempts=sorted({r.parent for s in a.sources for r in s.rglob('attempt_*/result.json')})
    rows=[summarize(v) for v in attempts];a.output.mkdir(exist_ok=True,parents=True)
    (a.output/'analysis.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 固定强布局逐弹诊断与完整图册','',
        '公开语义命中通过离线渲染ID对齐弹编号；目标认领为60px内唯一离线匹配，歧义不计。处置来自官方事实。控制状态为稀疏采样，阶段步数为逐步累计的无人机步，不能当作互斥任务时长。没有在线处置成功API，抓取消失不等于处置。','',
        '|地图/版本/批次|蓝原始/有效分|处置|发现|步数/墙钟秒|有效|','|---|---|---:|---:|---|---|']
    for row in rows:
        r=row['result'];lines.append(f"|{r['map']} / {r['blue_variant']} / {Path(row['attempt']).parents[5].name}|{r['blue_score']:.2f}/{r['blue_score'] if row['valid'] else '无效'}|{r['disposed_count']}|{row['discovered']}|{r['elapsed_steps']}/{r['elapsed_walltime']:.2f}|{row['valid']}|")
    for row in rows:
        r=row['result'];attempt=Path(row['attempt'])
        lines+=['',f"## {r['blue_variant']} — {attempt}",'',
            f"蓝种子{r['seed']}；实际区域{r['actual_conditions']}；布局SHA256 {r['layout_sha256']}；源码SHA256 {r['blue_source_sha256']}。",
            f"健康分{r['blue_health_score']}；蓝时间分{r['blue_time_score']}。原始蓝分{r['blue_score']}；有效蓝分{r['blue_score'] if row['valid'] else '空'}。红M/W/T/D/原始及有效总分均不适用，未运行红队。",
            f"触限={r.get('walltime_limited')}；控制器异常={r.get('controller_crashed')}；回调异常次数={r.get('engine_callback_errors')}。成功运输事件统计：{ {k:v for k,v in (row['transport'] or {}).items() if k!='events'} }。",
            f"资源{row['resources']}；观察器自身累计{row['observer_seconds']}秒（包括在墙钟中，不扣除）。分函数累计秒{row['function_seconds']}（control包含子函数，不可相加）。",
            '', '下表首抓到处置包含中途脱手/重抓，不等于一次成功运输耗时；后者见上方逐事件统计。phase_steps仅是无人机累计步，首批死亡机陈旧帧问题已在后续诊断修复，旧值不用于精确耗时。',
            '', '|弹号|发现/认领/近弹步|抓取（机@步）|到中心附近|释放次数/脱手次数|官方处置步|发现→首抓/首抓→处置|结束状态|',
            '|---:|---|---|---|---|---|---|---|']
        for b in row['bombs']:
            grabs=','.join(f"{e['drone']}@{e['step']}" for e in b['pickups']) or '无'
            lines.append(f"|{b['bomb']}|{b['first_seen']}/{b['first_claim']}/{b['first_near']}|{grabs}|{b['first_center_near']}|{len(b['releases'])}/{len(b['losses'])}|{b['disposed']}|{b['discovery_to_pickup']}/{b['pickup_to_dispose']}|{b['reason']}|")
        folder=attempt.parents[4]
        report=folder/'report.md'
        if report.exists():lines+=['',report.read_text(encoding='utf-8')]
    (a.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([dict(variant=r['result']['blue_variant'],score=r['result']['blue_score'],valid=r['valid'],
        disposed=r['result']['disposed_count'],discovered=r['discovered'],failures=r['phase_failures'],seconds=r['result']['elapsed_walltime']) for r in rows],ensure_ascii=False))


if __name__=='__main__':main()
