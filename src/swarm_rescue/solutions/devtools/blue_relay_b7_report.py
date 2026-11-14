"""Read-only discovery/relay audit. Offline bomb IDs never enter the policy."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(attempt):
    r=read(attempt/'result.json');t=read(attempt/'telemetry.json') if (attempt/'telemetry.json').exists() else {}
    life=t.get('lifecycle',{});bombs=life.get('bombs',[]);horizon=r.get('elapsed_steps',0)
    recorded=bool(bombs)
    valid=(r.get('status')=='ok' and r.get('exit_code')==0 and r.get('blue_score') is not None
           and not any(r.get(k) for k in ('walltime_limited','controller_crashed','engine_callback_errors','smoke')))
    discovered=sorted(b['first_seen'] for b in bombs if b['first_seen'] is not None)
    events=[];stats=Counter();opportunities=[];samples=repeat=deaths=0
    for ident,drone in life.get('b7_relay',{}).get('drones',{}).items():
        stats.update(drone.get('stats',{}));deaths+=int(drone.get('health',1)<=0)
        for event in drone.get('events',[]):
            # Unique spatial match to initial layout only; carry/loss identity
            # itself continues to come from the existing official observer.
            point=event.get('point');matches=[]
            if point is not None:
                matches=[b['bomb'] for b in bombs if np.linalg.norm(np.array(b['initial'])-point)<60]
            events.append(dict(event,drone=ident,bomb=matches[0] if len(matches)==1 else None,
                association='unique_initial_within_60px' if len(matches)==1 else 'ambiguous_or_missing'))
        cov=drone.get('coverage')
        if cov:
            shape=cov['shape'];opportunities.append(np.unpackbits(np.frombuffer(base64.b64decode(cov['packed']),np.uint8))[:np.prod(shape)].reshape(shape).astype(bool))
            samples+=cov['observed_cell_samples'];repeat+=cov['repeated_cell_samples']
    stages=[]
    for b in bombs:
        associated=[e for e in events if e['bomb']==b['bomb']]
        def first(kind):
            values=[e['step'] for e in associated if e['kind']==kind];return min(values) if values else None
        reason=('disposed' if b['disposed'] is not None else 'undiscovered' if b['first_seen'] is None else
                'no_pickup' if not b['pickups'] else 'unconfirmed_loss' if b['losses'] and b['losses'][-1]['step']>b['pickups'][-1]['step'] else 'return_unfinished')
        stages.append(dict(b,offer_built=first('offer_packet_built'),offer_received=first('offer_received'),
            receiver_accepted=first('receiver_accepted'),takeover_confirmed=first('takeover_confirmed'),
            fallbacks=[e for e in associated if e['kind'] in ('fallback_self','passive_fallback_self')],reason=reason))
    coverage={}
    if opportunities:
        total=np.sum(opportunities,axis=0)
        coverage=dict(unique_opportunity_cells=int((total>0).sum()),multi_drone_overlap_cells=int((total>1).sum()),
            observed_cell_samples=samples,repeated_cell_samples=repeat,scope='estimated/absolute-trusted intervals only; not bomb detection percentage')
    return dict(attempt=str(attempt.resolve()),result=r,valid=valid,
        discovery_milestones={str(k):discovered[k-1] if len(discovered)>=k else None for k in (1,3,5,8,10)},
        discovered=len(discovered) if recorded else None,bombs_total=r.get('bomb_count'),undiscovered=len(bombs)-len(discovered) if recorded else None,
        restricted_mean_discovery_step=(sum(b['first_seen'] if b['first_seen'] is not None else horizon for b in bombs)/len(bombs)) if bombs else None,
        censor_horizon=horizon,censor_note='missing bombs charged at observed horizon; lower bound, not mean of discovered only',
        picked_unique=sum(bool(b['pickups']) for b in bombs) if recorded else None,official_disposed=life.get('official_disposed'),
        discovered_no_pickup=sum(b['first_seen'] is not None and not b['pickups'] for b in bombs) if recorded else None,
        discovered_not_disposed=sum(b['first_seen'] is not None and b['disposed'] is None for b in bombs) if recorded else None,
        discovered_no_relay_ack=sum(b['first_seen'] is not None and b['takeover_confirmed'] is None for b in stages) if 'b7_relay' in life else None,
        relay_supported='b7_relay' in life,relay_stats=dict(stats),events=events,bombs=stages,
        transport=t.get('transport'),coverage=coverage,deaths=deaths if 'b7_relay' in life else None,
        collision_callbacks={i:d.get('collision_callbacks') for i,d in life.get('b6_diagnostics',{}).get('drones',{}).items()},
        resources=read(attempt/'resources.json') if (attempt/'resources.json').exists() else None)


def write_report(rows,output,historical=False):
    lines=['# B7 发现速度与接力专项诊断','',
        '历史资料重新统计；没有运行B7完整对局。' if historical else '本批实际对局；无效局保留原始分，有效分为空。',
        '接力消息事件按初始弹位60px内唯一候选离线关联；歧义留空，不冒充可靠身份。发现/抓取/处置沿用既有官方逐弹观察器。',
        'packet_built只是构造消息；offer_received证明到达；takeover_confirmed证明发现机收到承诺。持有消失不是处置。',
        '未发现弹按本局观察截止步计入限制平均发现步数，同时单列漏搜；此均值是截尾下界，不与已发现弹平均混用。','',
        '|地图/版本/种子|有效|蓝原始/有效分|发现/唯一抓取/处置|t1/t3/t5/t8/t10|漏搜|限制平均发现步|步数/秒|',
        '|---|---|---|---|---|---|---|---|']
    for row in rows:
        r=row['result'];times='/'.join(str(v) if v is not None else '未发现' for v in row['discovery_milestones'].values())
        lines.append(f"|{r.get('map')}/{r.get('blue_variant')}/{r.get('seed')}|{row['valid']}|{r.get('blue_score')}/{r.get('blue_score') if row['valid'] else '无效'}|"
            f"{row['discovered']}/{row['picked_unique']}/{row['official_disposed']}|{times}|{row['undiscovered']}|{row['restricted_mean_discovery_step']}|{r.get('elapsed_steps')}/{r.get('elapsed_walltime')}|")
        lines += ['',f"逐弹与事件：{row['attempt']}",
            f"未抓取/未处置（已发现）：{row['discovered_no_pickup']}/{row['discovered_not_disposed']}；未收到接力确认：{row['discovered_no_relay_ack']}（没有接力事件日志时为未知，不能解释为任务失败）。",
            f"健康/时间分：{r.get('blue_health_score')}/{r.get('blue_time_score')}；死亡：{row['deaths']}；运输事件：{row['transport']}。",
            f"等待/切换/失败计数：{row['relay_stats']}；覆盖代理：{row['coverage']}；碰撞回调（非独立接触数）：{row['collision_callbacks']}。",
            f"资源：{row['resources']}；红地图/墙/时间/难度/总分及红种子不适用，固定布局没有本轮红执行。",'',
            '|弹|发现|消息构造/送达|接收机认领/发现机确认|首抓|官方处置|状态|',
            '|---|---|---|---|---|---|---|']
        for b in row['bombs']:
            lines.append(f"|{b['bomb']}|{b['first_seen']}|{b['offer_built']}/{b['offer_received']}|{b['receiver_accepted']}/{b['takeover_confirmed']}|{b['pickups'][0]['step'] if b['pickups'] else None}|{b['disposed']}|{b['reason']}|")
        lines += ['','```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{row['attempt']}'",'```','']
    (output/'relay_analysis.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'relay_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    attempts=sorted({p.parent for p in a.source.rglob('attempt_*/result.json')})
    rows=[summarize(path) for path in attempts];write_report(rows,a.output,a.historical)
    receipt={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for path in attempts for p in (path/'result.json',path/'telemetry.json') if p.exists()}
    (a.output/'inputs.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('attempt','valid','discovery_milestones','discovered','picked_unique','official_disposed','restricted_mean_discovery_step')} for r in rows]))


if __name__=='__main__':main()
