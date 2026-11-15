"""B8 offline report supplement; existing lifecycle provides reliable bomb IDs."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .blue_relay_b7_report import summarize as discovery_summary, read


def summarize(attempt):
    row=discovery_summary(attempt)
    t=read(attempt/'telemetry.json') if (attempt/'telemetry.json').exists() else {}
    probes=t.get('lifecycle',{}).get('b8_navigation',{}).get('drones',{})
    stats=Counter();phases=Counter();reasons=Counter();deaths=0;transport=[]
    for d in probes.values():
        stats.update(d.get('stats',{}));phases.update(d.get('phase_steps',{}));deaths+=int(d.get('health',1)<=0)
        for sample in d.get('samples',[]):
            reasons[sample.get('navigation',{}).get('reason','unknown')]+=1
        transport += [s for s in d.get('samples',[]) if s.get('carrying')]
    row['navigation']=dict(stats=dict(stats),phase_steps=dict(phases),sampled_reasons=dict(reasons),
        deaths=deaths if probes else None,probe_seconds=sum(d.get('probe_seconds',0) for d in probes.values()),
        control_seconds=sum(d.get('control_seconds') or 0 for d in probes.values()) if probes else None,
        transport_samples=len(transport) if probes else None,
        limitation='sample counts are not exact stage durations; B0 control timing absent; no strict determinism guarantee')
    return row


def write_report(rows,output):
    output=Path(output)
    (output/'navigation_analysis.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# B8 空载与携弹导航对照','',
        '固定历史强布局；红地图/墙/时间/难度/总分和红种子不适用。正式入口未切换。',
        '只把完整有效局计为有效蓝分；失败、超时、零处置均保留。体感变快和原生组件通过不等于整局胜出。',
        '原始逐弹/控制/碰撞/资源见各attempt；碰撞回调不是独立接触数，载荷消失不是处置。','',
        '|地图/版本/蓝种子|有效|原始/有效蓝分|发现/不同抓取/处置|第1/3/5/8/10发现步|健康/蓝时间分|步数/秒|',
        '|---|---|---|---|---|---|---|']
    if not any(str(row['result'].get('blue_variant','')).startswith('b8_') for row in rows):
        lines.insert(2,'**历史完整局重新分析；本报告没有B8完整对局或新候选得分。**')
    for row in rows:
        r=row['result'];life=row.get('bomb_stages',row.get('stages',[]))
        milestones=' / '.join(str(row['discovery_milestones'][str(k)]) if row['discovery_milestones'][str(k)] is not None else ('未知' if row['discovered'] is None else '未发现') for k in (1,3,5,8,10))
        lines.append(f"|{r.get('map_name',r.get('map'))}/{r.get('blue_variant')}/{r.get('seed')}|{row['valid']}|{r.get('blue_score')}/{r.get('blue_score') if row['valid'] else '无'}|{row.get('discovered')}/{row.get('picked_unique')}/{r.get('disposed_count')}|{milestones}|{r.get('blue_health_score')}/{r.get('blue_time_score')}|{r.get('elapsed_steps')}/{r.get('elapsed_walltime')}|")
    for row in rows:
        a=Path(row['attempt']);r=row['result'];t=read(a/'telemetry.json') if (a/'telemetry.json').exists() else {}
        life=t.get('lifecycle',{});transport=t.get('transport',{})
        lines+=['',f"## {a}",'',f"导航：{row['navigation']}",f"运输事件及成功运输步数：{transport}",
            f"漏搜：{row.get('undiscovered')}；未发现按本局截止计入的限制平均发现步：{row.get('restricted_mean_discovery_step')}。",
            f"资源：{row.get('resources')}；有效性与完整源结果：navigation_analysis.json。",'',
            '|弹|发现|首次抓取|官方处置|抓取/脱手/释放次数|发现到抓取/最后抓取到处置步|',
            '|---|---|---|---|---|---|']
        for b in life.get('bombs',[]):
            picks=b.get('pickups',[]);first=picks[0]['step'] if picks else None
            found=b.get('first_seen');disposed=b.get('disposed')
            search_delay=first-found if first is not None and found is not None else None
            duration=disposed-picks[-1]['step'] if disposed is not None and picks else None
            lines.append(f"|{b['bomb']}|{found}|{first}|{disposed}|{len(picks)}/{len(b.get('losses',[]))}/{len(b.get('releases',[]))}|{search_delay}/{duration}|")
        lines+=['','三类完整图册见同目录report.md。红回放不适用；蓝回放：','```powershell',
            "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{a}'",'```','']
    (output/'navigation_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    attempts=sorted(path.parent for path in a.source.rglob('attempt_*/result.json'))
    write_report([summarize(attempt) for attempt in attempts],a.output)


if __name__=='__main__':main()
