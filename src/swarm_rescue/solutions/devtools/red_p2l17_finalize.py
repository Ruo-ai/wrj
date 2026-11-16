"""One-time offline delivery audit for the frozen 2026-11-16 RL17 experiment."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from swarm_rescue.solutions.devtools.red_p2l17_replay import write
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base


def main():
    root=Path.cwd();batch=root/'output/sf13/rl17_m89_once_20261116'
    out=batch.with_name(batch.name+'_figures');dev=root/'output/sf13/rl17_dev_20261116'
    doc=base.read(out/'summary.json');checks=[];regions=[];radio=[]
    for m in ('Map08','Map09'):
        runs={c:batch/f'runs/{m}_r01/{c}/red/attempt_001' for c in ('rl11_lines','rl17_repair')}
        with np.load(runs['rl11_lines']/'mapper_legal_inputs.npz') as a,np.load(runs['rl17_repair']/'mapper_legal_inputs.npz') as b:
            equality={k:bool(np.array_equal(a[k],b[k],equal_nan=True)) for k in a.files}
            assert set(a.files)==set(b.files) and all(equality.values())
        r=base.read(runs['rl17_repair']/'result.json');h,w=r['expected_shape']
        source=base.read(next((runs['rl17_repair']/'artifacts').glob('*_exploration.json')))
        pixels=np.asarray(source['grid'],np.uint8)
        replay=root/f'output/sf13/rl17_repair_replay{m[-2:]}_20261116'
        exact=bool(np.array_equal(pixels,np.load(replay/'rl17_repair_pixels.npy')))
        assert exact
        checks.append(dict(map=m,all_input_arrays=equality,full_candidate_pixels_equal_offline=exact))
        for c,run in runs.items():
            p=base.read(run/'p2l17_diagnostics.json');published={}
            for e in p['radio']:
                if e['event']=='local_publish':published[e['source'],e['version']]=e['sha256']
            receipts=[e for e in p['radio'] if e['event']=='received']
            bad=[e for e in receipts if published.get((e['source'],e['version']))!=e['sha256'] or e['version']>e['step']]
            assert not bad
            radio.append(dict(map=m,controller=c,received=len(receipts),source_hash_mismatches=bad))
            result=base.read(run/'result.json');pred=np.asarray(base.read(next((run/'artifacts').glob('*_exploration.json')))['grid'],bool)
            png=np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
            truth=np.all(png[base.COMPARISON_HEADER_HEIGHT:base.COMPARISON_HEADER_HEIGHT+h,:w]==40,axis=2).astype(np.uint8)
            scorer=base.MapExplorationScorer();surface,_=scorer.split_regions(truth)
            charged=pred&(scorer.distance_field(truth)>scorer.config.dead_zone)
            spec=base.read(base.read(run/'request.json')['map_file'])
            for z in spec['elements']:
                if z['type']!='no_gps_zone':continue
                l,t=map(int,z['top_left']);r,b=map(int,z['bottom_right']);sl=np.s_[max(0,t):min(h,b+1),max(0,l):min(w,r+1)]
                regions.append(dict(map=m,controller=c,bounds=[l,t,r,b],surface=int(surface[sl].sum()),
                    hits=int((surface.astype(bool)&pred)[sl].sum()),charged=int(charged[sl].sum())))
    pictures=list(out.glob('*.png'))
    for p in pictures:
        with Image.open(p) as img:img.verify()
    missing=[]
    for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',(out/'report.md').read_text(encoding='utf-8')):
        path=Path(target)
        if not path.is_absolute():path=out/path
        if not path.is_file():missing.append(str(path))
    assert not missing and len(pictures)==33 and len(doc['rows'])==4 and not doc['failures']
    protected=base.read(dev/'protected_sha256.json')
    changed=[p for p,digest in protected.items() if not Path(p).is_file() or base.digest(p)!=digest]
    assert not changed
    audit=dict(checks=checks,regions=regions,radio=radio,pngs_verified=len(pictures),missing_images=missing,
        protected_files=len(protected),protected_changed=changed,complete_red_phases=4,blue_phases=0)
    write(dev/'final_audit.json',audit)

    report=root/'复赛阶段_FS4_红队RL17_无GPS点到线补充与同输入核验_20261116.md'
    content=report.read_text(encoding='utf-8')
    content=content.replace('> 完整四局正在核验；最终结论与图册在本文件后半部分补充。以下行为测试与离线数字已实际完成，不是计划值。',
        '**本轮完成：25项行为测试，4次双地图器同输入离线比较，6组128步短测（首批诊断失败保留），4/4有效完整红阶段、0蓝。RL17-R两图地图分+0.6034/+1.0544；完整输入、轨迹、弹位与实际来源版本均相同。保留RL17-R继续验证，RL11不替换；RL17-A拒绝采用。缺测位姿均值未改善，探索/布弹/蓝队难度与完整总分提升均未证明。**')
    tail=['','## 10. 四局完整红阶段：最终结果','',
        f'批次：`{batch.relative_to(root).as_posix()}`，4场各一次、无重试。种子Map08=200806601，Map09=2477227837。',
        f'[完整逐局图册]({(out/"report.md").as_posix()}) · [机器可读明细]({(out/"summary.json").as_posix()}) · [交付核验]({(dev/"final_audit.json").as_posix()})',
        '', '|图/版本|红有效|M|W|T|弹|步|墙钟s|健康|墙/机接触|真墙命中|受罚误画|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in doc['rows']:
        tail.append(f"|{r['map']}/{r['controller']}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['red_steps']}|{r['red_walltime']:.3f}|{r['health']:.2f}|{r['contacts_wall']}/{r['contacts_drone']}|{r['wall_hits']}|{r['charged_pixels']}|")
    tail+=['','全部原始回调错误0、controller_crashed=False、walltime_limited=False，诊断errors=[]；收据和源码冻结哈希通过，官方评分复算一致。只因退出码0不会判有效。蓝队难度D、完整比赛原始总分与有效总分均未评测。',
        '完整两图逐数组（含固定射线角和语义）比较一致，录像完整结构数组与实际弹位一致，实际收到版本一致；候选提交像素与前述离线R结果精确一致。墙钟差+0.294/+1.360秒没有在这四局改变观测或触发提前截止。这是本轮因果隔离证据，不代表更多图/负载下始终相同。',
        '', '|图/版本|缺测机步|位置均值/P95 px|角均值/P95 °|地图update累计s|新增匹配s|单次update最大ms|Job峰值MB|',
        '|---|---:|---|---|---:|---:|---:|---:|']
    for r in doc['rows']:
        p,a=r['full_outage_position'],r['full_outage_heading_degrees']
        tail.append(f"|{r['map']}/{r['controller']}|{p['count']}|{p['mean']:.3f}/{p['p95']:.3f}|{a['mean']:.3f}/{a['p95']:.3f}|{r['mapper_seconds']:.4f}|{r['incremental_p2l_seconds']:.5f}|{r['mapper_max_seconds']*1000:.2f}|{r['resources']['peak_job_commit_bytes']/1e6:.2f}|")
    tail+=['','此表误差为新诊断每一步、控制后物理推进前对齐，区别于离线表的12步采样。增量匹配没有触发时间预算退回；Map09-R的45.75ms峰值是整次地图update（含继承的缓存恢复等），不是仅新增点线求解耗时，不能混作4ms门槛被忽略。单轮峰值/墙钟不证明最坏情况保证。',
        '', '|图/无GPS区像素框/版本|真实墙面|命中|受罚误画|', '|---|---:|---:|---:|']
    for r in regions:tail.append(f"|{r['map']}/{r['bounds']}/{r['controller']}|{r['surface']}|{r['hits']}|{r['charged']}|")
    tail+=['','灰区指标是离线分区审计，不是额外官方分。Map08全图信用减少0.01552分、扣分减少0.61888分，净+0.60336；Map09信用增加0.98738分、扣分减少0.06706分，净+1.05444。仍有明显重影/假墙，不能宣称特殊区已画准。',
        '', '## 11. 接收链与已证实/推测/待验证','',
        '**已证实：** R只改独立提交地图估计，固定输入和收到版本下有小幅地图收益；Map08灰区命中合计23658→23634，仍有局部退步；Map09灰区22048→23101。每局墙体分、时间分、实际落弹、碰撞接触和平均健康均不变。',
        '**已证实的送达缺口：** Map08提交者0号仍收到来源3的3900、来源4的3930版本，分别落后635/605步；Map09最旧版本落后70步。R没有改善送达。按接收日志逐笔核对：每个收到版本的载荷SHA都能匹配该来源本机发布SHA，无未来版本。广播次数/重复接收并非独立准确性证据。',
        '**推测而非根因定论：** 少量新接受改变后续参考、缓存姿态与恢复投影，可能改善提交重叠分布，同时恶化部分位姿均值。地图分与瞬时位姿均值方向相反，不能把收益简单归因“漂移减少”；本轮没有把融合、重投和后处理做完整因果消融。',
        '**待验证：** 更多独立种子下地图收益与真墙召回；单墙/重复结构错误接受对恢复重投的影响；原参考合并可能跨越短开口；语义量程外动态回波；来源共同偏差与迟到快照。这些不通过直接恢复RL13-L/RL16-B整包来解决。',
        '**采用判断：** RL11继续作为开发对照；RL17-A拒绝，RL17-R保留实验候选，不切正式入口。下一步先复核重复性和接受事件，不新增导航、探索或布弹机制。用户运行后续命令或发送“继续”前，不再启动评测。',
        '', '## 12. 探索、提交、轨迹、墙体、炸弹对比','',
        '以下均为本轮完整四局新图，上Map08、下Map09，左RL11、右RL17-R。轨迹/弹位相同是实际数组核验结果，不是复用旧图冒充新局。完整图册另含每局真值、本机终图、接收版本、提交图与漏画/误画。']
    for k,title in [('exploration','雷达观察范围（采样近似）'),('mapping','实际提交建图与误差'),('trajectories','完整实际轨迹'),('walls','官方墙体覆盖'),('bombs','实际炸弹位置')]:
        tail+=['',f'![{title}]({(out/f"comparison_{k}.png").as_posix()})']
    tail+=['','## 13. 全部回放、保护与最终交接','',
        '每条命令从项目根目录运行；无蓝阶段，因此没有本轮蓝回放。']
    for r in doc['rows']:
        tail+=['',f"{r['map']} / {r['controller']} / r{r['round']} / seed {r['red_seed']}",'','```powershell',
            "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{r['attempt']}'",'```']
    tail+=['',f'最终核验：{len(protected)}个既有跟踪文件SHA无变化；正式入口、RL11、旧候选、蓝队、规则、地图、原报告和根目录提交包均保持原样。33张完整局新PNG可解码，图册所有图片链接存在。新增离线收尾工具 `src/swarm_rescue/solutions/devtools/red_p2l17_finalize.py`，不参与在线控制。',
        'RL11 SHA256 `56474b5b14af1ce653060f8a082b4e24fdf44a5ded6a4dde7aa5e0415083bbf6`；RL17点线组件 `d4e7c107de6277cb8bc0b1c2bf7d59118fea6420091441597900fd9792ba4c70`；RL17-R `1785348ed19b2e9d68cbdbf59384746717cd2d38cdd3d5a57b35dfc31a730573`。',
        '先前两批短测原始全组图册（均为短测，不算成绩）：',
        f'[首批诊断封装失败，原样保留]({(root/"output/sf13/rl17_smoke_20261116/report.md").as_posix()})；[修复后的3组短测]({(root/"output/sf13/rl17_smoke_fixed_20261116/report.md").as_posix()})。',
        '本轮没有调用AI模型/API控制比赛，没有按图名或真值写在线规则；源代码冻结ZIP只是本地评测留证，不是新的team742提交包。没有提交、推送、打包或修改正式入口。',
        '', '用户后续最短命令仍为 `.\\red_rl17_evaluate.cmd`：两版本×Map08/09×1轮=4红0蓝，新时间戳目录与完整图册。此后等待用户结果或“继续”。']
    report.write_text(content+'\n'.join(tail)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
