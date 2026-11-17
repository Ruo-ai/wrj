"""Create an independent complete R20 report from verified existing outputs."""
import argparse
from pathlib import Path
import statistics

from PIL import Image

from swarm_rescue.solutions.devtools.red_pair19_batch import read, write, digest, REPO
from swarm_rescue.solutions.devtools.red_pair19_results_review import table, fmt, LABELS, KINDS

ARMS=('rl11_lines','rl17_repair','r20_budget')
LABELS['r20_budget']='R20-B'
METRICS={
 'map_score':('地图分M',1),'wall_hits':('真墙命中H',1),'wall_missed':('漏画L',-1),
 'charged_pixels':('受罚误画C',-1),'wall_score':('墙体分W',1),'time_score':('时间分T',1),
 'bombs':('炸弹数',1),'sampled_lidar_free_percent':('雷达采样覆盖%',1),
 'wall_seconds':('引擎耗时秒',-1),'process_seconds':('进程耗时秒',-1),
 'stalled_steps':('卡住触发计数步',-1),'yield_steps':('避让计数步',-1),
 'max_source_age':('最陈旧来源年龄步',-1),'mean_source_age':('平均来源年龄步',-1),
 'last_bomb_step':('末弹确认引擎步',-1),'wall_contacts':('墙接触次数',-1),
 'drone_contacts':('机间接触次数',-1),'mean_health':('平均健康',1),
 'sum_outage_steps':('五机累计绝对缺测步',-1),
 'lifetime_passage_episodes':('局部任务次数',0),'sum_local_active_ticks':('局部任务活动步',0),
 'steps':('阶段结束步',0),'official_trajectory_exploration':('引擎轨迹探索指标',1)}


def describe(values):
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values))


def sign(x):return f'{x:+.4f}' if isinstance(x,float) else f'{x:+d}'


def put(path,lines):
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')


def embed(path,label):return ['',f'![{label}]({path.as_posix()})','']


def key(row):return row['map'],row['round'],row['controller']


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();batch=a.batch.resolve();out=a.review.resolve();report=a.report.resolve()
    if report.exists():raise FileExistsError(report)
    summary=read(out/'atlas/summary.json');original=read(batch/'summary.json');audit=read(out/'detailed_audit.json')
    if summary!=original:raise ValueError('Independent summary differs from original batch summary')
    if not audit['valid'] or summary['valid_count']!=24 or read(out/'atlas/figure_errors.json'):raise ValueError('Incomplete audited atlas')
    plan=read(batch/'plan.json');detail={key(d):d for d in read(out/'run_details.json')};rows=[]
    for row in summary['rows']:
        d=detail[key(row)];row=dict(row)
        for k in METRICS:
            if k in d:row[k]=d[k]
        row['wall_contacts']=d['collision_contacts'].get('wall',0);row['drone_contacts']=d['collision_contacts'].get('drone',0)
        rows.append(row)
    groups=sorted({(r['map'],r['round'],r['seed']) for r in rows})
    contrasts=[]
    for name,number,seed in groups:
        arms={r['controller']:r for r in rows if r['map']==name and r['round']==number}
        for baseline,candidate in [('rl17_repair','r20_budget'),('rl11_lines','rl17_repair'),('rl11_lines','r20_budget')]:
            delta={k:arms[candidate][k]-arms[baseline][k] for k in METRICS}
            regressions={k:v for k,v in delta.items() if METRICS[k][1] and v*METRICS[k][1]<-1e-9}
            contrasts.append(dict(map=name,round=number,seed=seed,baseline=baseline,candidate=candidate,delta=delta,regressions=regressions))
    stats={c:{k:describe([r[k] for r in rows if r['controller']==c]) for k in METRICS} for c in ARMS}
    paired={f'{c}-{b}':{k:describe([x['delta'][k] for x in contrasts if x['baseline']==b and x['candidate']==c]) for k in METRICS}
            for b,c in [('rl17_repair','r20_budget'),('rl11_lines','rl17_repair'),('rl11_lines','r20_budget')]}
    write(out/'comparisons.json',dict(rows=rows,contrasts=contrasts,per_version=stats,paired=paired))
    sheets=out/'contact_sheets';sheets.mkdir(exist_ok=False)
    for name in sorted({r['map'] for r in rows}):
        for kind in KINDS:
            with Image.open(out/f'atlas/{name}_r1_{kind}.png') as im1, Image.open(out/f'atlas/{name}_r2_{kind}.png') as im2:
                joined=Image.new('RGB',(max(im1.width,im2.width),im1.height+im2.height),'white')
                joined.paste(im1,(0,0));joined.paste(im2,(0,im1.height));joined.save(sheets/f'{name}_{kind}_r1_r2.png')
    appendix=['# R20 逐机完整附表','','所有行来自本批实际记录；局部图、收到的来源图、最终提交图分别列示。',
              'H/L/C分别为真墙命中/漏画/受罚误画。来源年龄=提交控制步−实际接收版本。单机指标不可相加成团队得分。',
              '诊断导航误差为每12步采样；末弹时钟为引擎步；补充时钟为控制器步，二者相差暖机偏移。','']
    for name,number,seed in groups:
        appendix+=['',f'## {name} / r{number} / seed {seed}','']
        ds=[detail[(name,number,c)] for c in ARMS]
        appendix+=table(['版本/机','覆盖%','局部M/H/L/C','实际收到来源M/H/L/C','接收版本/年龄','末尾本地发布版本','未送达本地格/陈旧来源格'],[
            [f'{LABELS[d["controller"]]}/d{x["owner"]}',fmt(x['sampled_visible_free_percent']),
             '/'.join(fmt(x['local'][k]) for k in ('map_score','hits','missed','charged')),
             '/'.join(fmt(x['received'][k]) for k in ('map_score','hits','missed','charged')),
             f"{x['source_version']}/{x['source_age']}",x['final_local_published_version'],f"{x['missing_local_cells']}/{x['outdated_received_cells']}"] for d in ds for x in d['drones']])
        appendix+=['','未送达本地格=末尾本地图有、提交者收到的该来源图没有；陈旧来源格为反向差集，二者不等于错墙数量。','']
        appendix+=table(['版本/机','任务数/活动步','补充步','卡住/避让计数','确认放弹步','缺测步','导航误差P95/最大px','角色控制步','路径释放/恢复原因'],[
            [f'{LABELS[d["controller"]]}/d{x["owner"]}',f"{x['passage_episodes']}/{x['rn7'].get('active_ticks',0)}",x['renewals'],
             f"{x['stalled_steps']}/{x['yield_steps']}",x['placement_engine_steps'],x['outage_steps'],
             f"{x['navigation_error_p95']:.2f}/{x['navigation_error_max']:.2f}",x['roles'],f"{x['route_releases']} / {x['recovery_reasons']}"] for d in ds for x in d['drones']])
    put(out/'per_drone_appendix.md',appendix)
    lines=['# R20 三版本配对红阶段结果核验与探索归因','',
           '2026-11-17。仅分析已有批次，不启动新飞行、不修改正式入口或任何旧文件。',
           f'原始批次：`{batch}`。当前HEAD：`{read(out/"initial_audit.json")["head"]}`，开始时工作区干净。',
           '**结论：不建议采用当前R20-B补充预算方案。预算确实增加了局部任务，但没有稳定增加探索，且出现地图、布弹质量和墙接触退步。RL11保留稳定对照；RL17-R仍是候选，正式入口不切换。**',
           '4张训练地图×2种子×3版本，共24局完整红队阶段；不是完整红蓝对抗。蓝队难度分D、炸弹难度分及完整总分：未评测；原始partial round_score不作为总分报告。','',
           '## 核验、版本与实际约束','',
           f"24/24完整有效，24份done/integrity收据、{len(plan['source_sha256'])}项冻结文件、冻结源码字节及当前依赖环境一致。3份批次冻结文件与准备目录逐字节一致。",
           f"{audit['checked_npz_crc']}个NPZ通过CRC，{audit['checked_png']}个原始PNG通过读取；11102次实际规划前状态复算选中路线全部相符，无截断、无诊断错误。",
           '24张提交图的M/H/L/C从像素独立复算一致；每局实际收到的来源图并集经原后处理逐像素等于提交图。发布/接收来源、版本、哈希和时间顺序一致，没有提前使用未来信息。T按原曲线复算，W按已保存官方分解公式复核；未改官方评分。',
           '原生5400步/180秒；5架红机、每机2弹；单CPU、Windows Job提交内存上限4,000,000,000字节；外层240秒，串行。所有局实际实例化NO_COM_ZONE与NO_GPS_ZONE。',
           f"进程累计{sum(r['process_seconds'] for r in rows)/60:.2f}分钟，引擎累计{sum(r['wall_seconds'] for r in rows)/60:.2f}分钟；引擎最大{max(r['wall_seconds'] for r in rows):.3f}秒，进程最大{max(r['process_seconds'] for r in rows):.3f}秒，峰值Job提交内存{max(r['resources']['peak_job_commit_bytes'] for r in rows)/1e6:.2f}MB。",
           '记录的各运行文件时间区间不重叠；其他窗口是否争抢CPU无法从现有记录追溯。因此单次微小耗时差不能解释为确定的性能提升或退步机制。',
           '退出0且有效24局；超时0、控制器异常0、回调异常0、日志警告0、哈希不符0、未提交0、少弹0。墙接触是行为退步，不等同进程异常。','']
    lines+=table(['版本','实际控制器','源码SHA256'],[[LABELS[c],next(r for r in rows if r['controller']==c)['loaded_controller'],next(r for r in rows if r['controller']==c)['controller_source_sha256']] for c in ARMS])
    lines+=['']+table(['地图','划分','r1/r2种子','地图SHA256'],[[r['name'],'原训练集','2026111701 / 2026111702',plan['source_sha256'][r['map_path']]] for r in plan['map_rows']])
    lines+=['','局限：这些是已用于诊断的训练地图，不是未见测试集；每图仅两个种子，不能宣称统计显著或全地图普遍有效。RL11/RL17-R在本批8组中的实际轨迹、炸弹状态和雷达采样数组均一致；R20-B轨迹已改变，不能把三者称作同一输入回放。',
            '', '## 三版本分项统计','',
            '下表每格为“均值 / 中位数 / 最小 / 最大”。分数为百分点；覆盖为自由像素的采样比例。跨图尺寸不同，像素数量均值只作描述。卡住计数是控制器每60步检测位移不足25px时累计60，不是连续物理静止时长。来源年龄更小也可能是飞得更近，不能单独作为探索成功依据。','']
    lines+=table(['指标']+[LABELS[c] for c in ARMS],[[label]+[' / '.join(fmt(stats[c][k][v]) for v in ('mean','median','min','max')) for c in ARMS] for k,(label,_) in METRICS.items()])
    lines+=['','## 配对差值与所有退步','',
            '主对照R20-B−RL17-R：地图分4升4降，均值−1.5072、中位−1.2354；雷达采样覆盖3升5降，均值−7.1676个百分点；真墙命中均值−1774.5；W均值−0.3920，T均值+5.5456。',
            'R20-B平均局部任务40→95.5，卡住触发计数570→637.5；全部8组引擎耗时数值增加，平均+3.6639秒。任务次数增加本身不是有效探索。',
            'RL17-R−RL11：M有4升1降3平，平均+0.2949；真墙命中平均−95.5，W/T/覆盖/布弹相同。Map37/r1 M−0.1848、命中−914；Map55/r1与Map24两轮也有命中退步，Map37/r2受罚误画+1977。不能据此宣称稳定全面优于RL11。','']
    for baseline,candidate in [('rl17_repair','r20_budget'),('rl11_lines','rl17_repair'),('rl11_lines','r20_budget')]:
        cs=[x for x in contrasts if x['baseline']==baseline and x['candidate']==candidate]
        lines+=['',f'### {LABELS[candidate]} − {LABELS[baseline]}：全部8组','']
        selected=('map_score','wall_hits','wall_missed','charged_pixels','wall_score','time_score','bombs','sampled_lidar_free_percent','wall_seconds')
        lines+=table(['地图/轮次']+[METRICS[k][0] for k in selected],[[f"{x['map']}/r{x['round']}"]+[sign(x['delta'][k]) for k in selected] for x in cs])
        lines+=['','配对差值分布（每项均为8对）：','']
        lines+=table(['指标','均值','中位数','最小','最大'],[[label]+[fmt(paired[f'{candidate}-{baseline}'][k][v]) for v in ('mean','median','min','max')] for k,(label,_) in METRICS.items()])
        lines+=['','按地图汇总两种子的平均差值：','']
        lines+=table(['地图']+[METRICS[k][0] for k in selected],[[name]+[fmt(statistics.mean(x['delta'][k] for x in cs if x['map']==name)) for k in selected] for name in sorted({x['map'] for x in cs})])
        lines+=['','所有不利方向（含耗时数值增大；来源年龄/缺测/避让变化是诊断指标，不作独立因果判断）：','']
        for x in cs:
            text='；'.join(METRICS[k][0]+sign(v) for k,v in x['regressions'].items()) or '无列示指标退步'
            lines.append(f"- {x['map']}/r{x['round']}/seed{x['seed']}：{text}。")
    lines+=['','### 最差有效局与突出异常','']
    for c in ARMS:
        r=min((r for r in rows if r['controller']==c),key=lambda r:r['map_score'])
        lines.append(f"- {LABELS[c]}最低地图分：{r['map']}/r{r['round']}/seed{r['seed']}，M={r['map_score']:.4f}。")
    lines += ['- 最大地图退步：Map37/r2，R20-B比RL17-R少10.6141分、少11126个真墙命中、覆盖少13.2546个百分点；W增加4.0919不能抵消这些事实。',
              '- 最大覆盖退步：Map37/r1，覆盖少25.4155个百分点，M少6.5777，受罚误画多10653。',
              '- 新墙接触：Map72/r2，R20-B记录17次墙接触，两个对照均为0；卡住计数180→780，地图仅+0.3001。原诊断仅保存团队接触计数，不能把17次精确归到某架机或某个动作；健康采样也不能替代接触逐事件记录。',
              '- 正例也有代价：Map24/r1，M+9.2998、覆盖+5.2142个百分点，但T−7.1693、引擎+12.1591秒、来源最大年龄64→1685步；Map24/r2则M−4.2755、W−7.4776，表现不稳定。','',
              '## Map72/r1：提交更及时，但深处探索减少','',
              'RL17-R→R20-B：覆盖27.3773%→16.8807%，M9.6836→11.1881，H13016→13457，C35326→30501，W29.9886→24.7111。局部任务40→89，活动步855→2309。',
              '原版4号覆盖27.3773%，末尾本地图命中26232，但提交者只收到v2670、命中12754的旧来源图。候选4号覆盖约14.35%，末尾本地图命中12717、实际收到命中12699：版本更近，深处观测却少了。不能把来源年龄1865→77步解释为成功完成深处探索和回传。',
              'M增加伴随受罚误画减少，新增命中仅441；这与集中在较小范围、地图更一致相容，但不能单靠分数分解证明全部因果。最终收到的并集已逐像素验证，不使用假想“全机最新图”。',
              '实际规划前快照仍显示入口中心经常被膨胀阻挡。0/3/4号候选分别95/99、83/87、79/85次规划处于该状态，中心可达分别1/0/2次；并未修复导航图断连。4号入口附近目标11次，却未复现原版的大范围探索。',
              '末弹确认3444→3676步，反而晚232步；但阶段4536→3888步、T22.8571→40，说明更早提交/结束与更早放完炸弹不同。引擎104.5283→107.7528秒，结束步更少也不等于计算更快。',
              '', '## Map55/r1：探索略增，漂移与误画加重','',
              'RL17-R→R20-B：覆盖26.2805%→28.1508%，M16.4214→13.6505，H24777→24068，C62979→76226。局部任务40→112，活动步1059→3102；卡住计数1440→1020，但累计绝对缺测6481→7667个机步。',
              '候选3号覆盖22.56%→25.71%，绝对缺测2052→3802步；每12步导航误差采样P95约47.66→227.78px。其本地图受罚误画27632→52167，提交者实际收到的该来源误画25505→39791。4号导航误差P95约85.14→224.70px。送达改善未阻止地图质量退步。',
              '候选实际来源版本[4535,4530,4500,3660,4410]，年龄[0,5,35,875,125]；原版为[4535,3840,3720,3450,4230]，年龄[0,695,815,1085,305]。单机来源分数不能相加，最终M下降已由真实提交图核实。',
              '末弹3673→3338步，早335步；W31.3233→38.1047，布弹覆盖确有收益。阶段仍4536步，T保持22.8571；引擎116.7986→119.0135秒。布弹改善和地图退步必须同时保留。','']
    focus=read(out/'focus_planning.json')
    for name in ('yh_map_fs_72','yh_map_fs_55'):
        lines+=['',f'### {name}/r1/seed2026111701：逐机送达和导航证据','']
        ds=[detail[(name,1,c)] for c in ARMS]
        lines+=table(['版本/机','雷达覆盖%','任务次数','补充步','接收版本/年龄','本地H/C','已接收H/C','缺测步','导航误差P95px','放弹引擎步'],[
            [f'{LABELS[d["controller"]]}/d{x["owner"]}',fmt(x['sampled_visible_free_percent']),x['passage_episodes'],x['renewals'],
             f"{x['source_version']}/{x['source_age']}",f"{x['local']['hits']}/{x['local']['charged']}",f"{x['received']['hits']}/{x['received']['charged']}",x['outage_steps'],fmt(x['navigation_error_p95']),x['placement_engine_steps']] for d in ds for x in d['drones']])
        lines+=['','实际保存的规划前快照（不是整局均匀采样，不能与R19的188次采样分母直接比较）：','']
        lines+=table(['版本/机','规划次数','入口中心自由','原障碍/膨胀阻挡','中心可达','入口附近选中目标','空路线'],[
            [f'{LABELS[f["controller"]]}/d{x["owner"]}',x['plans'],x['free'],f"{x['raw_obstacle']}/{x['inflated']}",x['reachable'],x['near_gate_goal'],x['route_empty']]
            for f in focus if f['map']==name for x in f['drones']])
    lines+=['','## 判断与下一步','',
            '已证实：补充预算被实际触发、局部任务增加；8组中5组覆盖下降，平均地图分和真墙命中下降，并出现17次新增墙接触。本实现没有达到“稳定改善探索不足”的目标，不建议进入正式入口。',
            '推测：更多局部任务可能把行动机会用于房间内重复移动，或改变出门方向和进入特殊区的时机；送达更及时也可能来自探索范围收缩。这些是与结果相容的机制，不是已逐动作证明的唯一原因。',
            '下一步待验证：优先利用保存的目标、路线、入口栅格和实际收发版本，区分“有入口观测但规划断连”与“已进入深处但地图未送达”；若继续局部任务研究，应检验任务是否产生新区域进展及失败退出。此次不追加参数补丁、不改恢复权重、不运行新批次。','',
            '## 每张地图、每轮、每个分项及图册','',
            '以下全部是本次R20实际完整红阶段新图；每张图从左到右RL11、RL17-R、R20-B。每组提供提交地图、采样探索、轨迹、墙体覆盖、炸弹五类。图例和地图/轮次/种子在图内；蓝队难度分D与完整总分逐行均为未评测。','']
    for name,number,seed in groups:
        rs=[next(r for r in rows if r['map']==name and r['round']==number and r['controller']==c) for c in ARMS]
        lines+=['',f'### {name} / r{number} / seed {seed}','']
        lines+=table(['版本','实际特殊区','M','H','L','C','W','T','弹','D/完整总分','完成/有效'],[[LABELS[r['controller']],'+'.join(r['actual_zones']),fmt(r['map_score']),r['wall_hits'],r['wall_missed'],r['charged_pixels'],fmt(r['wall_score']),fmt(r['time_score']),r['bombs'],'未评测/未评测',f"{r['status']}/{r['valid']}"] for r in rs])
        lines+=['']+table(['版本','雷达采样%','引擎轨迹指标','结束步','引擎/进程秒','峰值MB','卡住/避让','局部任务/活动步','最陈旧来源步','末弹步','墙/机接触','平均健康'],[
            [LABELS[r['controller']],fmt(r['sampled_lidar_free_percent']),fmt(r['official_trajectory_exploration']),r['steps'],f"{r['wall_seconds']:.3f}/{r['process_seconds']:.3f}",f"{r['resources']['peak_job_commit_bytes']/1e6:.2f}",f"{r['stalled_steps']}/{r['yield_steps']}",f"{r['lifetime_passage_episodes']}/{r['sum_local_active_ticks']}",r['max_source_age'],r['last_bomb_step'],f"{r['wall_contacts']}/{r['drone_contacts']}",fmt(r['mean_health'])] for r in rs])
        for kind in KINDS:lines+=embed(out/f'atlas/{name}_r{number}_{kind}.png',f'R20实际批次 {name} r{number} seed{seed} RL11/RL17-R/R20-B {kind}')
        lines+=['','本组实际录像回放（每条独立运行）：','','```powershell',f"Set-Location -LiteralPath '{REPO}'"]
        for r in rs:lines.append(fr".\replay_map.cmd '{r['attempt']}'")
        lines+=['```']
    lines+=['','## 可追溯产物','',
            f'- [逐机完整附表：全部24局×5机]({(out/"per_drone_appendix.md").as_posix()})',
            f'- [独立复算汇总与有效性]({(out/"atlas/summary.json").as_posix()})',
            f'- [附加完整性/评分/收发核验]({(out/"detailed_audit.json").as_posix()})',
            f'- [全部对比、退步与统计]({(out/"comparisons.json").as_posix()})',
            f'- [实际来源图/逐机归因数据]({(out/"run_details.json").as_posix()})',
            f'- [本轮文件保护复核]({(out/"final_audit.json").as_posix()})',
            '不切换正式入口、不提交、不推送、不打包，旧结果原样保留。']
    put(report,lines)
    write(out/'report_manifest.json',dict(report=str(report),sha256=digest(report),contact_sheets=[str(p) for p in sheets.glob('*.png')]))
    print('REPORT',report,'groups',len(groups),'figures',len(groups)*5,'per-drone rows',len(rows)*5,flush=True)


if __name__=='__main__':main()
