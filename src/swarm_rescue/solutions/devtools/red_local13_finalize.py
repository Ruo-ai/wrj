"""RL13 eight-run overview and evidence-based final report; offline only."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from PIL import Image
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_lines11_review as lines
from swarm_rescue.solutions.devtools import red_consistent12_review as panels
from swarm_rescue.solutions.devtools.red_local13_review import NAMES


ROOT=Path.cwd().resolve()
OUT=ROOT/'output/sf13/rl13_m89_review_20261115'
BATCH=ROOT/'output/sf13/rl13_m89_once_20261115'
DEV=ROOT/'output/sf13/rl13_dev_20261115'
CONTROLLERS=['rl11_lines','rl13_entry','rl13_segments','rl13_local']
REPORT=ROOT/'复赛阶段_FS4_红队RL13_入口与局部图单因素实测_20261115.md'


def assets(row):
    run=Path(row['attempt']);result=base.read(run/'result.json');request=base.read(run/'request.json')
    h,w=result['expected_shape'];head=base.COMPARISON_HEADER_HEIGHT;sep=base.COMPARISON_SEPARATOR_WIDTH
    png=np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
    truth=np.all(png[head:head+h,:w]==40,axis=2).astype(np.uint8)
    wall=np.asarray(Image.open(next((run/'artifacts').glob('*_wall_destruction.png'))).convert('RGB'))
    with np.load(next((run/'artifacts/recordings').glob('*.npz')),allow_pickle=True) as f:trace=f['drone_states'].copy()
    with np.load(run/'local13_diagnostics.npz') as f:
        visible=np.zeros(((h+3)//4,(w+3)//4),bool)
        for key in f.files:visible|=f[key].astype(bool)
    visible=np.repeat(np.repeat(visible,4,0),4,1)[:h,:w]
    return dict(truth=truth,merged=png[head:head+h,2*(w+sep):2*(w+sep)+w],
                wall_panel=wall[head:head+h,:w],trajectories=trace,visible=visible,
                bombs=base.read(next((run/'artifacts').glob('*_bombs.json')))['bombs'],
                radius=result['phase1']['wall_breakdown']['blast_radius'],spec=base.read(request['map_file']),w=w,h=h)


def main():
    assert not REPORT.exists(), 'Refuse existing report'
    summary=base.read(OUT/'summary.json');rows=summary['rows'];assert len(rows)==8 and not summary['failures']
    data={(r['map'],r['controller']):assets(r) for r in rows};panels.NAMES=NAMES
    for kind,label in [('exploration','雷达探索范围估计'),('mapping','完整提交建图'),('trajectories','完整实际轨迹'),('walls','官方墙体覆盖'),('bombs','实际炸弹位置')]:
        destination=OUT/f'overview_{kind}.png';assert not destination.exists()
        fig,axes=base.plt.subplots(4,2,figsize=(18,27),squeeze=False)
        fig.subplots_adjust(left=.015,right=.985,top=.945,bottom=.045,wspace=.06,hspace=.45)
        for mindex,m in enumerate(['Map08','Map09']):
            for cindex,c in enumerate(CONTROLLERS):
                ax=axes[mindex*2+cindex//2,cindex%2];r=next(r for r in rows if r['map']==m and r['controller']==c);d=data[m,c]
                panels.panel(ax,'mapping' if kind=='exploration' else kind,r,d)
                if kind=='exploration':
                    pixels=np.ones((*d['truth'].shape,3))*.94;pixels[d['visible']]=[.57,.85,.73]
                    pixels[d['truth'].astype(bool)]=[.12,.12,.12];ax.imshow(pixels);base.areas(ax,d['spec'])
        fig.suptitle('Map08 / Map09 · 四版各一次完整红队：'+label,fontproperties=lines.FONT,fontsize=23,y=.982)
        fig.text(.5,.96,'上半部 Map08，下半部 Map09；每图依次：RL11 / 入口对齐 / 分段运动 / 局部图。',ha='center',fontproperties=lines.FONT,fontsize=13)
        note={'exploration':'绿色为每8步、4px雷达可见区采样估计；含射线间插值，不能等同完整观察覆盖或地图得分。',
              'mapping':'绿：命中；蓝：漏墙；深红：收费误画；黄：免罚误画。',
              'trajectories':'颜色为录像槽位。灰色无GPS/罗盘，黄色无通信；轨迹不等于雷达覆盖。',
              'walls':'绿：覆盖内墙；蓝：覆盖边界墙；灰：未覆盖；红：爆炸范围。',
              'bombs':'编号为实际落点产物顺序，圆为爆炸范围；圆重叠不等于墙面收益重复。'}[kind]
        fig.text(.5,.023,note,ha='center',fontproperties=lines.FONT,fontsize=13)
        fig.text(.5,.009,'八局均完整有效、各10弹；蓝队难度与完整比赛总分未评测。',ha='center',fontproperties=lines.FONT,fontsize=12)
        base.save(fig,destination)
    protected=base.read(DEV/'protected_before.json')
    changed=[name for name,h in protected.items() if not (ROOT/name).is_file() or base.digest(ROOT/name)!=h]
    verification=dict(protected_files=len(protected),changed=changed,full_red_phases=8,blue_phases=0,
        valid=sum(r['red_valid'] for r in rows),callback_errors=[r['callback_errors'] for r in rows],
        diagnostic_errors=[r['local13']['errors'] for r in rows],input_comparisons=summary['input_comparisons'],
        tracked_diff=subprocess.check_output(['git','diff','--stat'],text=True),
        root_zip_sha256=base.digest(ROOT/'team742_preliminary.zip'))
    assert not changed,changed
    with (OUT/'verification.json').open('x',encoding='utf-8') as f:json.dump(verification,f,indent=2,ensure_ascii=False)
    doc=['# RL13：RL11基础上的入口、分段运动与短局部图完整实测','',
         '2026-11-15。**继续保留 RL11，不采用本轮三个候选，不组合、不切正式入口。** 本轮并未取得稳定建图提升。完成 Map08 / Map09 × RL11 / RL13-E / RL13-S / RL13-L × 各1轮，共8场完整红队，全部有效、10弹；0场蓝队。不是用冒烟或旧成绩代替完整地图结果。','',
         '## 1. 最新状态与保护','',
         '接手HEAD为 a531f14882cb9937b1c2136f8f8f285590d1ab0e，工作区干净；旧交接的HEAD/蓝队状态不覆盖本地事实。RL11 SHA256仍为 56474b5b14af1ce653060f8a082b4e24fdf44a5ded6a4dde7aa5e0415083bbf6。正式place仍为进阶2，rescue仍为示例；根目录ZIP仍是上一轮按用户要求放置的RL11包，本轮不打包。',
         f'保留 {len(protected)} 个既有源码、配置、根目录文档与包，最终哈希变化0。未改蓝队、共享定位/通信/评测、评分、物理、地图、入口或旧报告。CIM进程查询不可用，可用Get-Process未报告其他Python评测；八局由同一驱动严格串行。进程可见性有权限限制，不能据此证明整机完全无其他负载。',
         '已依次核对快速交接、总方案、图件规范、RL11/RL12报告、实际源码与评分。RL11已包含S6停转、逐步里程计、A3缓存/恢复重锚定和版本快照，本轮没有重新发明这些机制。','',
         '## 2. 接口与单因素实现','',
         '合法缺测getter为None，需检查NaN；NoGpsZone同时禁GPS和罗盘，NoComZone独立禁通信，不是物理墙。GPS标准差5px、罗盘4°且AR(1)=0.98；里程计每步为实际距离/上一机体系运动方向/实际转角，噪声标准差0.2px/8°/1°；雷达181束、360°、230px、标准差2.5px。控制指令是力/旋转控制量，不是位移。上述核对来自本地传感器与动力学代码。入口姿态与角度累计都不能视作准确真值。','',
         '|候选|新增文件与开关|实际改变|',
         '|---|---|---|',
         '|RL13-E|my_drone_red_entry13.py，ENTRY_ALIGN；CENTERING=False|仅改变S6的入口相对方向建议；RL11导航位姿和提交地图不改|',
         '|RL13-S|同文件，ADAPTIVE_SEGMENTS|短目标不反复启停、可见净空内选择80～140px段长、实际转角的一步提前量；巡航目标仍4px/步|',
         '|RL13-L|my_drone_red_local13.py，LOCAL_BLOCKS；BLOCK_SPAN=48，ENTRY_HOLD=24|只改暂存扫描的恢复重投，运动/导航/放弹不改；保留3px图及原后处理|','',
         '**E入口对齐：**正常时保存最近可靠估计、非零不确定度和最多16条合法墙段；失效后原S6连续3帧确认。每4步拟合任意方向有限墙段，需两侧相反、近似平行、有至少35px向前共同长度、宽52～180px，且接近当前导航方向。连续3次方向/宽度一致、前方可制动且旋转净空足够，才建议相对朝向。单侧墙、开口、动态遮挡、无可观测配对均回退。没有把相对墙角当成全局罗盘。主动居中保留独立开关但关闭，未宣称验收。入口墙快照仅作记录，不另加未经验证的全局重定位。',
         '**S分段执行：**复用原NORMAL→BRAKE→SETTLE→OBSERVE→TURN→ACCEL/CRUISE及有限等待/冷却。原来很近的粗路径点也会进入启停流程；本候选在加速制动预算不足时先交还原导航，有空间时再选可见段长。用实际里程计角增量作转向提前量，不通过低速宣称精度提高。所有加减速、转向侧滑仍由原MotionEstimate累计；原近场避障、队友等待上限、转向净空、生命/布弹/收尾抢占不删除。',
         '**L局部图：**复用RL11每4步的合法墙/自由射线缓存、每8步墙段配准；入口不确定扫描最多暂缓发布24步，连续两次墙段匹配可提早解除，未确认则回到原RL11低权重临时图，完整观测不丢弃。恢复GPS与罗盘后，将暂存片段按48步或0.18rad转角切短块，每块使用一个SE(2)刚体修正，保留块内相对距离/角度。仅重投这次失定位扫描，历史图不整体搬动，不填未知墙。没有新增独立无GPS定位器；未恢复时仍为RL11暂存图，块间缝隙、块内非刚性漂移和带噪声的恢复锚点仍是局限。它不能解决导航进门。',
         '**送达隔离：**保留RL11已观察导航图、既有集合路径和硬提交时限；特殊运动仍在进度0.60后交还原控制，布弹/集合优先。未增加主动循迹返通信区或新增确认协议，不宣称已解决可靠退路。独立记录收到的来源/版本、发布与本机终图差异。实际S组送达回退证明仅保留原时间预算并不足够，下一专项应独立处理。','',
         '## 3. 实际测试与公平性','',
         '30项行为检查通过，1.94秒；前两轮28/29项也通过，XML全部保留。覆盖任意斜墙、单墙/开口回退、±π、关开关等价、加减速/侧滑累计、短路点回退、制动、动态队友有限等待、缓存不丢、恢复不双计和旧可信图不平移。合成用例含带噪雷达和理想运动增量，不能代替真实噪声比赛。',
         '3个候选各128步原生冒烟：3.252/3.232/3.189秒；新增雷达覆盖诊断另有1次128步链路检查。冒烟均无诊断异常，但只标短测，不记完整有效成绩。新诊断为控制返回后的被动观察器，真值从不返回控制器。','',
         '|相同完整合法输入离线重放|RL11|RL13-L|差值|',
         '|---|---:|---:|---:|',
         '|Map08|63.141498|63.188014|+0.046516|',
         '|Map09|73.367575|73.221123|−0.146452|','',
         '重放使用上轮RL11真实完整GPS/罗盘/里程计/181束雷达/语义输入，固定实际提交收到的来源版本；基线各机图和提交像素精确复现。每地图器9.39～10.20秒。这是离线官方公式诊断，不是新比赛成绩。其后仍执行了本轮完整八局。','',
         '实际八局数组核对：L两图与RL11的完整轨迹、完整合法输入和实际弹位完全相同；E的Map09也全部相同；E的Map08及S的两图均不同。E的“guided_ticks”统计方向建议，不等于最终动作真的改变（Map09有11次建议却没有改变轨迹）。同种子本身没有被当作同输入证据。','',
         '## 4. 八局完整红阶段结果','',
         'Map08种子200806601、NO_GPS_ZONE，1660×1122；Map09种子2477227837、NO_GPS_ZONE+NO_COM_ZONE，1700×1100。每图每版1次、均attempt_001、无重试；官方5400步/180秒、5红机×2弹、1CPU、4,000,000,000字节限制。所有原始结果、日志、录像、源码快照和失败状态保留。没有蓝阶段，D、完整原始/有效总分统一未评测。','',
         '|地图/版|红有效|M|W|T|弹|实际步|引擎墙钟秒|进程秒|墙/机接触|',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{r['red_valid']}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['placed_count']}|{r['red_steps']}|{r['red_walltime']:.3f}|{r['resources']['process_seconds']:.3f}|{r['contacts_wall']}/{r['contacts_drone']}|")
    doc+=['',f"八局无控制器崩溃、回调异常、超时或审计失败。Job峰值提交内存 {min(r['resources']['peak_job_commit_bytes'] for r in rows)/1e6:.1f}～{max(r['resources']['peak_job_commit_bytes'] for r in rows)/1e6:.1f} MB。接触计数不是扣血次数；进程秒含启动/产物，不能与引擎墙钟混同。本机限制不等于平台性能。",'',
          '|地图/版|真墙面命中|收费误画像素|信用/扣分（百分点）|灰区墙命中|灰区假墙像素|灰区采样可见自由区%|失定位位置均值px/角均值°|',
          '|---|---:|---:|---|---:|---:|---:|---|']
    for r in rows:
        e=r['line_accuracy'];position=e['outage_position_summary']['mean'];heading=e['outage_heading_summary']['mean']
        doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{r['wall_hits']}|{r['charged_pixels']}|{100*r['map_credit']:.4f}/{100*r['map_penalty']:.4f}|{r['gray_surface_hits']}/{r['gray_true_surface']}|{r['gray_false_pixels']}|{100*r['gray_sampled_visible_free_fraction']:.2f}|{position:.3f}/{heading:.3f}|")
    doc+=['','灰区假墙包括免罚误画，不等同收费像素。收费像素个数也不是扣分，官方还按离真实墙的距离计权。可见区为真实雷达射线每8步采样、4px栅格和射线间插值的估计，并非连续精确覆盖；实际墙端点每40步采样另有下界指标。不同轨迹组误差来自各自经历的区域，不能作为相同位姿的精度对照。','',
          '|地图/版|入口首次缺测次数|入口地图器位置误差均值px|入口角误差均值°|最旧已收到来源延迟步|本机终图未包含在对应来源快照的格数合计|',
          '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        entries=[e for d in r['local13']['drones'].values() for e in d['entries']];n=len(entries)
        pos=sum(e['mapper_position_error'] for e in entries)/max(1,n);angle=sum(e['mapper_heading_error_degrees'] for e in entries)/max(1,n)
        ages=[s['age'] for s in r['source_snapshots'] if s['age'] is not None]
        missing=sum(s['own_final_absent_from_source_snapshot'] for s in r['source_snapshots'])
        doc.append(f"|{r['map']}/{NAMES[r['controller']]}|{n}|{pos:.3f}|{angle:.3f}|{max(ages,default=0)}|{missing}|")
    doc+=['','首次缺测次数包含反复穿越边界，未经稳定判定的单帧缺测也计入离线入口审计；不同组不是同一组入口事件。末项按来源计数，不是去重后全队丢失面积，也不等于这些格一定正确。逐源表在完整图册。','',
          '## 5. 证实、假设与采用结论','',
          '- **E不采用。** Map08增加2015个真墙面命中，却增加29994个收费误画像素；信用+2.0848、扣分+5.6202个百分点，地图分净降3.5354。灰区墙命中反而减少351、灰区假墙增加19559。区内平均角误差5.186°→4.914°，位置误差15.836→18.790px；角度稍好不等于最终地图更好。Map09逐输入/轨迹/弹位均与原版相同，没有提升。',
          '- **S不采用。** 两图地图分分别下降16.5499和21.8426。Map09已收到2/3号来源分别落后2495/2045步，对应本机终图有10922/4070格未包含于收到的快照。离线无视通信地并合各机最终渲染图，公式值约66.2831，仍低于RL11的73.3676；这个离线值不是官方成绩或可达上界。说明送达和本机图质量都需要处理，不能只怪通信。',
          '- **L不采用。** 两图M变动+0.0465/−0.1465，均与相同输入离线重放一致；位置/角误差、轨迹、雷达输入、弹位保持原样。短块刚体重投没有解决入口偏差与长段非刚性漂移，当前没有双图收益证据。',
          '- **仍待验证：**在有可信退路和足够返程时间时如何主动恢复地图送达；导航图失真导致入口不可达的因果；初始绝对角偏置、连续几何退化、恢复锚点噪声分别贡献多少。单局不能拆成精确因果比例，不能保证所有入口都可达或特殊区都画准。',
          '- **下一步优先级：**在RL11上独立做送达/退路专项，再考虑有可观测性约束的局部位姿与导航图一致性。当前不把这批失败项组合，也不批量搜索阈值。原RL11回退选择器为rl11_lines；三个Off类和测试保留。','',
          '## 6. 全部八局图示','',
          '每张总图上半部Map08、下半部Map09；各地图依次为RL11、入口对齐、分段运动、局部图。所有图均来自本轮完整实跑，原始图未覆盖。']
    for kind in ['exploration','mapping','trajectories','walls','bombs']:
        doc+=['',f'![八局 {kind}]({(OUT/f"overview_{kind}.png").as_posix()})']
    doc+=['',f'[可放大逐候选对照、逐源送达和八条回放命令]({(OUT/"report.md").as_posix()})；[原始逐组图册]({(BATCH/"report.md").as_posix()})；[全部机器可读指标]({(OUT/"summary.json").as_posix()})。','',
          '## 7. 文件、实跑命令及后续命令','',
          '新增Python均在solutions：my_drone_red_entry13.py、my_drone_red_local13.py；devtools下test_red_local13.py、red_local13_batch.py、red_local13_diagnostics.py、red_local13_replay.py、red_local13_review.py、red_local13_finalize.py。前者在线仅合法观测，后者离线使用真值；不导入AI/外部API、新框架或第三方代码。',
          '已运行命令（这些目录已完成，不要再次续跑）：','',
          '```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",
          '.\\.venv\\Scripts\\python.exe -B -m pytest src/swarm_rescue/solutions/devtools/test_red_local13.py src/swarm_rescue/solutions/devtools/test_red_lines11.py -q -p no:cacheprovider',
          '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_local13_batch --output output/sf13/rl13_m89_once_20261115 --maps Map08 Map09 --controllers rl11_lines rl13_entry rl13_segments rl13_local --rounds 1 --run',
          '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_local13_review --batch output/sf13/rl13_m89_once_20261115 --output output/sf13/rl13_m89_review_20261115','```','',
          '本轮无蓝队阶段，因此没有本轮蓝回放，难度与完整总分未评测。红回放：']
    for r in rows:
        doc+=['',f"{r['map']} / {NAMES[r['controller']]}：",'```powershell',f".\\replay_map.cmd '{r['attempt']}'",'```']
    doc+=['','不建议现在扩大失败候选。若用户希望独立复核，可以手动运行下面命令：Map08/09、每版1轮，共8红0蓝，新目录；本轮未自动启动第二批，不做参数搜索或100图。删去--run只生成计划。','',
          '```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",
          "$rl13Out = 'output/sf13/rl13_recheck_' + (Get-Date -Format 'yyyyMMdd_HHmmss')",
          "if (Test-Path -LiteralPath $rl13Out) { throw '输出目录已存在' }",
          '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_local13_batch --output $rl13Out --maps Map08 Map09 --controllers rl11_lines rl13_entry rl13_segments rl13_local --rounds 1 --run',
          "if ($LASTEXITCODE -ne 0) { throw '评测中断，保留全部失败产物' }",
          '$rl13Review = $rl13Out + "_review"',
          '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_local13_review --batch $rl13Out --output $rl13Review',
          "Invoke-Item (Join-Path $rl13Review 'report.md')",'```','',
          '只有这两张既有开发图参与本轮，70训练/15验证/15测试划分及100图均未改动。保留所有低分结果、原始日志和源码快照，不提交、不推送、不打包。']
    with REPORT.open('x',encoding='utf-8') as f:f.write('\n'.join(doc)+'\n')
    print(json.dumps(dict(report=str(REPORT),overview_count=5,verification=verification),ensure_ascii=True))


if __name__=='__main__':main()
