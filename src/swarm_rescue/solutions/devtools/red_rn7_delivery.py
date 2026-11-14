"""Finalize RN7 read-only audit and native evidence after the six real games."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools.optimization.evaluation.runner import validate_sources


def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,default=Path('output/sf13/rn7_m89_review'))
    p.add_argument('--output',type=Path,default=Path('output/sf13/rn7_delivery'));args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    review=base.read(args.review/'summary.json');batch=Path('output/sf13/rn7_m89_once')
    plan=base.read(batch/'plan.json');validate_sources(plan['source_sha256'])
    native=base.read('output/sf13/rn7_native_v3/results.json')
    fig,axes=base.plt.subplots(1,len(native),figsize=(18,4.4))
    for ax,row in zip(axes,native):
        for y in (row['width']/2,-row['width']/2):ax.plot([0,160],[y,y],color='black',lw=3)
        if row['deadend']:ax.plot([160,160],[-row['width']/2,row['width']/2],color='black',lw=3)
        trace=row['trace'];ax.plot([r['x'] for r in trace],[r['y'] for r in trace],lw=1)
        active=[r for r in trace if r['active']]
        ax.scatter([r['x'] for r in active],[r['y'] for r in active],s=3,color='tomato')
        ax.set(xlim=(-70,205),ylim=(-80,80),aspect='equal')
        ax.set_title(f"{row['case']}\npassed={row['passed']}, contacts={row['contacts']}",fontsize=9)
    fig.suptitle('Native component fixtures, one final run each. Red dots: RN7 active. Not official map scores.')
    base.save(fig,args.output/'native_cases.png')
    protected=json.loads(Path('output/sf13/rn7_dev/protected_before.json').read_text(encoding='utf-8-sig'))
    changed=[];checked=0
    for item in protected:
        file=Path(item['path'])
        if not file.exists() or base.digest(file).lower()!=item['sha256'].lower():changed.append(item['path'])
        checked+=1
    tests=[]
    for path in sorted(Path('output/sf13/rn7_dev').glob('tests*.xml')):
        suite=ET.parse(path).getroot().find('testsuite')
        tests.append(dict(file=str(path),**suite.attrib))
    # Width evidence from ORIGINAL official truth images. This is offline only.
    widths={}
    gates={}
    for r in review['rows']:
        w,h=(1660,1122) if r['map']=='Map08' else (1700,1100)
        targets=[(1330,1024),(1632,1059)] if r['map']=='Map08' else [(491,564),(138,319)]
        run=Path(r['attempt']);motion=base.read(run/'motion_diagnostics.json')['drones']
        narrow=base.read(run/'narrow_diagnostics.json')['drones'];records=[]
        with np.load(next((run/'artifacts/recordings').glob('*.npz')),allow_pickle=True) as z:
            t=z['drone_states'];x=t['x']+w/2;y=h/2-t['y']
            for px,py in targets:
                distances=np.where(t['active'],np.hypot(x-px,y-py),np.inf)
                candidates=[(math.hypot(s['truth'][0]+w/2-px,h/2-s['truth'][1]-py),i,s)
                            for i,d in motion.items() for s in d['samples']]
                distance,ident,sample=min(candidates,key=lambda v:v[0])
                entered=[e for e in narrow[ident]['events'] if e['reason']=='entered' and e['tick']<=sample['tick']]
                records.append(dict(offline_gate_canvas=[px,py],minimum_recorded_center_distance=float(distances.min()),
                    nearest_diagnostic_sample_distance=distance,drone_id=ident,sample=sample,
                    narrow_episodes_used_at_sample=len(entered),maximum_episodes=8))
        gates[r['map']+'/'+r['controller']]=records
    for map_name in ('Map08','Map09'):
        r=next(r for r in review['rows'] if r['map']==map_name)
        run=Path(r['attempt']);result=base.read(run/'result.json');h,w=result['expected_shape']
        img=np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
        truth=np.all(img[24:24+h,:w]==40,axis=2)
        ranges=[(1024,1250,1370),(1059,1550,1659)] if map_name=='Map08' else [(564,400,570),(319,70,205),(231,50,175)]
        observed=[]
        for y,l,rgt in ranges:
            free=~truth[y,l:rgt+1];starts=np.flatnonzero(free&~np.r_[False,free[:-1]])
            ends=np.flatnonzero(free&~np.r_[free[1:],False])
            observed.append(dict(row=y,window=[l,rgt],free_runs=[dict(start=int(l+a),end=int(l+b),pixels=int(b-a+1)) for a,b in zip(starts,ends)]))
        widths[map_name]=observed
    verification=dict(protected_files_checked=checked,changed=changed,source_snapshot_validated=True,
        plan_sha256=base.digest(batch/'plan.json'),tests=tests,offline_widths=widths,
        offline_gate_diagnostics=gates,
        actual_real_attempts=len(list((batch/'runs').rglob('attempt_*'))),blue_phases=0)
    text=(args.review/'report.md').read_text(encoding='utf-8')
    paths=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
    for path in paths:
        with Image.open(Path(path.strip('<>'))) as im:im.verify()
    verification['verified_image_links']=len(paths)
    with (args.output/'verification.json').open('x',encoding='utf-8') as f:json.dump(verification,f,indent=2)
    notes=['# RN7 窄路专项开发与验证记录','',
        '本轮新增 A3+RN7、B+RN7、C+RN7 三个独立类；原 A3 / S6-B / S6-C / advanced5_narrow、蓝队、共享文件、正式入口未由本任务修改。没有提交、推送、打包或切换分支。',
        '当前正式放弹入口仍为 Advanced2；RN7 仅由专用评测注册。新增 Python 均在 solutions 目录。','',
        '## 证据与实现','',
        '- 源码事实：原导航为 20px 栅格、十字障碍膨胀，局部避障圆半径通常25px、紧通道22px。旧 N0 局部 BODY_RADIUS=10 / TRACK_RADIUS=12，不满足当前机身外接半径16.4012px，因此未直接恢复整包。',
        '- RN7 复用 N0 的保守雷达扇区与有界路径搜索，但采用 2px、440×440px 当前机体系局部快照；机身外接半径16.4012、跟踪18.5、规划19.9142px。未知不可通行，路径整段检查，无对共享地图的改写。',
        '- 两侧需要实测静态墙端点，不能以未知边界充当墙。候选端点偏好低访问、新观察和前进方向。每机最多8段、每段170步，任务失败冷却，队友等待7+3×机号步后交还原控制器。',
        '- 路点每步用合法里程计距离、相对方向、转角转换；包含加减速和侧滑，不积分指令。导航位置/提交地图算法原样保留。新扫描不通时先刹车并局部重规划，可在已观察空间内横向修正局部路点12px；不修正全局定位。',
        '- 保留放弹/探索主流程：生命≤10、放弹尝试、距目标弹点55px、归队、提交后、进度≥68%时不接管。B/C 的 S6 运动在 RN7 接管期间让出，合法传感器累计和原建图仍运行。开关 NARROW_ENABLED=False（rn7_off）；三个基线/对照版本均保留。',
        '- 不能声称 Map09 左上角缺探索已被证明是窄路误判：主要入口约118px，旧录像缺少每帧规划栅格。Map08 最底部则确有约50～60px口与通道。精确官方真值栅格开口见 verification.json（仅离线）。',
        '- 当前精细观察仍较保守：46px斜通道可能被拒绝；50px带真实噪声原生场景未通。不能承诺所有物理可通通道均可通过。','',
        '## 实际漏搜诊断与采用结论','',
        'Map08 的 A3+RN7 与 B+RN7 最底部区域均未进入。B+RN7 的录像距第一处下方入口中心最近3.8188px，仍未穿入下方走廊。第2868次控制的诊断样本可确认无人机1就在入口附近：GPS/罗盘可用，原导航位置误差16.7612px、提交地图器4.0469px。该机早在第1222步已启动第8段窄路任务、第1246步结束，之后受8段总预算限制不再启用RN7。',
        '因此，存在“到达关键入口之前，窄路总预算已经用完”的直接代码与事件证据；但仅凭现有记录还不能证明这就是漏搜的唯一原因。下一步应先按空间去重/进展刷新预算，保留每段限时和队友等待上限，再对关键入口是否实际激活做验证；不应先全面缩小避障距离。',
        '本轮三个版本均保留为实验候选，不切换正式入口。单轮收益不代表稳健胜出；图册同时记录地图、墙体、时间、接触和区域进入结果。用户指定各图各版只跑一次，本轮不修改参数重跑挑分，不再启动额外批量。','',
        'Map08 三候选真实墙命中相对各自原版分别减少664/734/1598像素，误画扣分像素分别减少55544/27628/8696；地图分提高主要由误画减少带来，不能称为观察到更多真实墙。Map08 的 B/C 墙体分分别下降2.1520/14.0111，C组合拒绝直接采用。',
        'Map09 三候选真实墙命中分别增加18195/20097/21128像素，左上框选区域无人机帧数由64/8/0变为2769/2538/1268；实际录像可见三版都深入左上。B 在本轮 Map09 的地图分最高68.0871，但本轮只有单种子，也没有蓝队阶段。A3+RN7、B+RN7保留继续验证，C不作为通用替换推荐。','',
        '## 实际测试及保留的失败','',
        '首轮 tests_v1：13通过、2失败（46px斜通道过于乐观的通行期望）；保留XML。没有缩小安全半径强行通过。改为明确验证55px各方向与46px斜向保守拒绝。',
        '最终 tests_final：16项通过，真实计时见 verification.json。覆盖观测/未知、机身外接半径、整段路径、侧滑转角累计、前墙刹车、有限队友等待、任务交还、失效里程计。理想几何测试不替代真实物理。',
        '原生 v1/v2/v3 所有日志、结果保留；v1把越过终点但绕过25px通道的行为错记passed，v2修正为必须真正进入且未绕行。v1这个标签不作为通行证据。v3为最终代码小场景结果，5/5进程正常，全部零墙接触；50px理想、80px真实噪声通过，50px真实噪声未通过；25px拒绝并绕行，死胡同未穿墙。全部仅组件检查，不是官方地图成绩。','',
        f'![原生物理小场景]({(args.output.resolve()/"native_cases.png").as_posix()})','',
        '## 文件与复现','',
        '新增策略：src/swarm_rescue/solutions/my_drone_red_rn7.py；专用 test_red_rn7.py、red_rn7_native.py、red_rn7_batch.py、red_rn7_review.py、red_rn7_delivery.py 位于 solutions/devtools。旧文件保护校验和冻结源码审计见 verification.json。',
        '已运行命令（下列完整对局已完成，不要当作续跑命令再次启动）：','',
        '```powershell',"Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",
        '.\\.venv\\Scripts\\python.exe -B -m pytest src/swarm_rescue/solutions/devtools/test_red_rn7.py -q -p no:cacheprovider',
        '.\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.red_rn7_batch --output output/sf13/rn7_m89_once --maps Map08 Map09 --controllers rn7_a rn7_b rn7_c --rounds 1 --run','```','',
        '任务量6完整红局、0蓝局，Map08/09各三候选单轮；资源1CPU/4GB、5400步/180秒。当前阶段不再自动启动评测。完整蓝队难度及总分均未评测。']
    notes+=['','## 官方像素分解及资源','',
        '|地图/候选|真实墙命中像素|误画扣分像素|地图credit / penalty|Job峰值MB|进程秒|回调异常|控制器崩溃|时间触限|',
        '|---|---:|---:|---|---:|---:|---:|---|---|']
    for r in review['rows']:
        resource=r['resources']
        notes.append(f"|{r['map']}/{r['controller']}|{r['wall_hits']}|{r['charged_pixels']}|{r['map_credit']:.6f}/{r['map_penalty']:.6f}|{resource.get('peak_job_commit_bytes',0)/1e6:.1f}|{resource.get('process_seconds',0):.2f}|{r['callback_errors']}|{r['controller_crashed']}|{r['walltime_limited']}|")
    notes+=['','## 保护与身份核验','',
        f'原已登记文件SHA256检查 {checked} 个；变化列表：{changed}。冻结源文件验证通过；正式对局attempt共{verification["actual_real_attempts"]}个，未重试。图册{len(paths)}个内嵌图链接已逐一用图片解码验证。',
        f'冻结计划SHA256：`{verification["plan_sha256"]}`。red_rn7_delivery.py是冻结之后新增的离线交付工具，不属于比赛策略。',
        '原来每组的结果收据由 review_row 复核，原提交图用官方地图公式重新评分并与原分一致；没有改写旧分、旧录像或旧源文件。',
        '正式入口、共享定位/通信/评分/物理、蓝队 B7 未由本任务修改。未运行蓝队，不给蓝队强度背书。后续若继续，先针对过早用完预算和带噪窄门拒绝修正，再由用户决定新的多地图/多种子验证与完整蓝对局；本轮结果不得替代该验收。']
    with (args.output/'development.md').open('x',encoding='utf-8') as f:f.write('\n'.join(notes)+'\n')
    report=Path('复赛阶段_FS4_红队RN7窄路_Map08_Map09单轮实测.md')
    with report.open('x',encoding='utf-8') as f:f.write(text+'\n\n'+'\n'.join(notes)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
