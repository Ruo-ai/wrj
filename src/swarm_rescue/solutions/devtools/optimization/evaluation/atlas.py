"""Shared offline round atlas; raw scores and missing observations remain explicit."""
from pathlib import Path
import json

from .runner import read, REPO, phase_valid


def optional(path):
    return read(path) if path.is_file() else {}


def fmt(value):
    if value is None:
        return '未知/未产生'
    return f'{value:.4f}' if isinstance(value, float) else str(value)


def round_section(pair, output, layout_image=None, historical=False):
    """Never infer bomb ownership, collection events, or completion from static images."""
    from .report import layout_plot
    red = Path(pair['red_run'])
    blue = Path(pair['blue_run']) if pair.get('blue_run') else None
    r, b = dict(pair['red']), dict(pair.get('blue') or {})
    for result, run in [(r,red),(b,blue)]:
        if run and (run/'run.log').is_file():
            result['engine_callback_errors']=max(result.get('engine_callback_errors',0),
                (run/'run.log').read_text(encoding='utf-8',errors='replace').count('Exception ignored from cffi callback'))
    rq = optional(red/'request.json')
    bq = optional(blue/'request.json') if blue else {}
    rr = optional(red/'resources.json')
    br = optional(blue/'resources.json') if blue else {}
    transport=optional(blue/'telemetry.json').get('transport',{}) if blue else {}
    pickup=transport.get('pickups');verified=transport.get('verified_disposals')
    duration=transport.get('disposed_transport_steps',[])
    transport_text=(f"{verified}/{pickup}（持有期间确认） / {sum(duration)/len(duration):.1f}步" if duration else
                    f"{verified}/{pickup}（持有期间确认） / 暂无成功样本") if transport else '未知（无事件记录）'
    p, full = r.get('phase1') or {}, b.get('phase1') or {}
    valid = bool(phase_valid(r, 'red') and phase_valid(b, 'blue') and not r.get('smoke'))
    label = f"{pair['name']} / {pair['controller']} / {b.get('blue_controller_class', 'unknown blue')} / r{pair['round']}"
    lines = ['', '### '+label, '',
        ('历史数据离线重绘；不是本轮新成绩。' if historical else '本批结果；短测不计有效成绩。'), '',
        f"红 attempt：`{red}`；蓝 attempt：`{blue}`。",
        f"红种子：{rq.get('seed', r.get('seed'))}；蓝种子：{bq.get('seed', b.get('seed'))}。",
        f"红类：{rq.get('controller_class', r.get('controller_class'))}；蓝类：{b.get('blue_controller_class', '未知')}。",
        f"实际条件：红 `{json.dumps(r.get('actual_conditions'), ensure_ascii=False)}`；蓝 `{json.dumps(b.get('actual_conditions'), ensure_ascii=False)}`。",
        f"红计划 SHA256：{rq.get('plan_sha256', '未知')}；蓝计划 SHA256：{bq.get('plan_sha256', '未知')}；布局 SHA256：{b.get('layout_sha256', '未知')}。", '',
        '|地图分|墙体分|时间分|原始难度|原始总分|有效总分|红步/秒|蓝步/秒|',
        '|---:|---:|---:|---:|---:|---:|---|---|',
        '|'+'|'.join(fmt(v) for v in [p.get('score_map'), p.get('score_wall'), p.get('score_time'),
            full.get('score_bomb'), full.get('round_score'), full.get('round_score') if valid else None])+
        f"|{r.get('elapsed_steps')}/{fmt(r.get('elapsed_walltime'))}|{b.get('elapsed_steps')}/{fmt(b.get('elapsed_walltime'))}|",
        f"有效完整成绩：{valid}；红状态={r.get('status')}，蓝状态={b.get('status')}；"
        f"触限(红/蓝)={r.get('walltime_limited')}/{b.get('walltime_limited')}；"
        f"回调异常(红/蓝)={r.get('engine_callback_errors', '未知')}/{b.get('engine_callback_errors', '未知')}；"
        f"控制器异常(红/蓝)={r.get('controller_crashed')}/{b.get('controller_crashed')}。", '',
        '|蓝分|处置|健康指标（官方原值）|发现/抓取次数|运输成功率/耗时|红/蓝峰值Job字节|红/蓝进程秒|',
        '|---:|---:|---:|---|---|---|---|',
        f"|{fmt(b.get('blue_score'))}|{fmt(b.get('disposed_count'))}|{fmt(b.get('blue_health_score'))}|{fmt(transport.get('discovered'))}/{fmt(pickup)}|{transport_text}|"
        f"{fmt(rr.get('peak_job_commit_bytes'))}/{fmt(br.get('peak_job_commit_bytes'))}|{fmt(rr.get('process_seconds'))}/{fmt(br.get('process_seconds'))}|", '']
    for title, pattern in [('建图/提交差异（不是完整探索轨迹）', '*_exploration_diff.png'),
                           ('墙体覆盖（原始覆盖率不同于归一墙体分）', '*_wall_destruction.png')]:
        images = sorted((red/'artifacts').glob(pattern))
        lines += [title, '']
        lines += [f'![{label} {title}](<{img.resolve().as_posix()}>)' for img in images] or [
            '暂无此图：本 attempt 未产生相应官方图件；短测/失败不补造建图或覆盖结果。']
        lines += ['']
    if layout_image is None:
        figures = output/'figures'
        figures.mkdir(exist_ok=True)
        layout_image = figures/f"{pair['name']}_{pair['controller']}_r{pair['round']:02d}_{red.name}.png"
        if not layout_image.exists():
            layout_plot(pair, layout_image)
    lines += ['实际初始落弹位置（编号仅为产物顺序，不推测所属机号；蓝队运输后的位置另列）', '',
              f'![{label} 实际炸弹位置](<{layout_image.resolve().as_posix()}>)', '']
    for team,run in [('红',red),('蓝',blue)]:
        if run:
            sample=optional(run/'telemetry.json').get('pose_samples',[])
            if sample:
                plot=output/'figures'/f"{pair['name']}_{pair['controller']}_r{pair['round']:02d}_{'red' if team=='红' else 'blue'}_pose.png"
                pose_plot(sample,plot)
                lines += [team+'队采样轨迹及定位误差（真值仅由离线工具记录；稀疏连线不表示逐步路径）','',
                    f'![{label} {team}定位审计](<{plot.resolve().as_posix()}>)','']
    for phase, run in [('红', red), ('蓝', blue)]:
        if run is None:
            lines += [phase+'阶段未运行，没有可回放录像。', '']
        else:
            lines += [phase+'队完整回放命令：', '', '```powershell',
                "Set-Location -LiteralPath '"+str(REPO).replace("'", "''")+"'",
                ".\\replay_map.cmd '"+str(run).replace("'", "''")+"'", '```', '']
    return lines


def pose_plot(samples,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    path.parent.mkdir(exist_ok=True)
    fig,(ax,err)=plt.subplots(1,2,figsize=(11,4.5))
    for identifier in sorted({s['drone'] for s in samples}):
        rows=[s for s in samples if s['drone']==identifier]
        truth=np.array([s['truth'] for s in rows]);estimate=np.array([s['estimate'] for s in rows])
        line,=ax.plot(truth[:,0],truth[:,1],'-o',ms=2,label=str(identifier))
        ax.plot(estimate[:,0],estimate[:,1],'--',color=line.get_color())
        err.plot([s['step'] for s in rows],[s['error'] for s in rows],color=line.get_color())
    ax.set(title='Offline sampled poses: truth solid / estimate dashed',xlabel='world x',ylabel='world y',aspect='equal')
    ax.legend(title='Drone',fontsize=6,ncol=2)
    err.set(title='Offline position error',xlabel='Step',ylabel='Error / px');err.grid(alpha=.25)
    fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)
