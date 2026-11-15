"""RL13 overview layout repair, new output only; no match data is changed."""
import hashlib
import json
from pathlib import Path
import numpy as np
from swarm_rescue.solutions.devtools import red_local13_finalize as source


def main():
    out=source.ROOT/'output/sf13/rl13_m89_figures_20261115'
    out.mkdir(parents=True,exist_ok=False)
    report_before=source.REPORT.read_bytes()
    rows=source.base.read(source.OUT/'summary.json')['rows']
    data={(r['map'],r['controller']):source.assets(r) for r in rows}
    source.panels.NAMES=source.NAMES
    for kind,label in [('exploration','雷达探索范围估计'),('mapping','完整提交建图'),('trajectories','完整实际轨迹'),('walls','官方墙体覆盖'),('bombs','实际炸弹位置')]:
        fig,axes=source.base.plt.subplots(4,2,figsize=(18,29),squeeze=False)
        fig.subplots_adjust(left=.015,right=.985,top=.90,bottom=.045,wspace=.06,hspace=.48)
        for mindex,m in enumerate(['Map08','Map09']):
            for cindex,c in enumerate(source.CONTROLLERS):
                ax=axes[mindex*2+cindex//2,cindex%2]
                row=next(r for r in rows if r['map']==m and r['controller']==c);d=data[m,c]
                source.panels.panel(ax,'mapping' if kind=='exploration' else kind,row,d)
                if kind=='exploration':
                    pixels=np.ones((*d['truth'].shape,3))*.94;pixels[d['visible']]=[.57,.85,.73]
                    pixels[d['truth'].astype(bool)]=[.12,.12,.12]
                    ax.imshow(pixels);source.base.areas(ax,d['spec'])
        fig.suptitle('Map08 / Map09 · 四版各一次完整红队：'+label,fontproperties=source.lines.FONT,fontsize=23,y=.982)
        subtitle=fig.text(.5,.958,'上半部 Map08，下半部 Map09；每图依次：RL11 / 入口对齐 / 分段运动 / 局部图。',
                          ha='center',fontproperties=source.lines.FONT,fontsize=13)
        note={'exploration':'绿：每8步/4px雷达可见区采样估计，含射线间插值；灰：无GPS/罗盘；黄：无通信。',
              'mapping':'绿：命中；蓝：漏墙；深红：收费误画；黄：免罚误画。',
              'trajectories':'颜色为录像槽位。灰：无GPS/罗盘；黄：无通信。轨迹不等于雷达观察覆盖。',
              'walls':'绿：覆盖内墙；蓝：覆盖边界墙；灰：未覆盖；红：爆炸范围。',
              'bombs':'编号是实际落点产物顺序，圆是爆炸半径；圆重叠不等于墙面收益重复。'}[kind]
        fig.text(.5,.023,note,ha='center',fontproperties=source.lines.FONT,fontsize=13)
        fig.text(.5,.009,'八局均完整有效、各10弹；蓝队难度与完整比赛总分未评测。',ha='center',fontproperties=source.lines.FONT,fontsize=12)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        assert all(not ax.title.get_window_extent(renderer).overlaps(subtitle.get_window_extent(renderer)) for ax in axes.ravel())
        source.base.save(fig,out/f'overview_{kind}.png')
    original=report_before.decode('utf-8');updated=original
    for kind in ['exploration','mapping','trajectories','walls','bombs']:
        updated=updated.replace((source.OUT/f'overview_{kind}.png').as_posix(),(out/f'overview_{kind}.png').as_posix())
    updated+='\n图件校验补记：总览另存至rl13_m89_figures_20261115修正标题间距；本报告引用修正版，初版图及全部原始评测不改。新增离线工具red_local13_figures_v2.py。\n'
    assert source.REPORT.read_bytes()==report_before,'Report changed concurrently; preserve it'
    source.REPORT.write_text(updated,encoding='utf-8')
    with (out/'layout_audit.json').open('x',encoding='utf-8') as f:
        json.dump(dict(title_subtitle_overlap=False,report_sha256_before=hashlib.sha256(report_before).hexdigest(),
            report_sha256_after=hashlib.sha256(source.REPORT.read_bytes()).hexdigest(),data_changed=False),f,indent=2)
    with (out/'report.md').open('x',encoding='utf-8') as f:
        f.write('# RL13八局总览（版式修正）\n\n只调整图件版式，数据与原始图均保留。\n\n')
        for kind in ['exploration','mapping','trajectories','walls','bombs']:
            f.write(f'![{kind}]({(out/f"overview_{kind}.png").as_posix()})\n\n')
        f.write(f'[逐候选对照、全部分项和回放]({(source.OUT/"report.md").as_posix()})\n')
    print(str(out))


if __name__=='__main__':main()
