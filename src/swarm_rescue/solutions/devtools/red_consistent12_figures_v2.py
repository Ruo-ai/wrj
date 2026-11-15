"""Layout-only RL12 atlas refresh; preserves all evaluation and prior figures."""
import argparse
from pathlib import Path
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools import red_consistent12_review as review


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    batch=Path('output/sf13/rl12_m89_once_20261115');ph=base.digest(batch/'plan.json')
    rows=[];assets={}
    for raw in base.read(batch/'results.json'):
        row,data=base.review_row(raw,ph);rows.append(row);assets[row['map'],row['controller']]=data
    maps=['Map08','Map09'];controllers=['rl11_lines','rl12_nav']
    for kind,label in [('mapping','提交建图'),('trajectories','实际完整轨迹'),('walls','官方墙体覆盖'),('bombs','实际炸弹位置')]:
        fig,axes=base.plt.subplots(2,2,figsize=(18,14),squeeze=False)
        fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.065,wspace=.06,hspace=.44)
        for y,m in enumerate(maps):
            for x,c in enumerate(controllers):
                row=next(r for r in rows if r['map']==m and r['controller']==c)
                review.panel(axes[y,x],kind,row,assets[m,c])
        fig.suptitle('RL11 与独立导航校正对照：'+label,fontproperties=review.FONT,fontsize=22,y=.985)
        note={'mapping':'绿：命中；蓝：漏墙；深红：收费误画；黄：免罚误画。',
              'trajectories':'轨迹不是雷达覆盖；灰：无GPS/罗盘；黄：无通信；颜色是录像槽位。',
              'walls':'绿：覆盖内墙；蓝：覆盖边界墙；灰：未覆盖；红：爆炸范围。',
              'bombs':'编号是实际落点产物顺序；圆是爆炸半径，重叠不等于有效墙面重复。'}[kind]
        fig.text(.5,.025,note+' 各图各版一次完整红队；蓝队及完整总分未评测。',ha='center',fontproperties=review.FONT,fontsize=13)
        base.save(fig,a.output/f'comparison_{kind}.png')
    doc=['# RL12完整四组图册（修正版式）','','仅修正标题间距，不更改任何数据、旧图和评测。',
         '2026-11-15；Map08 seed200806601，Map09 seed2477227837；各图各版一次完整红队，全部有效、10弹；蓝队难度与完整总分未评测。','',
         '|图|版|M|W|T|步|秒|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:doc.append(f"|{r['map']}|{review.NAMES[r['controller']]}|{r['map_score']:.4f}|{r['wall_score']:.4f}|{r['time_score']:.4f}|{r['red_steps']}|{r['red_walltime']:.3f}|")
    for kind in ['mapping','trajectories','walls','bombs']:
        doc+=['',f'![{kind}]({(a.output/f"comparison_{kind}.png").resolve().as_posix()})']
    for r in rows:
        doc+=['',f"{r['map']} / {r['controller']} / r1 / seed {r['red_seed']}",'','```powershell',
              "Set-Location -LiteralPath 'E:\\桌面\\gitee\\swarm-eod'",f".\\replay_map.cmd '{Path(r['attempt']).resolve()}'",'```']
    doc+=['','无本轮蓝录像。入口和通信专项图、源码收据、完整指标见原始审计目录rl12_m89_review_20261115及根目录RL12专属报告。']
    with (a.output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(doc)+'\n')
    print(str(a.output.resolve()))


if __name__=='__main__':main()
