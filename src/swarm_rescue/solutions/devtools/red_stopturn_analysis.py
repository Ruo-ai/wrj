"""Offline figures from retained native S6 fixtures; never used by controllers."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch


def background(ax,arrays):
    ax.imshow(arrays['truth'],cmap='Greys',vmin=0,vmax=1,extent=(-400,400,-300,300))
    ax.add_patch(Rectangle((-160,-230),300,460,facecolor='gray',alpha=.12,edgecolor='gray'))
    ax.set(xlim=(-400,400),ylim=(-300,300),xlabel='World x (px)',ylabel='World y (px)')
    ax.set_aspect('equal')


def save(fig,path):
    fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    lines=['# S6 原生微场景完整图册','','原生物理/真实NoGpsZone；A为基线运动组件，并非完整A3探索器。',
        '非正式比赛：地图数值是官方地图公式在合成场景上的分量×100，墙体/时间/难度及总分未评测。',
        '观察量仅为引擎实际激光端点命中的墙像素下界，不是整面墙可见率；阴影表示离线特殊区。',
        '所有场景未配置布弹任务、出生区或处置区，不能评价放弹收益。没有蓝队阶段。','',
        '|场景/组/噪声|流程无异常|路线完成|步数|失效步|导航/地图位置误差px|公式地图量|墙命中/扣分像素|碰撞|秒|',
        '|---|---|---|---:|---:|---|---:|---|---:|---:|']
    cases=[]
    for path in sorted(args.native.glob('*.json')):
        if path.name=='results.json':continue
        row=json.loads(path.read_text(encoding='utf-8'))
        if 'trace' not in row or not path.with_suffix('.npz').is_file():continue
        cases.append((path.stem,row,np.load(path.with_suffix('.npz'))))
        mean=lambda key:'未覆盖' if row[key] is None else f'{row[key]:.3f}'
        score=row['map_formula']
        lines.append(f"|{path.stem}|是|{row['complete']}|{row['steps']}|{row['outage_steps']}|"
            f"{mean('outage_nav_mean')}/{mean('outage_mapper_mean')}|{100*score['s_map']:.4f}|"
            f"{score['wall_hits']}/{score['charged_pixels']}|{row['collision_contacts']}|{row['seconds']:.3f}|")
    for name,row,arrays in cases:
        xy=np.array([r['true_position'] for r in row['trace']]);truth=arrays['truth'].astype(bool)
        pred=arrays['prediction'].astype(bool)
        rgb=np.ones((*truth.shape,3));rgb[truth]=(.4,.4,.4)
        rgb[pred&~truth]=(1.,.4,.25);rgb[pred&truth]=(.15,.7,.4)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        background(axes[0],arrays);axes[0].plot(xy[:,0],xy[:,1],color='royalblue',lw=1.5)
        axes[0].set_title(f'{name}: actual trajectory')
        axes[1].imshow(rgb,extent=(-400,400,-300,300));axes[1].set_title('A3 submitted-map component')
        axes[1].set(xlim=(-400,400),ylim=(-300,300));axes[1].legend(handles=[Patch(color='gray',label='Missed truth'),
            Patch(color=(.15,.7,.4),label='Exact hit'),Patch(color=(1,.4,.25),label='False drawing (may be uncharged)')],loc='upper right',fontsize=8)
        save(fig,args.output/(name+'_mapping.png'))
        fig,ax=plt.subplots(figsize=(8,5));background(ax,arrays)
        observed=arrays['observed_endpoints'];y,x=np.nonzero(observed)
        ax.scatter(x-400,300-y,s=1,c='tab:green',label='Actual sampled ray endpoints on walls')
        ax.set_title(f'{name}: walls / observed endpoint subset\nBlast coverage not evaluated (no deployment phase)')
        ax.legend(fontsize=8);save(fig,args.output/(name+'_walls.png'))
        fig,ax=plt.subplots(figsize=(8,5));background(ax,arrays)
        ax.text(0,0,'No bomb deployment phase\nNo actual bomb positions / blast circles',ha='center',va='center',bbox=dict(facecolor='white',alpha=.9))
        ax.set_title(name+': placement not evaluated');save(fig,args.output/(name+'_bombs.png'))
        lines += ['',f'## {name}', '',
            f"seed=742；原生噪声={row['noise']}；路线完成={row['complete']}。D与完整官方总分未评测。",
            f"![探索与建图]({name}_mapping.png)",f"![墙体与实际采样端点]({name}_walls.png)",
            f"![布弹未评测]({name}_bombs.png)",
            '原始每步轨迹、命令、里程计、状态、真实/估计位置可在同名JSON回放分析；不是GUI比赛录像。']
    selected=[case for case in cases if case[0].startswith('corner_') and case[1]['noise']]
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for col,(name,row,arrays) in enumerate(selected):
        ax=axes[0,col];background(ax,arrays);trace=row['trace']
        for key,color,label in [('true_position','royalblue','True path'),('nav_position','darkorange','Navigation estimate'),('map_position','purple','Mapper estimate')]:
            xy=np.array([r[key] for r in trace]);ax.plot(xy[:,0],xy[:,1],color=color,lw=1,label=label)
        ax.set_title(f"{row['group']} | {row['steps']} steps | fixture map {100*row['map_formula']['s_map']:.2f}")
        ax.legend(fontsize=8)
        ax=axes[1,col];t=[r['step'] for r in trace]
        for key,color,label in [('nav_error','darkorange','Navigation'),('map_error','purple','Mapper'),('position_error','teal','Private odometry')]:
            ax.plot(t,[r[key] for r in trace],color=color,lw=1,label=label)
        ax.fill_between(t,0,40,where=[not all(r['absolute']) for r in trace],color='gray',alpha=.12)
        ax.set(xlim=(0,420),ylim=(0,40),xlabel='Control step',ylabel='Position error (px)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Native corner fixture, seed 742, real sensor noise — not official match scores')
    save(fig,args.output/'comparison.png')
    lines+=['','![真实噪声A/B/C对比](comparison.png)']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
