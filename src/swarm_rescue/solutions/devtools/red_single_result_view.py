"""Render one retained red result, without running a match or editing artifacts."""
import argparse
from pathlib import Path
import numpy as np
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--map',required=True)
    parser.add_argument('--controller',required=True)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=[r for r in base.read(args.batch/'results.json') if r['map']==args.map
         and r['controller']==args.controller and r['round']==args.round]
    if len(raw)!=1:raise ValueError('Require exactly one retained result')
    row,data=base.review_row(raw[0],base.digest(args.batch/'plan.json'))
    args.output.mkdir(parents=True,exist_ok=False)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    w,h=data['w'],data['h'];trace=data['trajectories']
    for kind in ('trajectories','mapping'):
        fig,ax=base.plt.subplots(figsize=(12,8))
        if kind=='trajectories':
            ax.imshow(data['truth'],cmap='Greys',vmin=0,vmax=1)
            base.areas(ax,data['spec'])
            for slot in range(trace.shape[1]):
                active=trace['active'][:,slot]
                x=np.where(active,trace['x'][:,slot]+w/2,np.nan)
                y=np.where(active,h/2-trace['y'][:,slot],np.nan)
                jump=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40]
                x[jump]=np.nan;y[jump]=np.nan
                ax.plot(x,y,lw=.75,alpha=.8)
            title='实际探索轨迹';note='彩线为实际机体轨迹，不等于雷达覆盖；灰色无GPS，黄色无通信。'
        else:
            ax.imshow(data['merged'],interpolation='nearest')
            title='提交地图与真值对照';note='绿色命中；蓝色漏墙；深红误画扣分；黄色误画免扣；灰色无分内部。'
        ax.set(xlim=(0,w),ylim=(h,0),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f"{args.map} / 旧 RN7-B / {title}\n地图 {row['map_score']:.4f} | 墙体 {row['wall_score']:.4f} | {row['red_steps']} 步",fontproperties=font,fontsize=15)
        fig.text(.5,.035,note,ha='center',fontproperties=font,fontsize=11)
        fig.text(.5,.009,f"历史实测 r{args.round}，seed {row['red_seed']}；10枚弹，红阶段有效，蓝阶段未评测。",ha='center',fontproperties=font,fontsize=10)
        fig.subplots_adjust(left=.02,right=.98,top=.89,bottom=.09)
        base.save(fig,args.output/f'{args.map}_{args.controller}_{kind}.png')
    print(args.output.resolve())


if __name__=='__main__':main()
