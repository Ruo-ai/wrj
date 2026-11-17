"""New offline scientific figures from frozen RL18 audits; no policy mutations."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from swarm_rescue.solutions.devtools.red_recovery18_audit import read,load_npz,Metrics,ROI,EVENTS


COLORS={'rl11_lines':'#3477b7','rl17_repair':'#d08322','shadow':'#c33b49','odometry':'#888888'}
LABEL={'rl11_lines':'RL11','rl17_repair':'RL17-R','shadow':'RL18 shadow (rejected)','odometry':'Odometry'}


def save(fig,out,name):
    fig.savefig(out/name,dpi=145,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    stage=a.root/'stage_a_v2';shadow=a.root/'stage_b_v1'
    for zoom in (False,True):
        fig,axes=plt.subplots(2,2,figsize=(16,10))
        for i,name in enumerate(EVENTS):
            for j,c in enumerate(('rl11_lines','rl17_repair')):
                folder=stage/f'{name}_{c}';report=read(folder/'results.json')
                metric=Metrics(np.load(folder/'truth.npy'),ROI[name]);pred=np.load(folder/'pixels.npy')
                img=metric.rgb(pred)
                if zoom:img=img[metric.roi]
                axes[i,j].imshow(img);axes[i,j].axis('off');v=report['final'];q=report['request']
                axes[i,j].set_title(f'{name} | {LABEL[c]} | seed {q["seed"]} | r1\nM={v["map_score"]:.4f}; hits={v["hits"]}; charged={v["charged"]}'+
                    (f'\nROI {ROI[name]}: hits={v["roi_hits"]}; charged={v["roi_charged"]}' if zoom else ''),fontsize=10)
        fig.suptitle('NEW OFFLINE REPLAY — original submitted pixels reproduced exactly\n'+
                     ('Approximate screenshot regions (not exact original crop)' if zoom else 'Fixed recorded inputs and ACTUALLY received source versions'),fontsize=15)
        fig.text(.5,.01,'Green: true wall hit | Blue: missed surface | Red: charged error | Yellow: uncharged prediction | Gray: truth interior',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.03,1,.92));save(fig,a.output,'mapping_roi.png' if zoom else 'mapping_full.png')
    fig,axes=plt.subplots(4,4,figsize=(19,13));row=0
    for name,c in [('Map08','rl17_repair'),('Map09','rl11_lines')]:
        folder=stage/f'{name}_{c}';report=read(folder/'results.json');metric=Metrics(np.load(folder/'truth.npy'),ROI[name])
        for e in report['recovery']:
            for col,key in enumerate(('before','unanchored_commit','anchored_commit','after')):
                img=plt.imread(folder/f'd{e["owner"]}_t{e["step"]}_{key}.png')
                axes[row,col].imshow(img[metric.roi]);axes[row,col].axis('off');v=e['stages'][key]
                axes[row,col].set_title(f'{name} {LABEL[c]} d{e["owner"]} t{e["step"]} | {key}\nwhole-local hits={v["hits"]}; charged={v["charged"]}',fontsize=9)
            row+=1
    fig.suptitle('RECOVERY ATTRIBUTION — ROI image; counts refer to WHOLE SINGLE-DRONE map\nUnanchored commit: diagnostic counterfactual (.8 weight) | r1 | Map08 seed 200806601; Map09 seed 2477227837',fontsize=13)
    fig.text(.5,.01,'Each row keeps the same view. Green hits; blue misses; red charged; yellow uncharged. No new flight.',ha='center')
    fig.tight_layout(rect=(0,.025,1,.94));save(fig,a.output,'recovery_layers.png')
    fig,axes=plt.subplots(2,2,figsize=(16,9))
    for i,name in enumerate(EVENTS):
        for j,c in enumerate(('rl11_lines','rl17_repair')):
            folder=stage/f'{name}_{c}';z=load_npz(folder/'maps_poses.npz');truth=np.load(folder/'truth.npy')
            count=np.repeat(np.repeat(z['source_count'],3,0),3,1)[:truth.shape[0],:truth.shape[1]]
            colors=['#f7f7f7','#67a9cf','#2166ac','#fdb863','#e66101','#b2182b']
            axes[i,j].imshow(count,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,6),6))
            axes[i,j].axis('off');axes[i,j].set_title(f'{name} {LABEL[c]} | r1 | original source versions')
    fig.suptitle('ACTUAL RECEIVED WALL-SOURCE OVERLAP — one count per original source\n0 is NOT observed free | Map08 seed 200806601; Map09 seed 2477227837 | r1\nReliable peer-free evidence was NOT transmitted.',fontsize=13)
    fig.text(.5,.02,'White 0 | Light blue 1 | Blue 2 | Yellow 3 | Orange 4 | Red 5 sources. Correlated frames/relays add no votes.',ha='center')
    fig.tight_layout(rect=(0,.04,1,.91));save(fig,a.output,'source_overlap.png')
    fig,axes=plt.subplots(4,2,figsize=(16,13));row=0
    for name in EVENTS:
        z=load_npz(shadow/name/'traces.npz')
        for owner,end in EVENTS[name]:
            missing=z[f'missing_{owner}'];start=int(np.flatnonzero(~missing[:end-1])[-1]+1)
            ticks=np.arange(start+1,end)
            for c in COLORS:
                axes[row,0].plot(ticks,z[f'{c}_position_{owner}'][ticks-1],label=LABEL[c],color=COLORS[c],lw=1)
                axes[row,1].plot(ticks,z[f'{c}_heading_{owner}'][ticks-1],label=LABEL[c],color=COLORS[c],lw=1)
            for j in range(2):
                axes[row,j].set_title(f'{name} d{owner} | outage {start+1}..{end-1} | r1');axes[row,j].grid(alpha=.2)
                axes[row,j].set_ylabel('Position error / px' if j==0 else 'Absolute heading error / deg')
                axes[row,j].set_xlabel('Recorded control step')
            row+=1
    axes[0,0].legend(fontsize=9);fig.suptitle('RL18-B OFFLINE SHADOW — fixed legal tape; truth only for these plots\nMap08 seed 200806601; Map09 seed 2477227837 | r1\nCumulative pose regressed overall; no map/control integration',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,a.output,'shadow_failures.png')
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for i,(name,c) in enumerate([('Map08','rl17_repair'),('Map09','rl11_lines')]):
        d=read(stage/f'{name}_{c}/results.json')
        for j,(owner,end) in enumerate(EVENTS[name]):
            tr=next(r['rows'] for r in d['local_growth'] if r['owner']==owner)
            x=[v['step'] for v in tr];y=[v['charged'] for v in tr]
            axes[i,j].plot(x,y,label='Single-drone charged pixels',color='#c33b49')
            axes[i,j].axvline(end,color='black',ls='--',label='GPS + compass return')
            axes[i,j].set_title(f'{name} {LABEL[c]} d{owner} | r1');axes[i,j].set_xlabel('Step');axes[i,j].grid(alpha=.2)
            axes[i,j].legend(fontsize=9)
    fig.suptitle('ERROR EXISTS BEFORE RECOVERY — exact replay checkpoints; connecting lines are visual only\nMap08 seed 200806601; Map09 seed 2477227837 | r1',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,a.output,'growth_before_recovery.png')


if __name__=='__main__':main()
