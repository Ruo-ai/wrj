"""Read-only analysis of R8/R9 results; outputs must be new directories.

Engine hit endpoints are offline evidence, never passed to a policy. They are
sampled every four steps and cannot reconstruct full sensor observation area.
"""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from matplotlib.patches import Patch, Rectangle
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools.red_rn7_review import REGIONS, regions
from swarm_rescue.solutions.devtools.optimization.evaluation.runner import validate_sources


def dump(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args();args.output.mkdir(parents=True, exist_ok=False)
    rows=[];assets={};new=[]
    batches=('rn7_m89_once','r8_m89_single','r9_map08_single')
    for batch in batches:
        root=Path('output/sf13')/batch
        plan=base.read(root/'plan.json');validate_sources(plan['source_sha256'])
        sha=base.digest(root/'plan.json')
        for raw in base.read(root/'results.json'):
            if batch==batches[0] and raw['controller']!='rn7_b':continue
            row,data=base.review_row(raw,sha);run=Path(row['attempt'])
            row['region_drone_frames']=regions(data,row['map'])
            rows.append(row);assets[(row['map'],row['controller'])]=data
            if batch==batches[0]:continue
            new.append(row)
            with np.load(run/'observed_endpoints.npz') as arrays:
                observed=np.logical_or.reduce([arrays[key] for key in arrays.files])
            pred=np.asarray(base.read(next((run/'artifacts').glob('*_exploration.json')))['grid'],bool)
            surface,_=base.MapExplorationScorer().split_regions(data['truth'])
            endpoint_surface=observed&surface.astype(bool)
            data.update(observed=observed,pred=pred,endpoint_surface=endpoint_surface)
            metrics=dict(sampled_surface_pixels=int(endpoint_surface.sum()),
                sampled_surface_present=int((endpoint_surface&pred).sum()),
                sampled_surface_absent=int((endpoint_surface&~pred).sum()))
            metrics['regions']={}
            for name,(l,t,r,b) in REGIONS[row['map']].items():
                seen=endpoint_surface[t:b,l:r];image=pred[t:b,l:r]
                metrics['regions'][name]=dict(sampled_surface_pixels=int(seen.sum()),
                    sampled_surface_present=int((seen&image).sum()),sampled_surface_absent=int((seen&~image).sum()))
            row['offline_endpoint_metrics']=metrics
            row['diagnostic_errors']=base.read(run/'r8_evidence.json')['diagnostic_errors']
    # Three panels intentionally distinguish physical path, sampled laser hit
    # positions, and the final submitted scoring image. No coverage estimate.
    for row in new:
        data=assets[(row['map'],row['controller'])];w,h=data['w'],data['h']
        fig,axes=base.plt.subplots(1,3,figsize=(18,5.4))
        fig.subplots_adjust(left=.02,right=.99,top=.84,bottom=.15,wspace=.04)
        axes[0].imshow(data['truth'],cmap='Greys',vmin=0,vmax=1)
        base.areas(axes[0],data['spec'])
        trace=data['trajectories']
        for slot in range(trace.shape[1]):
            good=trace['active'][:,slot]
            x=np.where(good,trace['x'][:,slot]+w/2,np.nan)
            y=np.where(good,h/2-trace['y'][:,slot],np.nan)
            jump=np.r_[False,np.hypot(np.diff(x),np.diff(y))>40]
            x[jump]=np.nan;y[jump]=np.nan;axes[0].plot(x,y,lw=.5,alpha=.7)
        axes[0].set_title('Physical paths (not sensor coverage)')
        axes[1].imshow(data['truth'],cmap='Greys',vmin=0,vmax=1)
        hit=data['endpoint_surface'];present=hit&data['pred'];missing=hit&~data['pred']
        for mask,colour in ((present,'seagreen'),(missing,'crimson')):
            yy,xx=np.nonzero(mask);axes[1].scatter(xx,yy,s=.7,c=colour,linewidths=0)
        m=row['offline_endpoint_metrics']
        axes[1].set_title(f"Sampled true surface endpoints\n{m['sampled_surface_present']} present / {m['sampled_surface_absent']} absent")
        axes[2].imshow(data['merged'],interpolation='nearest')
        axes[2].set_title(f"Submitted map | M {row['map_score']:.4f}")
        for ax in axes:
            ax.set(xlim=(0,w),ylim=(h,0),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"{row['map']} / {row['controller']} / r1 / seed {row['red_seed']}")
        fig.legend(handles=[Patch(color='seagreen',label='Sampled endpoint also submitted'),
            Patch(color='crimson',label='Sampled endpoint absent from final map')],loc='lower center',ncol=2)
        fig.text(.5,.07,'Stride 4, exact surface pixels only; missing final pixels do not identify the cause of loss.',ha='center',fontsize=10)
        base.save(fig,args.output/f"{row['map']}_{row['controller']}_observation.png")
    # Detailed view of the requested Map08 lower entrance.
    selected=['rn7_b','r8_n','e9_n'];gate_rows=[]
    fig,axes=base.plt.subplots(1,3,figsize=(16,5))
    for ax,key in zip(axes,selected):
        data=assets[('Map08',key)];w,h=data['w'],data['h'];trace=data['trajectories']
        ax.imshow(data['truth'],cmap='Greys',vmin=0,vmax=1)
        info=dict(controller=key,region_frames=0,first_frame=None,last_frame=None,max_y=None)
        for slot in range(trace.shape[1]):
            x=trace['x'][:,slot]+w/2;y=h/2-trace['y'][:,slot];active=trace['active'][:,slot]
            ax.plot(np.where(active,x,np.nan),np.where(active,y,np.nan),lw=.9,alpha=.8)
            good=active&(x>=956)&(x<1654)&(y>=1027)&(y<1116)
            indices=np.flatnonzero(good)
            if len(indices):
                info['region_frames']+=len(indices)
                info['first_frame']=int(indices[0]) if info['first_frame'] is None else min(info['first_frame'],int(indices[0]))
                info['last_frame']=int(indices[-1]) if info['last_frame'] is None else max(info['last_frame'],int(indices[-1]))
                info['max_y']=float(y[good].max()) if info['max_y'] is None else max(info['max_y'],float(y[good].max()))
        gate_rows.append(info)
        ax.add_patch(Rectangle((956,1027),698,89,fill=False,color='magenta',ls='--'))
        ax.plot([1301,1356],[1024,1024],color='cyan',lw=3)
        ax.set(xlim=(1180,1420),ylim=(1116,950),aspect='equal')
        ax.set_title(f"{key} | lower region {info['region_frames']} drone-frames")
        ax.set_xlabel('Canvas x (px)');ax.set_ylabel('Canvas y (px)')
    fig.suptitle('Map08 lower entrance | cyan: 56 px physical opening | magenta: diagnostic region')
    fig.tight_layout();base.save(fig,args.output/'Map08_entrance_detail.png')
    dump(args.output/'summary.json',dict(rows=rows,gate_rows=gate_rows,blue_evaluated=False))
    tests=[]
    for name in ('tests_final.xml','tests_r9_final.xml'):
        root=ET.parse(Path('output/sf13/r8_dev')/name).getroot()
        tests.append(dict(file=name,**root.find('testsuite').attrib))
    protected=base.read('output/sf13/r8_dev/protected_before.json')
    changed=[p['path'] for p in protected if not Path(p['path']).exists() or base.digest(p['path']).lower()!=p['sha256'].lower()]
    pictures=[]
    for root in (Path('output/sf13/r8_m89_review'),Path('output/sf13/r9_map08_review'),args.output):
        for path in root.glob('*.png'):
            with Image.open(path) as image:image.verify()
            pictures.append(str(path))
    tracked_diff=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()
    verify=dict(protected_count=len(protected),protected_changed=changed,tracked_diff=tracked_diff,
        actual_new_red_attempts=len(new),blue_phases=0,source_receipts_verified=True,
        official_map_scores_reproduced=True,tests=tests,images_verified=pictures,
        errors=[r['diagnostic_errors'] for r in new])
    dump(args.output/'verification.json',verify)
    lines=['# 本轮探索、实际射线命中与提交地图分开核对','',
        '全部图来自已有六场新红阶段结果；本工具不启动仿真。每4步采样实际射线端点，是稀疏观察证据，不是完整传感器覆盖率。缺失的最终像素可能由定位、量化、擦除、阈值或传输合并造成，此数据不能把原因逐项分解。','',
        '|地图/候选|采样到的真实表面像素|最终提交命中|最终提交缺失|','|---|---:|---:|---:|']
    for row in new:
        m=row['offline_endpoint_metrics']
        lines.append(f"|{row['map']}/{row['controller']}|{m['sampled_surface_pixels']}|{m['sampled_surface_present']}|{m['sampled_surface_absent']}|")
    lines+=['','Map08入口细节：89是无人机帧数，不是89次进入；进入底部附近不等于整个底部已探索。',
        '',f'![入口]({(args.output.resolve()/"Map08_entrance_detail.png").as_posix()})']
    for row in new:
        name=f"{row['map']}_{row['controller']}_observation.png"
        lines+=['',f'![{name}]({(args.output.resolve()/name).as_posix()})']
    with (args.output/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(gate_rows=gate_rows,metrics=[dict(map=r['map'],controller=r['controller'],**r['offline_endpoint_metrics']) for r in new],
        protected_changed=changed,images=len(pictures)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
