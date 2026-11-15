"""Offline final evidence: actual source-map loss, retained negative ablations."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base
from swarm_rescue.solutions.devtools.red_lines11_review import FONT
from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper


def overlay(truth,pred):
    image=np.ones((*truth.shape,3))*.97
    image[truth.astype(bool)]=[.45,.6,.75]
    image[pred.astype(bool)&~truth.astype(bool)]=[.7,.15,.2]
    image[pred.astype(bool)&truth.astype(bool)]=[.2,.65,.25]
    return image


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    batch=Path('output/sf13/rl12_m89_once_20261115');review=base.read('output/sf13/rl12_m89_review_20261115/summary.json')
    raw=next(r for r in base.read(batch/'results.json') if r['map']=='Map09' and r['controller']=='rl12_nav')
    row,data=base.review_row(raw,base.digest(batch/'plan.json'));run=Path(row['attempt']);h,w=data['h'],data['w']
    doc=base.read(run/'exploration_diagnostics.json');submission=doc['submissions'][0]
    with np.load(run/'exploration_stages.npz') as f:
        local=f['local_final_pixels_2'];received=SubmissionMapper((w,h)).to_pixels(f[f"submitter_{submission['drone']}_source_2"])
    actual=np.asarray(base.read(next((run/'artifacts').glob('*_exploration.json')))['grid'],np.uint8)
    fig,axes=base.plt.subplots(1,3,figsize=(18,7.5))
    titles=['2号机实际最终本机地图','提交者收到的2号机旧图（步2280）','真实最终全队提交图（M 50.36）']
    for ax,pred,title in zip(axes,[local,received,actual],titles):
        ax.imshow(overlay(data['truth'],pred));ax.set(xlim=(120,1000),ylim=(600,0),aspect='equal');ax.axis('off')
        ax.set_title(title,fontproperties=FONT,fontsize=14)
    fig.suptitle('Map09：进入后采到的墙面没有全部送达提交者',fontproperties=FONT,fontsize=19)
    fig.text(.5,.025,'绿：与真实墙重合；蓝：真实墙未画；红：画在真实墙外（本图不区分是否扣分）。图来自实际本机/收到快照/提交产物。',ha='center',fontproperties=FONT,fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.93));base.save(fig,a.output/'Map09_source_delivery.png')
    rows=[];fig,axes=base.plt.subplots(2,4,figsize=(24,12))
    for y,name in enumerate(['Map08','Map09']):
        d=Path('output/sf13/rl12_dev_20261115')
        first=base.read(d/f'replay_{name}_v2/results.json');second=base.read(d/f'replay_{name}_stable/results.json')
        run=Path(first['source']);result=base.read(run/'result.json');h,w=result['expected_shape']
        png=np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
        truth=np.all(png[base.COMPARISON_HEADER_HEIGHT:base.COMPARISON_HEADER_HEIGHT+h,:w]==40,axis=2).astype(np.uint8)
        choices=[(first,0,'RL11 原地图器',d/f'replay_{name}_v2/rl11_pixels.npy'),
                 (first,1,'短片段回溯（未采用）',d/f'replay_{name}_v2/rl12_lag_pixels.npy'),
                 (first,2,'入区前持续方向校正（未采用）',d/f'replay_{name}_v2/rl12_heading_pixels.npy'),
                 (second,1,'复用A5恢复（未采用）',d/f'replay_{name}_stable/rl12_stable_pixels.npy')]
        for x,(doc,ix,title,path) in enumerate(choices):
            r=doc['records'][ix];ax=axes[y,x];pred=np.load(path)
            ax.imshow(overlay(truth,pred));ax.axis('off')
            ax.set_title(f"{name} · {title}\n离线地图公式值 {r['offline_map_score']:.4f}",fontproperties=FONT,fontsize=13)
            rows.append(dict(map=name,mapper=r['mapper'],offline_formula=r['offline_map_score'],seconds=r['seconds']))
    fig.suptitle('相同合法观测输入的建图器消融：离线诊断，不是新比赛成绩',fontproperties=FONT,fontsize=20)
    fig.tight_layout(rect=(0,.04,1,.95));base.save(fig,a.output/'offline_mapping_ablations.png')
    protected=base.read('output/sf13/rl12_dev_20261115/protected_before.json')
    changed=[r['path'] for r in protected if not Path(r['path']).is_file() or base.digest(r['path'])!=r['sha256']]
    verification=dict(protected=len(protected),changed=changed,all_red_valid=all(r['red_valid'] for r in review['rows']),
        callbacks=[r['callback_errors'] for r in review['rows']],
        navigation_diagnostic_errors=[r['navigation_diagnostic_errors'] for r in review['rows']],
        source_2_received_sequence=submission['sources']['2'],actual_submitting_drone=submission['drone'],
        offline_ablations=rows,new_files={str(p):base.digest(p) for p in Path('src/swarm_rescue/solutions').rglob('*12*.py') if 'consistent12' in p.name or 'navlines12' in p.name})
    with (a.output/'verification.json').open('x',encoding='utf-8') as f:json.dump(verification,f,indent=2)
    print(json.dumps(verification,ensure_ascii=True,indent=2))
    if changed:raise SystemExit('Protected existing files changed; inspect without overwriting')


if __name__=='__main__':main()
