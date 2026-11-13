"""Synthetic noisy sensor sequences, NOT native dynamics or official scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from swarm_rescue.solutions.my_drone_blue_b6_local_20261113 import BlueLocalWindow
from swarm_rescue.solutions.semifinal_localization import scan_points,wrap


ANGLES=np.linspace(-math.pi,math.pi,181)


def cast(walls,position,heading,noise=0.,rng=None):
    ray=np.column_stack((np.cos(ANGLES+heading),np.sin(ANGLES+heading)))
    d=np.full(len(ray),230.)
    for start,end in walls:
        start=np.asarray(start);segment=np.asarray(end)-start;delta=start-position
        den=ray[:,0]*segment[1]-ray[:,1]*segment[0]
        with np.errstate(divide='ignore',invalid='ignore'):
            t=(delta[0]*segment[1]-delta[1]*segment[0])/den
            u=(delta[0]*ray[:,1]-delta[1]*ray[:,0])/den
        valid=(abs(den)>1e-8)&(t>0)&(u>=0)&(u<=1)
        d=np.minimum(d,np.where(valid,t,np.inf))
    if noise:d=d+(rng or np.random.default_rng(0)).normal(0,noise,len(d))
    return d


def polygon(points):return list(zip(points,points[1:]+points[:1]))


def run_sequence(kind,seed):
    rng=np.random.default_rng(seed);steps=180
    if kind=='corridor':
        walls=[((-600,-40),(600,-40)),((-600,40),(600,40))]
        truth=[(np.array([i*.65,0.]),0.) for i in range(steps)]
    else:
        walls=polygon([(-200,-35),(35,-35),(35,200),(-35,200),(-35,35),(-200,35)])
        truth=[]
        for i in range(steps):
            if i<65:p=np.array([-140+2*i,0.]);h=0.
            elif i<85:p=np.array([-12.,0.]);h=(i-65)/20*math.pi/2
            else:p=np.array([-12.,(i-85)*1.5]);h=math.pi/2
            truth.append((p,h))
    baseline=truth[0][0].copy();corrected=baseline.copy();hb=hc=truth[0][1]
    window=BlueLocalWindow();rows=[];start=time.perf_counter()
    for tick,(point,heading) in enumerate(truth):
        if tick:
            previous,ph=truth[tick-1];delta=point-previous
            distance=np.linalg.norm(delta)+rng.normal(0,.2)
            alpha=wrap(math.atan2(delta[1],delta[0])-ph)+rng.normal(0,math.radians(8))
            rotation=wrap(heading-ph)+rng.normal(0,math.radians(1))+.003
            baseline+=distance*np.array([math.cos(hb+alpha),math.sin(hb+alpha)]);hb=wrap(hb+rotation)
            corrected+=distance*np.array([math.cos(hc+alpha),math.sin(hc+alpha)]);hc=wrap(hc+rotation)
        points=scan_points(cast(walls,point,heading,2.5,rng),ANGLES)
        gps,compass=(point,heading) if tick<12 else (None,None)
        if tick<12:baseline=point.copy();corrected=point.copy();hb=hc=heading
        corrected,hc=window.update(corrected,hc,points,gps,compass,tick)
        rows.append(dict(step=tick,truth=point.tolist(),baseline=baseline.tolist(),corrected=corrected.tolist(),
            baseline_error=float(np.linalg.norm(baseline-point)),candidate_error=float(np.linalg.norm(corrected-point)),
            baseline_yaw_error=abs(wrap(hb-heading)),candidate_yaw_error=abs(wrap(hc-heading)),info=window.info))
    return dict(kind=kind,seed=seed,steps=steps,seconds=time.perf_counter()-start,stats=dict(window.stats),
        baseline_final=rows[-1]['baseline_error'],candidate_final=rows[-1]['candidate_error'],
        baseline_p95=float(np.percentile([r['baseline_error'] for r in rows],95)),
        candidate_p95=float(np.percentile([r['candidate_error'] for r in rows],95)),rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[run_sequence(kind,seed) for kind in ('corridor','corner') for seed in (742,743,744)]
    import swarm_rescue.solutions.my_drone_blue_b6_local_20261113 as policy
    result=dict(scope='synthetic rays + specified odometry noise/bias; NOT physical passage or full-game evidence',
        source_sha256=hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest(),cases=rows)
    (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,r in zip(axes.flat,rows):
        ax.plot([p['baseline_error'] for p in r['rows']],label='odometry only')
        ax.plot([p['candidate_error'] for p in r['rows']],label='B6 local match')
        ax.set_title(f"{r['kind']} / seed {r['seed']}");ax.set_ylabel('position error (px)');ax.legend(fontsize=8)
    fig.suptitle('SYNTHETIC sensor sequences — no physics, payload or official score')
    fig.tight_layout();fig.savefig(a.output/'synthetic_errors.png',dpi=120);plt.close(fig)
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in rows]))


if __name__=='__main__':main()
