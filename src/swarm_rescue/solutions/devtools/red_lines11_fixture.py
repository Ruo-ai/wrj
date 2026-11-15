"""Synthetic identical-input accuracy/CPU evidence, not competition scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from swarm_rescue.solutions.devtools.test_red_lines11 import scan,ROOM
from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper
from swarm_rescue.solutions.semifinal_localization import transform,wrap


def inputs(case):
    rng=np.random.default_rng(742);noise=case!='ideal_oblique'
    segments=ROOM if case!='noisy_corridor' else ROOM[[0,2]]
    segments=np.array([transform(s,np.zeros(2),.31) for s in segments])
    rows=[];previous=None;gps_noise=np.zeros(2);compass_noise=0.
    for k in range(420):
        position=np.array([50*math.sin(k*.006),25*math.sin(k*.009)])
        heading=.7*math.sin(k*.011)+.2;odom=np.zeros(3)
        if previous is not None:
            p,h=previous;delta=position-p
            odom=np.array([np.linalg.norm(delta),wrap(math.atan2(delta[1],delta[0])-h),wrap(heading-h)])
        previous=(position,heading)
        if noise:odom+=rng.normal(0,[.2,math.radians(8),math.radians(1)])
        d,a=scan(segments,position,heading,rng if noise else None)
        missing=64<=k<360
        if not missing and noise:
            gps_noise=.98*gps_noise+rng.normal(0,5*math.sqrt(1-.98**2),2)
            compass_noise=.98*compass_noise+rng.normal(0,math.radians(4)*math.sqrt(1-.98**2))
        rows.append((None if missing else position+gps_noise,None if missing else heading+compass_noise,
                     odom,d,a,position,heading))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);result=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for case in ('ideal_oblique','noisy_room','noisy_corridor'):
        rows=inputs(case);digest=hashlib.sha256()
        for row in rows:
            for value in row[:5]:digest.update(b'None' if value is None else np.asarray(value,dtype='<f8').tobytes())
        fig,axes=plt.subplots(1,2,figsize=(11,4));traces={}
        for name,cls in (('RN7-B mapper',SubmissionMapper),('RL11 lines',WallLineMapper)):
            mapper=cls((500,500));trace=[];begin=time.perf_counter()
            for gps,heading,odom,d,a,truth,angle in rows:
                mapper.update(gps,heading,odom,d,a)
                trace.append((np.linalg.norm(mapper.position-truth),abs(wrap(mapper.heading-angle))*180/math.pi))
            seconds=time.perf_counter()-begin;trace=np.array(trace);traces[name]=trace
            result.append(dict(case=case,controller=name,seconds=seconds,input_sha256=digest.hexdigest(),
                outage_position_mean=float(trace[64:360,0].mean()),outage_heading_mean_deg=float(trace[64:360,1].mean()),
                outage_position_p95=float(np.percentile(trace[64:360,0],95)),outage_heading_p95_deg=float(np.percentile(trace[64:360,1],95)),
                stats=mapper.stats,pending_max_bound=mapper.MAX_PENDING))
            for i,ax in enumerate(axes):ax.plot(trace[:,i],label=name);ax.axvspan(64,360,color='gray',alpha=.1)
        axes[0].set_ylabel('position error / px');axes[1].set_ylabel('heading error / degrees')
        for ax in axes:ax.set_xlabel('synthetic step');ax.legend()
        fig.suptitle(case+' | identical legal synthetic inputs, not match results')
        fig.tight_layout();fig.savefig(args.output/f'{case}.png',dpi=160);plt.close(fig)
    with (args.output/'results.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
