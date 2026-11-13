"""B6 read-only historical evidence audit. Truth and map-specific coordinates stay offline."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def analyze(root,out):
    out.mkdir(parents=True,exist_ok=False)
    source=root/'output/sf13/b5_compare'
    variants=[]
    for p in sorted(source.glob('*/results.json')):
        for row in read(p):
            r=row['result'];attempt=Path(row['run']);t=read(attempt/'telemetry.json')
            variants.append(dict(variant=r['blue_variant'],result=r,valid=row['valid_blue'],
                transport=t['transport'],bombs=t['lifecycle']['bombs'],resources=read(attempt/'resources.json')))
    attempt=source/'adapted/runs/Map08_r01/advanced2/blue/attempt_001'
    t=read(attempt/'telemetry.json');frames=t['lifecycle']['frames']
    target=[f for f in frames if f['drone']==7 and 3700<=f['step']<=5300]
    bomb=t['lifecycle']['bombs'][5]
    z=np.load(next(attempt.rglob('*.npz')),allow_pickle=False);states=z['drone_states']
    timeline=[]
    for f in target:
        peers=states[min(f['step']-1,len(states)-1)];peers=peers[peers['active']]
        dist=np.hypot(peers['x']-f['truth'][0],peers['y']-f['truth'][1])
        nearest=np.sort(dist)
        timeline.append(dict(f,error=float(np.linalg.norm(np.asarray(f['position'])-f['truth'])),
                             nearest_other=float(nearest[1]) if len(nearest)>1 and nearest[0]<20 else None))
    carry=[f for f in timeline if f['carrying']]
    counts=dict(Counter(f['source'] for f in carry))
    summary=dict(source=str(attempt),bomb=bomb,destruction_log='Drone 7 destroyed, too much collision !' in (attempt/'run.log').read_text(encoding='utf-8'),
        carry_samples=len(carry),sources=counts,empty_route_samples=sum(f['route_points']==0 for f in carry),
        one_point_route_samples=sum(f['route_points']==1 for f in carry),gps_missing_samples=sum(f['gps_missing'] for f in carry),
        max_error=max(f['error'] for f in carry),median_error=float(np.median([f['error'] for f in carry])),
        snapshots={k:dict(step=v['step'],layers=list(v['layers'])) for k,v in t['lifecycle'].get('coverage_maps',{}).items()},
        gaps=['no historical raw lidar/odometer/estimated heading','only last own-grid snapshot, no raw evidence or remote merged map at pickup',
              'peer proximity is not proof of blockage','NPZ columns compact after removal; do not use them as persistent IDs'])
    receipt={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [attempt/'result.json',attempt/'telemetry.json',attempt/'run.log',next(attempt.rglob('*.npz'))]}
    (out/'evidence.json').write_text(json.dumps(dict(summary=summary,timeline=timeline,variants=variants,source_sha256=receipt),indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    steps=[f['step'] for f in timeline]
    for ax,key,label in zip(axes,('error','health','route_points','nearest_other'),('Offline pose error (px)','Health','Remaining route points','Nearest other body (px)')):
        ax.plot(steps,[f[key] for f in timeline],'.-',lw=.8);ax.set_ylabel(label)
        for tick in (3989,4047,5262):ax.axvline(tick,color='grey',ls='--',lw=.7)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('simulation step; dashed: discovery / pickup / loss')
    fig.suptitle('HISTORICAL B5 B0, drone 7 / bomb 6 — offline diagnosis, not a B6 result')
    fig.tight_layout();fig.savefig(out/'historical_timeline.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    image=plt.imread(root/'map_data/map_08.png')
    for ax in axes:ax.imshow(image,extent=(0,1660,1122,0))
    for ax,key in zip(axes,('truth','position')):
        q=np.array([f[key] for f in timeline]);ax.plot(q[:,0]+830,561-q[:,1],'-',lw=1)
        ax.plot(1560,936,'ro');ax.text(1560,936,' 6');ax.set_title('HISTORICAL '+('true path' if key=='truth' else 'estimated path'))
        ax.set(xlim=(850,1660),ylim=(1122,350))
    fig.tight_layout();fig.savefig(out/'historical_return.png',dpi=120);plt.close(fig)
    # Snapshot inspection is explicitly own-only, not reconstruction of the
    # actual planner's remote fusion or its subthreshold evidence counters.
    m=t['lifecycle']['coverage_maps']['7'];shape=m['shape'];layers={k:np.unpackbits(np.frombuffer(base64.b64decode(v),np.uint8))[:np.prod(shape)].reshape(shape).astype(bool) for k,v in m['layers'].items()}
    free=layers['lidar'];wall=layers['walls'];inflated=wall.copy();padded=np.pad(wall,1)
    for dr,dc in ((0,1),(2,1),(1,0),(1,2)):inflated|=padded[dr:dr+shape[0],dc:dc+shape[1]]
    fig,axes=plt.subplots(1,3,figsize=(13,5))
    for ax,data,title in zip(axes,(free,wall,free&~inflated),('Own free cells','Own binary wall evidence','Derived own-only allowed cells')):
        ax.imshow(data,extent=(0,1660,1122,0));ax.set_title(title)
    fig.suptitle(f'HISTORICAL last drone-7 snapshot step {m["step"]}; NOT exact planner reconstruction')
    fig.tight_layout();fig.savefig(out/'historical_own_grid.png',dpi=120);plt.close(fig)
    (out/'audit.md').write_text('# B6 historical audit\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+
        '\n\n'.join(f'![{name}](<{(out/name).resolve().as_posix()}>)' for name in ('historical_timeline.png','historical_return.png','historical_own_grid.png')),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[4];analyze(root,a.output.resolve())


if __name__=='__main__':main()
