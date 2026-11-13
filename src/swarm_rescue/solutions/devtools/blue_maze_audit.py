"""Map08-only OFFLINE diagnostic, never imported by online blue policies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def audit(folder):
    rows=[]
    for file in sorted(folder.rglob('telemetry.json')):
        data=json.loads(file.read_text(encoding='utf-8'))
        if 'Map08' not in str(file):continue
        frames=list({(f['drone'],f['step']):f for f in data.get('lifecycle',{}).get('frames',[])}.values())
        region=[]
        for f in frames:
            if not f.get('truth'):continue
            x,y=f['truth'][0]+830,561-f['truth'][1]
            if 900<x<1120 and 500<y<1000:
                region.append(dict(f,image_x=x,image_y=y,
                    pose_error=float(np.linalg.norm(np.asarray(f['truth'])-f['position']))))
        bydrone={}
        for ident in sorted({f['drone'] for f in region}):
            fs=[f for f in region if f['drone']==ident]
            bydrone[ident]=dict(samples=len(fs),first_step=min(f['step'] for f in fs),
                last_step=max(f['step'] for f in fs),max_x=max(f['image_x'] for f in fs),
                modes=dict(Counter(f['mode'] for f in fs)),carrying=sum(f['carrying'] for f in fs),
                gps_missing=sum(f['gps_missing'] for f in fs),route_empty=sum(not f['route_points'] for f in fs),
                pose_error_median=float(np.median([f['pose_error'] for f in fs])),
                pose_error_max=max(f['pose_error'] for f in fs),
                target_samples=[{k:f.get(k) for k in ('step','image_x','image_y','mode','carrying','target','route_end','route_points','command','pose_error')} for f in fs[::max(1,len(fs)//10)]])
        deep=[f for f in frames if f.get('truth') and f['truth'][0]+830>=1100 and 516<=561-f['truth'][1]<=997]
        replays=list(file.parent.rglob('*.npz'));replay=None
        if replays:
            with np.load(replays[0],allow_pickle=False) as archive:
                state=archive['drone_states'];x=state['x']+830;y=561-state['y']
                inside=(x>=957)&(x<=1581)&(y>=516)&(y<=997)&state['active']
                far=inside&(x>=1100)
                replay=dict(source=str(replays[0]),frames=len(state),deep_drone_steps=int(far.sum()),
                    inside_drone_steps=int(inside.sum()),
                    max_region_x=float(x[inside].max()) if inside.any() else None,
                    identity_note='Recorder columns can shift after a drone is removed; these are NOT stable drone IDs. Use lifecycle frames for identity.',
                    recording_columns_not_ids={str(i):dict(inside_steps=int(inside[:,i].sum()),deep_steps=int(far[:,i].sum()),
                              max_x=float(x[:,i][inside[:,i]].max())) for i in range(state.shape[1]) if inside[:,i].any()})
        rows.append(dict(source=str(file),region=bydrone,deep_samples=len(deep),replay=replay,
                         bombs=data.get('lifecycle',{}).get('bombs',[])))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=audit(a.folder);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for r in rows:print(json.dumps(dict(source=r['source'],deep_samples=r['deep_samples'],
        drones={i:{k:v for k,v in row.items() if k!='target_samples'} for i,row in r['region'].items()})))


if __name__=='__main__':main()
