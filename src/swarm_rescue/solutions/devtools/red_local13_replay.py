"""RL11 / RL13-L complete legal-input replay, retaining actual source versions."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from PIL import Image
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper
from swarm_rescue.solutions.my_drone_red_local13 import LocalBlockMapper
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base


def replay(run,out):
    out.mkdir(parents=True,exist_ok=False)
    meta=base.read(run/'line_diagnostics.json');diag=base.read(run/'exploration_diagnostics.json')
    result=base.read(run/'result.json');assert meta['mapper_inputs_complete'] and not meta['errors']
    with np.load(run/'mapper_legal_inputs.npz') as f:inputs={k:f[k] for k in f.files}
    with np.load(run/'exploration_stages.npz') as f:stages={k:f[k] for k in f.files}
    kinds={v:k for k,v in meta['kinds'].items()};objects={}
    for owner,tick,kind,distance,angle in inputs['semantic']:
        objects.setdefault((int(owner),int(tick)),[]).append(SimpleNamespace(
            entity_type=SimpleNamespace(name=kinds[int(kind)]),distance=float(distance),angle=float(angle)))
    submission=diag['submissions'][0];h,w=result['expected_shape']
    actual=np.asarray(base.read(next((run/'artifacts').glob('*_exploration.json')))['grid'],np.uint8)
    png=np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
    truth=np.all(png[base.COMPARISON_HEADER_HEIGHT:base.COMPARISON_HEADER_HEIGHT+h,:w]==40,axis=2).astype(np.uint8)
    records=[]
    for name,cls in [('rl11_lines',WallLineMapper),('rl13_local',LocalBlockMapper)]:
        started=time.perf_counter();sources={};exact={};stats={}
        for ident in diag['drones']:
            mapper=cls((w,h));data=inputs['drone_'+ident];angles=inputs['angles_'+ident]
            assert np.array_equal(data[:,0],np.arange(1,len(data)+1))
            for row in data:
                tick=int(row[0]);gps=row[5:7].copy() if row[1] else None
                compass=float(row[7]) if row[2] else None;odom=row[8:11].copy() if row[3] else None
                lidar=row[11:].copy() if row[4] else None
                mapper.update(gps,compass,odom,lidar,angles,objects.get((int(ident),tick),()))
                if tick==submission['sources'].get(ident):sources[ident]=mapper.mask().copy()
            exact[ident]=bool(np.array_equal(mapper.mask(),stages['local_final_'+ident]))
            stats[ident]=dict(mapper=mapper.stats,local=dict(getattr(mapper,'local_stats',{})))
        merged=np.zeros(mapper.shape,bool)
        for mask in sources.values():merged|=mask
        assert len(sources)==len(submission['sources'])
        pixels=mapper.to_pixels(merged);np.save(out/f'{name}_pixels.npy',pixels)
        score=base.MapExplorationScorer().score(truth,pixels,result['elapsed_walltime'])
        record=dict(mapper=name,seconds=time.perf_counter()-started,local_exact=exact,
            final_exact=bool(np.array_equal(pixels,actual)),offline_map_score=score.s_map*100,
            credit=score.credit,penalty=score.penalty,stats=stats)
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='stats'}),flush=True)
    assert records[0]['final_exact'] and all(records[0]['local_exact'].values())
    with (out/'results.json').open('x',encoding='utf-8') as f:
        json.dump(dict(records=records,input_sha256=base.digest(run/'mapper_legal_inputs.npz'),
            source=str(run.resolve()),source_sequences=submission['sources'],
            limitation='Same complete legal mapping tape; offline formula, not a new full game.'),f,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();replay(a.attempt,a.output)
