"""RL12 same-input mapper experiment and recovery audit; offline only."""
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper
from swarm_rescue.solutions.my_drone_red_consistent12 import ConsistentMapper, ContinuousHeadingMapper, StableLineMapper
from swarm_rescue.solutions.semifinal_localization import wrap
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base


def replay(run, output, recovery=False):
    output.mkdir(parents=True, exist_ok=False)
    meta = base.read(run/'line_diagnostics.json')
    assert meta['mapper_inputs_complete'] and not meta['errors']
    diag = base.read(run/'exploration_diagnostics.json'); result = base.read(run/'result.json')
    with np.load(run/'mapper_legal_inputs.npz') as f: inputs = {k:f[k] for k in f.files}
    with np.load(run/'exploration_stages.npz') as f: stages = {k:f[k] for k in f.files}
    motion = base.read(run/'motion_diagnostics.json')
    kinds = {v:k for k,v in meta['kinds'].items()}; objects = {}
    for owner,tick,kind,distance,angle in inputs['semantic']:
        objects.setdefault((int(owner),int(tick)), []).append(SimpleNamespace(
            entity_type=SimpleNamespace(name=kinds[int(kind)]),distance=float(distance),angle=float(angle)))
    submission = diag['submissions'][0]; h,w = result['expected_shape']
    actual = np.asarray(base.read(next((run/'artifacts').glob('*_exploration.json')))['grid'],np.uint8)
    png = np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
    truth = np.all(png[base.COMPARISON_HEADER_HEIGHT:base.COMPARISON_HEADER_HEIGHT+h,:w] == 40,axis=2).astype(np.uint8)
    records=[]
    variants=([('rl11',WallLineMapper),('rl12_stable',StableLineMapper)] if recovery else
              [('rl11',WallLineMapper),('rl12_lag',ConsistentMapper),('rl12_heading',ContinuousHeadingMapper)])
    for name,cls in variants:
        started=time.perf_counter(); sources={}; exact={}; audits=[]; errors=[]; stats={}
        for ident in diag['drones']:
            owner=int(ident); data=inputs['drone_'+ident]; angles=inputs['angles_'+ident]
            samples={s['tick']:s for s in motion['drones'][ident]['samples']}
            assert np.array_equal(data[:,0],np.arange(1,len(data)+1))
            mapper=cls((w,h)); sequence=submission['sources'].get(ident)
            for row in data:
                tick=int(row[0]); gps=row[5:7].copy() if row[1] else None
                compass=float(row[7]) if row[2] else None; odom=row[8:11].copy() if row[3] else None
                lidar=row[11:].copy() if row[4] else None
                recovering=mapper.missing>0 and gps is not None and compass is not None
                if recovering:
                    pp=mapper.position.copy(); ph=mapper.heading
                    if odom is not None:
                        pp+=odom[0]*np.array([math.cos(ph+odom[1]), math.sin(ph+odom[1])]); ph=wrap(ph+odom[2])
                    audits.append(dict(drone=owner,step=tick,missing=mapper.missing,pending=len(mapper.pending),
                        recovery_angle_deg=math.degrees(wrap(compass-ph)),recovery_distance=float(np.linalg.norm(gps-pp))))
                mapper.update(gps,compass,odom,lidar,angles,objects.get((owner,tick),()))
                if tick in samples and mapper.position is not None:
                    s=samples[tick]
                    errors.append(dict(drone=owner,step=tick,missing=bool(not(row[1] and row[2])),
                        position=float(np.linalg.norm(mapper.position-s['truth'])),
                        heading_deg=abs(math.degrees(wrap(mapper.heading-s['true_heading'])))))
                if tick==sequence: sources[ident]=mapper.mask().copy()
            exact[ident]=bool(np.array_equal(mapper.mask(),stages['local_final_'+ident]))
            stats[ident]=dict(mapper.stats)
        assert len(sources)==len(submission['sources'])
        merged=np.zeros(mapper.shape,bool)
        for mask in sources.values():merged|=mask
        pixels=mapper.to_pixels(merged);np.save(output/f'{name}_pixels.npy',pixels)
        score=base.MapExplorationScorer().score(truth,pixels,result['elapsed_walltime'])
        record=dict(mapper=name,seconds=time.perf_counter()-started,local_exact=exact,
            final_exact=bool(np.array_equal(pixels,actual)),offline_map_score=score.s_map*100,
            credit=score.credit,penalty=score.penalty,recovery=audits,sampled_errors=errors,stats=stats)
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('recovery','sampled_errors','stats')},ensure_ascii=True),flush=True)
    assert records[0]['final_exact'] and all(records[0]['local_exact'].values()),'RL11 did not reproduce exactly'
    with (output/'results.json').open('x',encoding='utf-8') as f:
        json.dump(dict(source=str(run.resolve()),input_sha256=base.digest(run/'mapper_legal_inputs.npz'),
            actual_source_sequences=submission['sources'],records=records,
            limitation='Same legal mapper inputs and actual source ages; offline scores only, not a new official game.'),f,indent=2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recovery',action='store_true')
    a=p.parse_args();replay(a.attempt,a.output,a.recovery)


if __name__=='__main__':main()
