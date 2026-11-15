"""Replay complete recorded legal mapper inputs, retaining actual source ages.

No flight or radio simulation. Counterfactual map values are offline diagnostics.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper
from swarm_rescue.solutions.devtools import red_stopturn_match_report as base


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);run=args.attempt
    meta=base.read(run/'line_diagnostics.json');assert meta['mapper_inputs_complete'] and not meta['errors']
    diag=base.read(run/'exploration_diagnostics.json');result=base.read(run/'result.json')
    with np.load(run/'mapper_legal_inputs.npz') as f:inputs={k:f[k] for k in f.files}
    with np.load(run/'exploration_stages.npz') as f:stages={k:f[k] for k in f.files}
    kinds={v:k for k,v in meta['kinds'].items()};objects={}
    for owner,tick,kind,distance,angle in inputs['semantic']:
        objects.setdefault((int(owner),int(tick)),[]).append(SimpleNamespace(
            entity_type=SimpleNamespace(name=kinds[int(kind)]),distance=float(distance),angle=float(angle)))
    submission=diag['submissions'][0] if diag['submissions'] else None
    h,w=result['expected_shape'];records=[];images={}
    for name,cls in (('baseline',SubmissionMapper),('lines',WallLineMapper)):
        started=time.perf_counter();sources={};local_exact={};received_exact={};sequence_found={}
        for ident in diag['drones']:
            owner=int(ident);data=inputs['drone_'+ident];angles=inputs['angles_'+ident]
            assert np.array_equal(data[:,0],np.arange(1,len(data)+1)),(ident,'missing input step')
            mapper=cls((w,h));sequence=submission['sources'].get(ident) if submission else None
            for row in data:
                step=int(row[0]);gps=row[5:7].copy() if row[1] else None;compass=float(row[7]) if row[2] else None
                odom=row[8:11].copy() if row[3] else None;lidar=row[11:].copy() if row[4] else None
                mapper.update(gps,compass,odom,lidar,angles,objects.get((owner,step),()))
                if step==sequence:sources[ident]=mapper.mask().copy()
            local_exact[ident]=bool(np.array_equal(mapper.mask(),stages['local_final_'+ident]))
            sequence_found[ident]=sequence is None or ident in sources
            if ident in sources:
                key=f"submitter_{submission['drone']}_source_{ident}"
                received_exact[ident]=bool(np.array_equal(sources[ident],stages[key]))
        assert all(sequence_found.values()),sequence_found
        record=dict(mapper=name,seconds=time.perf_counter()-started,local_exact=local_exact,
                    received_exact=received_exact,source_versions=submission['sources'] if submission else None)
        if submission:
            merged=np.zeros(mapper.shape,bool)
            for mask in sources.values():merged|=mask
            pixels=mapper.to_pixels(merged);images[name]=pixels;np.save(args.output/f'{name}_pixels.npy',pixels)
            actual=np.asarray(base.read(next((run/'artifacts').glob('*_exploration.json')))['grid'],np.uint8)
            record['final_pixels_exact']=bool(np.array_equal(actual,pixels))
            png=np.asarray(Image.open(next((run/'artifacts').glob('*_exploration_diff.png'))).convert('RGB'))
            truth=np.all(png[base.COMPARISON_HEADER_HEIGHT:base.COMPARISON_HEADER_HEIGHT+h,:w]==40,axis=2).astype(np.uint8)
            score=base.MapExplorationScorer().score(truth,pixels,result['elapsed_walltime'])
            record.update(offline_map_formula_value=score.s_map*100,credit=score.credit,penalty=score.penalty)
        records.append(record)
    original='lines' if result['controller_selector']=='rl11_lines' else 'baseline'
    reproduction=next(r for r in records if r['mapper']==original)
    exact=all(reproduction['local_exact'].values()) and all(reproduction['received_exact'].values())
    exact &= reproduction.get('final_pixels_exact',True)
    report=dict(source_attempt=str(run.resolve()),input_sha256=base.digest(run/'mapper_legal_inputs.npz'),
        exact_original_reproduction=exact,records=records,
        limitation='Counterfactual fixed legal inputs and source versions only. Not a new official match; no difficulty or full score.')
    with (args.output/'results.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if not exact:raise SystemExit('Original mapper did not reproduce: do not claim identical input replay validity')


if __name__=='__main__':main()
