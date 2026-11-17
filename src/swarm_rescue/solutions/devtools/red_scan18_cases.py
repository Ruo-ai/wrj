"""Keep synthetic successes AND failures; geometry is offline only."""
import argparse
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np

from swarm_rescue.solutions.devtools.test_red_lines11 import ROOM,scan
from swarm_rescue.solutions.devtools.red_scan18_shadow import make_scan,match_pair
from swarm_rescue.solutions.devtools.red_recovery18_audit import write,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];arrays={}
    corridor=np.array([[[-1000.,-70.],[1000.,-70.]],[[-1000.,70.],[1000.,70.]]])
    for case in ('corridor','corner','repeated_rooms_alias','single_wall','open','invalid','dynamic_known','dynamic_unknown','noisy_oblique_rejected'):
        rng=np.random.default_rng(742)
        walls=corridor if case=='corridor' else corridor[:1] if case=='single_wall' else ROOM
        truth=np.array([6.,-2.,.06]);prior=truth+np.array([2.,-2.,.035]);objects=[]
        d0,angles=scan(walls,np.zeros(2),0.,rng);d1,_=scan(walls,truth[:2],truth[2],rng)
        limitation=None
        if case=='open':d0[:]=230;d1[:]=230
        if case=='invalid':d0[:]=np.nan;d1[:]=np.nan
        if case=='repeated_rooms_alias':
            # Distinct, identical rooms 400 px apart: scans alone cannot recover
            # an odometry prior in the wrong alias. Outside normal 4-step basin.
            walls=np.concatenate((ROOM,ROOM+np.array([400.,0.])))
            truth=np.array([400.,0.,0.]);prior=np.zeros(3)
            d0,angles=scan(walls,np.zeros(2),0.);d1,_=scan(walls,truth[:2],0.)
            limitation='Deliberate wrong-alias prior outside normal motion basin; local acceptance is not global correctness.'
        if case.startswith('dynamic'):
            d0[np.abs(angles+.3)<.18]=55;d1[np.abs(angles-.3)<.18]=45
            if case=='dynamic_known':
                objects=[SimpleNamespace(distance=45.,angle=.3,entity_type=SimpleNamespace(name='DRONE'))]
            limitation='Only current semantic exclusion is available; old unlabelled foreground remains possible.'
        if case=='noisy_oblique_rejected':
            from swarm_rescue.solutions.devtools.red_scan18_shadow import rotation
            truth=np.array([3.,-2.,.06]);prior=truth+np.array([2.,-2.,.035]);rng=np.random.default_rng(742)
            d0,angles=scan(ROOM,np.zeros(2),.53,rng)
            d1,_=scan(ROOM,rotation(.53)@truth[:2],.59,rng)
        start=time.perf_counter()
        answer,info=match_pair(make_scan(d1,angles,objects),make_scan(d0,angles),prior)
        elapsed=time.perf_counter()-start
        used=prior if answer is None else answer
        error=used-truth;error[2]=(error[2]+np.pi)%(2*np.pi)-np.pi
        rows.append(dict(case=case,accepted=answer is not None,info=info,truth_relative=truth.tolist(),
                         prior=prior.tolist(),used=used.tolist(),position_error=float(np.linalg.norm(error[:2])),
                         angle_error_deg=float(abs(error[2])*180/np.pi),seconds=elapsed,limitation=limitation))
        arrays[case+'_reference']=d0;arrays[case+'_current']=d1
    arrays['angles']=angles;np.savez_compressed(a.output/'cases.npz',**arrays)
    write(a.output/'results.json',dict(rows=rows,source_sha256=digest(Path(__file__).with_name('red_scan18_shadow.py')),
          scope='Synthetic pair fixtures, no flight and no competition scores. Failures intentionally retained.'))
    print(__import__('json').dumps(rows),flush=True)


if __name__=='__main__':main()
