"""Actual failing drone2 legal tape: audit corrected path coordinates offline."""
import importlib.util,json,os
from pathlib import Path
os.environ['PYGLET_SHADOW_WINDOW']='false'
import numpy as np
from swarm_rescue.solutions.red_continuous38 import ContinuousMapper38
from swarm_rescue.solutions.my_drone_red_delivery39r2 import CorrectedWalkGraph39
from swarm_rescue.solutions.red_delivery39 import WalkGraph39

ROOT=Path(__file__).resolve().parents[4]
RUN=ROOT/'output/sf13/delivery39r1_run_20261122/runs/Map09_r03/candidate_delivery39r1/red/attempt_001'
OUT=ROOT/'output/sf13/delivery39r2_offline_20261122'
spec=importlib.util.spec_from_file_location('d39_legal_inputs',ROOT/'output/sf13/continuous38_replay_20261122.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();mapper=ContinuousMapper38((1700,1100));old=WalkGraph39();new=CorrectedWalkGraph39(mapper)
    samples={r['step']:r for r in json.loads((RUN/'entry37_diagnostics.json').read_text(encoding='utf-8'))['samples'] if r['drone']==2}
    difference=0.;audits=[]
    with np.load(RUN/'mapper_legal_inputs.npz',allow_pickle=False) as tape:
        for t,args in helper.inputs(tape,2):
            if t>3841:break
            mapper.update(*args)
            old.observe(mapper.position,mapper.heading,args[3],args[4],t)
            new.observe(mapper.position,mapper.heading,args[3],args[4],t)
            if t in samples:difference=max(difference,float(np.linalg.norm(mapper.position-samples[t]['navigation'])))
            if mapper.recovery38 is not None:audits.append(new.recoveries[-1])
    saved=json.loads((RUN/'delivery39_diagnostics.json').read_text(encoding='utf-8'))['drones']['2']
    # The original erroneous segment is reconstructed from the unchanged tape.
    oldpoints=np.array(old.points)
    original_exact=bool(np.array_equal(oldpoints,np.asarray(saved['walk_points'])[:len(oldpoints)]))
    assert original_exact and difference<1e-8,(original_exact,difference)
    assert new.stats['coordinate_recoveries']>0 and new.stats['corrected_walk_nodes']>0
    record=dict(new_flights=0,scope='Same saved legal tape, before candidate flight; no score inferred',
        old_graph_exactly_reproduced=original_exact,maximum_shared_pose_difference=difference,
        mapper_unchanged=True,recoveries=audits,old_nodes=len(old.points),new_nodes=len(new.points),
        original_bad_segment=[old.points[i].tolist() for i in (124,125,126)],
        passed=True)
    np.savez_compressed(OUT/'paths.npz',old=np.array(old.points),new=np.array(new.points))
    (OUT/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
