"""Fixed-layout orchestration must retain failures and never rerun red."""
from pathlib import Path
from . import joint_batch as joint,runner
from .test_evaluation import completed,red_result


def test_candidate_blue_reuses_verified_red_and_keeps_failed_rows(tmp_path,monkeypatch):
    reference=tmp_path/'reference';reference.mkdir()
    candidate=tmp_path/'candidate_blue';candidate.mkdir()
    runner.write(reference/'plan.json',dict(test=True))
    plan_sha=runner.file_hash(reference/'plan.json')
    schedule=[dict(name='fixture',controller=c,round=1,map_id='id',seed=742)
              for c in ('advanced2','advanced2_semifinal')]
    sources=[];paths=[]
    for item,result in zip(schedule,[red_result(),red_result(status='process_timeout')]):
        path=completed(reference/'runs'/item['controller']/'red',result,plan_sha)
        paths.append(path);sources.append(dict(map='fixture',controller=item['controller'],round=1,red_run=str(path)))
    runner.write(reference/'results.json',sources)
    runner.write(candidate/'plan.json',dict(schedule=schedule,source_sha256={}))
    monkeypatch.setattr(joint,'prepare',lambda _:None)
    monkeypatch.setattr(runner,'run',lambda _:None)
    monkeypatch.setattr(joint,'summarize',lambda _:None)
    from . import placement_audit
    monkeypatch.setattr(placement_audit,'audit',lambda _:None)
    calls=[];reports=[]
    def execute(output,plan,item,phase,red):
        assert phase=='blue';calls.append(red)
        return dict(status='ok'),candidate/'blue'
    monkeypatch.setattr(runner,'execute_phase',execute)
    monkeypatch.setattr(joint,'save_report',lambda output,plan,pairs:reports.append(list(pairs)))
    joint.run_matrix(tmp_path)
    assert calls==paths[:1] and len(reports[-1])==2
    assert reports[-1][1]['blue'] is None
    assert reports[-1][1]['red']['status']=='process_timeout'


def test_actual_wall_marginals_sum_to_union_including_duplicate_disks():
    import numpy as np
    from .placement_audit import increments
    weights=np.ones((3,5));weights[0,:]=.5
    a=np.zeros((3,5),bool);a[:,:3]=True
    b=np.zeros((3,5),bool);b[:,2:]=True
    masks=np.array([a,b,a])
    rows,total=increments(weights,masks,[0,1,2])
    assert rows[2]['marginal']==0 and sum(r['marginal'] for r in rows)==total==weights.sum()
