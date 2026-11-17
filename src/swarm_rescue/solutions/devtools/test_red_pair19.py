"""Behavioral checks for causal replay, experiment identity and invalid scores."""
import copy
from pathlib import Path

import numpy as np
import pytest

from swarm_rescue.solutions.devtools import red_pair19_batch as batch
from swarm_rescue.solutions.devtools import red_pair19_report as report
from swarm_rescue.solutions.devtools.red_recovery19_weights import check_radio, recovery_ticks
from swarm_rescue.solutions.devtools.red_recovery18_audit import evidence_grid, radio_states
from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper, anchor_frames


def radio():
    return [dict(event='local_publish',holder=0,source=0,version=1,step=1,sha256='a'),
        dict(event='local_publish',holder=1,source=1,version=2,step=2,sha256='b'),
        dict(event='received',holder=0,source=1,version=2,step=5,sha256='b'),
        dict(event='local_publish',holder=1,source=1,version=6,step=6,sha256='c')]


def test_no_future_or_unreceived_final_source():
    events=radio();submission=dict(drone=0,step=7,sources={'0':1,'1':2})
    _,state=check_radio(events,submission)
    assert state=={0:1,1:2}  # source 1's later local version is NOT received
    assert radio_states(events,0,[4,5])[4]=={0:1}
    assert radio_states(events,0,[4,5])[5]=={0:1,1:2}


@pytest.mark.parametrize('fault', ['future','missing_origin','payload','wrong_final'])
def test_bad_radio_is_rejected(fault):
    events=radio();submission=dict(drone=0,step=7,sources={'0':1,'1':2})
    if fault=='future':events[2]['step']=1
    if fault=='missing_origin':events.pop(1)
    if fault=='payload':events[2]['sha256']='different'
    if fault=='wrong_final':submission['sources']['1']=6
    with pytest.raises(ValueError):check_radio(events,submission)


def test_recovery_discovery_includes_short_and_empty_scan_intervals():
    d=np.zeros((9,12));d[:,0]=np.arange(1,10);d[:,1:3]=1
    d[1,1]=0;d[4:6,2]=0;d[7,1:3]=0
    assert recovery_ticks(d)==[3,7,9]


def test_recovery_preserves_arbitrary_frame_weight_and_geometry():
    mapper=SubmissionMapper((60,60));point=np.array([[0.,0.]]);empty=np.empty((0,2))
    frames=[(i,point,np.zeros(2),0.,weight,empty) for i,weight in enumerate((.2,.45,.63),1)]
    adjusted=anchor_frames(frames,(0,np.zeros(2)),np.zeros(2),0.,np.zeros(2),0.,4)
    assert [f[4] for f in adjusted]==[.2,.45,.63]
    retained=evidence_grid(mapper,adjusted);uniform=evidence_grid(mapper,adjusted,fixed_weight=.8)
    assert retained.max()==pytest.approx(1.28)
    assert uniform.max()==pytest.approx(2.4)
    assert retained.max()<2.4  # fewer false hits can also drop true evidence


def test_weight_applies_to_free_erasure_too():
    mapper=SubmissionMapper((60,60));point=np.array([[0.,0.]]);empty=np.empty((0,2))
    start=np.zeros(mapper.shape,np.float32);mapper._insert(start,point,3.)
    f=[(1,empty,np.zeros(2),0.,.2,point)]
    a=evidence_grid(mapper,f,start);b=evidence_grid(mapper,f,start,fixed_weight=.8)
    assert a.max()==pytest.approx(2.93)
    assert b.max()==pytest.approx(2.72)
    assert start.max()==3.  # original grid untouched


def test_original_entry_and_explicit_identities():
    batch.register()
    from swarm_rescue.solutions.my_drone_eval import drone_class_for_mode
    assert drone_class_for_mode('place').__name__=='MyDroneRedAdvanced2'
    a=batch.runner.ev.load_controller(batch.runner.ev.controller_identity('rl11_lines'))
    b=batch.runner.ev.load_controller(batch.runner.ev.controller_identity('rl17_repair'))
    assert a.__name__=='MyDroneRedLines11' and b.__name__=='MyDroneRedRepair17'
    assert a is not b and issubclass(b,a)


def test_no_data_means_no_score(tmp_path):
    item=dict(name='test',controller='rl11_lines',round=1,seed=42,map_file='map.json')
    row=report.evaluate_row(item,tmp_path/'absent',dict(source_sha256={'map.json':'sha'}),'sha')
    assert not row['valid'] and row['map_score'] is None and row['full_score'] is None


def test_missing_diagnostics_not_zero_exit_success(tmp_path,monkeypatch):
    attempt=tmp_path/'attempt_001';attempt.mkdir()
    monkeypatch.setattr(report.runner,'load_done',lambda *args: (dict(status='ok',exit_code=0),attempt))
    item=dict(name='test',controller='rl11_lines',round=1,seed=42,map_file='map.json')
    row=report.evaluate_row(item,attempt,dict(source_sha256={'map.json':'sha'}),'sha')
    assert not row['valid'] and not row['execution_ok'] and row['reasons']


def test_empty_and_negative_delta_aggregation():
    assert report.aggregate([])['map_score']['mean'] is None
    a=report.aggregate([{'map_score':-4.},{'map_score':1.}])['map_score']
    assert a==dict(n=2,mean=-1.5,median=-1.5,min=-4.,max=1.)


def test_frozen_receipt_rejects_edit(tmp_path):
    batch.write(tmp_path/'source_bytes.json',{})
    plan=dict(source_bytes_sha256=batch.digest(tmp_path/'source_bytes.json'))
    batch.write(tmp_path/'plan.json',plan)
    batch.write(tmp_path/'prepared.json',dict(plan_sha256=batch.digest(tmp_path/'plan.json')))
    batch.validate_plan(tmp_path,plan,check_live=False)
    (tmp_path/'plan.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError):batch.validate_plan(tmp_path,plan,check_live=False)


def test_full_offline_pipeline_with_delayed_receipts(tmp_path):
    """Synthetic fixture, NOT a flight: full final-merge branch and stale source."""
    from PIL import Image
    from swarm_rescue.solutions.devtools import red_recovery19_weights as weights
    from swarm_rescue.solutions.my_drone_red_repair17 import RepairMapper17
    from swarm_rescue.solutions.devtools.red_recovery18_audit import Metrics,legal_arguments
    run=tmp_path/'synthetic_only';run.mkdir();art=run/'artifacts';art.mkdir()
    inputs={'semantic':np.empty((0,5))};poses={};stages={};diag={'drones':{},'submissions':[]}
    events=[];matches=[];snapshots={};mappers={}
    for owner in (0,1):
        mapper=RepairMapper17((120,120));mappers[owner]=mapper
        data=np.zeros((48,192));data[:,0]=np.arange(1,49);data[:,1:5]=1
        data[8:32,1:3]=0;data[:,11:]=45.;angles=np.linspace(-np.pi,np.pi,181)
        mapper.matcher.match=lambda *a: (None,{'reason':'synthetic_fixed_rejection'})
        trace=[]
        for record in data:
            tick=int(record[0]);mapper.update(*legal_arguments(record,angles,[]))
            trace.append([tick,0,0,0,*mapper.position,mapper.heading,not bool(record[1])])
            if tick%8==0 and not record[1]:
                matches.append(dict(drone=owner,step=tick,decision=dict(accepted=False,offset=None,info={'reason':'synthetic_fixed_rejection'})))
            if tick==1 or tick%4==0:
                snapshots[owner,tick]=mapper.mask().copy()
                events.append(dict(event='local_publish',holder=owner,source=owner,step=tick,version=tick,sha256=f'{owner}:{tick}'))
        inputs[f'drone_{owner}']=data;inputs[f'angles_{owner}']=angles;poses[f'pose_{owner}']=np.array(trace)
        diag['drones'][str(owner)]={'history':[dict(step=12,published_sequence=12),dict(step=40,published_sequence=40)]}
        for tick in (12,40):stages[f'published_{owner}_{tick}']=snapshots[owner,tick]
        stages[f'local_final_{owner}']=mapper.mask()
    events.extend([dict(event='received',holder=0,source=1,step=22,version=20,sha256='1:20'),
                   dict(event='received',holder=0,source=1,step=44,version=40,sha256='1:40')])
    events.sort(key=lambda e:e['step'])
    submission=dict(drone=0,step=48,sources={'0':48,'1':40});diag['submissions']=[submission]
    for owner,version in ((0,48),(1,40)):stages[f'submitter_0_source_{owner}']=snapshots[owner,version]
    pred=mapper.to_pixels(snapshots[0,48]|snapshots[1,40]);truth=pred.copy()
    png=np.full((weights.COMPARISON_HEADER_HEIGHT+120,120,3),255,np.uint8)
    png[weights.COMPARISON_HEADER_HEIGHT:][truth.astype(bool)]=40
    Image.fromarray(png).save(art/'synthetic_exploration_diff.png')
    batch.write(art/'synthetic_exploration.json',dict(grid=pred.tolist()))
    metric=Metrics(truth,(0,0,119,119))
    batch.write(run/'result.json',dict(expected_shape=[120,120],phase1={'score_map':metric.count(pred)['map_score']}))
    batch.write(run/'request.json',dict(map='SYNTHETIC_TEST_ONLY',round=1,seed=0))
    batch.write(run/'line_diagnostics.json',dict(kinds={},errors=[],mapper_inputs_complete=True))
    batch.write(run/'exploration_diagnostics.json',diag)
    batch.write(run/'p2l17_diagnostics.json',dict(radio=events,matches=matches))
    np.savez_compressed(run/'mapper_legal_inputs.npz',**inputs)
    np.savez_compressed(run/'exploration_stages.npz',**stages)
    np.savez_compressed(run/'p2l17_diagnostics.npz',**poses)
    r=weights.replay(dict(run=str(run),map='SYNTHETIC_TEST_ONLY',events=[{},{}]),tmp_path/'synthetic_result')
    assert r['valid_full_replay'] and r['event_count']==2
    assert r['final']['actual_sources']=={0:48,1:40}
    assert r['holder_timeline'][0]['actual_sources']=={0:32,1:20}
    assert all(c['pose_max_abs']==0 for c in r['checks'])
