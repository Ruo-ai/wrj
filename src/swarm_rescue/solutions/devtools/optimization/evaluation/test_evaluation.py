"""G3 failure, receipt, pairing and resource checks; no formal simulations."""
import argparse
import os
from pathlib import Path
import sys

import pytest

from . import runner, report
from .resources import run_limited


def red_result(**overrides):
    row = dict(status='ok',exit_code=0,submitted=True,valid_shape=True,
               placed_count=10,expected_count=10,
               phase1=dict(score_map=80,score_wall=70,score_time=90,round_score=63))
    row.update(overrides)
    return row


def pair(tmp_path,controller='advanced2',score=83):
    return dict(name='fixture',map_id='id',map_file='unused',family='rooms',controller=controller,
        round=1,seed=42,red=red_result(),blue=dict(status='ok',exit_code=0,
        phase1=dict(round_score=score,score_bomb=100,score_status='final')),
        red_run=str(tmp_path/'red'),blue_run=str(tmp_path/'blue'))


@pytest.mark.parametrize('overrides',[dict(status='process_timeout'),dict(exit_code=1),
    dict(walltime_limited=True),dict(controller_crashed=True),dict(submitted=False),
    dict(valid_shape=False),dict(phase1=None)])
def test_red_failures_not_valid(overrides):
    assert not runner.phase_valid(red_result(**overrides),'red')


def test_smoke_and_incomplete_blue_never_have_full_score(tmp_path):
    item = pair(tmp_path)
    assert report.flatten(item)['total_score']==83
    item['red']['smoke'] = True
    assert report.flatten(item)['total_score'] is None
    item['red']['smoke'] = False
    item['blue']['phase1']['score_status'] = 'partial'
    row = report.flatten(item)
    assert not row['valid_score'] and row['total_score'] is None
    assert row['failure_reason']=='blue_missing_final_score'
    item['red']['walltime_limited'] = True
    assert report.flatten(item)['failure_reason']=='red_walltime_limit'


def test_statistics_and_paired_summary(tmp_path,monkeypatch):
    stats = report.statistics_of([10,20,30])
    assert stats['mean']==stats['median']==20
    assert stats['p10']==12 and stats['minimum']==10
    assert stats['population_stdev']==pytest.approx((200/3)**.5)
    assert report.statistics_of([]) is None
    rows = [pair(tmp_path),pair(tmp_path,'candidate',85),pair(tmp_path,'candidate',99)]
    rows[-1]['seed'] = 999  # Cannot be compared to seed 42.
    failed = pair(tmp_path,'candidate');failed['round']=2;failed['blue']=None
    rows.append(failed)
    plan = dict(configuration=dict(controllers=['advanced2','candidate']),schedule=rows)
    monkeypatch.setattr(report,'layout_plot',lambda p,path:path.write_bytes(b'fixture'))
    report.save_report(tmp_path,plan,rows)
    summary = runner.read(tmp_path/'summary.json')['candidate']
    assert summary['valid_scores']==2 and summary['failure_rate']==pytest.approx(1/3)
    assert summary['total']['mean']==92
    changes = runner.read(tmp_path/'comparison.json')
    assert len(changes)==1 and changes[0]['delta']==2
    failed['controller']='advanced2'
    report.save_report(tmp_path,dict(configuration=dict(controllers=['advanced2']),schedule=[failed]),[failed])
    assert runner.read(tmp_path/'summary.json')['advanced2']['total'] is None


def completed(directory,result,plan_sha):
    attempt = directory/'attempt_001';attempt.mkdir(parents=True)
    runner.write(attempt/'result.json',result)
    runner.write(attempt/'integrity.json',runner.receipt(attempt,plan_sha))
    runner.write(directory/'done.json',dict(attempt=attempt.name,
        receipt_sha256=runner.file_hash(attempt/'integrity.json')))
    return attempt


def test_receipt_rejects_tampering_and_plan_change(tmp_path):
    attempt = completed(tmp_path/'red',red_result(),'plan')
    assert runner.load_done(tmp_path/'red','plan')[0]['placed_count']==10
    with pytest.raises(ValueError,match='changed'):
        runner.load_done(tmp_path/'red','other-plan')
    runner.write(attempt/'result.json',red_result(placed_count=9))
    with pytest.raises(ValueError,match='changed'):
        runner.load_done(tmp_path/'red','plan')


def test_failed_retained_and_retry_preserves_old_attempt(tmp_path,monkeypatch):
    plan = dict(configuration=dict(smoke=False,timeout=1,memory_bytes=4_000_000_000))
    runner.write(tmp_path/'plan.json',plan)
    item = dict(name='fixture',map='Generatedfixture',round=1,seed=42,controller='advanced2',map_file='fixture')
    directory = tmp_path/'runs/fixture_r01/advanced2/red'
    old = completed(directory,dict(status='process_timeout',exit_code=1),runner.file_hash(tmp_path/'plan.json'))
    monkeypatch.setattr(runner,'run_limited',lambda *args:dict(status='process_timeout',exit_code=1))
    assert runner.execute_phase(tmp_path,plan,item,'red')[1]==old
    # Interrupted attempts have no commit marker and must not be overwritten either.
    (directory/'attempt_002').mkdir()
    result,attempt = runner.execute_phase(tmp_path,plan,item,'red',retry_failed=True)
    assert attempt.name=='attempt_003' and old.exists()
    assert result['status']=='process_timeout'


@pytest.fixture
def frozen_plan(tmp_path,monkeypatch):
    from . import scenarios
    monkeypatch.setattr(scenarios,'scenario',lambda *args:dict(stage='semifinal',zones=[],spawn=[100,100],size=[1700,1100]))
    dataset = tmp_path/'dataset';dataset.mkdir()
    data = dict(id='id',name='fixture',family='rooms',map_path='fixture.json',
                content_sha256='12345678'+'0'*56,geometry_sha256='f'*64,files={})
    for name,payload in [('dataset',dict(dataset_id='fixture')),('index',dict(maps=[data])),('train',dict(maps=[data]))]:
        runner.write(dataset/(name+'.json'),payload)
    source = tmp_path/'source.txt';source.write_text('frozen')
    monkeypatch.setattr(runner,'verify_dataset',lambda _:None)
    monkeypatch.setattr(runner,'environment',lambda:dict(test=True))
    monkeypatch.setattr(runner,'source_receipt',lambda _:{str(source):runner.file_hash(source)})
    args = argparse.Namespace(dataset=dataset,split='train',controllers=['advanced2','advanced5_narrow'],
        rounds=2,seed=2026,timeout=240,memory_bytes=4_000_000_000,limit=None,
        smoke=False,new=False,output=tmp_path/'batch')
    return args,source


def test_plan_seed_pairing_resume_and_source_freeze(frozen_plan):
    args,source = frozen_plan
    output,plan = runner.create_or_resume(args)
    assert [r['controller'] for r in plan['schedule']]==['advanced2','advanced5_narrow','advanced5_narrow','advanced2']
    assert plan['schedule'][0]['seed']==plan['schedule'][1]['seed']
    assert plan['schedule'][2]['seed']==plan['schedule'][3]['seed']!=plan['schedule'][0]['seed']
    assert runner.create_or_resume(args)==(output,plan)
    source.write_text('changed')
    with pytest.raises(ValueError,match='mismatch'):
        runner.create_or_resume(args)


def test_resume_rejects_different_rounds(frozen_plan):
    args,_ = frozen_plan
    runner.create_or_resume(args)
    args.rounds = 3
    with pytest.raises(ValueError,match='mismatch'):
        runner.create_or_resume(args)


def test_semifinal_plan_records_conditions(frozen_plan):
    _,plan=runner.create_or_resume(frozen_plan[0])
    assert plan['configuration']['protocol']=='semifinal-r0-v1'
    assert plan['schedule'][0]['scenario']['stage']=='semifinal'


def test_exact_blue_red_binding_on_resume(tmp_path):
    runner.write(tmp_path/'plan.json',dict(configuration=dict(smoke=False)))
    plan = runner.read(tmp_path/'plan.json')
    red = tmp_path/'red';red.mkdir();runner.write(red/'result.json',red_result())
    item = dict(name='fixture',round=1,controller='advanced2')
    directory = tmp_path/'runs/fixture_r01/advanced2/blue'
    completed(directory,dict(status='ok',exit_code=0,phase1=dict(score_status='final'),
        red_result_sha256='different'),runner.file_hash(tmp_path/'plan.json'))
    with pytest.raises(ValueError,match='different red attempt'):
        runner.execute_phase(tmp_path,plan,item,'blue',red)


@pytest.mark.skipif(os.name!='nt',reason='Windows Job Object enforcement')
def test_windows_memory_and_affinity_are_enforced(tmp_path):
    # Child cannot import engine until the job is attached and GO is received.
    script = '''import sys,ctypes as c
assert sys.stdin.readline().strip()=='GO'
from ctypes import wintypes as w
k=c.WinDLL('kernel32'); k.GetCurrentProcess.restype=w.HANDLE
k.GetProcessAffinityMask.argtypes=[w.HANDLE,c.c_void_p,c.c_void_p]
a,b=c.c_size_t(),c.c_size_t()
assert k.GetProcessAffinityMask(k.GetCurrentProcess(),c.byref(a),c.byref(b))
assert a.value.bit_count()==1
try: block=bytearray(384_000_000)
except MemoryError: print('enforced')
else: raise AssertionError('memory cap missing')
'''
    logpath = tmp_path/'resource.log'
    with logpath.open('w') as log:
        result = run_limited([sys.executable,'-S','-c',script],os.environ.copy(),log,10,256_000_000)
    assert result['exit_code']==0,logpath.read_text()
    # Verify allocation rejection, not a bound on the OS diagnostic peak counter.
    assert result['memory_limit_bytes']==256_000_000 and result['peak_job_commit_bytes']>0
    assert 'enforced' in logpath.read_text()


def test_process_timeout_is_terminated(tmp_path):
    with (tmp_path/'timeout.log').open('w') as log:
        result = run_limited([sys.executable,'-S','-c',
            "import sys,time;sys.stdin.readline();time.sleep(10)"],os.environ.copy(),log,.2)
    assert result['status']=='process_timeout' and result['exit_code']!=0


def test_native_merge_uses_all_four_components():
    from ...evaluate_provided_blue import merge
    from swarm_rescue.simulation.reporting.place_score_manager import difficulty_from_blue_score
    source = red_result()
    source['phase1'].update(score_bomb=0,score_status='partial',map_breakdown={},bomb_breakdown={},
        score_exploration_trajectory=0,wall_breakdown={},time_breakdown={})
    merged = merge(source,50,50,0)
    assert merged['score_status']=='final'
    assert merged['score_bomb']==pytest.approx(difficulty_from_blue_score(50))
    assert merged['round_score']==pytest.approx(80*.5+70*.2+90*.1+merged['score_bomb']*.2)
    assert merge(source,50,50,0,crashed=True) is None
