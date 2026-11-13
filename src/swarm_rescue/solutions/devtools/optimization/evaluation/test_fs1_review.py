from .fs1_review import comparison,events


def test_red_partial_improvement_is_not_full_score_improvement():
    row=dict(map='fixture',round=1,red_seed=1,blue_seed=2,red_valid=True,full_valid=False,
             M=80,W=40,T=30,total=None,placed=10)
    rows=[dict(row,controller='base'),dict(row,controller='candidate',W=45)]
    diff=comparison(rows,'base','candidate')[0]
    assert diff['delta_wall']==5 and diff['delta_red_objective']==1
    assert diff['delta_full'] is None
    rows[-1]['red_valid']=False
    assert comparison(rows,'base','candidate')[0]['delta_wall'] is None


def test_different_blue_seed_is_not_a_pair():
    rows=[dict(map='fixture',round=1,red_seed=1,blue_seed=s,controller=c)
          for s,c in [(2,'base'),(3,'candidate')]]
    assert comparison(rows,'base','candidate')==[]


def test_failure_logs_and_confirmed_events_are_not_dropped():
    log='noise\nV3_EVENT {"event":"placement_confirmed","fs1":{"record_count":1}}\nV3_EVENT {"event":"status"}'
    assert len(events(log))==1
    assert events(log)[0]['fs1']['record_count']==1
