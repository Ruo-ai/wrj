import numpy as np
from swarm_rescue.solutions.my_drone_blue_frontier import FrontierTaskGrid


def test_occluded_unknown_or_free_space_cannot_inflate_gain():
    g=FrontierTaskGrid((640,480));free=np.ones(g.shape,bool);walls=np.zeros(g.shape,bool)
    seen=np.ones(g.shape,bool);walls[:,12]=True;cell=(15,8)
    before=g.visibility_gain(cell,free,walls,seen)
    seen[:,13:]=False
    assert g.visibility_gain(cell,free,walls,seen)==before
    free[:,13:]=False
    assert g.visibility_gain(cell,free,walls,seen)==before


def test_unknown_stops_ray_and_known_observation_point_is_required():
    g=FrontierTaskGrid((640,480));g.free[10:20,3:10]=True
    path=g.route(g.point((15,5)),identifier=0,number=10)
    assert path
    allowed,_=g.layers()
    assert all(allowed[g.cell(p)] for p in path)
    assert g.task_state=='active'


def test_local_replan_retains_region_task_until_timeout_or_progress_failure():
    g=FrontierTaskGrid((640,480));g.free[4:26,3:27]=True
    p=g.point((15,8));g.route(p);task=g.task.copy()
    g.tick=24;g.route(p)
    np.testing.assert_array_equal(task,g.task)
    g.tick=150;g.route(p)
    assert g.stats['blocked']==1
    assert any(np.array_equal(task,q) for q,_ in g.cooldowns)


def test_reconnected_claim_conflict_is_finite_and_reassigns():
    g=FrontierTaskGrid((640,480));g.free[4:26,3:27]=True
    p=g.point((15,8));g.route(p);old=g.task.copy()
    g.priority_claims=[old.copy()];g.tick=24;g.route(p)
    assert g.stats['claim_conflict']==1
    assert g.cooldowns[0][1]==124
    g.priority_claims=[];g.tick=200;g.route(p)
    assert all(t>200 for _,t in g.cooldowns)
    assert not any(np.array_equal(old,q) for q,_ in g.cooldowns)


def test_three_coverage_meanings_remain_separate():
    g=FrontierTaskGrid((640,480));p=np.array([0.,0.]);a=np.linspace(-np.pi,np.pi,181)
    g.observe(p,0.,np.full(181,230.),a,np.zeros(181,bool))
    assert (g.visits>0).sum()==1
    assert g.free.sum()>g.searched.sum()>(g.visits>0).sum()
