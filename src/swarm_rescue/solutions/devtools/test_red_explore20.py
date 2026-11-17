"""Verify bounded renewal, mission protection, inherited clearance and trace replay."""
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_budget20 import RenewedPassage20
from swarm_rescue.solutions.my_drone_red_rn7 import PassageMotion
from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor
from swarm_rescue.solutions.devtools.test_red_rn7 import Coarse


def call(p,allow=True,width=55):
    d,a=corridor(width)
    return p.command(d,a,Coarse(),np.zeros(2),0.,0.,allow=allow)


def ready(tick=1200):
    p=RenewedPassage20();p.tick=tick;p.valid_odom=True;p.episodes=8;p.cooldown=0
    return p


def test_exhausted_original_stays_off_candidate_retries_observed_corridor():
    old=PassageMotion();old.tick=1200;old.valid_odom=True;old.episodes=8;old.cooldown=0
    assert call(old) is None
    new=ready();assert call(new) is not None
    assert new.renewals==1 and new.lifetime_episodes==9 and new.path


@pytest.mark.parametrize('reason',['time','preempted','odometry','cooldown','active'])
def test_no_renewal_before_time_or_during_protected_mission(reason):
    p=ready();allow=True
    if reason=='time':p.tick=1199
    if reason=='preempted':allow=False
    if reason=='odometry':p.valid_odom=False
    if reason=='cooldown':p.cooldown=1300
    if reason=='active':p.path=[np.array([80.,0.])];p.since=p.tick;p.last_progress=p.tick
    call(p,allow);assert p.renewals==0


def test_lifetime_maximum_24_not_unbounded_restart():
    p=ready();call(p);p.path=[];p.episodes=8;p.tick=2399;p.cooldown=0
    call(p);assert p.renewals==1
    p.tick=2400;call(p);assert p.renewals==2
    p.path=[];p.episodes=8;p.tick=5000;p.cooldown=0
    assert call(p) is None and p.lifetime_episodes==24


def test_renewal_does_not_relax_narrow_or_unknown_clearance():
    p=ready();assert call(p,width=25) is None;assert not p.path
    d,a=corridor(55);d[:]=np.nan;p=ready()
    assert p.command(d,a,Coarse(),np.zeros(2),0.,0.) is None


def test_before_renewal_outputs_and_state_match_original():
    a,b=PassageMotion(),RenewedPassage20()
    d,angles=corridor(55)
    for _ in range(140):
        for p in (a,b):p.update(np.array([.8,0.,0.]))
        ca=a.command(d,angles,Coarse(),np.zeros(2),0.,0.)
        cb=b.command(d,angles,Coarse(),np.zeros(2),0.,0.)
        assert ca==cb and a.stats==b.stats and a.episodes==b.episodes
        np.testing.assert_array_equal(a.path,b.path)


def test_plan_trace_preserves_blocked_tuple_cells_and_decision(tmp_path):
    from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
    from swarm_rescue.solutions.devtools.red_explore20_diagnostics import recording,verify_decisions
    grid=_M19_SurveyGrid((400,400),lambda *a,**k:None);grid.free[:]=True;grid.step=60
    path=tmp_path/'trace.json'
    with recording(path):
        first=grid.plan(np.zeros(2),np.zeros(2),0,5,[],{})
        second=grid.plan(np.zeros(2),np.zeros(2),0,5,[],{first[-1]:999})
    assert second[-1]!=first[-1]
    assert verify_decisions(path)==dict(plans=2,mismatches=[],errors=[],truncated=0)
