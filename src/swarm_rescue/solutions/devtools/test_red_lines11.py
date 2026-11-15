"""Deterministic, synthetic geometry; no competition truth in online classes."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_lines11 import (
    fit_lines,line_registration,WallLineMatcher,WallLineMapper,MyDroneRedLines11,
    MyDroneRedLines11Slow,MyDroneRedLines11Off)
from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper
from swarm_rescue.solutions.my_drone_red_rn7 import MyDroneRedRN7B
from swarm_rescue.solutions.semifinal_localization import transform,wrap


def scan(segments,position,heading,rng=None):
    angles=np.linspace(-math.pi,math.pi,181);ray=angles+heading
    directions=np.column_stack((np.cos(ray),np.sin(ray)));d=np.full(len(ray),230.)
    for p,q in np.asarray(segments):
        edge=q-p;rel=p-position
        denominator=directions[:,0]*edge[1]-directions[:,1]*edge[0]
        safe=np.where(np.abs(denominator)>1e-9,denominator,np.nan)
        distance=(rel[0]*edge[1]-rel[1]*edge[0])/safe
        fraction=(rel[0]*directions[:,1]-rel[1]*directions[:,0])/safe
        good=(distance>0)&(fraction>=0)&(fraction<=1)&(np.abs(denominator)>1e-9)
        d=np.minimum(d,np.where(good,distance,230.))
    if rng is not None:d+=rng.normal(0,2.5,len(d))
    return d,angles


ROOM=np.array([[[-160.,-100.],[180.,-100.]],[[180.,-100.],[180.,130.]],
               [[180.,130.],[-160.,130.]],[[-160.,130.],[-160.,-100.]]])


@pytest.mark.parametrize('angle',[0.,.41,math.pi-.02,-math.pi+.02])
def test_arbitrary_angle_registration_and_wrap(angle):
    refs=np.array([transform(s,np.zeros(2),angle) for s in ROOM])
    pos=np.array([12.,-8.]);heading=angle+.3
    lines=[transform(s-pos,np.zeros(2),-heading) for s in refs]
    prior=pos+np.array([4.,-3.]);prior_heading=heading+.045
    offset,info=line_registration(lines,refs,prior,prior_heading)
    assert offset is not None and info['rank']==2
    np.testing.assert_allclose(prior+offset[:2],pos,atol=1e-7)
    assert abs(wrap(prior_heading+offset[2]-heading))<1e-7


def test_parallel_corridor_only_constrains_normal_and_rotation():
    refs=np.array([[[-200.,-70.],[200.,-70.]],[[-200.,70.],[200.,70.]]])
    offset,info=line_registration(refs,refs,np.array([9.,5.]),.04)
    assert offset is not None and info['rank']==1
    assert offset[0]==pytest.approx(0.)
    assert offset[1]==pytest.approx(-5.) and offset[2]==pytest.approx(-.04)


def test_single_wall_and_inconsistent_angle_do_not_claim_full_pose():
    assert line_registration(ROOM[:1],ROOM[:1],np.zeros(2),0.)[0] is None
    damaged=[transform(ROOM[0],np.zeros(2),.12),transform(ROOM[1],np.zeros(2),-.12)]
    assert line_registration(damaged,ROOM,np.zeros(2),0.)[0] is None


def test_observed_segments_do_not_bridge_unknown_and_exclude_semantics():
    d,a=scan(ROOM,np.zeros(2),0.)
    lines=fit_lines(d,a)
    assert len(lines)>=3
    assert fit_lines(np.full(181,230.),a)==[]
    keep=d.copy();keep[(a>-.4)&(a<.4)]=230.
    for line in fit_lines(keep,a):
        assert not (line[:,0].mean()>175 and line[:,1].min()<-40 and line[:,1].max()>40)
    obj=SimpleNamespace(distance=100.,angle=math.pi/2)
    assert isinstance(fit_lines(d,a,[obj]),list)


def test_matching_cannot_insert_current_scan_or_invent_axis():
    match=WallLineMatcher((500,500));match.lines=list(ROOM)
    points=np.vstack(ROOM);match.add(points,np.zeros(2),.34)
    before=[p.copy() for p in match.references];match.match(points,np.zeros(2),.38)
    for a,b in zip(before,match.references):np.testing.assert_array_equal(a,b)
    assert match.stats['accepted']==1


def test_disabled_mapper_is_exact_a3():
    class Off(WallLineMapper):LINE_REGISTRATION=False
    a=SubmissionMapper((500,500));b=Off((500,500))
    rng=np.random.default_rng(742)
    for tick in range(56):
        pos=np.array([tick*.3,0.]);d,angles=scan(ROOM,pos,0.,rng)
        gps,compass=(pos,0.) if tick<24 or tick>47 else (None,None)
        args=(gps,compass,np.array([.3,0.,0.]),d,angles)
        a.update(*args);b.update(*args)
    np.testing.assert_array_equal(a.mask(),b.mask())
    np.testing.assert_array_equal(a.position,b.position);assert a.stats==b.stats


def test_mapping_candidate_preserves_navigation_and_placement(monkeypatch):
    token=dict(forward=.2,lateral=.1,rotation=.03)
    monkeypatch.setattr(MyDroneRedRN7B,'_navigation',lambda self:token)
    actor=object.__new__(MyDroneRedLines11)
    assert actor._navigation() is token
    assert MyDroneRedLines11._deployment is MyDroneRedRN7B._deployment
    assert not MyDroneRedLines11Slow.LINE_MAPPING and MyDroneRedLines11Slow.SLOW_OUTAGE
    assert not MyDroneRedLines11Off.LINE_MAPPING and not MyDroneRedLines11Off.SLOW_OUTAGE


def test_slow_guard_measured_motion_brakes_before_turn(monkeypatch):
    monkeypatch.setattr(MyDroneRedRN7B,'_navigation',lambda self:dict(forward=.7,lateral=.2,rotation=.35))
    actor=object.__new__(MyDroneRedLines11Slow)
    actor._l11_active=True;actor._progress=lambda:.3;actor._gathering=False;actor._attempt_step=None
    actor._rn7=SimpleNamespace(path=[]);actor._l11_stats={k:0 for k in ('turn_translation_brake','guard_ticks')}
    monkeypatch.setattr(MyDroneRedLines11Slow,'drone_health',property(lambda self:25))
    actor.odometer_values=lambda:np.array([4.,0.,0.])
    cmd=actor._navigation();assert cmd['forward']<0 and cmd['rotation']==0


def test_legal_recovery_confirmation_separate_sensor_returns(monkeypatch):
    monkeypatch.setattr(MyDroneRedRN7B,'_update_observation',lambda self:None)
    actor=object.__new__(MyDroneRedLines11)
    actor._l11_missing=actor._l11_good=0;actor._l11_active=False
    actor.measured_gps_position=lambda:None;actor.measured_compass_angle=lambda:0.
    for _ in range(3):actor._update_observation()
    assert actor._l11_active
    actor.measured_gps_position=lambda:np.zeros(2);actor.measured_compass_angle=lambda:None
    for _ in range(10):actor._update_observation()
    assert actor._l11_active
    actor.measured_compass_angle=lambda:0.
    for _ in range(7):actor._update_observation()
    assert actor._l11_active
    actor._update_observation();assert not actor._l11_active
