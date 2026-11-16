"""Synthetic behavior and integration contracts; truth only in this test file."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.devtools.test_red_lines11 import ROOM, scan
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper, WallLineMatcher, MyDroneRedLines11
from swarm_rescue.solutions.my_drone_red_mapping import MappingSnapshots, free_points, wall_points
from swarm_rescue.solutions.my_drone_red_p2l17 import (
    point_line_registration, PointLineMatcher17, PointLineMapper17,
    PointLineMapper17Off, MyDroneRedP2L17,
)
from swarm_rescue.solutions.semifinal_localization import scan_points, transform, wrap


def points(refs, angle=0., noise=False):
    d,a=scan(refs,np.zeros(2),angle,np.random.default_rng(742) if noise else None)
    return scan_points(d,a)


@pytest.mark.parametrize('angle',[0.,.41,math.pi-.02,-math.pi+.02])
@pytest.mark.parametrize('noise',[False,True])
def test_corner_with_noise_and_angle_wrap(angle,noise):
    refs=np.array([transform(s,np.zeros(2),angle) for s in ROOM])
    delta,info=point_line_registration(points(refs,angle,noise),refs,np.array([4.,-3.]),angle+.04)
    assert delta is not None,info
    assert info['rank']==2 and info['iterations']<=3
    assert np.linalg.norm(np.array([4.,-3.])+delta[:2])<(2 if noise else .01)
    assert abs(wrap(.04+delta[2]))<(math.radians(1.5) if noise else .001)


@pytest.mark.parametrize('walls',[1,2])
@pytest.mark.parametrize('angle',[0.,.57])
def test_single_wall_and_parallel_corridor_preserve_tangent(walls,angle):
    refs=np.array([[[-210.,-70.],[210.,-70.]],[[-210.,70.],[210.,70.]]])[:walls]
    refs=np.array([transform(s,np.zeros(2),angle) for s in refs])
    tangent=np.array([math.cos(angle),math.sin(angle)])
    normal=np.array([-math.sin(angle),math.cos(angle)])
    prior=9*tangent+5*normal
    delta,info=point_line_registration(points(refs,angle),refs,prior,angle+.035)
    assert delta is not None,info
    assert info['rank']==1
    assert abs(delta[:2]@tangent)<1e-9
    assert abs((prior+delta[:2])@normal)<.01


def test_repeated_similar_parallel_surfaces_reject_ambiguous():
    refs=np.array([[[-200.,-70.],[200.,-70.]],[[-200.,-58.],[200.,-58.]]])
    p=np.column_stack((np.linspace(-150,150,40),np.full(40,-64.)))
    delta,info=point_line_registration(p,refs,np.zeros(2),0.)
    assert delta is None and info['reason']=='support_or_ambiguity'
    assert info['ambiguous']==40


def test_opening_does_not_extrapolate_finite_wall():
    refs=np.array([[[-180.,70.],[-80.,70.]],[[80.,70.],[180.,70.]]])
    p=np.column_stack((np.linspace(-60,60,30),np.full(30,70.)))
    assert point_line_registration(p,refs,np.zeros(2),0.)[0] is None


def test_teammate_occlusion_and_invalid_lidar_do_not_become_wall_or_free():
    a=np.linspace(-math.pi,math.pi,181);d=np.full(181,np.nan)
    d[80:101]=75.;obj=SimpleNamespace(distance=75.,angle=0.,entity_type=SimpleNamespace(name='DRONE'))
    p=scan_points(d,a,[obj]);assert len(p)<12
    unknown=np.full(181,np.nan)
    assert not len(free_points(unknown,a)) and not len(wall_points(unknown,a))
    assert not len(scan_points(np.full(181,230.),a))


def test_bad_initial_pose_and_coupled_degeneracy_reject():
    assert point_line_registration(points(ROOM),ROOM,np.array([80.,80.]),.4)[0] is None
    tiny=np.tile([100.,30.],(30,1))
    delta,info=point_line_registration(tiny,ROOM,np.array([80.,0.]),0.)
    assert delta is None


def test_matching_is_read_only_and_cannot_validate_current_scan():
    matcher=PointLineMatcher17((500,500));matcher.lines=list(ROOM)
    p=points(ROOM);before=p.copy()
    assert matcher.match(p,np.zeros(2),0.)[0] is None
    matcher.add(p,np.zeros(2),0.);refs=np.array(matcher.references).copy()
    matcher.match(p,np.array([3.,-2.]),.02)
    np.testing.assert_array_equal(p,before)
    np.testing.assert_array_equal(matcher.references,refs)


@pytest.mark.parametrize('budget',[0.,-1.])
def test_timeout_and_total_budget_fallback_are_rl11(budget):
    new=PointLineMatcher17((500,500));old=WallLineMatcher((500,500))
    p=points(ROOM)
    for m in (new,old):m.lines=list(ROOM);m.add(p,np.zeros(2),0.)
    if budget==0:new.MAX_CALL_SECONDS=0.
    else:new.MAX_TOTAL_SECONDS=0.
    a,_=new.match(p,np.array([3.,-2.]),.02);b,_=old.match(p,np.array([3.,-2.]),.02)
    np.testing.assert_array_equal(a,b)
    assert new.p2l_stats['fallback_rl11']==1


def test_off_switch_exact_recovery_and_no_old_map_shift():
    a=WallLineMapper((500,500));b=PointLineMapper17Off((500,500))
    for tick in range(100):
        pos=np.array([tick*.2,0.]);d,ang=scan(ROOM,pos,0.)
        gps,compass=(pos,0.) if tick<24 or tick>=76 else (None,None)
        args=(gps,compass,np.array([.2,0.,0.]),d,ang)
        a.update(*args);b.update(*args)
    np.testing.assert_array_equal(a.mask(),b.mask());assert a.stats==b.stats
    assert not b.pending and b.stats['relocks']==1
    # Missing scans cannot move/erase previously confirmed evidence.
    old_hits=b.hits.copy()
    for _ in range(12):b.update(None,None,np.zeros(3),d,ang)
    np.testing.assert_array_equal(b.hits,old_hits)


def test_cache_cap_no_sensing_freeze_and_recovery():
    m=PointLineMapper17((500,500));m.MAX_PENDING=3
    d,a=scan(ROOM,np.zeros(2),0.)
    for k in range(60):m.update(np.zeros(2) if k<20 else None,0. if k<20 else None,np.zeros(3),d,a)
    assert len(m.pending)==3 and m.stats['dropped']>0 and m.mask().any()
    m.update(np.zeros(2),0.,np.zeros(3),d,a)
    assert not m.pending and m.stats['relocks']==1


def test_missing_communication_only_received_versions_are_submitted():
    m=PointLineMapper17((60,60));m.hits[3,4]=3
    sender=MappingSnapshots(m.shape,2,m);receiver=MappingSnapshots(m.shape,2,m)
    sender.publish(1,7,np.zeros(m.shape,bool))
    assert not receiver.submission().any()  # publishing is not delivery
    packet=sender.snapshots[1]
    assert receiver.accept(packet) and not receiver.accept(packet)
    assert receiver.snapshots[1][1]==7 and receiver.submission()[3,4]


def test_navigation_placement_deadlines_and_radio_inherited():
    for name in ('_navigation','_deployment','_submit','control','define_message_for_all','_receive'):
        assert getattr(MyDroneRedP2L17,name) is getattr(MyDroneRedLines11,name)


def test_repair_preserves_every_accepted_rl11_correction():
    from swarm_rescue.solutions.my_drone_red_repair17 import RepairMatcher17
    old=WallLineMatcher((500,500));new=RepairMatcher17((500,500));p=points(ROOM)
    for m in (old,new):m.lines=list(ROOM);m.add(p,np.zeros(2),0.)
    for shift in ([0.,0.],[3.,-2.],[-4.,5.]):
        a,ia=old.match(p,np.array(shift),.02);b,ib=new.match(p,np.array(shift),.02)
        assert a is not None
        np.testing.assert_array_equal(a,b);assert ia==ib
    assert new.p2l_stats['kept_rl11']==3 and new.p2l_stats['attempts']==0


def test_passive_radio_unwraps_real_nested_envelope():
    from swarm_rescue.solutions.devtools.red_p2l17_diagnostics import outgoing_snapshot
    snapshot=(2,150,b'payload')
    message=('red-v4',('red-v3',('red-v1',(),snapshot),None,None),None,())
    assert outgoing_snapshot(message) is snapshot
    assert outgoing_snapshot(None) is None
