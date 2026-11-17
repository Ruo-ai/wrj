"""Synthetic contracts for independent scan odometry and replay diagnostics."""
from types import SimpleNamespace
import numpy as np
import pytest

from swarm_rescue.solutions.devtools.test_red_lines11 import ROOM, scan
from swarm_rescue.solutions.devtools.red_scan18_shadow import (
    ShortWindow18, Scan, make_scan, match_pair, relative, compose, rotation,
)
from swarm_rescue.solutions.devtools.red_recovery18_audit import radio_states


@pytest.mark.parametrize('angle',[0.,.53,3.13,-3.13])
@pytest.mark.parametrize('noisy',[False,True])
def test_corner_relative_motion_and_wrapping(angle,noisy):
    rng=np.random.default_rng(742) if noisy else None
    ref=make_scan(*scan(ROOM,np.zeros(2),angle,rng))
    truth=np.array([3.,-2.,.06]); world=rotation(angle)@truth[:2]
    current=make_scan(*scan(ROOM,world,angle+truth[2],rng))
    answer,info=match_pair(current,ref,truth+np.array([2.,-2.,.035]))
    if noisy and angle==.53:
        # Retained failed-recall fixture: reverse overlap is 44/90 (< .5).
        # Do not weaken the estimator's gate to force an accepted result.
        assert answer is None and info['reason']=='reverse_support'
        return
    assert answer is not None,info
    # Extracted finite lines can include a boundary corner point; allow a
    # quarter pixel, while retaining the stricter exact-wall corridor test.
    assert np.linalg.norm(answer[:2]-truth[:2])<(2 if noisy else .25)
    assert abs(answer[2]-truth[2])<(np.deg2rad(1.5) if noisy else .002)


@pytest.mark.parametrize('walls',[1,2])
def test_single_wall_and_straight_corridor_preserve_unobservable_translation(walls):
    segments=np.array([[[-1000.,-70.],[1000.,-70.]],[[-1000.,70.],[1000.,70.]]])[:walls]
    ref=make_scan(*scan(segments,np.zeros(2),0.))
    current=make_scan(*scan(segments,np.array([8.,2.]),.03))
    prior=np.array([15.,5.,.06]); answer,info=match_pair(current,ref,prior)
    assert answer is not None,info
    assert info['rank']==1
    assert abs(answer[0]-prior[0])<1e-7
    assert abs(answer[1]-2)<.05 and abs(answer[2]-.03)<.002


def test_repeated_room_equal_surfaces_reject_instead_of_counting_two_votes():
    # Deliberate equally plausible parallel surfaces in two repeated rooms.
    segments=np.array([[[-200.,-70.],[200.,-70.]],[[-200.,-58.],[200.,-58.]]])
    points=np.column_stack((np.linspace(-150,150,40),np.full(40,-64.)))
    reference=Scan(points,segments);current=Scan(points,segments)
    answer,_=match_pair(current,reference,np.zeros(3))
    assert answer is None


@pytest.mark.parametrize('value',[np.nan,np.inf,0.,230.])
def test_open_space_and_invalid_rays_fall_back_to_odometry(value):
    angles=np.linspace(-np.pi,np.pi,181);ranges=np.full(181,value)
    with np.errstate(invalid='ignore'):
        estimator=ShortWindow18();estimator.update(np.zeros(2),0.,np.zeros(3),ranges,angles)
        for _ in range(12):estimator.update(None,None,np.array([1.,0.,.01]),ranges,angles)
    assert estimator.stats['accepted']==0
    assert abs(estimator.pose[2]-.12)<1e-10


def test_dynamic_foreground_is_excluded_and_does_not_bridge_occlusion():
    d,a=scan(ROOM,np.zeros(2),0.)
    affected=np.abs(a)<.15;d[affected]=50.
    obj=SimpleNamespace(distance=50.,angle=0.,entity_type=SimpleNamespace(name='DRONE'))
    candidate=make_scan(d,a,[obj])
    assert not np.any(np.abs(np.linalg.norm(candidate.points,axis=1)-50)<1)
    assert match_pair(candidate,make_scan(*scan(ROOM,np.zeros(2),0.)),np.zeros(3))[0] is not None


def test_finite_segments_cannot_fill_unobserved_doorway():
    segments=np.array([[[-180.,70.],[-80.,70.]],[[80.,70.],[180.,70.]]])
    points=np.column_stack((np.linspace(-60,60,30),np.full(30,70.)))
    assert match_pair(Scan(points,segments),Scan(points,segments),np.zeros(3))[0] is None


def test_mixed_radio_event_types_do_not_supply_future_or_duplicate_votes():
    events=[dict(event='local_publish',holder=0,step=1,source=0,version=1),
            dict(event='send_attempt',holder=0,step=2,source=1,version=2),
            dict(event='received',holder=0,step=4,source=1,version=2),
            dict(event='submitted',holder=0,step=5,versions={'0':1,'1':2})]
    states=radio_states(events,0,[1,3,5])
    assert states=={1:{0:1},3:{0:1},5:{0:1,1:2}}


def test_relative_compose_are_inverse_and_scan_matching_is_read_only():
    a=np.array([5.,-2.,3.13]);b=np.array([8.,1.,-3.12])
    np.testing.assert_allclose(compose(a,relative(a,b)),b,atol=1e-10)
    s=make_scan(*scan(ROOM,np.zeros(2),0.));before=s.points.copy()
    match_pair(s,s,np.array([2.,1.,.03]))
    np.testing.assert_array_equal(before,s.points)
