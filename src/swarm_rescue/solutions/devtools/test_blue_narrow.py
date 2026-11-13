"""Behavioral checks of native-model footprint and conservative local gates."""
import math
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_blue_narrow import HULL,NarrowPassage,corridor_geometry,lateral_support


def corridor(width,angle=0.,offset=0.,noise=0.,seed=742):
    a=np.linspace(-math.pi,math.pi,181);relative=a+angle
    sine=np.sin(relative)
    boundary=np.where(sine>0,width/2-offset,-width/2-offset)
    with np.errstate(divide='ignore',invalid='ignore'):d=np.minimum(230.,boundary/sine)
    d=np.clip(d,0,230)+np.random.default_rng(seed).normal(0,noise,len(a))
    return d,a


def test_native_collision_hull_and_orientation_are_not_a_radius_guess():
    from swarm_rescue.simulation.drone.drone_base import DroneBase
    actual=DroneBase()
    np.testing.assert_array_equal(HULL,[tuple(v) for v in actual.pm_shapes[0].get_vertices()])
    assert lateral_support(0)==(-12.,13.)
    assert lateral_support(math.pi/4)[1]-lateral_support(math.pi/4)[0]>31


@pytest.mark.parametrize('angle',[-.15,0.,.15])
def test_measured_parallel_walls_recover_width_and_heading(angle):
    d,a=corridor(35,angle,2,noise=.3)
    g=corridor_geometry(d,a,-angle)
    assert g is not None and abs(g['width']-35)<1.2
    assert abs(g['angle']+angle)<.03
    assert abs(g['center']+2)<1.


def test_single_wall_is_not_a_confirmed_doorway():
    d,a=corridor(35);d[a<0]=230
    assert corridor_geometry(d,a) is None


def test_payload_never_uses_empty_passage_admission_or_releases():
    d,a=corridor(35);gate=NarrowPassage()
    for t in range(15):assert gate.step(d,a,[0,0,0],t,carrying=True) is None
    assert gate.stats['attempts']==0


def test_too_narrow_stays_stopped_then_cools_down():
    d,a=corridor(25.5);gate=NarrowPassage()
    for t in range(10):cmd=gate.step(d,a,[0,0,0],t)
    assert gate.reason=='width_uncertain' and gate.stats['entered']==0
    assert cmd['forward']==0
    assert gate.step(d,a,[0,0,0],11) is None


def test_missing_scan_and_dynamic_blocker_end_attempt_without_blind_reversal():
    d,a=corridor(35)
    for failure in ('scan','peer'):
        gate=NarrowPassage();gate.step(d,a,[0,0,0],0)
        cmd=gate.step(np.full_like(d,np.nan) if failure=='scan' else d,a,[0,0,0],1,dynamic=failure=='peer')
        assert gate.state=='cooldown'
        assert cmd['forward']==cmd['lateral']==0 and cmd['grasper']==0


def test_no_progress_is_bounded_even_with_perfect_unchanging_scan():
    d,a=corridor(35,offset=-.5);gate=NarrowPassage()
    for t in range(130):
        gate.step(d,a,[0,0,0],t)
        if gate.reason:break
    assert gate.reason=='no_progress'
    assert gate.stats['passed']==0


def test_geometry_loss_is_not_reported_as_successful_exit():
    d,a=corridor(35,offset=-.5);gate=NarrowPassage()
    for t in range(12):gate.step(d,a,[0,0,0],t)
    for t in range(12,20):gate.step(np.full_like(d,230.),a,[0,0,0],t)
    assert gate.stats['passed']==0 and gate.reason=='geometry_lost'


def test_repeated_failures_have_a_whole_round_attempt_limit():
    gate=NarrowPassage();gate.stats['attempts']=gate.MAX_ATTEMPTS
    d,a=corridor(35)
    assert gate.step(d,a,[0,0,0],600) is None
    assert gate.stats['attempt_budget_exhausted']==1


def test_visible_bomb_approach_cannot_overwrite_checked_narrow_motion(monkeypatch):
    from types import SimpleNamespace
    from swarm_rescue.solutions.my_drone_blue_narrow import MyDroneBlueNarrow
    from swarm_rescue.solutions.my_drone_blue_transport import MyDroneBlueTransportCompute
    checked=dict(forward=.02,lateral=.01,rotation=.005,grasper=0)
    def inherited(drone):
        drone._narrow_action=checked.copy()
        return dict(forward=.3,lateral=0.,rotation=.7,grasper=1)
    monkeypatch.setattr(MyDroneBlueTransportCompute,'control',inherited)
    monkeypatch.setattr(MyDroneBlueNarrow,'grasped_bombs',lambda self:[])
    drone=MyDroneBlueNarrow.__new__(MyDroneBlueNarrow);drone._narrow=SimpleNamespace(state='transit')
    assert drone.control()==dict(checked,grasper=1)
