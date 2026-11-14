"""RN7 observed clearance and bounded measured-motion behavior."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_rn7 import LocalPassage,PassageMotion,rotate,HULL
from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor


class Coarse:
    visits=np.zeros((1,1),int)
    def cell(self,p):return (0,0)


@pytest.mark.parametrize('angle',[0.,math.pi/4,-math.pi/3])
def test_55px_observed_corridor_encloses_native_hull(angle):
    d,a=corridor(55,angle=angle);fine=LocalPassage(d,a)
    assert fine.BODY_RADIUS>=np.linalg.norm(HULL,axis=1).max()
    assert fine.TRACK_RADIUS>fine.BODY_RADIUS
    assert fine.line_clear(np.zeros(2),80*np.array([math.cos(angle),math.sin(angle)]))
    assert fine.squeeze[fine.cell((0,0))]


def test_46px_aligned_passes_but_oblique_keeps_observation_margin():
    d,a=corridor(46);assert LocalPassage(d,a).line_clear(np.zeros(2),np.array([80.,0.]))
    d,a=corridor(46,angle=math.pi/4)
    assert not LocalPassage(d,a).line_clear(np.zeros(2),80*np.array([2**-.5,2**-.5]))


def test_25px_door_rejected_and_missing_rays_unknown():
    d,a=corridor(25);assert not LocalPassage(d,a).line_clear(np.zeros(2),np.array([60,0]))
    d[:]=np.nan;fine=LocalPassage(d,a)
    assert not fine.free.any() and not fine.squeeze.any()


def test_unknown_edge_cannot_supply_wall_evidence():
    d,a=corridor(400);fine=LocalPassage(d,a)
    fine.free[:]=False;fine.free[100:120,50:170]=True;fine.static[:]=False
    fine._clearance=None
    assert not fine.squeeze.any()


def test_route_swept_through_observed_space_only():
    d,a=corridor(46);fine=LocalPassage(d,a)
    path=fine.route(np.zeros(2),np.array([1,0]),Coarse(),limit=165)
    assert path and np.linalg.norm(path[-1])>60
    assert all(fine.line_clear(x,y) for x,y in zip(path,path[1:]))


def test_actual_odom_moves_every_waypoint_during_turn_and_slip():
    p=PassageMotion();p.path=[np.array([50.,10.])]
    p.update(np.array([3.,.4,math.pi-.01]))
    expected=rotate(np.array([50.,10.])-3*np.array([math.cos(.4),math.sin(.4)]),-math.pi+.01)
    np.testing.assert_allclose(p.path[0],expected)
    assert np.linalg.norm(p.velocity)>0


def test_new_front_wall_brakes_and_releases_instead_of_blind_travel():
    p=PassageMotion();p.tick=100;p.since=100;p.path=[np.array([80.,0.])]
    p.valid_odom=True;p.velocity=np.array([4.,0.]);p.last_progress=100
    d,a=corridor(46,end=24)
    for _ in range(12):
        cmd=p.command(d,a,Coarse(),np.zeros(2),0,0);p.tick+=1
        assert cmd['forward']<0
    assert not p.path
    assert p.stats['fresh_scan_obstructed']==1


def test_peer_wait_is_bounded_and_brakes():
    p=PassageMotion();p.tick=100;p.since=100;p.path=[np.array([80.,0.])]
    p.valid_odom=True;p.velocity=np.array([2.,0.]);p.last_progress=100
    d,a=corridor(46)
    obj=SimpleNamespace(entity_type=SimpleNamespace(name='DRONE'),distance=40,angle=0)
    for _ in range(8):
        cmd=p.command(d,a,Coarse(),np.zeros(2),0,0,[obj]);p.tick+=1
        assert cmd['forward']<0
    assert not p.path and p.stats['peer_wait_budget']==1


@pytest.mark.parametrize('reason',['deadline','health','bomb','submitted'])
def test_mission_preemption_hands_original_controller_back(reason):
    p=PassageMotion();p.path=[np.array([80.,0.])];p.valid_odom=True
    d,a=corridor(46)
    assert p.command(d,a,Coarse(),np.zeros(2),0,0,allow=False) is None
    assert not p.path


def test_off_candidate_does_not_override_navigation():
    from swarm_rescue.solutions.my_drone_red_rn7 import MyDroneRedRN7Off
    assert not MyDroneRedRN7Off.NARROW_ENABLED


def test_failed_odometry_drops_local_path():
    p=PassageMotion();p.path=[np.array([80.,0.])];p.update(None)
    assert not p.path and not p.valid_odom
