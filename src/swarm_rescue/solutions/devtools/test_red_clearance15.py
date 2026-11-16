"""RC15 geometry/behavior checks; synthetic observations are not match scores."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_clearance15 import (
    HULL, ClearanceMap, ClearanceMotion, MyDroneRedClearance15, MyDroneRedClearance15Off)
from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor

class Coarse:
    visits=np.zeros((1,1),int)
    obstacle=np.zeros((1,1),np.float32)
    free=np.ones((1,1),bool)
    def cell(self,p):return (0,0)

def active():
    n=ClearanceMotion();n.tick=100;n.since=100;n.last_progress=100
    n.valid_odom=True;n.path=[np.array([80.,0.])];return n

def test_actual_native_polygon_not_interaction_radius():
    from swarm_rescue.simulation.drone.drone_base import DroneBase
    native=DroneBase();v=np.array([tuple(x) for x in native.pm_shapes[0].get_vertices()])
    np.testing.assert_array_equal(HULL,v)
    assert np.ptp(v[:,1])==25 and np.linalg.norm(v,axis=1).max()>16

def test_new_task_selects_observed_goal_and_emits_full_route():
    d,a=corridor(46);n=ClearanceMotion();n.tick=100;n.valid_odom=True
    n.command(d,a,Coarse(),np.zeros(2),0.,0.)
    assert n.path and n.stats['fine_goal_selected']==1
    assert n.audit[-1]['path'] and n.audit[-1]['margin']==2

@pytest.mark.parametrize('width,angle,passes',[(25,0,False),(29,0,False),(36,0,True),
    (46,0,True),(36,math.pi/4,False),(46,math.pi/4,True),(46,-math.pi/3,True)])
def test_full_hull_widths_and_diagonal_passages(width,angle,passes):
    d,a=corridor(width,angle=angle);f=ClearanceMap(d,a)
    assert f.line_clear(np.zeros(2),60*np.array([math.cos(angle),math.sin(angle)]))==passes

def test_rotation_sweep_rejects_intermediate_collision_even_when_endpoints_fit():
    d,a=corridor(36);f=ClearanceMap(d,a)
    assert f.swept_clear([0,0],[0,0],0,0)
    assert f.swept_clear([0,0],[0,0],math.pi/2,math.pi/2)
    assert not f.swept_clear([0,0],[0,0],0,math.pi/2)
    d,a=corridor(60);assert ClearanceMap(d,a).swept_clear([0,0],[0,0],0,math.pi/2)

def test_l_shaped_route_cannot_cut_unknown_corner():
    d,a=corridor(46,end=45);f=ClearanceMap(d,a)
    assert not f.swept_clear([0,0],[80,50],0,math.pi/2)
    assert not f.line_clear([0,0],[300,0])
    d[:]=np.nan;assert not ClearanceMap(d,a).configuration.any()

def test_does_not_mutate_sensor_input_and_shifted_pose_not_global_truth():
    d,a=corridor(46);saved=d.copy();f=ClearanceMap(d,a)
    np.testing.assert_array_equal(d,saved)
    assert f.line_clear([0,0],[60,0])
    assert not f.line_clear([0,12],[60,12])

@pytest.mark.parametrize('noise',[0.,2.5])
def test_noise_and_known_front_wall_cannot_be_ignored(noise):
    d,a=corridor(46,end=27);d+=np.random.default_rng(742).normal(0,noise,len(d))
    n=active();n.velocity=np.array([5.,0.]);c=n.command(d,a,Coarse(),np.zeros(2),0.,0.)
    assert c['forward']<0 and c['rotation']==0

def test_dynamic_peer_wait_and_no_progress_are_bounded():
    d,a=corridor(50);n=active();n.velocity=np.array([2.,0.])
    peer=SimpleNamespace(entity_type=SimpleNamespace(name='DRONE'),distance=40,angle=0)
    for _ in range(9):
        c=n.command(d,a,Coarse(),np.zeros(2),0,0,[peer]);n.tick+=1
        if c is not None:assert c['forward']<=0
    assert not n.path and n.stats['peer_wait_budget']==1
    n=active();n.best_remaining=80.;n.tick=137
    n.command(d,a,Coarse(),np.zeros(2),0,0)
    assert not n.path and n.stats['no_progress']==1

def test_residual_translation_and_wrapped_rotation_update_every_waypoint():
    from swarm_rescue.solutions.my_drone_red_rn7 import rotate
    n=active();n.update(np.array([3.,.4,math.pi-.01]))
    np.testing.assert_allclose(n.path[0],rotate(np.array([80.,0.])-3*np.array([math.cos(.4),math.sin(.4)]),-math.pi+.01))
    assert np.linalg.norm(n.velocity)>0

@pytest.mark.parametrize('why',['deadline','health','bomb','submission','bad_odometry'])
def test_preemption_is_not_blocked_by_narrow_task(why):
    n=active();d,a=corridor(46)
    if why=='bad_odometry':n.update(None)
    else:assert n.command(d,a,Coarse(),np.zeros(2),0,0,allow=False) is None
    assert not n.path

def test_exact_inheritance_for_mapping_deployment_and_off_switch():
    from swarm_rescue.solutions.my_drone_red_lines11 import MyDroneRedLines11
    assert MyDroneRedClearance15._deployment is MyDroneRedLines11._deployment
    assert MyDroneRedClearance15._update_observation is MyDroneRedLines11._update_observation
    assert not MyDroneRedClearance15Off.MIN_CLEARANCE_ENABLED
    assert MyDroneRedClearance15Off.NARROW_ENABLED
