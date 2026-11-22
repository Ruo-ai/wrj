import math
import numpy as np
import pytest

from swarm_rescue.solutions.red_entry37 import Entry37
from swarm_rescue.simulation.ray_sensors.distance_sensor import compute_ray_angles
from swarm_rescue.simulation.utils.constants import FOV_LIDAR_SENSOR, RESOLUTION_LIDAR_SENSOR

A = compute_ray_angles(math.radians(FOV_LIDAR_SENSOR), RESOLUTION_LIDAR_SENSOR)
D = np.full(RESOLUTION_LIDAR_SENSOR, 230.)


def observed(g, tick, x, good=True, compass=0., odom=(0.,0.,0.)):
    g.observe(tick,np.array([x,0.]),0.,np.array([x,0.]) if good else None,
              compass if good else None,np.array(odom))


def lost():
    g=Entry37()
    for t in range(1,41):
        observed(g,t,float(t*3),odom=(3.,0.,0.))
    for t in range(41,44):
        observed(g,t,float(t*3),False,odom=(3.,0.,0.))
    return g


def test_joint_sensor_loss_requires_recent_traversed_route():
    g=lost()
    command=g.step(D,A,True,.1)
    assert g.active and g.stats['attempts']==1
    assert command['forward'] < 0 and command['rotation']==0
    assert all(p[0]<=120 for p in g.return_path)
    h=Entry37();observed(h,100,10,False)
    assert h.step(D,A,True,.1) is None


@pytest.mark.parametrize('progress,eligible',[(.51,True),(.1,False)])
def test_exploration_and_deadline_reserves(progress,eligible):
    g=lost();assert g.step(D,A,eligible,progress) is None
    assert not g.active


def test_no_late_outage_return():
    g=lost();g.missing=100
    assert g.step(D,A,True,.1) is None


def test_same_entry_does_not_oscillate():
    g=lost();g.step(D,A,True,.1);g.release('test');g.missing=3
    assert g.step(D,A,True,.1) is None
    assert g.stats['attempts']==1


def test_brake_then_retrace_with_no_rotation():
    g=lost();g.step(D,A,True,.1)
    for t in range(44,58):
        observed(g,t,135,False)
        c=g.step(D,A,True,.1)
    assert g.state=='RETREAT'
    assert c['forward']<0 and c['rotation']==0


def test_invalid_odom_and_deadline_preempt():
    g=lost();g.step(D,A,True,.1)
    g.measured=None;assert g.step(D,A,True,.1) is None
    assert g.stats['finish_invalid_observation']==1
    g=lost();g.step(D,A,True,.1)
    assert g.step(D,A,True,.61) is None
    assert g.stats['finish_task_preempted']==1


def test_blocked_retrace_has_finite_exit():
    g=lost();g.step(D,A,True,.1);g.state='RETREAT'
    for t in range(44,100):
        observed(g,t,135,False)
        g.step(np.full_like(D,10.),A,True,.1)
        if not g.active:break
    assert g.stats['finish_blocked']==1


def test_calibration_requires_both_sensors_and_stationarity():
    g=lost();g.step(D,A,True,.1);g.transition('CALIBRATE','test')
    for t in range(44,49):
        observed(g,t,90,True,odom=(3.,0.,0.));g.step(D,A,True,.1)
    assert not g.samples
    g.observe(49,np.array([90.,0.]),0.,np.array([90.,0.]),None,np.zeros(3))
    g.step(D,A,True,.1)
    assert g.state=='RETREAT' and not g.calibrated


def test_full_recovery_alignment_reentry_and_no_map_mutation():
    g=lost();g.step(D,A,True,.1);g.transition('CALIBRATE','test')
    original=D.copy()
    for t in range(44,85):
        observed(g,t,90,True);g.step(D,A,True,.1)
        if g.state=='REENTER':break
    assert g.calibrated and g.state=='REENTER'
    assert g.stats['calibrations']==1
    observed(g,90,125,False,odom=(2.,0.,0.));g.step(D,A,True,.1)
    assert g.stats['reentries']==1
    np.testing.assert_array_equal(D,original)
    assert 'true_position' not in vars(g)


def test_noisy_recovery_does_not_declare_calibration():
    g=lost();g.step(D,A,True,.1);g.transition('CALIBRATE','test')
    for t in range(44,95):
        observed(g,t,90,True,compass=.3 if t%2 else -.3)
        g.step(D,A,True,.1)
    assert not g.calibrated


def test_large_turn_is_not_combined_with_translation():
    g=lost();g.step(D,A,True,.1);g.state='ALIGN';g.good=10;g.absolute=True
    g.target_heading=1.;g.velocity=np.array([3.,0.])
    c=g.step(D,A,True,.1)
    assert c['rotation']==0 and c['forward']<0


def test_episode_budget_is_not_extended_by_state_changes():
    g=lost();g.step(D,A,True,.1);g.tick+=g.MAX_EPISODE
    assert g.step(D,A,True,.1) is None
    assert g.stats['finish_budget']==1
