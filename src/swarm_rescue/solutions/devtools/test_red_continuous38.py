"""Observable geometry, immutable delivery and actual observation dispatch."""
import math
import numpy as np
import pytest
from unittest.mock import patch
from swarm_rescue.solutions.red_continuous38 import (
    ContinuousMapper38, SharedPose38, fixed_heading_translation, segmented_recovery)
from swarm_rescue.solutions.semifinal_localization import transform

LINES=[np.array([[-90.,60.],[90.,60.]]),np.array([[70.,-90.],[70.,90.]])]

def test_arbitrary_learned_axis_constrains_heading_without_world_axis_assumption():
    m=ContinuousMapper38((400,400));m.position=np.zeros(2);m.heading=.49
    m.structure28.axis=.3;m.step=m.pose_step38=3
    lines=[transform(x,np.zeros(2),-.3) for x in LINES]
    with patch('swarm_rescue.solutions.red_continuous38.fit_lines',return_value=lines):
        m.update(None,None,np.zeros(3),None,None)
    # Measured body wall orientation -.3, world reference .3 -> heading .6.
    assert abs(m.heading-.6)<abs(.49-.6)
    assert m.stats38['axis_corrections']==1

def test_no_learned_axis_does_not_invent_heading():
    m=ContinuousMapper38((400,400));m.position=np.zeros(2);m.heading=.49;m.pose_step38=3
    with patch('swarm_rescue.solutions.red_continuous38.fit_lines',return_value=LINES):
        m.update(None,None,np.zeros(3),None,None)
    assert m.heading==pytest.approx(.49) and not m.stats38['axis_corrections']

def test_corridor_translation_keeps_unobservable_direction():
    lines=[np.array([[-90.,60.],[90.,60.]]),np.array([[-90.,-60.],[90.,-60.]])]
    delta,info=fixed_heading_translation(lines,lines,np.array([17.,4.]),0.)
    assert info['rank']==1
    np.testing.assert_allclose(delta,[0,-4,0],atol=1e-9)

def test_rejection_never_reseeds_protected_reference_with_bad_pose():
    m=ContinuousMapper38((400,400));m.position=np.zeros(2);m.matcher.references=[x.copy() for x in LINES]
    with patch('swarm_rescue.solutions.red_continuous38.fit_lines',return_value=[]):
        for _ in range(40):m.update(None,None,np.zeros(3),None,None)
    np.testing.assert_array_equal(m.trusted_references38,LINES)
    assert m.stats38['reference_preserved_on_reject']==5
    assert m.stats['reference_resets']==0

def test_recovery_preserves_observed_heading_and_scan_confidence():
    frames=[(i,np.zeros((1,2)),np.array([float(i),0.]),.2,.2,np.zeros((0,2))) for i in range(32)]
    meta=[dict(heading_supported=True,uncertainty=1 if i<16 else 6) for i in range(32)]
    out=segmented_recovery(frames,meta,np.array([31.,0.]),np.array([45.,0.]),.5)
    assert all(x[3]==pytest.approx(.2) and x[4]==.2 for x in out)
    np.testing.assert_allclose(out[-1][2],[45,0])
    assert out[15][2][0]-15==pytest.approx(2.)

def test_shared_pose_updates_after_seal_without_mutating_snapshot():
    m=ContinuousMapper38((400,400));p=SharedPose38(m)
    m.update(np.zeros(2),0.,np.zeros(3),None,None);m.seal();old=m.mask();step=m.step
    m.update(None,None,np.array([3.,0.,.02]),None,None)
    a=p.update(None,None,None);b=p.update(None,None,None)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(a[0],[3,0])
    assert a[1]==pytest.approx(.02) and m.step==step and m.pose_step38==2
    np.testing.assert_array_equal(m.mask(),old)

def test_actual_controller_observation_chain_shares_pose_and_preserves_policy():
    from swarm_rescue.solutions.my_drone_red_continuous38 import MyDroneRedContinuous38
    from swarm_rescue.solutions.my_drone_red_entry37r2 import MyDroneRedEntry37R2
    from swarm_rescue.simulation.utils.misc_data import MiscData
    for name in ('_navigation','_deployment','_submit','_receive','define_message_for_all'):
        assert getattr(MyDroneRedContinuous38,name) is getattr(MyDroneRedEntry37R2,name)
    d=MyDroneRedContinuous38(identifier=0,misc_data=MiscData(size_area=(400,400),number_drones=5,max_timestep_limit=5400,max_walltime_limit=180))
    d._progress=lambda:.1
    d.measured_gps_position=lambda:np.zeros(2)
    d.measured_compass_angle=lambda:.3
    d.odometer_values=lambda:np.array([1.,0.,.01])
    d.lidar_values=lambda:np.full(181,230.)
    d.lidar_rays_angles=lambda:np.linspace(-math.pi,math.pi,181)
    d.semantic_values=lambda:()
    for tick in range(1,49):
        d._steps=tick;d.elapsed_timestep=tick+1
        d._update_observation()
        np.testing.assert_allclose(d._position,d._submission_mapper.position)
        np.testing.assert_allclose(d._segment_motion.pose.position,d._position)
        assert d._heading==d._segment_motion.pose.heading==d._submission_mapper.heading
    assert d._submission_mapper.pose_step38==48
