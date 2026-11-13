import math
import numpy as np
from swarm_rescue.solutions.my_drone_blue_recovery import BoundaryRecovery,corridor_gap


def setup():
    return BoundaryRecovery(0),np.zeros(2),np.array([300.,0.]),np.linspace(-math.pi,math.pi,181)


def test_complete_route_does_not_trigger_boundary_recovery():
    b,p,t,a=setup();d=np.full(len(a),60.)
    assert b.command(0,p,0,t,d,a,False) is None
    assert b.command(200,p,0,t,d,a,False) is None


def test_partial_route_requires_stagnation_and_keeps_grasp():
    b,p,t,a=setup();d=np.full(len(a),230.);d[np.abs(a)<.3]=55
    assert b.command(0,p,0,t,d,a,True) is None
    cmd=b.command(121,p,0,t,d,a,True)
    assert cmd['grasper']==1 and b.started==121
    assert b.side==1
    b.command(180,p,0,t,d,a,True)
    assert b.side==1


def test_narrow_or_missing_lidar_never_causes_blind_translation():
    for d in (np.full(181,20.),np.full(181,np.nan)):
        b,p,t,a=setup();b.command(0,p,0,t,d,a,True)
        cmd=b.command(121,p,0,t,d,a,True)
        assert cmd['forward']==cmd['lateral']==0 and cmd['grasper']==1


def test_timeout_releases_recovery_not_payload_and_has_cooldown():
    b,p,t,a=setup();d=np.full(181,60.)
    b.command(0,p,0,t,d,a,True);b.command(121,p,0,t,d,a,True)
    assert b.command(841,p,0,t,d,a,True) is None
    assert b.side==-1 and b.command(850,p,0,t,d,a,True) is None


def test_loss_of_payload_or_center_nearby_cancels_recovery():
    b,p,t,a=setup();d=np.full(181,60.)
    b.command(0,p,0,t,d,a,True);b.command(121,p,0,t,d,a,True)
    assert b.command(130,p,0,t,d,a,True,active=False) is None
    assert b.started is None and b.anchor is None


def test_no_recovery_when_real_progress_occurs():
    b,p,t,a=setup();d=np.full(181,60.)
    for step in range(0,600,60):
        p=np.array([step*.8,0.])
        assert b.command(step,p,0,t,d,a,True) is None


def test_departure_requires_progress_and_current_clear_corridor():
    b,p,t,a=setup();d=np.full(181,60.)
    b.command(0,p,0,t,d,a,True);b.command(121,p,0,t,d,a,True)
    assert b.command(200,p,0,t,np.full(181,230.),a,True) is not None
    assert b.command(220,np.array([80.,0.]),0,t,np.full(181,230.),a,True) is None
    assert b.reason=='observed_goal_corridor'


def test_corridor_requires_present_measurement():
    assert corridor_gap(np.array([]),np.array([]),0)==0
    assert corridor_gap(np.array([30.,200.]),np.array([0.,.5]),0)==30
