import math
from types import SimpleNamespace as NS
import numpy as np
from swarm_rescue.solutions.my_drone_blue_contact import ContactDelivery

NAV=dict(forward=.2,lateral=.1,rotation=-.2,grasper=1)
def cmd(c,step,distance,theta=0.,obstacle=None):
    hit=NS(distance=distance,angle=0.) if distance is not None else None
    rays=np.array([distance if obstacle is None else obstacle]) if distance is not None else np.array([230.])
    return c.command(step,True,hit,np.array([0.,0.,theta]),rays,np.zeros(len(rays)),NAV)

def test_distance_jitter_does_not_reset_actual_contact_turn():
    c=ContactDelivery();cmd(c,0,16.)
    for step in range(1,15):cmd(c,step,17. if step%2 else 22.,.02)
    assert c.active and math.isclose(c.turn,.28)
    assert cmd(c,15,30.)['grasper']==1 and not c.active

def test_short_loss_never_moves_or_releases_and_expires():
    c=ContactDelivery();cmd(c,0,16.);cmd(c,1,17.,.1)
    result=cmd(c,2,None)
    assert result==dict(forward=0.,lateral=0.,rotation=0.,grasper=1)
    assert c.turn==.1
    assert cmd(c,8,None)==NAV and not c.active

def test_commanded_rotation_alone_never_releases():
    c=ContactDelivery()
    for step in range(601):assert cmd(c,step,16.,0.)['grasper']==1
    assert c.retry_until==645 and cmd(c,602,16.)==NAV

def test_missing_lidar_never_replays_a_stale_moving_command():
    c=ContactDelivery()
    result=c.command(1,True,None,None,np.array([]),np.array([]),NAV)
    assert result==dict(forward=0.,lateral=0.,rotation=0.,grasper=1)

def test_release_requires_full_measured_turn_and_stable_near_observation():
    c=ContactDelivery()
    for step in range(50):assert cmd(c,step,16.,.1)['grasper']==1
    for step in range(50,65):result=cmd(c,step,16.,.1)
    assert result['grasper']==0 and c.state=='release_request_unconfirmed'

def test_other_drone_blocks_center_keeps_navigation_and_payload():
    c=ContactDelivery();assert cmd(c,0,65.,obstacle=25.)==NAV
    assert c.state=='obstructed_use_navigation'

def test_payload_disappearance_is_not_success():
    c=ContactDelivery();cmd(c,0,16.)
    assert c.command(1,False,None,None,np.array([]),np.array([]),NAV) is None
    assert c.state=='inactive' and c.turn==0
