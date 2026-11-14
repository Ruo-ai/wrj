"""S6 legal observation/state/rollback checks; no competition truth in policy."""
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from swarm_rescue.solutions import my_drone_red_stopturn as s6
from swarm_rescue.solutions.my_drone_red_mapping import MyDroneRedMapping

ANGLES = np.linspace(-math.pi, math.pi, 180, endpoint=False)
OPEN = np.full(180, 230.)


def initialized(radar=False):
    m = s6.SegmentMotion((800, 800), radar=radar)
    for _ in range(8): m.pose.update([0.,0.],0.,[0.,0.,0.],OPEN,ANGLES)
    return m


def tick(m, odom=(0.,0.,0.), gps=None, compass=None, desired=0., distances=OPEN, **kwargs):
    m.pose.update(gps, compass, odom, distances, ANGLES)
    return m.step(distances,ANGLES,desired,**kwargs)


def test_all_stages_integrate_actual_motion_in_previous_body_frame():
    m = initialized(); expected=np.zeros(2); heading=0.
    # Accelerating, cruising, decelerating, zero translation, and turning side-slip.
    for d,a,t in [(1.,0.,0.),(3.,0.,0.),(1.,0.,0.),(0.,0.,.2),(1.,math.pi/2,.3),(-.1,0.,-.1)]:
        expected += d*np.array([math.cos(heading+a),math.sin(heading+a)])
        heading = s6.wrap(heading+t)
        m.pose.update(None,None,[d,a,t])
    np.testing.assert_allclose(m.pose.position,expected,atol=1e-10)
    assert m.pose.heading == pytest.approx(heading)
    np.testing.assert_allclose(m.pose.velocity,s6.body_motion([-.1,0.,-.1]))


def test_stopped_rotation_noise_is_not_silently_ignored():
    m=initialized(); start=m.pose.heading_variance
    for _ in range(20): m.pose.update(None,None,[0.,0.,.01])
    assert m.pose.heading == pytest.approx(.2)
    assert m.pose.heading_variance>start


def test_angle_wrap_and_current_body_velocity():
    m=initialized(); m.pose.heading=math.pi-.02
    m.pose.update(None,None,[2.,math.pi/2,.04])
    assert m.pose.heading == pytest.approx(-math.pi+.02)
    np.testing.assert_allclose(m.pose.velocity,[2*math.sin(.04),2*math.cos(.04)])


def test_transient_loss_and_separate_recovery_require_stability():
    m=initialized()
    for _ in range(2): assert tick(m) is None
    assert m.state=='NORMAL'
    tick(m, gps=[0,0])  # Compass alone missing still enters.
    assert m.state in ('BRAKE','SETTLE')
    for _ in range(10): tick(m,gps=[0,0])
    assert m.state!='NORMAL' and not all(m.pose.absolute)
    for _ in range(7): tick(m,gps=[0,0],compass=0.)
    assert m.state=='RECOVER'
    assert tick(m,gps=[0,0],compass=0.) is None and m.state=='NORMAL'


def test_entry_records_uncertain_last_reliable_pose():
    m=initialized()
    for _ in range(3): tick(m)
    assert m.entry_anchor['step']==8
    assert m.entry_anchor['position_sigma']>=5
    assert m.entry_anchor['heading_sigma']>=math.radians(4)


def test_moving_must_brake_and_confirm_stop_before_turning():
    m=initialized()
    for _ in range(3): cmd=tick(m,odom=(4.,0.,0.),desired=math.pi/2)
    assert m.state=='BRAKE' and cmd['forward']<0 and cmd['rotation']==0
    for _ in range(2):
        cmd=tick(m,desired=math.pi/2)
        assert cmd['rotation']==0
    tick(m,desired=math.pi/2)
    assert m.state=='OBSERVE'
    for _ in range(2): tick(m,desired=math.pi/2)
    assert m.state=='TURN'
    assert tick(m,desired=math.pi/2)['rotation']>0


def test_approaching_wall_brakes_early():
    m=initialized()
    for _ in range(3): tick(m)
    m.state='CRUISE'; m.since=m.pose.step; m.origin=m.pose.position.copy();m.target_heading=m.pose.heading
    d=OPEN.copy(); frontal=np.abs(ANGLES)<.5;d[frontal]=50/np.cos(ANGLES[frontal])
    cmd=tick(m,odom=(4.,0.,0.),distances=d)
    assert m.state=='BRAKE' and cmd['forward']<0
    assert m.last_brake_distance>28


@pytest.mark.parametrize('width',[27.,35.])
def test_narrow_door_does_not_allow_unverified_in_place_rotation(width):
    m=initialized()
    for _ in range(3): tick(m)
    m.transition('OBSERVE','fixture');m.since-=3
    d=np.minimum(230.,(width/2)/np.maximum(.0001,np.abs(np.sin(ANGLES))))
    # The center ray is open; a conservative 22px sweep does not fit here.
    assert d[np.argmin(np.abs(ANGLES))]==230.
    assert not s6.turn_clear(d,ANGLES)
    assert m.step(d,ANGLES,.2) is None
    assert m.reason=='rotation_clearance'


def test_missing_and_partial_lidar_never_mean_free_space():
    assert s6.ray_clearance(None,ANGLES,0)==0
    d=OPEN.copy();d[np.abs(ANGLES)<.3]=np.nan
    assert s6.ray_clearance(d,ANGLES,0)==0 and not s6.turn_clear(d,ANGLES)
    m=initialized()
    for _ in range(3): tick(m)
    assert tick(m,odom=(2.,0.,0.),distances=None)['forward']<0


def test_peer_wait_has_finite_timeout_without_messages():
    m=initialized(); peer=NS(entity_type=NS(name='DRONE'),distance=40.,angle=0.)
    for _ in range(3): tick(m,objects=[peer])
    assert m.state=='WAIT'
    for _ in range(9): tick(m,objects=[peer])
    assert m.state=='COOLDOWN' and m.stats['fallback_peer_wait_budget']==1


def test_dead_end_and_unprogressive_turn_eventually_yield_to_original():
    m=initialized()
    for _ in range(3): tick(m)
    m.transition('TURN','fixture');m.target_heading=math.pi/2
    for _ in range(57): tick(m,desired=math.pi/2)
    assert m.state=='COOLDOWN' and m.reason=='state_timeout'


@pytest.mark.parametrize('reason',['low_health','deadline','deployment','submission','parent_escape'])
def test_original_task_safety_preemption_is_immediate(reason):
    m=initialized()
    for _ in range(3): tick(m)
    assert tick(m,allow=False) is None
    assert m.state=='COOLDOWN'


def test_recovery_correction_is_bounded_not_absolute_map_jump():
    m=initialized()
    for _ in range(30): m.pose.update(None,None,[0,0,0])
    for _ in range(7): m.pose.update([100,0],1.,[0,0,0])
    np.testing.assert_array_equal(m.pose.position,[0,0])
    m.pose.update([100,0],1.,[0,0,0])
    assert np.linalg.norm(m.pose.position)<=2 and abs(m.pose.heading)<=.04+1e-12


def test_corridor_rank_is_orientation_independent():
    x=np.linspace(-150,150,50)
    corridor=np.vstack((np.c_[x,np.full(50,50)],np.c_[x[::-1],np.full(50,-50)]))
    for theta in (0.,.3,.8,1.5):
        c,s=np.cos(theta),np.sin(theta)
        assert s6.wall_rank(corridor@np.array([[c,s],[-s,c]]))==1
    corner=np.vstack((np.c_[x,np.full(50,50)],np.c_[np.full(50,150),x[::-1]]))
    assert s6.wall_rank(corner)==2


def test_radar_matches_before_add_and_rejects_degenerate_geometry(monkeypatch):
    m=initialized(True); calls=[]
    class Match:
        frames=[];distance=None
        def match(self,*args): calls.append('match'); return np.array([0.,1.,.01]),{'observable':[True,True]}
        def add(self,*args): calls.append('add')
    m.pose.matcher=Match()
    monkeypatch.setattr(s6,'wall_rank',lambda p:2)
    for _ in range(8): m.pose.update(None,None,[0,0,0],OPEN,ANGLES)
    assert calls==['match','add'] and m.pose.stats['matches']==1
    calls.clear();monkeypatch.setattr(s6,'wall_rank',lambda p:1)
    for _ in range(8): m.pose.update(None,None,[0,0,0],OPEN,ANGLES)
    assert calls==['add'] and m.pose.stats['degenerate_or_sparse']==1


def test_submission_deployment_and_radio_are_retained():
    assert not s6.MyDroneRedStopTurnOff.SEGMENTED_MOTION
    assert not s6.MyDroneRedStopTurn.MOTION_RADAR and s6.MyDroneRedStopTurnRadar.MOTION_RADAR
    for name in ('control','_deployment','_submit','define_message_for_all','_receive'):
        assert getattr(s6.MyDroneRedStopTurn,name) is getattr(MyDroneRedMapping,name)


@pytest.mark.parametrize('override',[
    {'health':8}, {'progress':.60}, {'_attempt_step':3},
    {'_exploration_submitted':True}, {'_gathering':True},
    {'_escape_until':100}, {'_yield_until':100},
    {'_bomb_target':np.array([20.,0.])}, {'_route':[]},
])
def test_actual_navigation_adapter_respects_preemptions(monkeypatch,override):
    class Probe(s6.MyDroneRedStopTurn):
        def __init__(self): pass
        @property
        def drone_health(self): return self.health
        def _progress(self): return self.progress
        def lidar_values(self): return OPEN
        def lidar_rays_angles(self): return ANGLES
        def semantic_values(self): return []
    monkeypatch.setattr(MyDroneRedMapping,'_navigation',lambda self:dict(forward=.123,lateral=.234,rotation=0,grasper=1))
    obj=Probe();obj._segment_motion=initialized()
    for _ in range(3): tick(obj._segment_motion)
    defaults=dict(health=25,progress=.2,_steps=20,_route=[0],_position=np.zeros(2),_bomb_target=None,
        _exploration_submitted=False,_attempt_step=None,_gathering=False,_escape_until=0,_yield_until=0,
        _last_motion_angle=0.,_waypoint_progress=NS(last_advance=0),_explore=NS(point=lambda _:np.array([160.,0.])))
    for key,value in {**defaults,**override}.items(): setattr(obj,key,value)
    assert obj._navigation()==dict(forward=.123,lateral=.234,rotation=0,grasper=1)
    assert obj._segment_motion.state=='COOLDOWN'


def test_off_adapter_does_not_access_new_sensors_or_motion(monkeypatch):
    class Probe(s6.MyDroneRedStopTurnOff):
        def __init__(self): pass
    obj=Probe();token=object()
    monkeypatch.setattr(MyDroneRedMapping,'_navigation',lambda self:token)
    monkeypatch.setattr(MyDroneRedMapping,'_update_observation',lambda self:None)
    assert obj._navigation() is token
    obj._update_observation()  # no private estimator or sensor access needed
