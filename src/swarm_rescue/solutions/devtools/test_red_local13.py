"""RL13 mechanism checks; synthetic inputs are not competition scores."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper
from swarm_rescue.solutions.my_drone_red_local13 import LocalBlockMapper, rigid_blocks
from swarm_rescue.solutions.my_drone_red_entry13 import corridor_direction, EntryMotion, MeasuredSegmentMotion
from swarm_rescue.solutions.my_drone_red_stopturn import MotionEstimate, SegmentMotion
from swarm_rescue.solutions.semifinal_localization import transform, wrap
from swarm_rescue.solutions.devtools.test_red_lines11 import scan, ROOM


@pytest.mark.parametrize('angle',[0.,.37,math.pi-.02,-math.pi+.02])
def test_corridor_guide_is_relative_and_arbitrarily_oriented(angle):
    walls=np.array([[[-60.,-65.],[160.,-65.]],[[-60.,75.],[160.,75.]]])
    lines=[transform(w,np.zeros(2),angle) for w in walls]
    guide=corridor_direction(lines,angle+.15)
    assert guide is not None
    assert abs(wrap(guide['angle']-angle))<1e-9
    assert guide['width']==pytest.approx(140)
    assert abs(guide['center'])==pytest.approx(5)


def test_single_wall_opening_nonparallel_and_route_conflict_fall_back():
    left=np.array([[-60.,-65.],[160.,-65.]])
    right=left+[0,140]
    assert corridor_direction([left],0) is None
    assert corridor_direction([left,right+[0,100]],0) is None
    assert corridor_direction([left,right-400*np.array([1,0])],0) is None
    assert corridor_direction([left,transform(right,np.zeros(2),.2)],0) is None
    assert corridor_direction([left,right],math.pi/2) is None


def frames():
    points=np.array([[1.,2.],[3.,2.]],np.float32)
    return [(k,points,np.array([float(k),.2*k]),.01*k,.45,points/2) for k in range(4,85,4)]


def test_rigid_blocks_preserve_internal_geometry_and_do_not_mutate_tape():
    source=frames();before=[f[2].copy() for f in source]
    result,count=rigid_blocks(source,(0,np.zeros(2)),np.array([90.,18.]),.9,
                              np.array([98.,17.]),1.05,90,span=48,turn=2.)
    assert count==2 and len(result)==len(source)
    for i in range(1,12):
        assert np.linalg.norm(result[i][2]-result[0][2])==pytest.approx(np.linalg.norm(source[i][2]-source[0][2]))
        assert wrap(result[i][3]-result[0][3])==pytest.approx(wrap(source[i][3]-source[0][3]))
    for f,p in zip(source,before):np.testing.assert_array_equal(f[2],p)


def test_identity_recovery_and_heading_wrap_are_exact():
    source=frames()
    result,_=rigid_blocks(source,(0,np.zeros(2)),np.array([90.,18.]),math.pi-.03,
                         np.array([90.,18.]),math.pi-.03,90)
    for old,new in zip(source,result):
        np.testing.assert_allclose(old[2],new[2],atol=1e-12)
        assert abs(wrap(old[3]-new[3]))<1e-12
    result,_=rigid_blocks(source,(0,np.zeros(2)),np.array([90.,18.]),math.pi-.03,
                         np.array([90.,18.]),-math.pi+.03,90)
    assert max(abs(wrap(new[3]-old[3])) for old,new in zip(source,result))<.061


def test_disabled_local_mapper_matches_rl11_with_noise_and_partial_recovery():
    class Off(LocalBlockMapper):LOCAL_BLOCKS=False
    a=WallLineMapper((500,500));b=Off((500,500));rng=np.random.default_rng(742)
    for tick in range(160):
        pos=np.array([tick*.2,0.]);d,angles=scan(ROOM,pos,0.,rng)
        gps=pos if tick<40 or tick>130 else None
        compass=0. if tick<40 or tick>135 else None
        args=(gps,compass,np.array([.2,0.,0.]),d,angles)
        a.update(*args);b.update(*args)
        if tick%11==0:np.testing.assert_array_equal(a.mask(),b.mask())
    np.testing.assert_array_equal(a.hits,b.hits)
    assert a.stats==b.stats


def test_entry_scans_retained_and_recovery_does_not_double_count():
    mapper=LocalBlockMapper((500,500));d,angles=scan(ROOM,np.zeros(2),0.)
    for _ in range(16):mapper.update(np.zeros(2),0.,np.zeros(3),d,angles)
    old=mapper.hits.copy()
    for _ in range(32):mapper.update(None,None,np.zeros(3),d,angles)
    assert len(mapper.pending)==8
    np.testing.assert_array_equal(old,mapper.hits)
    expected=len(mapper.pending)
    mapper.update(np.zeros(2),0.,np.zeros(3),d,angles)
    assert mapper.stats['anchored']==expected
    assert mapper.local_stats['reprojected_scans']==expected
    assert mapper.stats['relocks']==1 and not mapper.pending


def test_entry_hold_is_bounded_without_erasing_or_losing_scans():
    mapper=LocalBlockMapper((500,500));mapper.entry={};mapper.missing=12
    mapper.hits[5,5]=4
    mapper.pending=frames()
    assert mapper.mask()[5,5] and len(mapper.pending)==21
    mapper.missing=24
    np.testing.assert_array_equal(mapper.mask(),WallLineMapper.mask(mapper))


def test_all_motion_increments_include_braking_turning_and_side_slip():
    pose=MotionEstimate((500,500));pose.update(np.zeros(2),math.pi-.02,np.zeros(3))
    expected=np.zeros(2);angle=math.pi-.02
    for d,alpha,theta in [(1.,0.,0.),(3.,.1,.02),(2.,0.,-.03),(.5,math.pi/2,.2),(.1,1.,-.1)]:
        expected+=d*np.array([math.cos(angle+alpha),math.sin(angle+alpha)]);angle=wrap(angle+theta)
        pose.update(None,None,np.array([d,alpha,theta]))
    np.testing.assert_allclose(pose.position,expected)
    assert pose.heading==pytest.approx(angle)


@pytest.mark.parametrize('cls,flag',[(EntryMotion,'ENTRY_ALIGN'),(MeasuredSegmentMotion,'ADAPTIVE_SEGMENTS')])
def test_motion_switch_off_is_original_segment_executor(cls,flag):
    base=SegmentMotion((500,500));candidate=cls((500,500));setattr(candidate,flag,False)
    d=np.full(181,230.);a=np.linspace(-math.pi,math.pi,181)
    for tick in range(80):
        args=(np.zeros(2),0.,np.zeros(3)) if tick<16 else (None,None,np.zeros(3))
        for m in (base,candidate):m.pose.update(*args)
        assert base.step(d,a,.2)==candidate.step(d,a,.2)
        assert base.state==candidate.state


@pytest.mark.parametrize('cls',[EntryMotion,MeasuredSegmentMotion])
def test_motion_preemption_missing_odometry_and_peer_wait_are_bounded(cls):
    m=cls((500,500),identifier=2);p=m.pose
    p.reliable=dict(step=1,position=np.zeros(2),heading=0.);p.missing=5
    p.position=np.zeros(2);p.odometry_valid=False;p.step=10
    d=np.full(181,230.);a=np.linspace(-math.pi,math.pi,181)
    assert m.step(d,a,0.) is None and m.state=='COOLDOWN'
    m.cooldown=0;p.odometry_valid=True;p.velocity=np.array([3.,0.]);p.step=100
    peer=SimpleNamespace(distance=35.,angle=0.,entity_type=SimpleNamespace(name='DRONE'))
    cmd=m.step(d,a,0.,objects=[peer]);assert cmd['forward']<0 and cmd['rotation']==0
    p.step+=15
    assert m.step(d,a,0.,objects=[peer]) is None
    assert m.step(d,a,0.,allow=False) is None


def test_segment_length_keeps_speed_and_requires_measured_space():
    m=MeasuredSegmentMotion((500,500));p=m.pose
    p.step=100;p.missing=30;p.absolute=(False,False);p.odometry_valid=True;p.position=np.zeros(2)
    m.state='OBSERVE';m.since=95;m.episode=80
    d=np.full(181,230.);a=np.linspace(-math.pi,math.pi,181)
    m.step(d,a,0.,target_distance=180.)
    assert m.state=='ACCEL' and m.SEGMENT_LENGTH==140
    assert m.CRUISE_SPEED==SegmentMotion.CRUISE_SPEED
    p.velocity=np.array([4.,0.]);p.step+=1
    d[np.abs(a)<.2]=35.
    cmd=m.step(d,a,0.,target_distance=180.)
    assert m.state=='BRAKE' and cmd['forward']<0 and cmd['rotation']==0


def test_short_waypoint_is_returned_without_start_brake_loop():
    m=MeasuredSegmentMotion((500,500));m.pose.missing=4;m.pose.odometry_valid=True
    m.pose.velocity=np.array([3.,0.]);m.pose.step=100
    assert m.step(np.full(181,230.),np.linspace(-math.pi,math.pi,181),0.,target_distance=20.) is None
    assert m.state=='COOLDOWN' and m.reason=='insufficient_segment_budget'


def test_last_legal_entry_wall_anchor_retained_after_loss():
    m=EntryMotion((500,500));d,a=scan(ROOM,np.zeros(2),.37)
    for _ in range(16):
        m.pose.update(np.zeros(2),.37,np.zeros(3));m.step(d,a,0.,allow=False)
    anchor=m.last_wall_anchor
    assert anchor and anchor['lines'] and anchor['heading_sigma']>0
    m.pose.update(None,None,np.array([.2,0.,.01]));m.step(d,a,0.,allow=False)
    assert m.entry_wall_anchor is anchor
