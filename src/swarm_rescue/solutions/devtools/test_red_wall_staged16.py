"""A/B independent synthetic legal-observation tests, not full-match scores."""
import math
import json
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_wall16 import SingleWallMotion,choose_wall,match_wall,supported_sweep
from swarm_rescue.solutions.my_drone_red_staged16 import StagedMapper,ValidatedSnapshots,overlap_check
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper,fit_lines
from swarm_rescue.solutions.semifinal_localization import transform,wrap
from swarm_rescue.solutions.devtools.test_red_lines11 import scan,ROOM

WALL=np.array([[-150.,-75.],[160.,-75.]])

@pytest.mark.parametrize('angle',[0.,.32,-.54,math.pi-.02])
def test_one_oblique_wall_four_directions_and_clearance(angle):
    wall=transform(WALL,np.zeros(2),angle);d,a=scan([wall],np.zeros(2),0.)
    result=choose_wall([wall],angle,d,a)
    assert result is not None and abs(wrap(result['direction']-angle))<1e-6
    assert abs(result['relation'])<1e-6
    toward=choose_wall([wall],angle-math.pi/2,d,a)
    # Perpendicular toward a wall 75px away may safely support the 42px
    # observation move; toward a 25px wall may never do so.
    close=transform(WALL+[0,50],np.zeros(2),angle);d,a=scan([close],np.zeros(2),0.)
    assert choose_wall([close],angle-math.pi/2,d,a) is None

def test_short_wall_occlusion_and_duplicate_associations_are_rejected():
    d,a=scan([WALL],np.zeros(2),0.)
    assert choose_wall([np.array([[0.,-75.],[35.,-75.]])],0.,d,a) is None
    assert match_wall(WALL,[WALL,WALL+[0,.1]]) is None
    assert match_wall(WALL,[WALL+[500,0]]) is None
    assert fit_lines(np.full(181,230.),a)==[]

def test_real_hull_rotation_sweep_and_unknown_space():
    walls=np.array([[[-200.,-18.],[200.,-18.]],[[-200.,18.],[200.,18.]]])
    d,a=scan(walls,np.zeros(2),0.)
    assert not supported_sweep(d,a,[0,0],math.pi/2)
    d[:]=np.nan;assert not supported_sweep(d,a,[1,0],0.)

def test_noisy_single_wall_is_fit_without_two_sided_requirement():
    d,a=scan([WALL],np.zeros(2),0.,np.random.default_rng(742))
    lines=fit_lines(d,a);assert lines
    assert choose_wall(lines,0.,d,a) is not None

def motion_pose(m,step=30,missing=3):
    p=m.pose;p.step=step;p.missing=missing;p.absolute=(False,False);p.odometry_valid=True
    p.position=np.zeros(2);p.velocity=np.zeros(2);p.angular_increment=0.;p.reliable=dict(step=20,position=np.zeros(2),heading=0.)

def test_stopped_aligned_wall_requires_retest_then_motion():
    m=SingleWallMotion((500,500));motion_pose(m);d,a=scan([WALL],np.zeros(2),0.)
    for k in range(18):
        m.pose.step=30+k;m.step(d,a,0.,160.,allow=True)
    assert m.wall_stats['alignment_verified']==1 and m.wall_stats['aligned_motion']>0
    json.dumps(list(m.wall_events))

@pytest.mark.parametrize('reason',['odometry','peer','task','timeout','wall_lost'])
def test_alignment_bounded_fallback(reason):
    m=SingleWallMotion((500,500));motion_pose(m);d,a=scan([WALL],np.zeros(2),0.)
    m.step(d,a,0.,allow=True)
    objects=[];allow=True
    if reason=='odometry':m.pose.odometry_valid=False
    if reason=='peer':objects=[SimpleNamespace(entity_type=SimpleNamespace(name='DRONE'),distance=40.,angle=0.)]
    if reason=='task':allow=False
    if reason=='timeout':m.pose.step+=126
    if reason=='wall_lost':
        m.align_state='TURN';m.wall=WALL.copy();m.good_wall_step=0;d[:]=230
    m.step(d,a,0.,objects=objects,allow=allow)
    assert m.align_state=='IDLE' and any(k.startswith('fallback_') for k in m.wall_stats)

def test_unique_nonparallel_overlap_and_shifted_submaps():
    r=overlap_check(ROOM,ROOM);assert r['rank']==2 and r['matches']>=3
    assert overlap_check(ROOM+np.array([45,35]),ROOM)['matches']==0
    assert overlap_check(ROOM,np.repeat(ROOM,2,axis=0))['ambiguous']>0
    corridor=np.array([WALL,WALL+[0,150]])
    assert overlap_check(corridor,corridor)['rank']==1

def feed(m,steps=32,gps=True):
    d,a=scan(ROOM,np.zeros(2),0.)
    for _ in range(steps):m.update(np.zeros(2) if gps else None,0. if gps else None,np.zeros(3),d,a)
    return d,a

def test_outage_drains_parent_provisional_without_changing_trusted_hits():
    m=StagedMapper((500,500));feed(m);old=m.hits.copy();feed(m,48,False)
    assert m.pending==[] and m.open_block is not None
    assert len(m.open_block['frames'])==12
    np.testing.assert_array_equal(m.hits,old)

def test_recovery_requires_eight_frames_and_preserves_old_map():
    m=StagedMapper((500,500));feed(m);feed(m,64,False)
    feed(m,1,True);assert m.recovery is not None and not m.stage_stats['recovery_checked']
    feed(m,7,True);assert m.stage_stats['recovery_checked']==1
    assert m.stage_stats['block_validated']>=1

def block(mapper,age=96,matched=12):
    points=np.array([[10.,10.],[13.,10.],[16.,10.]],np.float32)
    frames=[(k,points,np.zeros(2),0.,.45,np.empty((0,2),np.float32)) for k in range(age-44,age+1,4)]
    return dict(id=1,episode=1,origin=np.zeros(2),heading=0.,start=0,end=age,frames=frames,
        references=list(ROOM),lines=list(ROOM),matched=matched,match_checks=12,distance=20.,recovery=None)

def test_no_second_drone_short_supported_region_not_blanket_discarded():
    m=StagedMapper((500,500));grade,info=m.grade(block(m))
    assert grade=='short_entry_supported'
    b=block(m,900);b['distance']=600
    assert m.grade(b,True)[0] is None

def packet(source=1,seq=40,parents=None,origin='absolute_local'):
    return (source,seq,origin,(source,) if parents is None else parents,5.,math.radians(4),
        tuple(tuple(tuple(float(x) for x in p) for p in s) for s in ROOM))

def test_actual_message_only_fresh_versions_and_common_reference_not_independent():
    m=StagedMapper((500,500),0)
    assert not m.witnesses
    assert m.accept_witness(packet(),50,5)
    assert not m.accept_witness(packet(),50,5)
    assert not m.accept_witness(packet(2,41,parents=(0,2)),50,5)
    assert not m.accept_witness(packet(3,42,origin='relayed'),50,5)
    assert not m.accept_witness(packet(4,1),500,5)
    assert m.stage_stats['witness_duplicate']==1

def test_peer_match_needs_own_anchor_and_two_observable_directions():
    m=StagedMapper((500,500),0);m.step=200;m.accept_witness(packet(seq=200),200,5)
    b=block(m,age=180);assert m.grade(b)[0]=='entry_plus_independent_peer'
    b['references']=[];assert m.grade(b)[0] is None
    b['references']=list(ROOM);b['lines']=[WALL,WALL+[0,150]]
    assert m.grade(b)[0] is None

def test_deadline_and_capacity_are_bounded_and_reasoned():
    m=StagedMapper((500,500));m.step=950
    m.blocks=[block(m,900) for _ in range(m.MAX_BLOCKS+1)]
    for b in m.blocks:b['distance']=800
    m.validate(capacity=True);assert len(m.blocks)<=m.MAX_BLOCKS
    m.set_progress(.79);assert not m.blocks and m.stage_stats['block_not_submitted']>=1
    assert all(e['cells']>=0 for e in m.stage_events if e['event']=='block_not_submitted')

def test_snapshot_versions_navigation_unchanged_free_conflicts_defer_tentative():
    a=StagedMapper((60,60),0);b=StagedMapper((60,60),1);shape=a.shape
    sa=ValidatedSnapshots(shape,2,a);sb=ValidatedSnapshots(shape,2,b)
    nav=np.zeros(shape,bool);nav[5,5]=True;a.validated[8,8]=True;a.hits[9,9]=3
    sa.publish(0,30,nav);assert sb.accept(sa.snapshots[0]);assert not sb.accept(sa.snapshots[0])
    b.free_support[8,8]=5;sb.publish(1,31,np.zeros(shape,bool))
    assert sb.merged()[5,5] and sb.submission()[9,9]
    assert not sb.submission()[8,8] and sb.conflict_cells[0]==1

def test_a_b_are_separate_rl11_extensions_and_off_inherits_original_mapper():
    from swarm_rescue.solutions.my_drone_red_wall16 import MyDroneRedWall16
    from swarm_rescue.solutions.my_drone_red_staged16 import MyDroneRedStaged16
    from swarm_rescue.solutions.my_drone_red_lines11 import MyDroneRedLines11
    assert MyDroneRedWall16._deployment is MyDroneRedLines11._deployment
    assert MyDroneRedStaged16._navigation is MyDroneRedLines11._navigation
    assert MyDroneRedStaged16._deployment is MyDroneRedLines11._deployment


def test_odometry_counts_residual_translation_and_wraps_during_turn():
    m=SingleWallMotion((500,500));p=m.pose
    for _ in range(8):p.update(np.zeros(2),math.pi-.03,np.zeros(3))
    start=p.position.copy();heading=p.heading
    p.update(None,None,np.array([1.2,.22,.09]))
    np.testing.assert_allclose(p.position-start,1.2*np.array([math.cos(heading+.22),math.sin(heading+.22)]))
    assert abs(wrap(p.heading-(heading+.09)))<1e-10 and p.heading<0
    assert np.linalg.norm(p.velocity)==pytest.approx(1.2)
    p.update(np.zeros(2),None,np.zeros(3));assert p.good==0
    p.update(None,0.,np.zeros(3));assert p.good==0
    for _ in range(7):p.update(np.zeros(2),0.,np.zeros(3))
    assert p.good==7


def test_unstable_recovery_and_false_consensus_not_accepted():
    m=StagedMapper((500,500));b=block(m,age=400);b['distance']=500
    b['recovery']=dict(stable=False,distance_error=0.,angle_error=0.)
    assert m.grade(b)[0] is None
    b['recovery']=dict(stable=True,distance_error=90.,angle_error=0.)
    assert m.grade(b)[0] is None
    for value in (None,'invalid',-1.,float('nan')):
        bad=list(packet());bad[4]=value
        assert not m.accept_witness(tuple(bad),50,5)


def test_corner_turn_rejects_wall_and_rank_one_is_not_full_pose_validation():
    walls=np.array([[[22.,-80.],[22.,80.]],[[-80.,-22.],[22.,-22.]]])
    d,a=scan(walls,np.zeros(2),0.)
    assert not supported_sweep(d,a,[25,0],math.pi/2)
    m=StagedMapper((500,500));b=block(m,age=150)
    b['lines']=[WALL,WALL+[0,150]];b['references']=b['lines']
    assert m.grade(b,deadline=True)[0] is None
