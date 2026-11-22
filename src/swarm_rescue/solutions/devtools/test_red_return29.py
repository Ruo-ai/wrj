"""Regression fixtures for actual 28 failures; no simulator or hidden input."""
import math
from types import SimpleNamespace
import numpy as np

from swarm_rescue.solutions.red_return29 import ReturnPlanner29
from swarm_rescue.solutions.red_passage28 import HullPassage28
from swarm_rescue.solutions.my_drone_red_rn7 import rotate
from swarm_rescue.solutions.devtools.test_red_return28 import OpenLocal, AxisLocal, D, A


def obj(kind, distance, angle):
    return SimpleNamespace(entity_type=SimpleNamespace(name=kind), distance=float(distance), angle=float(angle))


def bombs():
    return [obj('BOMB',0,a) for a in np.linspace(-math.pi,math.pi,37)]


def observe(p, step, point=(240.,0.), distance=0., alpha=0., theta=0., heading=0., d=D, semantic=(), seconds=None):
    p.observe(step,np.asarray(point),heading,np.array([distance,alpha,theta]),d,A,semantic,seconds)


def started(identifier=0, factory=HullPassage28):
    p=ReturnPlanner29(identifier,factory);p.set_home([0.,0.])
    observe(p,1)
    p.trail=[np.array([x,0.]) for x in range(0,241,12)]
    p.trail_steps=list(range(1,len(p.trail)+1))
    p.start_return();p.velocity[:]=0
    return p


def command(p,d=D,semantic=(),peers=None):
    return p.command(p.position,p.heading,p.velocity,d,A,semantic,peers)


def test_inventory_requests_return_without_consuming_navigation_or_route():
    p=ReturnPlanner29(0,OpenLocal);p.set_home([0,0]);observe(p,1)
    assert not p.should_return(10,1,2)
    assert p.return_requested and not p.returning and p.status()['inventory_pending']==2
    assert command(p) is None
    count=len(p.trail);observe(p,2,[252.,0.])
    assert len(p.trail)>count
    assert p.should_return(10,1,0) and p.returning


def test_latest_return_start_also_cannot_override_remaining_inventory():
    p=ReturnPlanner29(0,OpenLocal);observe(p,1)
    p.should_return(5000,150,1);p.start_return('latest')
    assert not p.returning and p.return_requested
    p.should_return(5000,150,0);p.start_return('latest')
    assert p.returning


def test_explicit_request_does_not_start_or_mutate_route():
    p=ReturnPlanner29();observe(p,1);p.request_return('latest')
    assert p.return_requested and not p.returning and p.cursor is None


def test_clock_window_does_not_turn_one_planner_spike_into_long_rate_error():
    p=ReturnPlanner29(0,OpenLocal)
    for step in range(1,121):observe(p,step,seconds=.03*step+(1. if step>=120 else 0.))
    assert .03<p.seconds_per_step<.04


def test_bomb_all_round_occlusion_uses_recent_observed_free_only():
    p=started();d=np.zeros_like(D);sem=bombs();observe(p,2,d=d,semantic=sem)
    result=command(p,d,sem)
    assert p.status()['bomb_occlusion_memory_used']
    assert result['forward']<0 and result['rotation']==0
    assert np.isfinite(list(result.values())).all()


def test_missing_rays_without_bomb_semantics_never_become_free():
    p=started();d=np.zeros_like(D);observe(p,2,d=d)
    result=command(p,d)
    assert result['forward']==0 and not p.status()['bomb_occlusion_memory_used']


def test_cache_expires_and_does_not_refresh_from_occluded_frames():
    p=started();d=np.zeros_like(D);sem=bombs()
    for step in range(2,84):observe(p,step,d=d,semantic=sem)
    assert p._clear_frame29['step']==1
    assert command(p,d,sem)['forward']==0
    assert not p.status()['bomb_occlusion_memory_used']


def test_pose_or_odometry_gap_invalidates_cached_frame():
    p=started();d=np.zeros_like(D);sem=bombs();observe(p,4,d=d,semantic=sem)
    assert p._clear_frame29 is None and command(p,d,sem)['forward']==0


def test_cache_translation_moves_original_obstacle_in_correct_body_direction():
    d=D.copy();f=np.cos(A)>0;d[f]=np.minimum(d[f],50/np.cos(A[f]))
    p=ReturnPlanner29();observe(p,1,d=d)
    observe(p,2,d=np.zeros_like(D),semantic=bombs(),distance=10)
    fine=p._scan(np.zeros_like(D),A,bombs())
    assert fine.free[fine.cell([10.,0.])]
    assert not fine.free[fine.cell([45.,0.])]
    assert np.allclose(p._cache_shift29,[10,0])


def test_cache_rotation_respects_body_coordinates_and_does_not_reduce_margin():
    d=D.copy();f=np.cos(A)>0;d[f]=np.minimum(d[f],50/np.cos(A[f]))
    p=ReturnPlanner29();observe(p,1,d=d)
    observe(p,2,d=np.zeros_like(D),semantic=bombs(),theta=.1,heading=.1)
    fine=p._scan(np.zeros_like(D),A,bombs())
    assert fine.free[fine.cell(rotate([20.,0.],-.1))]
    assert not fine.free[fine.cell(rotate([60.,0.],-.1))]
    np.testing.assert_array_equal(fine.footprint(),HullPassage28.footprint())


def test_small_angle_affine_is_rigid_and_precise_at_far_observed_points():
    fine=HullPassage28(D,A)
    points=np.array([[0.,0.],[180.,180.],[-180.,180.],[180.,-180.],[-180.,-180.]])
    def pixels(p):return p*np.array([1.,-1.])/fine.CELL+np.array([fine.width/(2*fine.CELL)-.5,fine.height/(2*fine.CELL)-.5])
    for angle in (.05,-.05,.001,-.001):
        shift=np.array([3.35,-2.27]);matrix=ReturnPlanner29._cache_affine(fine,shift,angle)
        actual=np.c_[pixels(points),np.ones(len(points))]@matrix.T
        expected=pixels(rotate(points-shift,-angle))
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        np.testing.assert_allclose(matrix[:,:2].T@matrix[:,:2],np.eye(2),atol=1e-12)
        assert abs(np.linalg.det(matrix[:,:2])-1)<1e-12


def test_new_nonbomb_wall_overrides_cached_free_sector():
    p=started();d=np.zeros_like(D);sem=bombs()
    front=np.abs(A)<.25;d[front]=40/np.cos(A[front])
    observe(p,2,d=d,semantic=sem);fine=p._scan(d,A,sem)
    assert not fine.line_clear(np.zeros(2),np.array([60.,0.]))
    assert fine.static.any()


def test_near_wall_memory_preserves_unknown_space_but_permits_monotone_retreat():
    d=D.copy();below=np.sin(A)<0;d[below]=np.minimum(d[below],18/-np.sin(A[below]))
    p=ReturnPlanner29();observe(p,1,d=d)
    observe(p,2,d=np.zeros_like(D),semantic=bombs());p.velocity[:]=0
    fine=p._scan(np.zeros_like(D),A,bombs())
    assert p._clear_frame29['step']==1 and not p._full_scan
    result=p._clearance_recovery(fine)
    assert result is not None and result['lateral']>0
    assert p.last_reason=='bomb_memory_monotone_recovery'
    assert not p._full_scan


def test_new_drone_explicitly_preempts_bomb_sector_even_at_matching_distance():
    p=started();d=np.zeros_like(D);sem=bombs()+[obj('DRONE',8,0)]
    observe(p,2,d=d,semantic=sem);fine=p._scan(d,A,sem)
    assert not fine.free[fine.cell([25.,0.])]
    assert not p._bomb_rays(d,A,sem)[len(A)//2]


def test_return_closer_than_bomb_range_uncertainty_is_never_filled():
    p=started();d=D.copy();d[np.abs(A)<.15]=2.
    sem=[obj('BOMB',17,0)]
    observe(p,2,d=d,semantic=sem);fine=p._scan(d,A,sem)
    assert not p._bomb_rays(d,A,sem).any()
    assert not fine.line_clear(np.zeros(2),np.array([20.,0.]))
    assert not p.status()['bomb_occlusion_memory_used']


def test_short_verified_escape_leaves_bomb_before_cache_expires():
    p=started()
    for offset in range(1,21):
        if offset<=12:
            d=np.zeros_like(D);sem=bombs()
        else:
            projection=offset*np.cos(A);disc=144-offset*offset*np.sin(A)**2
            hit=(disc>=0)&(projection>0)
            d=D.copy();d[hit]=projection[hit]-np.sqrt(disc[hit])
            sem=[obj('BOMB',d[i],A[i]) for i in np.flatnonzero(hit)]
        observe(p,offset+1,[240.-offset,0],distance=1,alpha=math.pi,d=d,semantic=sem)
        result=command(p,d,sem)
        assert result['rotation']==0
        assert p.last_reason not in ('return_scan_unavailable','origin_clearance_no_safe_retreat')
    assert p._clear_frame29['step']==1 and p.status()['memory_age']<p.CACHE_STEPS


def test_lower_priority_peer_actively_moves_away_when_swept_path_is_clear():
    p=started(3,OpenLocal);p.step+=1
    result=command(p,semantic=[obj('DRONE',35,math.pi)],peers={1:(p.step,[200.,0.],None)})
    assert result['forward']>0 and p.last_reason=='priority_peer_yield_motion'


def test_high_priority_peer_is_not_blanket_braked_by_drone_behind():
    p=started(0,OpenLocal);p.step+=1
    result=command(p,semantic=[obj('DRONE',35,0)],peers={3:(p.step,[280.,0.],None)})
    assert result['forward']<0 and p.last_reason=='following_observed_return_route'


def test_sustained_odom_noise_without_heading_change_does_not_freeze():
    p=started(0,OpenLocal)
    for step in range(2,12):observe(p,step,theta=.035,heading=0)
    p.velocity[:]=0
    assert command(p)['forward']<0 and p.last_reason!='return_wait_fixed_heading'


def test_supported_real_turn_brakes_until_heading_settles():
    p=started(0,OpenLocal)
    for step in range(2,9):observe(p,step,theta=.045,heading=(step-1)*.045)
    command(p)
    assert p.last_reason=='return_wait_fixed_heading'
    for step in range(9,16):observe(p,step,theta=0,heading=.315)
    assert command(p)['rotation']==0 and p.last_reason!='return_wait_fixed_heading'


def test_independent_instances_never_share_path_or_occlusion_memory():
    a=started();b=started(1)
    assert a._clear_frame29 is not b._clear_frame29
    a._clear_frame29['ranges'][0]=1
    assert b._clear_frame29['ranges'][0]==D[0]


def test_only_currently_visible_earlier_breadcrumb_can_shorten_route():
    p=started(0,OpenLocal)
    p.trail=[np.array([0.,0.]),np.array([120.,0.])]+[np.array([1000.+i,500.]) for i in range(30)]+[np.array([240.,0.])]
    p.cursor=len(p.trail)-1;p.step=6
    result=command(p)
    assert p.cursor<len(p.trail)-1 and p.stats['visible_return_shortcut']
    assert result['forward']<0
