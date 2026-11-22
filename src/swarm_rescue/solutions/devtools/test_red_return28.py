"""Pure observed-route/budget/control fixtures; no native flight or truth input."""
import math
from types import SimpleNamespace

import numpy as np

from swarm_rescue.solutions.red_return28 import ReturnPlanner28
from swarm_rescue.solutions.red_passage28 import HullPassage28
from swarm_rescue.solutions.my_drone_red_rn7 import rotate

D=np.full(181,230.)
A=np.linspace(-math.pi,math.pi,181)


class OpenLocal:
    TRACK_RADIUS=18.5
    PLAN_RADIUS=19.914213562373096
    def __init__(self,*args):pass
    def line_clear(self,a,b):return np.linalg.norm(b-a)<=180
    def sample(self,points):return np.full(len(np.atleast_2d(points)),50.)
    def route(self,a,b,coarse,target=None,limit=165):return [np.asarray(a),np.asarray(target)]


class BlockedLocal(OpenLocal):
    def line_clear(self,a,b):return False
    def route(self,*args,**kwargs):return []


class AxisLocal(OpenLocal):
    def line_clear(self,a,b):
        delta=np.asarray(b)-a
        return min(abs(delta))<1 and np.linalg.norm(delta)<=180


def observe(planner,step,point,heading=0.,distance=2.,rotation=0.,lidar=D,angles=A,seconds=None):
    planner.observe(step,np.asarray(point,float),heading,np.array([distance,0.,rotation]),lidar,angles,(),seconds)


def line_planner(factory=OpenLocal,home=(0.,0.),length=240):
    planner=ReturnPlanner28(0,factory);planner.set_home(home)
    for step,x in enumerate(range(0,length+1,12),1):observe(planner,step,(x,0.))
    planner.velocity[:]=0
    return planner


def command(planner,point=None,**kwargs):
    return planner.command(planner.position if point is None else np.asarray(point,float),planner.heading,
        planner.velocity,kwargs.pop('lidar',D),kwargs.pop('angles',A),kwargs.pop('semantic',()),kwargs.pop('peers',None))


def test_ordinary_exploration_returns_no_override_and_tracks_only_own_positions():
    a=line_planner();b=line_planner(length=120)
    assert command(a) is None
    assert len(a.trail)==21 and len(b.trail)==11
    assert a.trail is not b.trail
    np.testing.assert_array_equal(a.trail[0],[0,0])
    assert a.remaining_path_length()==240


def test_budget_uses_recorded_path_not_euclidean_shortcut():
    p=ReturnPlanner28(0,AxisLocal);p.set_home([0,0])
    points=[(0,0),(12,0),(24,0),(36,0),(48,0),(60,0),(72,0),(84,0),
            (84,12),(84,24),(84,36),(84,48),(84,60),(84,72),(84,84)]
    for i,point in enumerate(points,1):observe(p,i,point)
    assert abs(p.remaining_path_length()-168)<1e-8
    assert p.remaining_path_length()>np.linalg.norm(points[-1])
    assert not p.should_return(2000,100,0)
    required=p.last_budget['required_steps']
    assert required>=540+168/2*1.55
    assert p.should_return(required-1,100,0)
    assert p.returning


def test_wall_clock_is_independently_tighter_than_remaining_steps():
    p=line_planner()
    assert p.should_return(5000,5,0)
    assert p.last_budget['effective_remaining_steps']<200


def test_inventory_reserves_time_without_disabling_original_deployment():
    a=line_planner();b=line_planner()
    a.should_return(5000,100,0);b.should_return(5000,100,2)
    assert b.last_budget['required_steps']-a.last_budget['required_steps']==180
    assert 'place_bomb' not in command_after_start(a)


def command_after_start(p):
    p.start_return();return command(p)


def test_returns_along_own_reverse_path_with_no_world_heading_assumption():
    p=line_planner();p.start_return()
    result=command(p)
    assert result['forward']<0 and result['rotation']==0
    assert abs(result['lateral'])<1e-8
    assert p.last_target[0]<p.position[0]
    assert np.linalg.norm(np.asarray(p.last_target)-p.position)<=140


def test_return_command_is_covariant_under_legal_world_pose_rotation():
    first=line_planner();first.start_return();ref=command(first)
    angle=.71;p=ReturnPlanner28(0,OpenLocal);p.set_home(rotate([0.,0.],angle))
    for i,x in enumerate(range(0,241,12),1):observe(p,i,rotate([float(x),0.],angle),angle)
    p.velocity[:]=0;p.start_return();actual=command(p)
    np.testing.assert_allclose([actual['forward'],actual['lateral']],[ref['forward'],ref['lateral']],atol=1e-8)
    assert actual['rotation']==0


def test_l_corner_uses_observed_leg_when_diagonal_is_not_clear():
    p=ReturnPlanner28(0,AxisLocal);p.set_home([0,0])
    points=[(x,0) for x in range(0,85,12)]+[(84,y) for y in range(12,85,12)]
    for i,point in enumerate(points,1):observe(p,i,point)
    p.velocity[:]=0;p.start_return();result=command(p)
    assert abs(result['forward'])<1e-8 and result['lateral']<0
    assert p.last_target[0]==84


def test_loop_is_erased_only_with_current_scan_verified_connector():
    points=[(0,0),(12,0),(24,0),(36,0),(48,0),(60,0),(60,12),(48,12),(36,12),(24,12),(12,12),(0,12)]
    good=ReturnPlanner28(0,OpenLocal);bad=ReturnPlanner28(0,BlockedLocal)
    for i,point in enumerate(points,1):observe(good,i,point);observe(bad,i,point)
    assert len(good.trail)<len(bad.trail)
    assert good.stats['loop_nodes_erased']>0 and not bad.stats['loop_nodes_erased']
    np.testing.assert_array_equal(good.trail[0],[0,0])


def test_no_trail_growth_after_return_and_cursor_advances_only_on_reached_waypoint():
    p=line_planner();p.start_return();count=len(p.trail)
    command(p);target=np.asarray(p.last_target)
    old=p.cursor
    observe(p,p.step+1,target);p.velocity[:]=0
    command(p)
    assert len(p.trail)==count and p.cursor<old


def test_relocation_jump_is_not_invented_as_traversed_connector():
    p=line_planner(length=60);before=len(p.trail)
    observe(p,p.step+1,[500,500],distance=0)
    assert p.status()['pose_gap'] and len(p.trail)==before
    assert p.should_return(5000,100,0)
    assert command(p)=={'forward':-0.,'lateral':-0.,'rotation':0.}


def test_capacity_does_not_delete_start_or_turns_and_triggers_return():
    p=ReturnPlanner28(0,OpenLocal);p.MAX_NODES=4
    for i,x in enumerate(range(0,61,12),1):observe(p,i,[x,0])
    assert len(p.trail)==4
    np.testing.assert_array_equal(p.trail[0],[0,0])
    assert p.should_return(5000,100,0)


def test_missing_scan_brakes_without_following_unverified_history():
    p=line_planner();p.start_return();p.step+=1
    r=command(p,lidar=None)
    assert r=={'forward':-0.,'lateral':-0.,'rotation':0.}
    assert p.last_reason=='return_scan_unavailable'


def test_both_unavailable_pose_and_odometry_fail_closed():
    p=line_planner();p.start_return();p.valid_odometry=False
    assert command(p)['rotation']==0 and p.last_reason=='return_odometry_unavailable'
    r=p.command(None,None,[1,0],D,A)
    assert r['rotation']==0 and p.last_reason=='return_pose_unavailable'


def test_single_typical_odometry_noise_does_not_stall_return_but_confirmed_rotation_does():
    p=line_planner();p.start_return()
    observe(p,p.step+1,p.position,rotation=.03);p.velocity[:]=0
    assert command(p)['forward']<0
    observe(p,p.step+1,p.position,rotation=.03)
    observe(p,p.step+1,p.position,rotation=.03)
    command(p)
    assert p.last_reason=='return_wait_fixed_heading'
    for _ in range(3):observe(p,p.step+1,p.position,rotation=0.)
    p.velocity[:]=0
    assert command(p)['forward']<0


def test_large_single_rotation_immediately_brakes_and_keeps_rotation_zero():
    p=line_planner();p.start_return();observe(p,p.step+1,p.position,rotation=.09)
    assert command(p)['rotation']==0 and p.last_reason=='return_wait_fixed_heading'


def test_rendezvous_does_not_claim_delivery_only_brakes_locally():
    p=line_planner(length=60);p.start_return()
    assert command(p)['rotation']==0 and p.arrived
    assert p.last_reason=='at_rendezvous'
    assert 'final_complete' not in p.status()


def test_only_recent_peer_near_shared_home_is_usable_as_relay_contact():
    p=line_planner(length=156);p.start_return()
    command(p,peers={1:(p.step,[40.,0.],None)})
    assert p.last_reason=='at_near_home_relay'
    command(p,peers={1:(p.step-9,[40.,0.],None)})
    assert p.last_reason!='at_near_home_relay'
    command(p,peers={1:(p.step,[140.,0.],None)})
    assert p.last_reason!='at_near_home_relay'


def test_fresh_close_peer_preempts_route_following():
    p=line_planner();p.start_return()
    obj=SimpleNamespace(entity_type=SimpleNamespace(name='DRONE'),distance=30,angle=0.)
    command(p,semantic=[obj])
    assert p.last_reason=='return_peer_clearance'


def wall_lidar(offset=15.):
    # Horizontal wall below body: y=-offset; all other directions open.
    d=np.full_like(A,230.)
    hit=np.sin(A)<-1e-6
    d[hit]=np.minimum(230.,offset/-np.sin(A[hit]))
    return d


def test_real_hull_recovery_never_relaxes_footprint_and_moves_away_from_near_wall():
    p=line_planner(HullPassage28);p.start_return();p.step+=1;p.velocity[:]=0
    d=wall_lidar(18.)
    fine=HullPassage28(d,A)
    assert not fine.line_clear(np.zeros(2),np.zeros(2))
    result=command(p,lidar=d)
    assert p.last_reason=='monotone_clearance_recovery'
    assert result['lateral']>0 and result['rotation']==0
    assert np.linalg.norm([result['forward'],result['lateral']])<=1


def test_missing_rays_do_not_authorize_overlap_recovery():
    p=line_planner(HullPassage28);p.start_return();p.step+=1;p.velocity[:]=0
    d=wall_lidar(18.);d[20]=np.nan
    command(p,lidar=d)
    assert p.last_reason=='origin_clearance_no_safe_retreat'


def test_recovery_settles_existing_momentum_before_escape():
    p=line_planner(HullPassage28);p.start_return();p.step+=1;p.velocity[:]=[2,0]
    command(p,lidar=wall_lidar(18.))
    assert p.last_reason=='clearance_recovery_settle'


def test_recorded_free_path_is_rechecked_when_new_wall_closes_it():
    p=line_planner(HullPassage28);p.start_return();p.step+=1;p.velocity[:]=0
    # A surrounding tight enclosure offers no monotone exit through known rays.
    command(p,lidar=np.full(181,12.))
    assert p.last_reason=='origin_clearance_no_safe_retreat'
    assert not p.arrived


def test_original_trail_and_home_are_copied_from_caller_owned_arrays():
    p=ReturnPlanner28(0,OpenLocal);home=np.array([0.,0.]);point=np.array([10.,20.])
    p.set_home(home);observe(p,1,point)
    home[:]=99;point[:]=99
    np.testing.assert_array_equal(p.home,[0,0]);np.testing.assert_array_equal(p.trail[0],[10,20])
