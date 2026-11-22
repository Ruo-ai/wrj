"""Deployment authorization and bounded departures using native ray formats."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.red_egress36 import Egress36
from swarm_rescue.solutions.red_passage28 import HullPassage28
from swarm_rescue.simulation.ray_sensors.distance_sensor import compute_ray_angles
from swarm_rescue.simulation.utils.constants import FOV_LIDAR_SENSOR, RESOLUTION_LIDAR_SENSOR, MAX_RANGE_LIDAR_SENSOR

A=compute_ray_angles(math.radians(FOV_LIDAR_SENSOR),RESOLUTION_LIDAR_SENSOR)
D=np.full(RESOLUTION_LIDAR_SENSOR,float(MAX_RANGE_LIDAR_SENSOR))


def obj(name,distance,angle):
    return SimpleNamespace(entity_type=SimpleNamespace(name=name),distance=distance,angle=angle)


BOMBS=tuple(obj('BOMB',0.,float(a)) for a in np.linspace(-math.pi,math.pi,35))


def prepare(p,step=1,lidar=D,position=(0.,0.),heading=0.,odom=(0.,0.,0.)):
    return p.prepare(np.array(position),heading,np.array(odom),lidar,A,(),2,step)


def update(p,step=2,lidar=None,sem=BOMBS,inventory=1,position=(0.,0.),heading=0.,odom=(0.,0.,0.)):
    if lidar is None:
        lidar=np.zeros(181)
    return p.update(np.array(position),heading,np.array(odom),lidar,A,sem,inventory,step)


def test_native_scan_prepared_hull_margin_and_no_input_mutation():
    p=Egress36();d=D[:,None].copy();before=d.copy()
    assert prepare(p,lidar=d)
    np.testing.assert_array_equal(d,before)
    fine=p._plan['fine'];end=p._plan['endpoint']
    np.testing.assert_array_equal(fine.footprint(),HullPassage28.footprint())
    assert fine.line_clear(np.zeros(2),end)
    assert fine.sample(np.linspace(np.zeros(2),end,50)).min()>=fine.PLAN_RADIUS+p.EXTRA_MARGIN
    assert np.linalg.norm(end)==pytest.approx(p.CORRIDOR_LENGTH)


def test_no_safe_pre_drop_corridor_means_no_placement_authorization():
    p=Egress36()
    narrow=np.minimum(D,12./np.maximum(1e-9,np.abs(np.sin(A))))
    assert not prepare(p,lidar=narrow)
    assert not prepare(p,lidar=np.zeros(181))
    assert update(p) is None
    assert p.stats.get('motion_command_ticks',0)==0


def test_inventory_decrement_and_near_bomb_are_both_required():
    p=Egress36();assert prepare(p)
    assert update(p,inventory=2) is None
    assert not p.active
    q=Egress36();assert prepare(q)
    command=update(q,sem=())
    assert q.failed and q.reason=='egress_unconfirmed_occluder'
    assert command==dict(forward=0.,lateral=0.,rotation=0.)


def test_confirmed_self_bomb_all_zero_scan_uses_real_cached_corridor():
    p=Egress36();assert prepare(p)
    raw=np.zeros(181);before=raw.copy()
    c=update(p,lidar=raw)
    assert p.active and c['forward']<0 and c['rotation']==0
    np.testing.assert_array_equal(raw,before)
    assert p.status()['current_scan']['long_rays']==0
    assert p.status()['last_stopping_certificate']['cache_control_step']==1
    assert p.status()['last_stopping_certificate']['fresh_control_step']==2
    # Independently validate the cached stopping sweep of the actual command.
    stop=p.status()['last_stopping_certificate']
    assert p._plan['fine'].line_clear(np.array(stop['cached_start']),np.array(stop['cached_end']))


def test_unknown_near_wall_rays_stop_despite_self_bomb_elsewhere():
    p=Egress36();assert prepare(p)
    # Leave a semantic sector unclaimed by any BOMB; its near ray is opaque.
    sem=tuple(o for o in BOMBS if abs(o.angle)<2.5)
    c=update(p,sem=sem)
    assert p.active and p.reason=='current_nonself_ray_obstruction'
    assert c['forward']==0 and p.stats.get('motion_command_ticks',0)==0


def test_new_near_drone_overrides_own_bomb_authorization():
    p=Egress36();assert prepare(p)
    c=update(p,sem=(*BOMBS,obj('DRONE',12.,math.pi)))
    assert p.reason=='current_nonbomb_object_obstruction'
    assert c['forward']==0 and p.stats.get('motion_command_ticks',0)==0


def test_missing_rays_are_not_labeled_self_bomb_free_space():
    p=Egress36();assert prepare(p)
    d=np.zeros(181);d[50]=np.nan
    c=update(p,lidar=d)
    assert p.reason=='egress_unattributed_missing_rays'
    assert c['forward']==0


def test_duplicate_update_does_not_integrate_odom_twice_and_command_is_copy():
    p=Egress36();assert prepare(p)
    a=update(p,odom=(.5,0.,0.),position=(.5,0.))
    offset=p._offset.copy();distance=p._distance
    b=update(p,odom=(.5,0.,0.),position=(.5,0.))
    np.testing.assert_array_equal(p._offset,offset)
    assert p._distance==distance and a==b
    b['forward']=999
    assert p.command()['forward']!=999


@pytest.mark.parametrize('case,expected',[
    ('gap','egress_missing_odometry_interval'),
    ('pose','egress_localization_disagreement'),
    ('turn','egress_heading_budget'),
    ('jump','egress_odometry_jump'),
])
def test_cache_is_never_extended_across_invalid_motion(case,expected):
    p=Egress36();assert prepare(p)
    args=dict(step=2)
    if case=='gap':args['step']=3
    if case=='pose':args['position']=(40.,0.)
    if case=='turn':args['odom']=(0.,0.,.11)
    if case=='jump':args['odom']=(6.,0.,0.)
    update(p,**args)
    assert p.failed and p.reason==expected
    assert p.stats.get('motion_command_ticks',0)==0


def test_unconfirmed_placement_cache_expires_without_blind_activation():
    p=Egress36();assert prepare(p)
    for tick in range(2,8):
        update(p,step=tick,inventory=2,lidar=D,sem=())
    assert not p.active and p.state=='cleared'
    assert update(p,step=8,inventory=1) is None
    assert p.stats.get('motion_command_ticks',0)==0


def test_no_progress_latches_failure_without_renewing_cache():
    p=Egress36();assert prepare(p)
    for tick in range(2,35):
        update(p,step=tick)
    assert p.failed and p.reason=='egress_no_odometry_progress'
    ticks=p.stats['motion_command_ticks']
    for tick in range(35,41):
        update(p,step=tick)
    assert p.stats['motion_command_ticks']==ticks
    assert not prepare(p,step=41)


def test_failed_episode_only_recovers_when_fresh_scan_proves_hull_clear():
    p=Egress36();assert prepare(p)
    update(p,step=3)
    assert p.failed
    assert update(p,step=4,lidar=D,sem=()) is None
    assert not p.failed and p.reason=='late_scan_recovered' and p._plan is None


def test_two_fresh_clear_frames_return_unmodified_control_to_navigation():
    p=Egress36();assert prepare(p);update(p)
    update(p,step=3,lidar=D,sem=())
    assert update(p,step=4,lidar=D,sem=()) is None
    assert p.state=='cleared' and p.reason=='fresh_hull_cleared'
    assert update(p,step=5,lidar=D,sem=()) is None
    json.dumps(p.status())


def test_direction_follows_previously_travelled_reverse_body_vector():
    p=Egress36();assert prepare(p,odom=(.3,math.pi/2,0.))
    endpoint=p._plan['endpoint']
    assert endpoint[1]<-40
    c=update(p,heading=0.)
    assert c['lateral']<0


def test_stopping_sweep_cannot_leave_prevalidated_space():
    p=Egress36();assert prepare(p);update(p)
    # Force an adversarial offset in a test: the whole predicted stop must
    # still be checked against the original corridor range and known hull.
    p._offset=np.array([-49.,0.]);p.POSITION_CHECK=100.
    before=p.stats['motion_command_ticks']
    c=update(p,step=3)
    assert p.failed and p.reason in ('egress_corridor_exhausted_without_clear_scan','egress_cached_stopping_sweep_rejected')
    assert p.stats['motion_command_ticks']==before


def _deployment_harness(inventory):
    # Do not construct a drone, physics object, arena or simulator. Exercise
    # the actual participant method with the minimal legal controller state.
    from swarm_rescue.solutions.my_drone_red_mesh36 import MyDroneRedMesh36
    d=object.__new__(MyDroneRedMesh36)
    d.identifier=0;d._steps=2;d._position=np.zeros(2)
    d.carried_bombs_count=lambda:inventory
    d._m36_delivery_due=True
    d._initial_inventory=2;d._last_inventory=2
    d._attempt_step=1;d._attempt_position=np.zeros(2)
    d._attempts=1;d._total_attempts=7;d._next_attempt=0
    d._placements=[];d._failed_place=None
    d._event=lambda *a,**kw:None
    d._m36_event=lambda *a,**kw:None
    d._m36_sensor_args=lambda:()
    d._progress=lambda:.8
    d.semantic_values=lambda:()
    d._r35_brake=lambda:dict(forward=0.,lateral=0.,rotation=0.)
    return d


@pytest.mark.parametrize('remaining',(1,0))
def test_real_deployment_confirmation_does_not_overwrite_egress_first_command(remaining):
    d=_deployment_harness(remaining)
    d._last_inventory=remaining+1
    d._m36_egress=SimpleNamespace(command=lambda:dict(forward=.5,lateral=-.2,rotation=0.))
    command=dict(forward=.5,lateral=-.2,rotation=0.,place_bomb=0)
    d._deployment(command)
    assert command==dict(forward=.5,lateral=-.2,rotation=0.,place_bomb=0)
    assert d._last_inventory==remaining and d._attempt_step is None
    assert len(d._placements)==1


def test_real_deployment_precheck_rejection_does_not_consume_actual_attempt_limit():
    d=_deployment_harness(2)
    d._attempt_step=None;d._attempts=0
    d._m36_egress=SimpleNamespace(command=lambda:None,prepare=lambda *args:False)
    command=dict(forward=.1,lateral=0.,rotation=0.,place_bomb=0)
    d._deployment(command)
    assert command['place_bomb']==0 and d._attempt_step is None
    assert d._total_attempts==7 and d._attempts==0
    assert d._bomb_target is None and d._route==[]
