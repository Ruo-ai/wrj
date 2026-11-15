"""Small B8 tests: no native simulation, no cache or existing output writes."""
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from swarm_rescue.solutions.blue_nav_b8 import (
    BODY_RADIUS, PAYLOAD_RADIUS, body_velocity, stopping_speed, scan_geometry,
    PayloadPlanner, SpeedGovernor)
from swarm_rescue.solutions.my_drone_eval_blue import MyDroneBlueAdapted, RescueGrid
from swarm_rescue.solutions.my_drone_blue_nav_b8 import MyDroneBlueB8Empty, MyDroneBlueB8Payload, MyDroneBlueB8Off
from swarm_rescue.solutions.devtools.test_blue_relay_b7 import make_drone


ANGLES=np.linspace(-math.pi, math.pi, 181)
OPEN=np.full(len(ANGLES),230.)


def command(forward=.72, rotation=0.):
    return dict(forward=forward,lateral=0.,rotation=rotation,grasper=0)


def test_units_and_rotation_do_not_require_compass():
    np.testing.assert_allclose(body_velocity([4., .2, .2]), [4.,0.])
    assert body_velocity([np.nan,0.,0.]) is None
    assert stopping_speed(0,.35)==0
    assert stopping_speed(40,.10)<stopping_speed(40,.35)


def test_payload_swing_envelope_exceeds_empty_hull():
    assert 16 < BODY_RADIUS < 17
    assert 45 < PAYLOAD_RADIUS < 46
    # Same observed gap supports the empty body but NOT arbitrary payload swing.
    distances=np.minimum(230.,40./np.maximum(.001,np.abs(np.sin(ANGLES))))
    assert scan_geometry(distances,ANGLES,0.,BODY_RADIUS)['room']>0
    assert scan_geometry(distances,ANGLES,0.,PAYLOAD_RADIUS)['room']==0


def test_scan_gaps_do_not_become_free_space():
    d=OPEN.copy();d[np.abs(ANGLES)<.2]=np.nan
    assert not scan_geometry(d,ANGLES,0.,BODY_RADIUS)['visible']
    assert scan_geometry([],[],0.,BODY_RADIUS)['room']==0


def test_speed_reference_ramps_in_open_space_and_brakes_for_wall():
    gov=SpeedGovernor();refs=[]
    for _ in range(30):
        gov.update(command(),OPEN,ANGLES,[0.,0.,0.]);refs.append(gov.reference)
    assert np.max(np.diff(refs))<=.351 and refs[-1]>8.4
    d=OPEN.copy();d[np.abs(ANGLES)<.4]=45.
    out=gov.update(command(),d,ANGLES,[9.,0.,0.])
    assert out['forward']<0 and gov.last['desired_speed']<3.


@pytest.mark.parametrize('condition',['turn','teammate','uncertain'])
def test_risk_reduces_target_before_throttle(condition):
    gov=SpeedGovernor();gov.reference=9.
    opts=dict(turn=1.2) if condition=='turn' else dict(dynamic=True) if condition=='teammate' else dict(pose_uncertain=True)
    gov.update(command(),OPEN,ANGLES,[7.,0.,0.],**opts)
    assert gov.last['desired_speed']<=3.5


def test_invalid_odometry_does_not_blindly_accelerate():
    gov=SpeedGovernor()
    out=gov.update(command(),OPEN,ANGLES,None)
    assert out['forward']==0 and out['lateral']==0
    assert gov.last['reason']=='missing_odometry_coast'


def test_payload_brakes_earlier_and_preserves_grasp():
    d=OPEN.copy();d[np.abs(ANGLES)<.4]=90.
    empty,payload=SpeedGovernor(),SpeedGovernor()
    for g in (empty,payload):g.reference=9.
    empty.update(command(),d,ANGLES,[4.,0.,0.])
    out=payload.update(dict(command(),grasper=1),d,ANGLES,[4.,0.,0.],carrying=True)
    assert payload.last['target_speed']<empty.last['target_speed']
    assert out['grasper']==1 and out['forward']<0


def test_payload_fixed_attitude_monotone_egress_only():
    d=OPEN.copy();d[np.abs(ANGLES)>2.95]=42.
    assert scan_geometry(d,ANGLES,0.,PAYLOAD_RADIUS)['egress']
    assert not scan_geometry(d,ANGLES,math.pi,PAYLOAD_RADIUS)['egress']
    gov=SpeedGovernor();out=gov.update(command(rotation=.4),d,ANGLES,[0,0,0],carrying=True)
    assert out['rotation']==0 and gov.reference<=.55


def grid():
    g=RescueGrid((800,512));g.free[:]=True
    return g


def test_payload_plan_complete_observed_approach():
    g=grid();p=PayloadPlanner(g);r=p.plan(np.array([-230.,0.]),np.array([250.,0.]))
    assert r.complete and r.points and r.min_clearance>=PAYLOAD_RADIUS+3
    assert np.linalg.norm(r.points[-1]-[250.,0.])<=100
    assert r.estimated_steps>r.length/3.1


def test_unknown_or_wall_barrier_cannot_be_turned_into_direct_route():
    for blocked in ('unknown','wall'):
        g=grid()
        if blocked=='wall':g.evidence[:,24]=8
        else:g.free[:,24:27]=False
        p=PayloadPlanner(g);r=p.plan(np.array([-230.,0.]),np.array([250.,0.]))
        assert not r.complete and not r.points and r.reason=='return_disconnected'
        assert not p.segment(np.array([-100.,0.]),np.array([100.,0.]))


def test_longer_clear_route_can_beat_shorter_tight_route():
    g=grid();p=PayloadPlanner(g);p.layers()
    # Isolate the documented path-cost objective, not a fabricated full-game win.
    p.clearance[:]=120
    tight=[np.array([x,0.]) for x in (-150,-50,50,150)]
    wide=[np.array([-150.,0.]),np.array([-150.,100.]),np.array([150.,100.]),np.array([150.,0.])]
    for point in tight[1:-1]:p.clearance[g.cell(point)]=PAYLOAD_RADIUS+5
    sl,sc,_=p._evaluate(tight);wl,wc,_=p._evaluate(wide)
    assert wl>sl and wc<sc


def test_unknown_center_and_near_wall_start_are_explicit():
    g=grid();p=PayloadPlanner(g)
    assert p.plan(np.array([0.,0.]),None).reason=='unknown_center'
    g.free[:]=False;g.free[12:17,20:25]=True
    r=p.plan(np.array([0.,0.]),np.array([200.,0.]))
    assert not r.complete and not r.points


def test_breadcrumb_crossing_new_obstacle_is_rejected():
    g=grid();p=PayloadPlanner(g);p.layers()
    assert p.segment(np.array([-100.,0.]),np.array([100.,0.]))
    g.evidence[:,24]=8;p.layers()
    assert not p.segment(np.array([-100.,0.]),np.array([100.,0.]))


def test_disabled_candidate_is_B0_control_and_message_equivalent():
    a,b=make_drone(MyDroneBlueAdapted),make_drone(MyDroneBlueB8Off)
    for step in range(25):
        a.elapsed_timestep=b.elapsed_timestep=step
        assert a.control()==b.control()
        assert a.define_message_for_all()==b.define_message_for_all()


def test_empty_candidate_does_not_change_carrying_control():
    a,b=make_drone(MyDroneBlueAdapted),make_drone(MyDroneBlueB8Empty)
    for d in (a,b):d._inputs.held=[object()]
    for step in range(25):
        a.elapsed_timestep=b.elapsed_timestep=step
        assert a.control()==b.control()


def test_payload_candidate_does_not_change_empty_control():
    a,b=make_drone(MyDroneBlueAdapted),make_drone(MyDroneBlueB8Payload)
    for step in range(25):
        a.elapsed_timestep=b.elapsed_timestep=step
        assert a.control()==b.control()


def test_terminal_and_release_commands_are_not_overwritten():
    from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
    a,b=make_drone(MyDroneBlueAdapted),make_drone(MyDroneBlueB8Payload)
    for d in (a,b):
        d._inputs.held=[object()]
        d._inputs.objects=[NS(entity_type=DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER,distance=12.,angle=0.)]
    for step in range(32):
        a.elapsed_timestep=b.elapsed_timestep=step
        assert a.control()==b.control()
    assert b._b8_info['reason']=='b0_terminal_preserved'


def test_payload_disconnection_brakes_instead_of_using_goal_fallback():
    d=make_drone(MyDroneBlueB8Payload)
    d._position=np.array([-80.,0.]);d._heading=0.;d._tick=10;d._center_hit=None
    d._grid.free[:]=True;d._grid.evidence[:,18]=8
    cmd=d._navigate(np.array([200.,0.]),OPEN,ANGLES,True)
    assert cmd['forward']==0 and cmd['lateral']==0 and cmd['grasper']==1
    assert not d._b8_info['complete']


def test_no_online_bomb_or_truth_access():
    from pathlib import Path
    base=Path(__file__).parents[1]
    for name in ('blue_nav_b8.py','my_drone_blue_nav_b8.py'):
        text=(base/name).read_text(encoding='utf-8')
        for forbidden in ('true_position(', 'true_velocity(', '._pm_body', 'Map08', 'Map09', 'oracle_layout', '.disposed', 'grasped_bombs()[0]'):
            assert forbidden not in text


def test_probe_does_not_change_final_control_and_records_real_states():
    from swarm_rescue.solutions.devtools.blue_nav_b8_diagnostics import instrument
    a,b=make_drone(MyDroneBlueB8Empty),make_drone(instrument(MyDroneBlueB8Empty))
    for step in range(16):
        a.elapsed_timestep=b.elapsed_timestep=step
        assert a.control()==b.control()
    assert b._b8_probe_rows and b._probe_frame['b8']['mode']=='empty_search'


def test_drop_is_unconfirmed_and_clears_payload_route_state():
    d=make_drone(MyDroneBlueB8Payload);d._inputs.held=[object()]
    d.elapsed_timestep=1;d.control()
    d._inputs.held=[];d.elapsed_timestep=2;d.control()
    assert d._b8_stats['pickup_observed']==1
    assert d._b8_stats['possession_ended_unconfirmed']==1
    assert 'disposed' not in d._b8_stats and d._b8_recovery is None


def test_return_stall_uses_only_rechecked_trace_and_has_timeout():
    d=make_drone(MyDroneBlueB8Payload);d._grid.free[:]=True
    d._position=np.array([-80.,0.]);d._heading=0.;d._center_hit=None
    d._b8_anchor=d._position.copy();d._b8_progress_tick=0;d._tick=120
    d._b8_trace.append((10,np.array([-120.,0.])))
    d._navigate(np.array([200.,0.]),OPEN,ANGLES,True)
    assert d._b8_stats['stagnation']==1 and d._b8_recovery_until==165
    # New wall blocks the remembered route; do not reuse merely because visited.
    d._grid.evidence[:,8]=8;d._b8_planner.layers();d._tick=130
    d._navigate(np.array([200.,0.]),OPEN,ANGLES,True)
    assert d._b8_recovery is None


def test_batch_guard_precedes_output_and_source_freeze(monkeypatch,tmp_path):
    from swarm_rescue.solutions.devtools import blue_nav_b8_run as run
    out=tmp_path/'batch'
    monkeypatch.setattr('sys.argv',['b8','--run','--output',str(out)])
    with pytest.raises(SystemExit):run.main()
    assert not out.exists()


def test_report_preserves_timeout_and_missing_discoveries(tmp_path):
    import json
    from swarm_rescue.solutions.devtools.blue_nav_b8_report import summarize,write_report
    result=dict(status='ok',exit_code=0,walltime_limited=True,elapsed_steps=120,blue_score=99,
        blue_variant='b8_empty',map='fixture',bomb_count=2,disposed_count=0)
    (tmp_path/'result.json').write_text(json.dumps(result),encoding='utf-8')
    bombs=[dict(bomb=i,initial=[i*100,0],first_seen=10 if i==1 else None,pickups=[],losses=[],releases=[],disposed=None) for i in (1,2)]
    (tmp_path/'telemetry.json').write_text(json.dumps(dict(lifecycle=dict(bombs=bombs))),encoding='utf-8')
    row=summarize(tmp_path);write_report([row],tmp_path)
    assert not row['valid'] and row['discovered']==1 and row['picked_unique']==0
    assert row['restricted_mean_discovery_step']==65 and row['navigation']['deaths'] is None
    assert '99/无' in (tmp_path/'navigation_report.md').read_text(encoding='utf-8')


def test_center_approach_is_not_selected_across_occluding_wall():
    g=grid();g.evidence[:,24]=8
    p=PayloadPlanner(g)
    # Current safe component is within 100px of the target, but a wall between
    # them prevents a centre observation. Never return a zero-point 'complete'.
    r=p.plan(np.array([-90.,0.]),np.array([20.,0.]))
    assert not r.complete and not r.points
