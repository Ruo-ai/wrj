"""Behavior fixtures for the isolated regional candidates; no strength claims."""
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.semifinal_localization import LocalMatcher,PoseGuard,scan_points
from swarm_rescue.solutions.my_drone_blue_semifinal import MyDroneBlueSemifinal,TrustedRescueGrid
from swarm_rescue.solutions.my_drone_red_semifinal import MyDroneRedSemifinal,TrustedSurveyGrid
from .test_fs1_red import fake_drone


def corner():
    a=np.linspace(-80,80,24)
    return np.vstack((np.column_stack((a,np.full(24,100))),np.column_stack((np.full(24,110),a))))


def test_match_corrects_corner_with_a_bounded_search():
    m=LocalMatcher((600,600));p=corner();m.add(p,np.zeros(2),0)
    offset,info=m.match(p,np.array([8.,-6.]),0)
    assert offset is not None and np.linalg.norm(offset[:2]+[8,-6])<5
    assert info['support']>.9 and info['candidates']<=160


def test_flat_wall_preserves_unobservable_axis_and_bad_scan_rejected():
    p=np.column_stack((np.linspace(-120,120,40),np.full(40,100)))
    m=LocalMatcher((600,600));m.add(p,np.zeros(2),0)
    offset,info=m.match(p,np.array([8.,-6.]),0)
    assert not info['observable'][0]
    assert offset is None or offset[0]==0
    assert m.match(corner()+[180,140],np.zeros(2),0)[0] is None
    assert m.match(p[:5],np.zeros(2),0)[0] is None


def test_dynamic_semantics_are_not_match_anchors():
    obj=SimpleNamespace(entity_type=SimpleNamespace(name='BOMB'),distance=100,angle=0)
    points=scan_points([100,100,230,np.nan],[0,1,2,3],[obj])
    assert points.shape==(1,2)


def test_pose_warmup_no_self_validation_and_relock():
    g=PoseGuard((600,600));g.scan=corner()
    for _ in range(8):g.update(np.zeros(2),0,(0,0,0))
    assert g.trusted and len(g.matcher.frames)==1
    # Empty scans must not perpetually validate accumulated odometry.
    g.scan=np.empty((0,2))
    for _ in range(30):g.update(None,None,(2,0,0))
    assert not g.trusted and len(g.matcher.frames)==1  # empty grace scans are not added
    g.update(np.zeros(2),0,(0,0,0))
    assert g.relocked and not g.trusted and not g.matcher.frames
    for _ in range(3):g.update(np.zeros(2),0,(0,0,0))
    assert g.trusted and np.linalg.norm(g.position)<1


def test_trust_guards_both_map_layers():
    guard=SimpleNamespace(trusted=False)
    blue=TrustedRescueGrid((400,400),guard)
    red=TrustedSurveyGrid((400,400),lambda *a,**k:None,guard)
    args=(np.zeros(2),0,np.full(30,100.),np.linspace(-3,3,30),np.zeros(30,bool))
    blue.observe(*args);red.observe(*args)
    assert not blue.free.any() and not red.free.any()
    guard.trusted=True;blue.observe(*args);red.observe(*args)
    assert blue.free.any() and red.free.any()


def task_drone():
    d=object.__new__(MyDroneBlueSemifinal);d.identifier=1;d._tick=100
    d._pose_guard=SimpleNamespace(trusted=True);d._position=np.zeros(2)
    d._bombs=[[100,0,90,True]];d._peers={};d._deferred=[];d._target=None;d._mode='search'
    return d


def test_claims_expire_and_busy_drone_cannot_own_every_task():
    d=task_drone();d._peers={0:(100,np.array([90,0]),'collect',np.array([250,0]))}
    assert np.array_equal(d._choose_bomb(),[100,0])
    d._peers={0:(100,np.array([90,0]),'collect',np.array([100,0]))}
    assert d._choose_bomb() is None
    d._tick=146
    assert np.array_equal(d._choose_bomb(),[100,0])


def test_low_confidence_restricts_targets_to_current_semantics():
    d=task_drone();d._pose_guard.trusted=False;d._visible_bombs=[]
    assert d._choose_bomb() is None
    d._visible_bombs=[(SimpleNamespace(distance=12),np.array([12,0]))]
    assert np.array_equal(d._choose_bomb(),[12,0])
    old=[b.copy() for b in d._bombs]
    d._record_bomb(np.array([200.,0]),100,True)
    assert d._bombs==old


def test_delivery_requires_actual_net_rotation_and_lost_grasp_is_not_success(monkeypatch):
    from swarm_rescue.solutions.my_drone_eval_blue import MyDroneBlueAdapted
    monkeypatch.setattr(MyDroneBlueAdapted,'control',lambda _:dict(forward=.5,lateral=.4,rotation=.65,grasper=0))
    d=task_drone();d._center_hit=SimpleNamespace(distance=10);d._transport_carrying=False
    d._transport_start=None;d._release_requested=False;d._last_delivery_heading=None
    d._turn_progress=0.;d._escape_sign=1;d._heading=0
    monkeypatch.setattr(MyDroneBlueSemifinal,'grasped_bombs',lambda _:[1])
    delta=[0.];monkeypatch.setattr(MyDroneBlueSemifinal,'odometer_values',lambda _:(0,0,delta[0]))
    events=[];d._transport_event=lambda e,**kw:events.append((e,kw))
    for _ in range(40):assert d.control()['grasper']==1
    delta[0]=.1
    for _ in range(6):assert d.control()['grasper']==1
    delta[0]=-.1
    for _ in range(6):assert d.control()['grasper']==1
    assert d._turn_progress<.01
    delta[0]=.1
    for _ in range(14):cmd=d.control()
    assert cmd['grasper']==0 and d._release_requested
    monkeypatch.setattr(MyDroneBlueSemifinal,'grasped_bombs',lambda _:[])
    d.control()
    assert events[-1][0]=='grasp_ended' and not any('success' in e for e,_ in events)


def test_confirmed_union_invalidates_regional_cache_and_claim_decay(monkeypatch):
    monkeypatch.setattr(MyDroneRedSemifinal,'size_area',property(lambda _: (180,180)))
    d=fake_drone(MyDroneRedSemifinal);d.identifier=1
    f=d._coverage_field();d._placements=[(0.,0.)]
    assert f.marginal((0,0))>0 and d._coverage_field().marginal((0,0))==0
    d._placements=[];d._bomb_claims={0:(100,(0.,0.))}
    claimed=d._coverage_field();d._steps=190
    assert d._coverage_field().value((0,0))>claimed.value((0,0))
    assert claimed.marginal((0,0))==d._coverage_field().marginal((0,0))


def test_area_selection_respects_remaining_travel_and_confirmed_overlap():
    from swarm_rescue.solutions.my_drone_red_semifinal import choose_area_placement
    from swarm_rescue.solutions.my_drone_red_advanced2 import _M09_ExplorationGrid
    from .test_fs1_red import field
    grid=_M09_ExplorationGrid((180,180));grid.free[:]=True
    assert choose_area_placement(grid,np.zeros(2),field(),0) is None
    plan=choose_area_placement(grid,np.zeros(2),field([(0,0)]),300)
    assert plan is not None and plan.gain>0 and np.linalg.norm(plan.target)>=65


def test_oracle_greedy_union_and_invalid_blue_never_count_as_success():
    from .optimization.evaluation.oracle_layouts import greedy
    from .optimization.evaluation.joint_batch import oracle_valid
    walls=np.zeros((128,128),np.uint8);walls[:,60:68]=1
    allowed=np.ones_like(walls,bool);allowed[:,40:88]=False
    bombs,gains,total,union=greedy(walls,np.zeros_like(walls),allowed,32,count=2)
    assert len(bombs)==2 and total==sum(gains)==walls[union].sum()
    good=dict(status='ok',exit_code=0,blue_score=95)
    assert oracle_valid(good)
    for flag in ('walltime_limited','engine_callback_errors','controller_crashed'):
        assert not oracle_valid(dict(good,**{flag:True}))


def test_mixed_map_selection_excludes_holdouts():
    from .optimization.evaluation.scenarios import training_rows
    from .optimization.evaluation.oracle_layouts import MAPS,DATASET
    from .optimization.evaluation.runner import read
    rows=training_rows(MAPS,DATASET)
    assert [r['name'] for r in rows]==MAPS
    holdout=read(DATASET/'validation.json')['maps'][0]['id']
    name=next(r['name'] for r in read(DATASET/'index.json')['maps'] if r['id']==holdout)
    with pytest.raises(ValueError,match='training map'):training_rows([name],DATASET)


def test_native_red_deployment_confirms_inventory_not_claims():
    from .test_fs1_red import test_native_short_control_and_real_deployment_confirmation
    test_native_short_control_and_real_deployment_confirmation(MyDroneRedSemifinal)


@pytest.mark.parametrize('cls',[MyDroneBlueSemifinal,MyDroneRedSemifinal])
def test_native_control_missing_absolute_and_reacquisition(cls,monkeypatch):
    import random
    from .evaluate_red import configure_ray_backend
    from swarm_rescue import launcher
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    configure_ray_backend('cpu');np.random.seed(742);random.seed(742)
    world=launcher.Map09(cls,(ZoneType.NO_GPS_ZONE,ZoneType.NO_COM_ZONE),5)
    try:
        if cls is MyDroneRedSemifinal:
            for d in world.drones:d.enable_bomb_deployer(world,2)
        world.playground.step()
        gps=cls.measured_gps_position;compass=cls.measured_compass_angle
        for step in range(1,57):
            missing=17<=step<49
            monkeypatch.setattr(cls,'measured_gps_position',lambda d:None if missing else gps(d))
            monkeypatch.setattr(cls,'measured_compass_angle',lambda d:None if missing else compass(d))
            commands={}
            for d in world.drones:
                d.elapsed_timestep=step;d.elapsed_walltime=0
                commands[d]=d.control()
                assert all(np.isfinite(float(v)) for v in commands[d].values())
            world.playground.step(all_commands=commands)
        for d in world.drones:
            guard=d._pose_guard if cls is MyDroneBlueSemifinal else d._pose_filter
            assert guard.absolute_missing_steps==0 and guard.trusted
    finally:world.playground.window.close()
