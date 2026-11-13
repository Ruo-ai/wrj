"""Area estimator, union semantics, execution gate and native-path checks."""
from types import SimpleNamespace
import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_fs1 import (
    supported_area, AreaCoverageField, release_quality, MyDroneRedFS1Area, MyDroneRedFS1Gain,
)
from swarm_rescue.solutions import my_drone_red_advanced2 as base


def field(placed=(), claims=(), area=None):
    if area is None:
        area=np.full((60, 60),9.,np.float32)
    return AreaCoverageField(area,3,(180,180),placed,claims,24)


def test_area_keeps_supported_thickness_and_never_fills_room():
    obs=base._M18_ObservationMapRefined((180,180))
    obs.evidence[20:40,20:40]=8
    obs.evidence[24:36,24:36]=0
    before=obs.evidence.copy()
    area=supported_area(obs,np.zeros(obs.shape,bool))
    assert not area[24:36,24:36].any()
    assert area.sum()==(20*20-12*12)*9
    assert np.array_equal(before,obs.evidence)
    thin=field(area=np.where(area>0,9.,0.))
    thicker=field(area=np.full((60,60),9.))
    assert thicker.value((0,0))>thin.value((0,0))


def test_free_and_nonwall_contradictions_veto_remote():
    obs=base._M18_ObservationMapRefined((180,180))
    obs.evidence[30,30]=-2
    obs.evidence[20,20]=8
    obs.static_nonwall[20,20]=2
    obs.evidence[40,40]=4
    peer=np.ones(obs.shape,bool)
    area=supported_area(obs,peer)
    assert area[30,30]==area[20,20]==0
    assert area[40,40]==pytest.approx(3)
    assert area[10,10]==pytest.approx(9*.35)
    assert np.array_equal(peer,np.ones(obs.shape,bool))


def test_partial_border_cells_cannot_inflate_area():
    obs=base._M18_ObservationMapRefined((181,182));obs.evidence[:]=8
    area=supported_area(obs,np.zeros(obs.shape,bool))
    assert area.sum()==181*182 and area[-1,-1]==2


def test_confirmed_union_duplicate_order_and_claim_separation():
    a,b=(-10.,0.),(10.,0.)
    f=field([a,b,a]);g=field([b,a])
    assert np.array_equal(f.unclaimed,g.unclaimed)
    assert f.marginal(a)==0
    promised=field([], [a])
    assert promised.marginal(a)==field().marginal(a)>promised.value(a)>0
    assert field([a],[a]).value(a)==0


def test_gain_equals_union_difference_and_diminishes():
    a,b,c=(-15.,0.),(15.,0.),(0.,20.)
    gross=field()
    f=field([a]);g=field([a,b])
    union_ab=gross.standalone[gross.disk(a)|gross.disk(b)].sum()
    union_a=gross.standalone[gross.disk(a)].sum()
    assert f.marginal(b)==pytest.approx(float(union_ab-union_a))
    assert g.marginal(c)<=f.marginal(c)<=gross.marginal(c)


def test_empty_circle_overlap_is_not_a_wall_penalty():
    area=np.zeros((60,60),np.float32)
    area[30,39]=9  # x=28.5: new disk only, despite disks overlapping in empty space
    assert field([(-15,0)],area=area).marginal((15,0))==field(area=area).marginal((15,0))==9


def test_coarse_convolution_matches_exact_grid_centers():
    f=field([(-15,0)],[(15,0)])
    for r,c in [(20,20),(30,30),(40,40),(0,0)]:
        p=(f.x[r,c],f.y[r,c])
        assert f.sampled_value(p)==pytest.approx(f.value(p),abs=.001)


def test_release_gate_separates_claims_and_allows_deadline():
    f=field([(0,0)])
    assert not release_quality(f,(0,0),0,.7)['allow']
    assert release_quality(f,(0,0),60,.76)['reason']=='deadline_fallback'
    assert release_quality(field([],[(0,0)]),(0,0),0,.5)['allow']
    assert release_quality(field(),(0,0),60,.5)['displacement']==15


def fake_drone(cls):
    d=object.__new__(cls)
    d._map=base._M18_ObservationMapRefined((180,180));d._map.evidence[:]=8
    d._radio_maps=base._M01_SnapshotStore(d._map.shape,5)
    d._bomb_claims={};d._bomb_records={};d._placements=[]
    d._steps=100;d._advanced_field=None;d._field_step=-100;d._fs1_signature=None
    return d


def test_confirmed_fact_invalidates_cache_but_claim_expires(monkeypatch):
    monkeypatch.setattr(MyDroneRedFS1Area,'size_area',property(lambda _: (180,180)))
    d=fake_drone(MyDroneRedFS1Area)
    d.identifier=1
    f=d._coverage_field();assert f.marginal((0,0))>0
    d._placements=[(0.,0.)]
    assert d._coverage_field().marginal((0,0))==0
    d._steps=500
    assert d._coverage_field().marginal((0,0))==0  # never expires with radio age
    d._placements=[];d._bomb_claims={0:(500,(0.,0.))}
    claimed=d._coverage_field()
    d._steps=590
    assert d._coverage_field().value((0,0))>claimed.value((0,0))


def test_baselines_and_formal_entry_remain_frozen():
    import json,hashlib
    from .optimization.evaluation.runner import REPO
    frozen=json.loads((REPO/'output/semifinal_fs0_20261113/baseline/identity.json').read_text(encoding='utf-8'))
    for name in ('my_drone_red_advanced2.py','my_drone_red_advanced5_narrow.py','my_drone_eval_blue.py','my_drone_eval.py'):
        key='src/swarm_rescue/solutions/'+name
        assert hashlib.sha256((REPO/key).read_bytes()).hexdigest()==frozen['source_sha256'][key]


@pytest.mark.parametrize('cls',[MyDroneRedFS1Area,MyDroneRedFS1Gain])
def test_native_short_control_and_real_deployment_confirmation(cls):
    from .evaluate_red import configure_ray_backend
    from swarm_rescue import launcher
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    configure_ray_backend('cpu')
    np.random.seed(742)
    world=launcher.Map08(cls,(ZoneType.NO_GPS_ZONE,),5)
    try:
        for drone in world.drones:drone.enable_bomb_deployer(world,2)
        world.playground.step()
        # Exercise actual action -> inventory -> confirmation path at deadline.
        # This time jump is a behavior fixture, never a score or a complete run.
        for step in range(1,41):
            commands={}
            for drone in world.drones:
                drone.elapsed_timestep=4200+step;drone.elapsed_walltime=0
                commands[drone]=drone.control()
                assert all(np.isfinite(float(v)) for v in commands[drone].values())
            world.playground.step(all_commands=commands)
        assert sum(len(d._placements) for d in world.drones)>0
        for d in world.drones:
            # The last physics step may have just accepted an action. Give the
            # controller its next inventory observation before checking facts.
            d._steps+=1
            d._deployment({})
            assert len(d._placements)==2-d.carried_bombs_count()
            for p in d._placements:
                assert d._coverage_field(force=True).marginal(p)==0
    finally:
        world.playground.window.close()


def test_area_planner_selects_reachable_positive_residual():
    grid=base._M09_ExplorationGrid((400,400));grid.free[:]=True
    grid.obstacle[:,10]=3
    area=np.zeros((100,100),np.float32);area[:,50:52]=16
    f=AreaCoverageField(area,4,(400,400),[(-40,100)],[],48)
    position=grid.point((10,5))
    plan=base._M17_choose_placement(grid,position,f,None)
    assert plan is not None and plan.gain>0
    assert plan.gain==f.value(plan.target)
    assert all(c<10 and grid.obstacle[r,c]<2 for r,c in plan.route)


def test_gate_does_not_skip_confirmation_or_pending_request(monkeypatch):
    monkeypatch.setattr(MyDroneRedFS1Gain,'size_area',property(lambda _: (180,180)))
    d=fake_drone(MyDroneRedFS1Gain);d.identifier=0
    d._attempt_step=99;d._last_inventory=2;d._position=np.zeros(2)
    d._bomb_target=None;d._next_attempt=0
    monkeypatch.setattr(MyDroneRedFS1Gain,'carried_bombs_count',lambda _:1)
    called=[]
    monkeypatch.setattr(base.MyDroneRedAdvanced2,'_deployment',lambda *args:called.append(True))
    monkeypatch.setattr(MyDroneRedFS1Gain,'_coverage_field',lambda *args,**kw:pytest.fail('must process confirmation first'))
    d._deployment({})
    assert called==[True]
