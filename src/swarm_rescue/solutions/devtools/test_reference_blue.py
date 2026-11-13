"""Behavioural contracts for a finite, timely and legal semifinal reference blue."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_eval_blue import (
    MyDroneBlue, RescueGrid, steering_choice, LIDAR_RANGE)


def test_max_range_and_nonfinite_readings_never_create_walls():
    g=RescueGrid((800,800));a=np.linspace(-math.pi,math.pi,181)
    for value in [LIDAR_RANGE,LIDAR_RANGE+5,np.inf,np.nan,0]:
        for _ in range(4):g.observe(np.zeros(2),0,np.full(len(a),value),a,np.zeros(len(a),bool))
    assert not (g.evidence>0).any()
    g.observe(np.zeros(2),0,None,None,None)


def test_real_near_wall_is_retained_and_dynamic_hits_excluded():
    a=np.array([0.]);pos=np.zeros(2)
    real=RescueGrid((800,800));dynamic=RescueGrid((800,800))
    for _ in range(4):
        real.observe(pos,0,np.array([100.]),a,np.array([False]))
        dynamic.observe(pos,0,np.array([100.]),a,np.array([True]))
    assert real.evidence[real.cell([100,0])]>=2
    assert not dynamic.evidence.any()
    assert not real.free[real.cell([260,0])]


@pytest.mark.parametrize('carrying',[False,True])
def test_vector_steering_matches_scalar_observed_corridors(carrying):
    rng=np.random.default_rng(711)
    a=np.linspace(-math.pi,math.pi,181)
    for _ in range(30):
        d=rng.uniform(4,LIDAR_RANGE,len(a));d[rng.choice(len(a),8,replace=False)]=np.nan
        desired=rng.uniform(-math.pi,math.pi);c=43 if carrying else 31
        valid=np.isfinite(d);vv,aa=d[valid],a[valid];best=None
        for angle in np.linspace(-math.pi,math.pi,49,endpoint=False):
            rel=(aa-angle+math.pi)%(2*math.pi)-math.pi
            forward=vv*np.cos(rel);inside=(np.abs(rel)<.65)&(np.abs(vv*np.sin(rel))<c)
            gap=float(forward[inside].min()) if inside.any() else LIDAR_RANGE
            value=-2.8*abs((angle-desired+math.pi)%(2*math.pi)-math.pi)+min(gap,140)/100-8*(gap<c)
            if best is None or value>best[0]:best=(value,angle,gap)
        assert steering_choice(d,a,desired,carrying)==pytest.approx(best[1:])
    assert steering_choice(np.array([np.nan]),np.array([0.]),0,carrying) is None


def test_cached_layers_do_not_leak_route_or_caller_mutations():
    g=RescueGrid((320,320));g.free[:]=True
    first,_=g.layers();first[:]=False
    second,_=g.layers();assert second.all()
    for _ in range(3):g.observe(np.zeros(2),0,np.array([80.]),np.array([0.]),np.array([False]))
    third,_=g.layers();assert not third[g.cell([80,0])]


def test_route_stays_in_observed_reachable_space():
    g=RescueGrid((320,320));g.free[3:17,3:17]=True;g.evidence[3:14,10]=4
    start=g.point((7,5));end=g.point((7,14));route=g.route(start,end)
    allowed,_=g.layers()
    assert route and all(allowed[g.cell(p)] for p in route)
    assert any(g.cell(p)[0]>=15 for p in route)
    assert np.linalg.norm(route[-1]-end)<20


def bidder(enhanced=True):
    return SimpleNamespace(ENHANCED=enhanced,identifier=0,_position=np.array([0.,0.]),
        _peers={},_bombs=[[100.,0.,10,True]],_tick=20,BOMB_TTL=700,_deferred=[],_mode='search',_target=None)


def test_busy_peer_does_not_block_unrelated_bomb_assignment():
    b=bidder();b._peers={1:(20,np.array([90.,0.]),'collect',np.array([90.,200.]))}
    assert MyDroneBlue._choose_bomb(b)==pytest.approx([100,0])
    b.ENHANCED=False
    assert MyDroneBlue._choose_bomb(b) is None


def test_existing_claim_respected_and_own_task_survives_nearer_idle_peer():
    b=bidder();b._peers={1:(20,np.array([200.,0.]),'collect',np.array([100.,0.]))}
    assert MyDroneBlue._choose_bomb(b) is None
    b._peers={1:(20,np.array([90.,0.]),'search',None)};b._mode='collect';b._target=np.array([100.,0.])
    assert MyDroneBlue._choose_bomb(b)==pytest.approx([100,0])
    b._deferred=[(np.array([100.,0.]),100)]
    assert MyDroneBlue._choose_bomb(b) is None


def test_replay_pins_engine_and_map_but_not_live_controllers(monkeypatch):
    from swarm_rescue.solutions.devtools.optimization.evaluation import runner,replay
    selected=[]
    monkeypatch.setattr(runner,'validate_sources',lambda s:selected.append(s))
    sources={'src/swarm_rescue/solutions/my_drone_eval_blue.py':'a',
             'src/swarm_rescue/simulation/gui_map/replay_view.py':'b',
             'map_data/map_09.json':'c',
             'src/swarm_rescue/solutions/devtools/optimization/generated_maps/a.py':'d'}
    replay.validate_replay_sources(sources)
    assert selected==[{k:v for k,v in sources.items() if v!='a'}]


@pytest.mark.parametrize('width',[56,64])
def test_swept_payload_can_advance_slowly_in_observed_straight_corridor(width):
    from swarm_rescue.solutions.my_drone_eval_blue import swept_steering
    a=np.linspace(-math.pi,math.pi,181)
    d=np.minimum(LIDAR_RANGE,width/2/np.maximum(np.abs(np.sin(a)),1e-9))
    angle,gap,tight=swept_steering(d,a,0,True)
    assert angle==pytest.approx(0) and gap>8 and tight
    # The old formulation stops in the same observed corridor.
    _,old_gap=steering_choice(d,a,0,True)
    assert old_gap<43


def test_swept_payload_rejects_too_tight_corridor_and_front_obstacle():
    from swarm_rescue.solutions.my_drone_eval_blue import swept_steering
    a=np.linspace(-math.pi,math.pi,181)
    d=np.minimum(LIDAR_RANGE,20/np.maximum(np.abs(np.sin(a)),1e-9))
    angle,gap,_=swept_steering(d,a,0,True)
    assert gap<=8 or abs(angle)>1.2
    d=np.full(len(a),10.)
    assert swept_steering(d,a,0,False)[1]<=8
    assert swept_steering(np.full(len(a),np.nan),a,0,True) is None


def test_ignored_engine_callback_invalidates_an_otherwise_complete_score():
    from swarm_rescue.solutions.devtools.optimization.evaluation.runner import phase_valid
    result=dict(status='ok',exit_code=0,phase1=dict(score_status='final'))
    assert phase_valid(result,'blue')
    result['engine_callback_errors']=1
    assert not phase_valid(result,'blue')


def test_default_reference_entrypoints_do_not_select_unaccepted_enhancements():
    import swarm_rescue.solutions.my_drone_eval_blue as blue
    from swarm_rescue.solutions.devtools.evaluate_provided_blue import reference_blue_class
    assert blue.REFERENCE_BLUE_CLASS is blue.MyDroneBlueAdapted
    assert blue.MyDroneEval is blue.MyDroneBlueAdapted
    assert blue.drone_class_for_mode('rescue') is blue.MyDroneBlueAdapted
    assert reference_blue_class() is blue.MyDroneBlueAdapted
    assert reference_blue_class('adapted') is blue.MyDroneBlueAdapted
    assert reference_blue_class('enhanced') is blue.MyDroneBlue
    with pytest.raises(ValueError):reference_blue_class('unknown')


@pytest.mark.parametrize('carrying',[False,True])
@pytest.mark.parametrize('swept_empty',[False,True])
def test_transport_ablation_preserves_empty_navigation_and_task_settings(monkeypatch,carrying,swept_empty):
    import swarm_rescue.solutions.my_drone_eval_blue as blue
    calls=[]
    def old(*args):
        calls.append('adapted');return 0.,150.
    def swept(*args):
        calls.append('swept');return 0.,150.,False
    monkeypatch.setattr(blue,'steering_choice',old)
    monkeypatch.setattr(blue,'swept_steering',swept)
    b=SimpleNamespace(ENHANCED=True,SWEPT_PAYLOAD=MyDroneBlue.SWEPT_PAYLOAD,SWEPT_EMPTY=swept_empty,
        _peers={},_tick=1,_route_step=1,_route=[np.array([100.,0.])],
        _position=np.zeros(2),_heading=0.,_debug=False,identifier=0,
        _grid=SimpleNamespace(shortcut=lambda p,r:r),_anchor=None,_escape_until=0)
    cmd=MyDroneBlue._navigate(b,np.array([100.,0.]),np.array([200.]),np.array([0.]),carrying)
    assert calls==(['swept'] if swept_empty and not carrying else ['adapted'])
    assert cmd['forward']==pytest.approx(.62 if carrying else .72)
    assert cmd['grasper']==int(carrying) and b.ENHANCED
