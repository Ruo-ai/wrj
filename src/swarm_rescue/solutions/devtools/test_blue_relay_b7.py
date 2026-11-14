"""Light in-memory tests only: no engine loop, red imports or old-file writes."""
import copy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from swarm_rescue.solutions.blue_relay_b7_protocol import RelayLease,RelayPacket
from swarm_rescue.solutions.my_drone_blue_relay_b7 import (
    MyDroneBlueRelayDecision,MyDroneBlueRelayConfirmed,MyDroneBlueRelayOff,RelayView)
from swarm_rescue.solutions.my_drone_eval_blue import MyDroneBlueAdapted,RescueGrid


def packet(lease,tick):return lease.identifier,tick,lease.extension(tick,True,25)
def update(lease,packets,tick,**kw):return lease.update(packets,tick,lambda p:50.,lambda p:50.,**kw)


def pair():
    a,b=RelayLease(0),RelayLease(1);assert a.start((20.,0.),1,0)
    update(b,[packet(a,0)],1);assert b.inc['phase']=='accepted'
    assert a.out['phase']=='offered'
    update(a,[packet(b,1)],2);assert a.out['phase']=='confirmed'
    return a,b


def test_received_acceptance_is_required_before_confirmation():
    a,b=pair();assert a.stats['takeover_confirmed']==1 and b.stats['receiver_accepted']==1


def test_dropped_offer_does_not_produce_indefinite_deferral():
    a=RelayLease(0);a.start((20.,0.),1,0);update(a,[],9)
    assert a.out['phase']=='cancelled' and a.fallback['reason']=='no_ack'
    assert a.blocked((20.,0.),10)


def test_dropped_ack_cancel_reaches_receiver_without_releasing_payload():
    a,b=RelayLease(0),RelayLease(1);a.start((20.,0.),1,0)
    update(b,[packet(a,0)],1);update(a,[],9);update(b,[packet(a,9)],10)
    assert b.inc is None and b.terminal['phase']=='cancelled'
    a,b=pair();update(b,[],3,carrying=True);a.fail_out(4,'link_loss')
    update(b,[packet(a,4)],5,carrying=True)
    assert b.inc['phase']=='held'


def test_expired_lease_and_duplicate_ack_do_not_keep_scouting():
    a,b=pair();duplicate=packet(b,1)
    for tick in (3,10,20,32):update(a,[duplicate],tick)
    assert a.fallback['reason']=='ack_expired'


def test_stalled_receiver_releases_claim():
    a,b=pair();update(b,[],92)
    assert b.inc is None and b.terminal['phase']=='cancelled'
    update(a,[packet(b,92)],93)
    assert a.fallback['reason']=='receiver_cancelled'


def test_busy_or_untrusted_receiver_does_not_accept():
    a,b=RelayLease(0),RelayLease(1);a.start((20.,0.),1,0)
    update(b,[packet(a,0)],1,available=False);assert b.inc is None
    update(b,[packet(a,2)],3,healthy=False);assert b.inc is None


def test_competing_receivers_resolve_by_deterministic_priority():
    a,b=RelayLease(0),RelayLease(1);low,high=RelayLease(2),RelayLease(3)
    a.start((20.,0.),2,0);b.start((22.,0.),3,0)
    update(low,[packet(a,0)],1);update(high,[packet(b,0)],1)
    update(high,[packet(low,1)],2);update(low,[packet(high,2)],3)
    assert low.inc['phase']=='accepted' and high.inc is None


def test_payload_observation_is_distinct_from_official_disposal():
    a,b=pair();update(b,[],3,carrying=True);update(a,[packet(b,3)],4)
    assert a.out is None and a.stats['peer_pickup_confirmed']==1
    update(b,[],5,carrying=False)
    assert b.terminal['phase']=='ended_unconfirmed'
    assert 'disposed' not in b.stats


def test_messages_are_backward_compatible_and_detached_from_live_tasks():
    a=RelayLease(0);a.start((20.,0.),1,0)
    base=('blue-v1',0,0,(0.,0.),'search',None,(),None,None)
    message=RelayPacket(base,a.extension(0,True,25));a.out['phase']='cancelled'
    assert tuple(message)==base and len(message)==9
    assert message.relay['offer']['phase']=='offered'


def make_drone(cls,identifier=0):
    from swarm_rescue.simulation.utils.misc_data import MiscData
    d=cls(identifier=identifier,misc_data=MiscData(size_area=(512,256),number_drones=10,max_timestep_limit=5400,max_walltime_limit=180))
    d._inputs=NS(gps=np.array([-80.,0.]),compass=0.,held=[],objects=[])
    d.measured_gps_position=lambda:d._inputs.gps;d.measured_compass_angle=lambda:d._inputs.compass
    d.odometer_values=lambda:np.zeros(3);d.grasped_bombs=lambda:d._inputs.held
    d.lidar_values=lambda:np.full(181,230.);d.lidar_rays_angles=lambda:np.linspace(-np.pi,np.pi,181)
    d.semantic_values=lambda:d._inputs.objects;d.semantic_is_disabled=lambda:False
    return d


def context(cls):
    d=make_drone(cls);d._position=np.array([-80.,0.]);d._home=d._position.copy();d._tick=100
    d._relay_trusted=True;d._center=np.array([-150.,0.]);d._grid.free[2:14,2:25]=True
    point=np.array([-32.,0.]);d._bombs=[[-32.,0.,100,True]]
    d._relay_hits=[(NS(distance=48.,angle=0.),point)]
    return d,point


def test_reachable_frontier_has_gain_but_unknown_or_disconnected_bomb_rejected():
    d,point=context(MyDroneBlueRelayDecision)
    assert d._frontier(point) is not None
    assert d._can_accept(np.array([220.,0.])) is None
    d._grid.evidence[:,10]=8;d._grid._revision+=1;d._tick+=24
    assert d._can_accept(point) is None


@pytest.mark.parametrize('condition',['no_peer','no_frontier','late','no_gps','unknown_return'])
def test_no_intentional_skip_without_all_prerequisites(condition):
    d,point=context(MyDroneBlueRelayConfirmed)
    if condition=='no_frontier':d._grid.free[:]=True
    elif condition=='late':d._tick=5350;d._bombs[0][2]=5350
    elif condition=='no_gps':d._relay_trusted=False
    elif condition=='unknown_return':d._center=None
    np.testing.assert_equal(d._choose_bomb(),point)
    assert d._relay.out is None and not d._relay_wait


def test_existing_collect_report_allows_only_bounded_passive_deferral():
    d,point=context(MyDroneBlueRelayDecision)
    base=('blue-v1',1,100,(-80.,-16.),'collect',tuple(point),((-32.,0.,100,True),),None,None)
    d._peers={1:(100,np.array([-80.,-16.]),'collect',point)}
    d._relay_meta={1:(100,dict(trusted=True,health=25),base)}
    assert d._choose_bomb() is None and d._relay_explore is not None
    d._tick=114
    np.testing.assert_equal(d._choose_bomb(),point)
    assert d._relay.fallback is not None


def test_closed_switch_is_control_and_message_equivalent_to_B0():
    base=make_drone(MyDroneBlueAdapted);off=make_drone(MyDroneBlueRelayOff)
    for tick in range(1,33):
        for d in (base,off):d.elapsed_timestep=tick
        assert base.control()==off.control()
        assert base.define_message_for_all()==off.define_message_for_all()
    np.testing.assert_equal(base._grid.evidence,off._grid.evidence)


def test_carrying_controls_remain_B0_for_both_candidates():
    for cls in (MyDroneBlueRelayDecision,MyDroneBlueRelayConfirmed):
        base=make_drone(MyDroneBlueAdapted);candidate=make_drone(cls)
        for tick in range(1,25):
            for d in (base,candidate):d.elapsed_timestep=tick;d._inputs.held=[object()]
            assert base.control()==candidate.control()


def test_policy_sources_have_no_online_truth_or_map_special_cases():
    from pathlib import Path
    import swarm_rescue.solutions.my_drone_blue_relay_b7 as policy
    text=Path(policy.__file__).read_text(encoding='utf-8')
    for word in ('true_position','true_angle','Map08','Map09','1560','read_text','np.load','requests.','default_rng'):
        assert word not in text


def test_two_controller_interface_smoke_with_delayed_messages(tmp_path):
    """32 control steps, static legal sensor fixtures; NOT native flight/score."""
    from swarm_rescue.solutions.my_drone_eval_blue import DroneSemanticSensor
    import json
    drones=[make_drone(MyDroneBlueRelayConfirmed,i) for i in (0,1)]
    locations=[np.array([-80.,0.]),np.array([-110.,-16.])];bomb=np.array([-32.,0.])
    previous=[None,None];trace=[]
    for tick in range(1,33):
        for i,d in enumerate(drones):
            d._inputs.gps=locations[i];d._center=np.array([-150.,0.]);d._center_step=tick
            d._grid.free[2:14,2:25]=True;d._grid._revision+=1
            delta=bomb-locations[i]
            d._inputs.objects=[NS(entity_type=DroneSemanticSensor.TypeEntity.BOMB,distance=np.linalg.norm(delta),angle=np.arctan2(delta[1],delta[0]),grasped=tick>=23)]
            d._inputs.held=[object()] if i==1 and tick>=23 else []
            d.communicator._received_messages=[(None,previous[1-i])] if previous[1-i] is not None else []
        built=[d.define_message_for_all() for d in drones]  # GUI builds before control.
        for i,d in enumerate(drones):
            d.elapsed_timestep=tick;cmd=d.control()
            trace.append(dict(step=tick,drone=i,mode=d._mode,command=cmd,source=d._command_source,
                out=copy.deepcopy(d._relay.out),incoming=copy.deepcopy(d._relay.inc)))
        previous=built
    (tmp_path/'relay_interface_smoke.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    assert drones[0]._relay.stats['takeover_confirmed']>=1
    assert any(r['drone']==0 and r['source']=='relay_scout' and r['command']['grasper']==0 for r in trace)
    assert drones[1]._relay.stats['pickup_observed']==1
    assert drones[0]._relay.stats['peer_pickup_confirmed']==1


def test_extended_diagnostics_do_not_change_final_controls():
    from swarm_rescue.solutions.devtools.blue_relay_b7_diagnostics import instrument
    base=make_drone(MyDroneBlueRelayConfirmed);probe=make_drone(instrument(MyDroneBlueRelayConfirmed))
    for tick in range(1,26):
        base.elapsed_timestep=probe.elapsed_timestep=tick
        assert base.control()==probe.control()
    assert probe._b7_coverage is not None and probe._b7_coverage['observed_cell_samples']>0


def test_adapter_requires_explicit_red_idle_before_any_freeze(monkeypatch,tmp_path):
    import sys
    from swarm_rescue.solutions.devtools.blue_relay_b7_run import main
    output=tmp_path/'must_not_exist'
    monkeypatch.setattr(sys,'argv',['b7','--run','--output',str(output)])
    with pytest.raises(SystemExit) as error:main()
    assert error.value.code==2 and not output.exists()


def test_discovery_report_keeps_missing_bombs_and_invalid_raw_score(tmp_path):
    import json
    from swarm_rescue.solutions.devtools.blue_relay_b7_report import summarize
    (tmp_path/'result.json').write_text(json.dumps(dict(status='ok',exit_code=0,blue_score=50,walltime_limited=True,elapsed_steps=100,bomb_count=3)),encoding='utf-8')
    bombs=[dict(bomb=i+1,initial=[20*i,0],first_seen=seen,first_claim=None,first_near=None,pickups=[],losses=[],releases=[],disposed=None,first_center_near=None) for i,seen in enumerate((10,50,None))]
    (tmp_path/'telemetry.json').write_text(json.dumps(dict(lifecycle=dict(bombs=bombs,official_disposed=0))),encoding='utf-8')
    row=summarize(tmp_path)
    assert not row['valid'] and row['result']['blue_score']==50
    assert row['discovery_milestones']['3'] is None and row['undiscovered']==1
    assert row['restricted_mean_discovery_step']==pytest.approx(160/3)
    assert row['discovered_no_relay_ack'] is None


def test_missing_telemetry_is_unknown_instead_of_zero_discovery(tmp_path):
    import json
    from swarm_rescue.solutions.devtools.blue_relay_b7_report import summarize
    (tmp_path/'result.json').write_text(json.dumps(dict(status='failed',exit_code=1,elapsed_steps=0,bomb_count=10)),encoding='utf-8')
    row=summarize(tmp_path)
    assert row['discovered'] is None and row['picked_unique'] is None and not row['valid']


def test_one_handoff_does_not_authorize_skipping_another_bomb():
    d,point=context(MyDroneBlueRelayConfirmed)
    d._relay.start(point,1,100);d._relay.out.update(phase='confirmed',ack=100)
    second=np.array([-72.,40.]);d._relay_hits.append((NS(distance=41.,angle=1.3),second))
    d._bombs.append([-72.,40.,100,True])
    np.testing.assert_equal(d._choose_bomb(),second)
    assert d._relay_explore is None and d._relay.out['phase']=='confirmed'
    update(d._relay,[],101,carrying=True)
    assert d._relay.out['phase']=='confirmed'
