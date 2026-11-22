"""Mesh36 real nested messages, actual renderer and task-boundary regressions."""
import hashlib
import inspect
import json

import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_mesh36 import MyDroneRedMesh36
from swarm_rescue.solutions.my_drone_red_rendezvous35 import MyDroneRedRendezvous35
from swarm_rescue.solutions.red_boundary33 import FinalBoundary33
from swarm_rescue.solutions.devtools.test_red_rendezvous35_controller import (
    fixture as prior_fixture,committed_book,
)
from swarm_rescue.solutions.devtools.test_red_frontier34_controller import exchange
from swarm_rescue.solutions.devtools.test_red_fusion27_delivery import fleet,deliver,SIZE,EPOCH


def fixture(source=0,snapshots=None,policy=False,sealed=True):
    old=prior_fixture(source,snapshots=snapshots,policy=policy,sealed=sealed)
    cls=type('Mesh36Fixture',(MyDroneRedMesh36,),{'identifier':source,'drone_health':25})
    d=object.__new__(cls);d.__dict__.update(old.__dict__)
    d._m36_init()
    d._fusion27=FinalBoundary33(SIZE,wall_width=11)
    if hasattr(d._explore,'event'):d._explore.event=d._event
    assert '_event' not in d.__dict__ and 'kind' in inspect.signature(d._event).parameters
    d._f34_budget.update(remaining_steps=2000.)
    return d


def five(rounds=70):
    stores=fleet();drones=[fixture(i,snapshots=stores[i]) for i in range(5)]
    edges={(i,j) for i in range(5) for j in range(5) if abs(i-j)==1}
    exchange(drones,rounds,edges)
    return drones


def emitted(capsys):
    return [json.loads(line.split('V3_EVENT ',1)[1]) for line in
            capsys.readouterr().out.splitlines() if 'V3_EVENT ' in line]


def moving(d):
    d._position=np.array([500.,500.]);d._home=np.zeros(2)
    d._progress=lambda:.9
    d._f34_mode='return'
    d._m36_nav_target=np.zeros(2)
    d._f34_nav.command=lambda *a:dict(forward=.3,lateral=0.,rotation=0.)
    d._f34_nav.estimate_route=lambda *a,**kw:dict(exact=True,steps=10.)
    d._m36_book.observe_direct(1,int(d.elapsed_timestep))


def test_real_nested_five_aircraft_chain_exchanges_actual_maps_and_rendered_image_acks(capsys):
    drones=five()
    images=[]
    for i,d in enumerate(drones):
        state=d._exchange_state();mesh=state['mesh36']
        assert state['source_maps_complete'] and state['final_complete']
        assert state['acknowledged_sources']==list(range(5))
        assert mesh['all_bidirectionally_connected']
        assert d._m36_book.stats['render_hash_computed']==1
        assert not d._m36_final_pixels.flags.writeable
        assert d._m36_final_pixels.shape==(60,90)
        images.append(d._m36_final_pixels)
        for source in range(5):assert d._radio_maps.snapshots[source]==drones[source]._radio_maps.own_final
        assert mesh['render_hash']==hashlib.sha256(d._m36_final_pixels.tobytes()).hexdigest()
    for image in images[1:]:np.testing.assert_array_equal(image,images[0])
    assert drones[0]._m36_book.direct_neighbors(drones[0].elapsed_timestep)==[1]
    assert 4 not in drones[0]._m36_book.direct_neighbors(drones[0].elapsed_timestep)
    rows=emitted(capsys)
    assert len([r for r in rows if r['event']=='mesh36_final_render_verified'])==5
    assert len([r for r in rows if r['event']=='rendezvous35_payload_verified_here'])==25


def test_wire_has_one_source_payload_and_unwrap_preserves_native_navigation_fields():
    a,b=fixture(0),fixture(1)
    p=a.define_message_for_all()
    assert p[0]=='mesh36-v1' and p[3][0]=='rendezvous35-v1'
    frontier=p[3][1];structural=frontier[1];fusion=structural[1]
    assert fusion[0]=='fusion27-v1' and fusion[2] is None
    def payloads(value):
        if isinstance(value,tuple):
            if len(value)==3 and type(value[0]) is int and type(value[1]) is int and isinstance(value[2],bytes):
                return [value]
            return [item for child in value for item in payloads(child)]
        return []
    assert payloads(p)==[p[4][3]], 'Only Mesh36 transports source map bytes, including deep legacy frames'
    assert p[4][3]==a._radio_maps.own_final
    b.communicator.received_messages=[(object(),p)] # Native sender is opaque, not a peer-state oracle.
    b._receive()
    assert b._radio_maps.snapshots[0]==a._radio_maps.own_final
    assert b._status[0]==(a._steps,0)
    np.testing.assert_array_equal(b._peers[0][1],a._position)
    assert b._bomb_records[0]==(a._steps,tuple(a._placements))
    assert b._m36_book.direct_neighbors(b.elapsed_timestep)==[0]


@pytest.mark.parametrize('offset',[4,-1])
def test_expired_or_future_outer_envelope_does_not_create_contact(offset):
    a,b=fixture(0),fixture(1);p=a.define_message_for_all()
    stale=(p[0],p[1],b.elapsed_timestep-offset,p[3],p[4])
    b.communicator.received_messages=[(object(),stale)]
    b._receive()
    assert b._m36_book.direct_neighbors(b.elapsed_timestep)==[]
    assert 0 not in b._radio_maps.snapshots


def test_nonzero_holder_submits_actual_consensus_image_before_old_zero_leader_deadline(capsys):
    drones=five()
    winners=[min(d._m36_book.acks.values(),key=lambda ack:(ack[4],ack[0]))[0] for d in drones]
    assert len(set(winners))==1 and winners[0]!=0
    d=drones[winners[0]]
    for peer in drones:
        peer._progress=lambda:.9
        peer._submit()
    assert sum(len(peer._fixture_submissions) for peer in drones)==1
    assert len(d._fixture_submissions)==1
    np.testing.assert_array_equal(d._fixture_submissions[0],d._m36_book.final_pixels)
    assert d._s28_submission['submitter']==winners[0] and d._s28_submission['final_complete']
    assert d._s28_submission['render_selected_submitter']==winners[0]
    assert d._s28_submission['submitted_pixels_sha256']==d._m36_book.local_ack[3]
    assert d._s28_submission['acknowledged_sources']==list(range(5))
    d._submit();assert len(d._fixture_submissions)==1
    assert any(e['event']=='mesh36_submitted' and e['submitter']==winners[0] for e in emitted(capsys))


def test_complete_source_maps_without_other_render_acks_do_not_submit_early():
    stores=fleet()
    for s in stores[1:]:deliver(stores[0],s)
    d=fixture(0,snapshots=stores[0]);d._progress=lambda:.9
    state=d._exchange_state()
    assert state['source_maps_complete'] and state['acknowledged_sources']==[0]
    d._submit();assert not d._fixture_submissions


def test_incomplete_deadline_submission_remains_explicitly_incomplete():
    d=fixture(0);d._progress=lambda:.975
    d._submit()
    assert len(d._fixture_submissions)==1
    assert not d._s28_submission['final_complete']
    assert d._s28_submission['reason']=='deadline_incomplete_fallback'
    assert set(d._s28_submission['missing'])=={'1','2','3','4'}


def test_one_neighbor_at_meeting_no_longer_causes_unlimited_exchange_hold():
    d=fixture(0);moving(d)
    d._position=np.zeros(2);d._r35_last_rx[1]=d._steps
    d._m36_previous_direct={1}
    for _ in range(25):
        assert d._f34_return_motion()['forward']==.3
        d._steps+=1;d.elapsed_timestep+=1
    assert d._f34_mode=='return'
    assert d._m36_stats['connected_exchange_ticks']==0


def test_full_bidirectional_team_connectivity_holds_then_expired_link_resumes_motion():
    drones=five();d=drones[0];moving(d)
    assert d._f34_return_motion()['forward']==0.
    assert d._m36_phase=='whole_team_exchange'
    d._steps+=d._m36_book.edge_ttl+1;d.elapsed_timestep+=d._m36_book.edge_ttl+1
    assert d._f34_return_motion()['forward']==.3
    assert d._f34_mode=='return' and d._m36_connected_since is None


def test_actual_recently_lost_link_can_be_repaired_after_topology_expires_before_contact_deadline():
    a,b=fixture(0),fixture(1)
    a._position=np.array([0.,0.]);b._position=np.array([30.,0.])
    exchange([a,b],3)
    a._f34_nav.estimate_route=lambda *args,**kwargs:dict(exact=True,steps=15.)
    a._f34_budget['remaining_steps']=1000.
    assert 1 in a._m36_previous_direct
    a._steps+=a._m36_book.edge_ttl+2;a.elapsed_timestep+=a._m36_book.edge_ttl+2
    a.communicator.received_messages=[];a._receive()
    assert 1 in a._m36_lost and 1 not in a._m36_previous_direct
    component=set(a._m36_book.status(a.elapsed_timestep,a._radio_maps)['component'])
    assert 1 not in component
    target=a._m36_choose_repair(component)
    assert target is not None and a._m36_repair[0]==1
    np.testing.assert_array_equal(target,[30.,0.])
    a._steps=a._m36_repair[2]
    assert a._m36_choose_repair(component) is None
    assert a._m36_repair is None


def test_old_noncontact_position_report_cannot_start_link_repair():
    d=fixture(0);moving(d)
    d._f34_reports[4]=(4,d._steps,(500.,500.),None,(-1,)*5,(),(),'return',0)
    assert d._m36_choose_repair({0}) is None
    assert d._m36_repair is None


def test_common_fallback_ignores_unilateral_failure_and_switches_on_same_clock():
    drones=[fixture(i) for i in range(5)]
    for d in drones:
        d._r35_book=committed_book(int(d.identifier));d._progress=lambda:.9
        d._r35_primary_failed=int(d.identifier)%2==0
        np.testing.assert_array_equal(d._r35_target(),[0.,0.])
        assert not d._r35_fallback
    for d in drones:
        d._progress=lambda:.931
        np.testing.assert_array_equal(d._r35_target(),[100.,0.])
        assert d._r35_fallback


def test_reinstall_of_own_trace_keeps_unconsumed_suffix():
    d=fixture(0)
    d._m36_route_kind='own_trace_fresh_hull'
    d._f34_return_points=[np.array([900.,0.]),np.array([600.,0.]),np.array([200.,0.])]
    d._f34_nav.set_observed_route(d._f34_return_points)
    d._f34_nav._observed_cursor35=2
    d._f34_nav.estimate_route=lambda *a,**kw:dict(exact=False,steps=None)
    d._m36_install_route(np.zeros(2))
    np.testing.assert_array_equal(d._f34_nav.remaining_observed_route(),[[200.,0.]])


def test_budget_accounts_remaining_bombs_and_departure_before_inventory_is_empty():
    d=fixture(policy=True,sealed=False);del d.__dict__['_f34_budget_update']
    d._f34_nav.estimate_route=lambda *a,**kw:dict(exact=True,steps=100.)
    inventory=[2];d.carried_bombs_count=lambda:inventory[0]
    d._f34_budget_tick=-1000;d._f34_budget_update();two=dict(d._f34_budget)
    inventory[0]=0;d._steps+=30;d._f34_budget_update();zero=dict(d._f34_budget)
    assert two['deployment_reserve_steps']==360 and two['egress_reserve_steps']==180
    assert two['required_steps']-zero['required_steps']==540
    assert two['required_steps']==100+360+180+150+150+180
    assert two['remaining_steps']==min(two['step_remaining'],two['wall_seconds_remaining']*two['conservative_steps_per_second'])


def test_budget_reserves_an_active_final_bomb_egress_and_the_agreed_fallback_leg():
    d=fixture(policy=True,sealed=False);del d.__dict__['_f34_budget_update']
    d._r35_book=committed_book(0);d._progress=lambda:.5
    d.carried_bombs_count=lambda:0;d._m36_egress.active=True
    calls=[]
    def route(*a,**kw):
        calls.append(np.asarray(a[2]).copy())
        return dict(exact=True,steps=100. if len(calls)==1 else 45.)
    d._f34_nav.estimate_route=route
    d._f34_budget_tick=-1000;d._f34_budget_update()
    assert d._f34_budget['egress_reserve_steps']==90.
    assert d._f34_budget['agreed_fallback_reserve_steps']==45.
    assert d._f34_budget['required_steps']==100+45+90+150+150+180
    d._f34_budget_update();assert len(calls)==2


def deployment_fixture():
    d=fixture(policy=True,sealed=False)
    d._initial_inventory=d._last_inventory=2
    d._attempt_step=None;d._attempt_position=None;d._attempts=0
    d._next_attempt=0;d._total_attempts=0;d._failed_place=None
    d._placements=[];d._m36_delivery_due=True
    inventory=[2];d.carried_bombs_count=lambda:inventory[0]
    return d,inventory


@pytest.mark.parametrize('initial',[2,1])
def test_actual_native_deployment_requests_only_after_prevalidated_exit_then_confirms_inventory(capsys,initial):
    d,inventory=deployment_fixture()
    inventory[0]=d._last_inventory=initial
    if initial==1:d._placements=[(-500.,-500.)]
    already_placed=len(d._placements)
    command=d._empty_command();d._deployment(command)
    assert command['place_bomb']==1 and command['rotation']==0.
    assert d._m36_egress.state=='prepared' and not d._m36_egress.active
    assert len(d._placements)==already_placed and d._attempt_step==d._steps
    d._steps+=1;d.elapsed_timestep+=1;inventory[0]=initial-1
    # Simulate the observable inventory decrement, not a placement request ACK.
    d._m36_egress._command=dict(forward=.1,lateral=0.,rotation=0.)
    d._deployment(d._empty_command())
    assert len(d._placements)==already_placed+1 and d._attempt_step is None and d._last_inventory==initial-1
    rows=emitted(capsys)
    assert sum(e['event']=='placement_requested' for e in rows)==1
    assert sum(e['event']=='placement_confirmed' for e in rows)==1


def test_missing_safe_departure_cancels_native_drop_before_physics():
    d,inventory=deployment_fixture();d.lidar_values=lambda:np.zeros(181)
    command=d._empty_command();d._deployment(command)
    assert command['place_bomb']==0
    assert d._attempt_step is None and d._attempt_position is None
    assert not d._placements and inventory[0]==2
    assert d._m36_stats['drop_deferred_no_exit']==1


def test_active_egress_prevents_new_placement_without_suppressing_pending_confirmation():
    d,inventory=deployment_fixture()
    d._m36_egress._command=dict(forward=.1,lateral=0.,rotation=0.)
    command=d._empty_command();d._deployment(command)
    assert command['place_bomb']==0 and d._attempt_step is None and d._total_attempts==0
