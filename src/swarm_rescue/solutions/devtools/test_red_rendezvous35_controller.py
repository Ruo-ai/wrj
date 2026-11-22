"""Agreed rendezvous and actual native message/delivery integration tests."""
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_rendezvous35 import MyDroneRedRendezvous35
from swarm_rescue.solutions.my_drone_red_frontier34r1 import MyDroneRedFrontier34R1
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.red_rendezvous35_protocol import RendezvousBook35, digest35
from swarm_rescue.solutions.devtools.test_red_frontier34r1 import fixture as prior_fixture
from swarm_rescue.solutions.devtools.test_red_frontier34_controller import exchange
from swarm_rescue.solutions.devtools.test_red_fusion27_delivery import fleet, deliver


def fixture(source=0, snapshots=None, policy=False, sealed=True):
    previous = prior_fixture(source, snapshots=snapshots, policy=policy, sealed=sealed)
    cls = type('Rendezvous35Fixture', (MyDroneRedRendezvous35,),
               {'identifier':source,'drone_health':25})
    d = object.__new__(cls)
    d.__dict__.update(previous.__dict__)
    d._r35_init()
    if hasattr(d._explore,'event'):
        d._explore.event = d._event
    assert '_event' not in d.__dict__
    d._f34_budget = dict(route_exact=True,required_steps=400.,remaining_steps=1800.,
                        arrival_remaining_steps=1500.,arrival_required_steps=250.)
    d.lidar_values = lambda: np.full(181,200.)
    d.lidar_rays_angles = lambda: np.linspace(-np.pi,np.pi,181)
    d.semantic_values = lambda: ()
    d.odometer_values = lambda: np.zeros(3)
    d._map = SimpleNamespace(mask=lambda: np.zeros(d._radio_maps.shape,bool))
    return d


def committed_book(source=0, clock=100, meeting=(0.,0.), fallback=(100.,0.)):
    books = [RendezvousBook35(i,5) for i in range(5)]
    books[0].propose(clock,meeting,fallback)
    for b in books[1:]:
        b.ingest(books[0].packet(),clock)
    for b in books:
        assert b.accept(clock+1,True)
    for b in books[1:]:
        books[0].ingest(b.packet(),clock+2)
    assert books[0].commit is not None
    for b in books[1:]:
        b.ingest(books[0].packet(),clock+3)
    return books[source]


def emitted(capsys):
    return [json.loads(line.split('V3_EVENT ',1)[1])
            for line in capsys.readouterr().out.splitlines() if 'V3_EVENT ' in line]


def test_nested_native_packets_exchange_all_final_bytes_manifests_and_strict_ack(capsys):
    stores = fleet()
    drones = [fixture(i,snapshots=stores[i]) for i in range(5)]
    exchange(drones,30)
    for d in drones:
        state = d._exchange_state()
        assert state['final_complete'] and state['acknowledged_sources'] == list(range(5))
        assert state['received_versions'] == {str(i):4643 for i in range(5)}
        assert len(d._r35_local_receipts) == 5
        assert len(d._r35_receipts) == 25
        for i in range(5):
            assert d._radio_maps.snapshots[i] == stores[i].own_final
            receipt = d._r35_local_receipts[(int(d.identifier),i)]
            assert receipt[2:4] == (4643,stores[i].payload_digest(i))
    rows = emitted(capsys)
    assert len([r for r in rows if r['event']=='rendezvous35_payload_verified_here']) == 25
    assert all(r['step'] >= 4644 for r in rows)


def test_native_map_packets_forward_through_chain_without_direct_remote_pose():
    drones = [fixture(i) for i in range(5)]
    edges = {(i,j) for i in range(5) for j in range(5) if abs(i-j)==1}
    exchange(drones,100,edges)
    assert all(d._exchange_state()['final_complete'] for d in drones)
    assert 4 not in drones[0]._peers and 4 not in drones[0]._r35_last_rx
    assert drones[0]._radio_maps.snapshots[4] == drones[4]._radio_maps.own_final


def test_packet_unwrap_preserves_real_direct_peer_inventory_and_bombs():
    sender,receiver = fixture(0),fixture(1)
    packet = sender.define_message_for_all()
    assert packet[0]=='rendezvous35-v1'
    assert packet[1][0]=='frontier34-v1'
    assert packet[1][1][0]=='structural28-v1'
    assert packet[1][1][1][0]=='fusion27-v1'
    receiver.communicator.received_messages = [(0,packet)]
    receiver._receive()
    np.testing.assert_array_equal(receiver._peers[0][1],sender._position)
    assert receiver._status[0] == (sender._steps,0)
    assert receiver._bomb_records[0] == (sender._steps,tuple(sender._placements))
    assert receiver._r35_last_rx[0] == receiver._steps
    assert receiver._radio_maps.snapshots[0] == sender._radio_maps.own_final


def test_malformed_metadata_cannot_crash_unwrap_or_create_verified_payload():
    d = fixture(1)
    d.communicator.received_messages = [(0,('rendezvous35-v1',('frontier34-v1',None,None),None,None))]
    d._receive()
    assert set(d._r35_local_receipts) == {(1,1)}
    assert not d._exchange_state()['source_maps_complete']


def test_reported_receipt_cannot_replace_actual_payload_or_self_verified_receipt():
    d = fixture(1)
    remote = fixture(0)
    packet = remote.define_message_for_all()
    forged = ((0,4,4643,'a'*64,d.elapsed_timestep),
              (1,4,4643,'a'*64,d.elapsed_timestep))
    d.communicator.received_messages = [(0,(*packet[:3],forged))]
    d._receive()
    assert (0,4) in d._r35_receipts
    assert (1,4) not in d._r35_local_receipts and (1,4) not in d._r35_receipts
    assert '4' in d._exchange_state()['missing']
    assert not d._exchange_state()['final_complete']


def test_five_controllers_commit_only_after_independent_local_route_acceptance():
    drones = [fixture(i,policy=True,sealed=False) for i in range(5)]
    for d in drones:
        d._steps=60;d.elapsed_timestep=61
        d._f34_nav.estimate_route = lambda *a,**kw: dict(exact=True)
    drones[0]._r35_agreement()
    assert drones[0]._r35_book.proposal is not None
    assert drones[0]._r35_book.commit is None
    exchange(drones,2)
    assert all(d._r35_book.proposal is not None for d in drones)
    assert not any(d._r35_book.commit is not None for d in drones)
    for d in drones[1:]:
        d._r35_agreement()
    exchange(drones,5)
    assert all(d._r35_book.commit is not None for d in drones)
    assert all(d._r35_book.status()['local_accepted'] for d in drones)
    assert all(set(d._r35_book.accepts)==set(range(5)) for d in drones)


def test_local_accept_requires_reachable_primary_and_fallback_not_reported_accept():
    d = fixture(1,policy=True,sealed=False)
    leader = RendezvousBook35(0,5)
    leader.propose(100,(0.,0.),(100.,0.))
    d._r35_book.ingest(leader.packet(),d.elapsed_timestep)
    exact = iter((True,False))
    d._f34_nav.estimate_route = lambda *a,**kw: dict(exact=next(exact))
    d._r35_agreement()
    assert not d._r35_book.status()['local_accepted']
    forged = tuple((i,101,leader.proposal[8]) for i in range(5))
    commit = (0,102,leader.proposal[8],forged)
    d._r35_book.ingest((leader.proposal,forged,commit,()),d.elapsed_timestep)
    assert d._r35_book.commit is None and 1 not in d._r35_book.accepts


def test_agreement_target_is_fixed_not_a_stale_or_fresh_peer_position():
    d = fixture(1)
    d._r35_book = committed_book(1)
    d._progress = lambda:.9
    d._f34_reports[0] = (0,d._steps,(-900.,800.),None,(-1,)*5,(),(),'explore',0)
    np.testing.assert_array_equal(d._r35_target(),[0.,0.])
    d._progress = lambda:.94
    np.testing.assert_array_equal(d._r35_target(),[100.,0.])
    assert d._r35_fallback


def test_uncommitted_proposal_does_not_redirect_old_shared_home():
    d = fixture(1)
    d._f27_home = (0,2,(15.,25.))
    d._r35_book.proposal = committed_book(1).proposal
    assert d._r35_book.commit is None
    np.testing.assert_array_equal(d._r35_target(),[15.,25.])


def test_common_fallback_remains_active_after_clock_or_primary_failure():
    d = fixture(1)
    d._r35_book = committed_book(1)
    d._progress = lambda:.89
    d._r35_primary_failed = True
    np.testing.assert_array_equal(d._r35_target(),[100.,0.])
    d._r35_primary_failed = False
    np.testing.assert_array_equal(d._r35_target(),[100.,0.])


def test_final_epoch_preempts_exploration_even_with_ample_estimated_budget():
    d = fixture(policy=True)
    d._r35_agreement = lambda:None
    d._f34_policy()
    assert d._f34_mode=='return' and d._gathering
    assert d._f34_return_reason=='final_epoch_preemption'
    assert d._r35_phase=='to_primary'


@pytest.mark.parametrize('field', ['remaining_steps','arrival_remaining_steps'])
def test_full_exchange_or_arrival_budget_can_independently_preempt(field):
    d = fixture(policy=True,sealed=False)
    d._r35_agreement = lambda:None
    d._f34_budget[field]=0.
    d._f34_policy()
    assert d._f34_mode=='return'
    assert d._f34_return_reason=='meeting_exchange_return_confirmation_budget'


def test_pending_physical_bomb_confirmation_is_not_cancelled_by_deadline():
    d = fixture(policy=True,sealed=False)
    d._r35_agreement=lambda:None
    d._f34_budget['remaining_steps']=0.
    d._attempt_step=d._steps-1
    d._f34_policy()
    assert d._f34_mode=='explore'
    assert d._d32_stats['r35_delivery_waiting_inventory_confirmation']==1


def test_delivery_budget_charges_go_exchange_return_confirm_and_recovery():
    d = fixture(policy=True,sealed=False)
    del d.__dict__['_f34_budget_update']
    d._r35_book=committed_book(0)
    d._progress=lambda:.5
    d._position=np.array([500.,500.])
    # Own recorded terminal differs from the shared fallback/home. The
    # installed return itinerary must be charged before its final known leg.
    d._f34_crumbs=[np.array([0.,200.]),np.array([0.,0.]),np.array([500.,0.])]
    calls=[]
    def estimate(*a,**kw):
        calls.append((np.asarray(a[1]).copy(),np.asarray(a[2]).copy()))
        return dict(exact=True,steps=(50.,70.,90.)[(len(calls)-1)%3],mode='fixture_known_route')
    d._f34_nav.estimate_route=estimate
    d._f34_budget_tick=-1000
    d._f34_budget_update()
    b=d._f34_budget
    executed=1200./d.RETURN_SPEED34+2*6+70.
    assert b['travel_steps']==pytest.approx(executed)
    assert b['travel_steps']>50 and b['return_to_fallback_steps']==90
    assert b['required_steps']==pytest.approx(executed+120+90+120+180)
    assert b['arrival_required_steps']==pytest.approx(executed+180)
    np.testing.assert_array_equal(calls[1][0],[0.,200.])
    np.testing.assert_array_equal(calls[1][1],[0.,0.])
    assert len(calls)==3
    d._f34_budget_update()
    assert len(calls)==3


def test_inflight_budget_counts_only_unconsumed_observed_itinerary():
    d=fixture(policy=True,sealed=False)
    del d.__dict__['_f34_budget_update']
    d._r35_book=committed_book(0)
    d._progress=lambda:.5
    d._f34_mode='return'
    d._position=np.array([100.,50.])
    d._f34_crumbs=[np.array([5000.,5000.]),np.array([4000.,4000.])]
    d._f34_nav.set_observed_route([[5000.,5000.],[100.,100.],[0.,100.]])
    d._f34_nav._observed_cursor35=1
    d._f34_nav.estimate_route=lambda *a,**kw:dict(exact=True,steps=70.,mode='fixture_known_route')
    d._f34_budget_tick=-1000
    d._f34_budget_update()
    assert d._f34_budget['traced_length_px']==pytest.approx(150.)
    assert d._f34_budget['traced_turn_count']==1
    assert d._f34_budget['travel_steps']==pytest.approx(150./d.RETURN_SPEED34+6+70.)


def test_exchange_uses_actual_recent_rx_not_geometry_or_forwarded_position():
    d = fixture(1)
    d._position=np.array([25.,0.]);d._home=np.zeros(2)
    d._progress=lambda:.9
    d._f34_nav.command=lambda *a:dict(forward=.2,lateral=0.,rotation=0.)
    d._f34_reports[0]=(0,d._steps,(0.,0.),None,(-1,)*5,(),(),'return',0)
    assert d._f34_return_motion()['forward']==.2
    assert d._f34_mode=='return'
    d._r35_last_rx[0]=d._steps
    assert d._f34_return_motion()['forward']==0.
    assert d._f34_mode=='exchange'
    d._steps+=9
    assert d._f34_return_motion()['forward']==.2
    assert d._f34_mode=='return'


def test_blocked_task_exits_to_agreed_fallback_pauses_then_retries():
    d = fixture(1)
    d._r35_book=committed_book(1)
    d._progress=lambda:.9
    d._position=np.array([500.,0.])
    d._f34_nav.failed=True
    d._f34_return_points=[np.array([400.,0.]),np.array([200.,0.])]
    assert d._f34_return_motion()['forward']==0.
    assert d._r35_failures==1 and d._r35_fallback
    assert d._r35_retry_until==d._steps+20
    d._steps+=1
    assert d._f34_return_motion()['forward']==0.
    assert d._r35_phase=='fresh_observation_pause'
    d._steps+=20
    d._f34_nav.command=lambda *a:dict(forward=.2,lateral=0.,rotation=0.)
    assert d._f34_return_motion()['forward']==.2
    assert d._r35_phase=='to_fallback'


@pytest.mark.parametrize('trigger',['blocked_primary','common_window'])
def test_mission_reset_preserves_remaining_suffix_without_revisiting_deep_start(trigger):
    d=fixture(1)
    d._r35_book=committed_book(1)
    d._position=np.array([500.,0.])
    d._f34_mode='return'
    original=[np.array(p,float) for p in ((900.,0.),(700.,0.),(400.,0.),(200.,0.))]
    d._f34_return_points=[p.copy() for p in original]
    d._f34_nav.set_observed_route(original)
    d._f34_nav._observed_cursor35=2
    if trigger=='blocked_primary':
        d._progress=lambda:.9
        d._f34_nav.failed=True
        d._f34_return_motion()
        assert d._r35_failures==1 and d._r35_retry_until==d._steps+20
    else:
        d._progress=lambda:.94
        d._r35_target()
    assert d._r35_fallback
    np.testing.assert_array_equal(d._f34_return_points,original[2:])
    np.testing.assert_array_equal(d._f34_nav.remaining_observed_route(),original[2:])
    assert d._f34_nav._observed_cursor35==0
    assert not d._f34_nav.failed


def test_delivery_refresh_keeps_odom_lidar_and_publication_once_without_old_search(monkeypatch):
    d=fixture(policy=True,sealed=False)
    d._f34_mode='return'
    d._steps=3000;d.elapsed_timestep=3001
    d._f34_policy=lambda:None
    d.odometer_values=lambda:np.array([2.,0.,0.])
    d._velocity=np.array([1.,0.])
    observes=[];publishes=[];motion=[]
    d._explore.observe=lambda *a: observes.append(a)
    d._radio_maps.publish=lambda *a: publishes.append(a)
    monkeypatch.setattr(MyDroneRedRepair17,'_navigation',lambda *a:pytest.fail('Cancelled exploration search invoked'))
    def move():
        motion.append(d._velocity.copy())
        return dict(forward=.1,lateral=0.,rotation=0.)
    d._f34_return_motion=move
    d._navigation()
    np.testing.assert_allclose(motion[0],[1.6,0.])
    assert len(observes)==1 and len(publishes)==1
    assert observes[0][2].shape==(181,) and observes[0][4].shape==(181,)
    assert publishes[0][0:2]==(0,3000)


def test_invalid_delivery_odometry_never_brakes_using_stale_velocity():
    d=fixture(policy=True,sealed=False)
    d._velocity=np.array([3.,-4.])
    d.odometer_values=lambda:np.array([np.nan,0.,0.])
    d._r35_refresh_delivery_observation()
    assert not d._r35_velocity_valid
    assert d._r35_brake()==dict(forward=0.,lateral=0.,rotation=0.)


def test_known_complete_map_without_global_matching_ack_cannot_submit():
    stores=fleet()
    for s in stores[1:]:deliver(stores[0],s)
    d=fixture(snapshots=stores[0])
    assert d._exchange_state()['source_maps_complete']
    d._submit()
    assert not d._fixture_submissions
    assert not d._exchange_state()['final_complete']


def test_sealing_deployment_progress_and_submission_barrier_are_inherited():
    for name in ('_update_observation','_deployment','_progress','_exchange_state',
                 'FREEZE_AT','DEADLINE_AT'):
        assert getattr(MyDroneRedRendezvous35,name) is getattr(MyDroneRedFrontier34R1,name)
    assert 'kind' in inspect.signature(MyDroneRedRendezvous35._event).parameters


def test_only_leader_proposes_and_missing_local_accept_prevents_commit():
    peer=RendezvousBook35(1,5)
    assert not peer.propose(100,(0.,0.),(100.,0.))
    leader=RendezvousBook35(0,5)
    assert leader.propose(100,(0.,0.),(100.,0.))
    assert not leader.accept(101,False)
    fake=tuple((i,102,leader.proposal[8]) for i in range(5))
    leader.ingest((leader.proposal,fake,None,()),103)
    assert leader.commit is None and 0 not in leader.accepts


def test_accepted_own_receipt_cannot_be_overwritten_by_relayed_older_claim():
    b=RendezvousBook35(0,5)
    b.propose(100,(0.,0.),(100.,0.))
    b.accept(110,True)
    own=b.accepts[0]
    b.ingest((b.proposal,((0,101,b.proposal[8]),),None,()),120)
    assert b.accepts[0]==own and b._local_accept==own


def test_frozen_plan_rejects_valid_but_conflicting_reproposal():
    b=committed_book(2)
    original=b.proposal;commit=b.commit
    fields=(0,2,150,(500.,500.),(700.,700.),.88,.93,.965)
    altered=(*fields,digest35(fields))
    b.ingest((altered,(),None,()),200)
    assert b.proposal==original and b.commit==commit
    assert b.rejections>=1


def test_forwarding_keeps_original_accept_clock_and_rejects_future_accept():
    origin=RendezvousBook35(0,5)
    origin.propose(100,(0.,0.),(100.,0.));origin.accept(101,True)
    relay=RendezvousBook35(1,5);peer=RendezvousBook35(2,5)
    relay.ingest(origin.packet(),200)
    peer.ingest(relay.packet(),300)
    assert peer.proposal[2]==100 and peer.accepts[0][1]==101
    peer.ingest((peer.proposal,((3,401,peer.proposal[8]),),None,()),400)
    assert 3 not in peer.accepts


def test_plan_commit_does_not_count_as_final_map_or_final_set_ack():
    d=fixture()
    d._r35_book=committed_book(0)
    assert d._r35_book.commit is not None
    state=d._exchange_state()
    assert not state['source_maps_complete'] and not state['final_complete']
    assert state['acknowledged_sources']==[]
    d._submit()
    assert not d._fixture_submissions


def test_all_participants_use_same_fallback_after_common_window():
    targets=[]
    for i in range(5):
        d=fixture(i)
        d._r35_book=committed_book(i)
        d._progress=lambda:.931
        targets.append(d._r35_target())
    for point in targets:
        np.testing.assert_array_equal(point,[100.,0.])
