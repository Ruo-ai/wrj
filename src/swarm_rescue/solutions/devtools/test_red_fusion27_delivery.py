"""Independent pure-policy tests: no map, GUI, drone physics, or simulation."""
from collections import Counter
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_fusion27 import (
    FinalizableRepairMapper17, FinalSnapshots27, MyDroneRedFusion27,
)
from swarm_rescue.solutions.my_drone_red_repair17 import RepairMapper17, MyDroneRedRepair17


SIZE = (90, 60)
EPOCH = 4644
SEQUENCE = EPOCH - 1


def store(source=0, epoch=EPOCH, inventory=0, frozen=True, size=SIZE):
    mapper = FinalizableRepairMapper17(size)
    mapper.hits[3+source:6+source, 2:15] = 4.
    now = [epoch]
    result = FinalSnapshots27(mapper.shape, 5, mapper, lambda: now[0])
    result._fixture_clock = now
    nav = np.zeros(mapper.shape, bool)
    nav[1+source, 2:10] = True
    if frozen:
        result.freeze(source, epoch-1, epoch, nav, inventory)
    return result, nav


def fleet():
    return [store(i)[0] for i in range(5)]


def deliver(receiver, sender, manifest_first=False):
    source = sender.own_final[0]
    if manifest_first:
        receiver.accept_manifest(sender.manifests[source])
        receiver.accept(sender.own_final)
    else:
        receiver.accept(sender.own_final)
        receiver.accept_manifest(sender.manifests[source])


def full_leader():
    stores = fleet()
    for sender in stores[1:]:
        deliver(stores[0], sender)
    return stores[0]


def controller_fixture(source=0, progress=.9, snapshots=None):
    # Override public descriptors only on this pure Python fixture subclass.
    cls = type('DeliveryFixture27',(MyDroneRedFusion27,),{'identifier':source,'drone_health':25})
    drone = object.__new__(cls)
    drone._radio_maps = snapshots or store(source)[0]
    drone._steps = SEQUENCE + 20
    drone.elapsed_timestep = drone._steps + 1
    drone.elapsed_walltime = 150.
    drone._progress = lambda: progress
    drone._exploration_submitted = False
    drone._f27_stats = Counter()
    drone._f27_delivery = {}
    drone._f27_submission = None
    drone._f27_home = None
    drone._fixture_submissions = []
    drone._fixture_events = []
    drone.submit_exploration_map = lambda pixels: drone._fixture_submissions.append(pixels.copy())
    drone._event = lambda event,**fields: drone._fixture_events.append((event,fields))
    class Renderer:
        last_stats = {'fixture_renderer':True}
        def fuse(self, masks):
            self.received = {i:m.copy() for i,m in masks.items()}
            return np.logical_or.reduce(list(masks.values()))
    drone._fusion27 = Renderer()
    return drone


def messaging_fixture(source=0, snapshots=None):
    drone=controller_fixture(source=source,snapshots=snapshots)
    drone._n=5
    drone._position=np.array([20.+source,30.])
    drone._route=[]
    drone._placements=[(3.,4.)]
    drone._bomb_target=None
    drone._bomb_records={}
    drone._bomb_claims={}
    drone._status={source:(drone._steps,0)}
    drone._peers={}
    drone._explore=SimpleNamespace(shape=(3,4),shared_visited=np.zeros((3,4),bool),point=lambda p:np.asarray(p))
    drone._visited_snapshot=np.packbits(np.eye(3,4,dtype=bool).ravel()).tobytes()
    drone.communicator=SimpleNamespace(received_messages=[])
    return drone


def test_mapper_seal_is_immutable_copy_and_idempotent(monkeypatch):
    mapper = FinalizableRepairMapper17(SIZE)
    mapper.hits[2:8,2:12] = 4.
    before = mapper.mask().copy()
    mapper.step = 71
    mapper.seal()
    assert mapper.final_mapper_step == 71
    assert not mapper.final_mask.flags.writeable
    mapper.hits[:] = 0
    returned = mapper.mask()
    returned[:] = False
    mapper.step = 72
    mapper.seal()
    assert mapper.final_mapper_step == 71
    np.testing.assert_array_equal(mapper.mask(), before)
    monkeypatch.setattr(RepairMapper17, 'update', lambda *args,**kwargs: pytest.fail('Sealed mapper updated evidence'))
    mapper.update(None,None,None,None,None)
    assert mapper.stats['after_cutoff_observations_not_mapped'] == 1
    np.testing.assert_array_equal(mapper.mask(), before)


def test_unsealed_mapper_still_calls_original_sensor_update(monkeypatch):
    calls=[]
    monkeypatch.setattr(RepairMapper17,'update',lambda self,*args,**kw:calls.append((args,kw)))
    mapper=FinalizableRepairMapper17(SIZE)
    mapper.update('gps','compass','odometry','lidar','angles')
    assert calls == [(('gps','compass','odometry','lidar','angles'),{})]


def test_freeze_owns_exact_nav_and_submission_bytes_and_stable_sequence():
    result,nav=store()
    own=result.own_final
    expected=np.packbits(nav.ravel()).tobytes()+np.packbits(result.mapper.mask().ravel()).tobytes()
    assert own==(0,SEQUENCE,expected)
    assert result.manifests[0]==(0,EPOCH,SEQUENCE,hashlib.sha256(expected).hexdigest(),0)
    nav[:]=True
    result.mapper.hits[:]=30
    result._fixture_clock[0]+=50
    result.publish(0,SEQUENCE+50,nav)
    result.freeze(0,SEQUENCE+50,EPOCH+50,nav,1)
    assert result.own_final==own==result.snapshots[0]
    assert result.manifests[0][1:3]==(EPOCH,SEQUENCE)
    assert result.protocol_stats['own_finalized']==1
    assert result.protocol_stats['immutable_republish']==1


def test_all_five_same_epoch_exact_payloads_are_required():
    stores=fleet(); leader=stores[0]
    for sender in stores[1:4]:deliver(leader,sender)
    assert leader.delivery()['missing']=={'4':'missing_final_manifest'}
    deliver(leader,stores[4])
    state=leader.delivery()
    assert state['final_complete'] and state['missing']=={}
    assert set(state['received_versions'])==set(map(str,range(5)))
    assert set(state['received_versions'].values())=={SEQUENCE}


@pytest.mark.parametrize('manifest_first',[False,True])
def test_payload_and_manifest_can_arrive_in_either_order(manifest_first):
    leader,_=store(); sender,_=store(1)
    if manifest_first:
        assert leader.accept_manifest(sender.manifests[1])
        assert leader.delivery()['missing']['1']=='missing_payload'
        assert leader.accept(sender.own_final)
    else:
        assert leader.accept(sender.own_final)
        assert leader.delivery()['missing']['1']=='missing_final_manifest'
        assert leader.accept_manifest(sender.manifests[1])
    assert '1' not in leader.delivery()['missing']
    assert not leader.delivery()['final_complete']


def test_manifest_alone_never_fakes_delivered_maps():
    stores=fleet(); leader=stores[0]
    for sender in stores[1:]:leader.accept_manifest(sender.manifests[sender.own_final[0]])
    assert not leader.delivery()['final_complete']
    assert set(leader.delivery()['missing'].values())=={'missing_payload'}


def test_old_payload_and_final_manifest_are_not_a_final_delivery():
    leader,_=store(); sender,_=store(1)
    _,seq,payload=sender.own_final
    assert leader.accept((1,seq-30,payload))
    assert leader.accept_manifest(sender.manifests[1])
    assert leader.delivery()['missing']['1']=='final_version_not_received'
    assert leader.accept(sender.own_final)
    assert '1' not in leader.delivery()['missing']


def test_wrong_digest_never_satisfies_final_barrier():
    leader,_=store(); sender,_=store(1)
    assert leader.accept(sender.own_final)
    m=list(sender.manifests[1]);m[3]='0'*64
    assert leader.accept_manifest(tuple(m))
    assert leader.delivery()['missing']['1']=='final_digest_mismatch'


def test_nonzero_final_inventory_blocks_complete_even_with_all_maps():
    stores=fleet(); stores[4]=store(4,inventory=1)[0]
    for sender in stores[1:]:deliver(stores[0],sender)
    assert stores[0].delivery()['missing']=={'4':'inventory_remaining_at_cutoff'}
    assert not stores[0].delivery()['final_complete']


def test_frozen_receiver_rejects_other_epoch():
    leader,_=store(); other,_=store(1,epoch=EPOCH-10)
    assert leader.accept(other.own_final)
    assert not leader.accept_manifest(other.manifests[1])
    assert leader.delivery()['missing']['1']=='missing_final_manifest'


def test_prefreeze_foreign_epoch_is_pruned_at_own_epoch():
    receiver,nav=store(frozen=False)
    old,_=store(1,epoch=EPOCH-10)
    assert receiver.accept_manifest(old.manifests[1])
    assert not receiver.delivery()['final_complete']
    receiver.freeze(0,SEQUENCE,EPOCH,nav,0)
    assert 1 not in receiver.manifests
    good,_=store(1)
    deliver(receiver,good)
    assert '1' not in receiver.delivery()['missing']


@pytest.mark.parametrize('field,value',[(0,-1),(0,5),(0,True),(1,EPOCH+1),(1,-1),(1,True),(2,bytearray(b'bad'))])
def test_invalid_or_future_snapshot_is_rejected(field,value):
    receiver,_=store(); sender,_=store(1)
    snapshot=list(sender.own_final);snapshot[field]=value
    assert not receiver.accept(tuple(snapshot))
    assert 1 not in receiver.snapshots


def test_same_version_different_payload_and_postfinal_newer_version_are_rejected():
    receiver,_=store(); sender,_=store(1)
    deliver(receiver,sender)
    source,seq,payload=sender.own_final
    changed=bytes([payload[0]^1])+payload[1:]
    assert not receiver.accept((source,seq,changed))
    assert not receiver.accept((source,seq+1,payload))
    assert receiver.snapshots[1]==sender.own_final
    assert receiver.protocol_stats['conflicting_payload']==1
    assert receiver.protocol_stats['post_final_snapshot_rejected']==1


def test_duplicate_relays_add_no_maps_votes_or_versions():
    leader=full_leader()
    before={i:m.copy() for i,m in leader.mapping_masks.items()}
    for _ in range(10):
        for source in range(5):
            assert not leader.accept(leader.snapshots[source])
            assert leader.accept_manifest(leader.manifests[source])
    assert leader.delivery()['final_complete']
    assert len(leader.mapping_masks)==5
    for source in before:np.testing.assert_array_equal(leader.mapping_masks[source],before[source])


@pytest.mark.parametrize('field,value',[(0,-1),(0,5),(0,True),(1,EPOCH+1),(1,0),(1,True),
    (2,EPOCH+1),(2,-1),(2,True),(3,'g'*64),(3,'0'*63),(3,b'0'*64),(4,-1),(4,True)])
def test_malformed_or_future_manifest_is_rejected(field,value):
    receiver,_=store(); sender,_=store(1)
    manifest=list(sender.manifests[1]);manifest[field]=value
    assert not receiver.accept_manifest(tuple(manifest))
    assert 1 not in receiver.manifests


def test_conflicting_manifest_cannot_relabel_existing_final():
    receiver,_=store(); sender,_=store(1)
    good=sender.manifests[1]
    assert receiver.accept_manifest(good)
    conflict=(*good[:3],'1'*64,good[4])
    assert not receiver.accept_manifest(conflict)
    assert receiver.manifests[1]==good


def test_radio_relay_delivers_distant_final_without_shared_memory():
    leader,relay,distant,*rest=fleet()
    deliver(relay,distant)
    assert 2 not in leader.snapshots
    leader.accept_manifest(relay.manifests[2])
    leader.accept(relay.snapshots[2])
    assert leader.snapshots[2]==distant.own_final
    assert '2' not in leader.delivery()['missing']
    assert not leader.delivery()['final_complete']
    assert leader.manifests is not relay.manifests and leader.mapping_masks is not relay.mapping_masks


@pytest.mark.parametrize('progress',[.72,.84,.86,.9749])
def test_old_parent_submit_calls_are_deferred_until_protocol_or_deadline(progress):
    drone=controller_fixture(progress=progress)
    drone._submit()
    assert not drone._exploration_submitted and not drone._fixture_submissions
    assert drone._f27_stats['parent_submit_deferred']==1
    assert drone._f27_submission is None


def test_leader_submits_complete_real_final_sources_without_republishing():
    leader=full_leader(); before=dict(leader.snapshots)
    drone=controller_fixture(progress=.9,snapshots=leader)
    drone._submit();drone._submit()
    assert len(drone._fixture_submissions)==1 and drone._exploration_submitted
    assert leader.snapshots==before
    submission=drone._f27_submission
    assert submission['final_complete'] and submission['missing']=={}
    assert submission['reason']=='all_final_maps_received'
    assert set(drone._fusion27.received)==set(range(5))
    assert submission['submitted_pixels_sha256']==hashlib.sha256(drone._fixture_submissions[0].astype(bool).tobytes()).hexdigest()


def test_deadline_fallback_is_explicitly_incomplete_and_keeps_exact_missing_list():
    drone=controller_fixture(progress=.975)
    drone._submit()
    assert drone._exploration_submitted and len(drone._fixture_submissions)==1
    state=drone._f27_submission
    assert not state['final_complete']
    assert state['reason']=='deadline_incomplete_fallback'
    assert set(state['missing'])=={'1','2','3','4'}
    assert set(drone._fusion27.received)=={0}


def test_complete_nonleader_waits_for_its_backup_slot():
    peer=full_leader()
    drone=controller_fixture(source=1,progress=.98,snapshots=peer)
    drone._submit()
    assert not drone._exploration_submitted
    drone._progress=lambda:.984
    drone._submit()
    assert drone._exploration_submitted and drone._f27_submission['final_complete']


def test_empty_unfrozen_store_cannot_claim_final_complete():
    receiver,_=store(frozen=False)
    state=receiver.delivery()
    assert not state['final_complete'] and state['epoch_elapsed'] is None
    assert set(state['missing'])==set(map(str,range(5)))


def test_gathering_uses_radio_received_home_and_preserves_original_navigation(monkeypatch):
    drone=controller_fixture(progress=.65)
    drone._gathering=False;drone._route=[(1,2)];drone._home=np.array([50.,60.])
    drone._f27_home=(0,20,(10.,15.))
    seen=[]
    monkeypatch.setattr(MyDroneRedRepair17,'_navigation',lambda self:seen.append(self._home.copy()) or {'forward':.2})
    assert drone._navigation()=={'forward':.2}
    assert drone._gathering and drone._route==[]
    np.testing.assert_array_equal(seen[0],[10.,15.])
    assert drone._f27_stats['gather_started_control']==drone._steps


def test_hold_requires_local_clearance_and_brakes_inside_compact_radio_cluster():
    drone=controller_fixture(progress=.7)
    drone._position=np.array([69.,0.]);drone._home=np.zeros(2);drone._peers={}
    drone._gathering=True;drone.carried_bombs_count=lambda:0;drone._holding=False
    drone._route=[(1,2)];drone._bomb_target=np.ones(2);drone._velocity=np.array([2.,-2.])
    drone._map=SimpleNamespace(mask=lambda:np.zeros(drone._radio_maps.shape,bool))
    command={}; distances=np.full(24,100.);dynamic=np.zeros(24,bool)
    assert drone._try_hold(distances,dynamic,command)
    assert command=={'forward':-1.,'lateral':1.,'rotation':0.}
    assert drone._route==[] and drone._bomb_target is None
    assert not drone._try_hold(np.full(24,20.),dynamic,{})
    assert not drone._holding


def test_controller_changes_only_declared_protocol_navigation_hooks():
    assert MyDroneRedFusion27.__bases__==(MyDroneRedRepair17,)
    for name in ('_deployment','_progress','_navigate_motion','_start_recovery','_release_stale_route'):
        assert getattr(MyDroneRedFusion27,name) is getattr(MyDroneRedRepair17,name)


def test_real_wrapped_message_keeps_parent_inventory_peer_visited_and_bomb_records():
    sender=messaging_fixture(0);receiver=messaging_fixture(1)
    sender._f27_home=(0,2,(20.,30.))
    packet=sender.define_message_for_all()
    assert packet[0]=='fusion27-v1' and packet[1][0]=='red-v4'
    original_received=[(object(),packet)]
    receiver.communicator.received_messages=original_received
    receiver._receive()
    assert receiver.communicator.received_messages is original_received
    assert receiver._radio_maps.snapshots[0]==sender._radio_maps.own_final
    assert receiver._radio_maps.manifests[0]==sender._radio_maps.manifests[0]
    assert '0' not in receiver._radio_maps.delivery()['missing']
    assert receiver._status[0]==(sender._steps,0)
    np.testing.assert_array_equal(receiver._peers[0][1],sender._position)
    assert receiver._bomb_records[0]==(sender._steps,tuple(sender._placements))
    assert receiver._bomb_claims[0]==(sender._steps,None)
    np.testing.assert_array_equal(receiver._explore.shared_visited,np.eye(3,4,dtype=bool))
    assert receiver._f27_home==sender._f27_home


def test_real_message_relay_preserves_origin_final_and_leader_home():
    leader=messaging_fixture(0);relay=messaging_fixture(1);distant=messaging_fixture(2)
    leader._f27_home=(0,2,(20.,30.))
    relay.communicator.received_messages=[(object(),leader.define_message_for_all())]
    relay._receive()
    relay._steps=4657  # odd extra-payload slot chooses source 0 from [0, 1]
    distant.elapsed_timestep=4658
    distant.communicator.received_messages=[(object(),relay.define_message_for_all())]
    distant._receive()
    assert distant._radio_maps.snapshots[0]==leader._radio_maps.own_final
    assert distant._radio_maps.manifests[0]==leader._radio_maps.manifests[0]
    assert distant._f27_home==leader._f27_home
    assert '0' not in distant._radio_maps.delivery()['missing']


def test_extra_payload_rotation_does_not_starve_relayed_final_sources():
    sender=messaging_fixture(0,snapshots=full_leader())
    sources=[]
    for step in range(4700,4720):
        sender._steps=step
        sources.append(sender.define_message_for_all()[2][0])
    assert set(sources)==set(range(5))
    assert all(sources.count(source)>=2 for source in range(1,5))


@pytest.mark.parametrize('home',[(1,2,(3.,4.)),(0,5000,(3.,4.)),(0,2,(float('nan'),4.)),(0,2,(3.,))])
def test_invalid_leader_home_is_not_used_for_navigation(home):
    receiver=messaging_fixture(1)
    receiver.communicator.received_messages=[(object(),('fusion27-v1',None,None,(),home))]
    receiver._receive()
    assert receiver._f27_home is None


def test_received_home_is_instance_local_and_does_not_change_after_first_valid_leader_home():
    first=messaging_fixture(1);second=messaging_fixture(2)
    packet=('fusion27-v1',None,None,(),(0,2,(3.,4.)))
    first.communicator.received_messages=[(object(),packet)]
    first._receive()
    assert first._f27_home==(0,2,(3.,4.)) and second._f27_home is None
    first.communicator.received_messages=[(object(),('fusion27-v1',None,None,(),(0,3,(8.,9.))))]
    first._receive()
    assert first._f27_home==(0,2,(3.,4.))
