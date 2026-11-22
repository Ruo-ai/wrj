"""Real frozen-store payload forwarding and rendered-image agreement; no flight."""
import hashlib

import numpy as np
import pytest

from swarm_rescue.solutions.red_mesh36 import MeshBook36, hash36
from swarm_rescue.solutions.red_boundary33 import FinalBoundary33
from swarm_rescue.solutions.devtools.test_red_fusion27_delivery import (
    EPOCH, SEQUENCE, SIZE, store, fleet, deliver,
)


def packet(*, edges=(), reports=(), snapshot=None, manifests=(), acks=()):
    return ('mesh36-v1',edges,reports,snapshot,manifests,acks)


def render(s):
    return FinalBoundary33(SIZE,wall_width=11).fuse(s.mapping_masks)


def run_round(books,stores,clock,edges,confirm=True):
    for s in stores:s._fixture_clock[0]=clock
    outgoing=[b.packet(clock,s) for b,s in zip(books,stores)]
    for sender,receiver in edges:
        books[receiver].ingest(sender,outgoing[sender],clock,stores[receiver])
    if confirm:
        for b,s in zip(books,stores):
            if s.delivery()['final_complete']:
                b.confirm_render(s,render(s) if b.final_pixels is None else b.final_pixels,clock)


def complete_store(source=0):
    stores=fleet()
    for other in stores:
        if other.own_final[0]!=source:deliver(stores[source],other)
    return stores[source]


def test_five_real_sources_and_render_acks_cross_four_hops_without_direct_leader_contact():
    stores=fleet();books=[MeshBook36(i) for i in range(5)]
    edges=[(a,b) for a in range(5) for b in range(5) if abs(a-b)==1]
    for clock in range(EPOCH,EPOCH+70):run_round(books,stores,clock,edges)
    for i,(b,s) in enumerate(zip(books,stores)):
        state=b.status(EPOCH+69,s)
        assert state['source_maps_complete'] and state['render_complete']
        assert state['component']==state['strong_component']==list(range(5))
        assert state['render_ack_sources']==list(range(5))
        assert s.snapshots=={j:stores[j].own_final for j in range(5)}
        np.testing.assert_array_equal(b.final_pixels,render(s))
        assert b.local_ack[3]==hashlib.sha256(b.final_pixels.tobytes()).hexdigest()
        assert b.stats['render_hash_computed']==1
        assert b.stats['render_confirmation_cache_hit']>0
        assert len(b.packet(EPOCH+69,s)[1])<=20
    assert books[0].direct_neighbors(EPOCH+69)==[1]
    assert 4 not in books[0].direct_neighbors(EPOCH+69)


def test_chain_cut_expires_original_edges_even_while_their_rows_keep_being_relayed():
    stores=fleet();books=[MeshBook36(i,edge_ttl=3,report_ttl=5) for i in range(5)]
    chain=[(a,b) for a in range(5) for b in range(5) if abs(a-b)==1]
    for clock in range(EPOCH,EPOCH+10):run_round(books,stores,clock,chain,False)
    cut=[(a,b) for a,b in chain if {a,b}!={2,3}]
    for clock in range(EPOCH+10,EPOCH+25):run_round(books,stores,clock,cut,False)
    for i,b in enumerate(books):
        expected=[0,1,2] if i<=2 else [3,4]
        state=b.status(EPOCH+24,stores[i])
        assert state['component']==state['strong_component']==expected
        assert not state['all_bidirectionally_connected']
    assert (2,3) not in books[0].edges and (3,2) not in books[4].edges


def test_forwarded_edge_and_version_report_retain_original_clock_and_do_not_create_direct_peer():
    s,_=store(0);b=MeshBook36(0,edge_ttl=10,report_ttl=10)
    old=EPOCH-4
    message=packet(edges=((2,3,old),),reports=((3,old,(-1,-1,-1,old,-1)),))
    b.ingest(1,message,EPOCH,s)
    assert b.edges[(2,3)]==old and b.reports[3][1]==old
    assert b.direct_neighbors(EPOCH)==[1]
    forwarded=b.packet(EPOCH,s,include_payload=False)
    assert (2,3,old) in forwarded[1]
    b.ingest(1,message,EPOCH+7,s)
    assert (2,3) not in b.edges and 3 not in b.reports


def test_forwarded_own_contact_claim_and_future_clocks_cannot_forge_connectivity():
    s,_=store(0);b=MeshBook36(0)
    b.ingest(1,packet(edges=((0,4,EPOCH),(4,3,EPOCH+1))),EPOCH,s)
    assert b.direct_neighbors(EPOCH)==[1]
    assert (0,4) not in b.edges and (4,3) not in b.edges
    assert b.status(EPOCH,s)['component']==[0]


def test_one_way_chain_is_not_reported_as_mutual_or_strong_connectivity():
    s,_=store(0);b=MeshBook36(0)
    b.ingest(1,packet(edges=((1,2,EPOCH),(2,3,EPOCH),(3,4,EPOCH))),EPOCH,s)
    state=b.status(EPOCH,s)
    assert state['can_receive_from']==[0,1,2,3,4]
    assert state['can_send_to']==state['strong_component']==state['component']==[0]
    assert not state['all_strongly_connected']


def test_source_version_replays_are_deduplicated_without_rebuilding_store(monkeypatch):
    stores=fleet();b=MeshBook36(0)
    calls=[];original=stores[0].accept
    def accept(value):calls.append(value);return original(value)
    monkeypatch.setattr(stores[0],'accept',accept)
    p=packet(snapshot=stores[1].own_final,manifests=(stores[1].manifests[1],))
    assert b.ingest(1,p,EPOCH,stores[0])
    for _ in range(5):assert not b.ingest(2,p,EPOCH,stores[0])
    assert len(calls)==1 and b.stats['duplicate_snapshot']==5
    assert stores[0].snapshots[1]==stores[1].own_final


def test_old_and_conflicting_versions_cannot_replace_received_final_bytes():
    stores=fleet();b=MeshBook36(0)
    origin,version,blob=stores[1].own_final
    b.ingest(1,packet(snapshot=(origin,version,blob)),EPOCH,stores[0])
    assert not b.ingest(1,packet(snapshot=(origin,version-1,blob)),EPOCH,stores[0])
    altered=bytes([blob[0]^1])+blob[1:]
    assert not b.ingest(1,packet(snapshot=(origin,version,altered)),EPOCH,stores[0])
    assert stores[0].snapshots[1]==stores[1].own_final
    assert b.stats['old_snapshot']==b.stats['conflicting_snapshot']==1


def test_missing_neighbor_version_is_prioritized_before_redundant_source():
    s=complete_store(0);b=MeshBook36(0)
    versions=(SEQUENCE,SEQUENCE,SEQUENCE,SEQUENCE,-1)
    b.ingest(1,packet(reports=((1,EPOCH,versions),)),EPOCH,s)
    assert b.select_payload(EPOCH,s)[0]==4
    state=b.status(EPOCH,s)
    assert state['neighbors_needing_sources']=={'1':[4]}


def test_reports_cannot_supply_missing_payload_and_manifest_alone_is_not_complete():
    stores=fleet();b=MeshBook36(0)
    manifests=tuple(s.manifests[i] for i,s in enumerate(stores))
    p=packet(reports=tuple((i,EPOCH,(SEQUENCE,)*5) for i in range(5)),manifests=manifests)
    b.ingest(1,p,EPOCH,stores[0])
    assert set(stores[0].snapshots)=={0}
    assert b.confirm_render(stores[0],np.ones((60,90)),EPOCH) is None
    assert not b.status(EPOCH,stores[0])['render_complete']
    assert b.local_ack is None and b.final_pixels is None


def test_matching_render_acks_cannot_fake_own_ack_or_actual_local_payloads():
    complete=complete_store(0);valid=MeshBook36(0)
    ack=valid.confirm_render(complete,render(complete),EPOCH)
    forged=tuple((i,*ack[1:]) for i in range(5))
    s,_=store(0);b=MeshBook36(0)
    b.ingest(1,packet(acks=forged),EPOCH,s)
    assert b.local_ack is None and 0 not in b.acks
    state=b.status(EPOCH,s)
    assert state['render_ack_sources']==[] and not state['render_complete']
    assert b.stats['forged_own_render_ack_ignored']==1


def test_local_complete_inputs_do_not_equal_complete_render_ack_without_local_render():
    s=complete_store(0);b=MeshBook36(0)
    state=b.status(EPOCH,s)
    assert state['source_maps_complete'] and not state['local_render_confirmed']
    assert not state['render_complete'] and state['render_ack_sources']==[]


def test_same_input_manifest_with_different_actual_render_does_not_agree():
    a,b=MeshBook36(0),MeshBook36(1)
    sa,sb=complete_store(0),complete_store(1)
    first=render(sa);second=first.copy();second[10,10]=~second[10,10]
    aa=a.confirm_render(sa,first,EPOCH);ab=b.confirm_render(sb,second,EPOCH)
    assert aa[2]==ab[2] and aa[3]!=ab[3]
    a.ingest(1,b.packet(EPOCH,sb),EPOCH,sa)
    assert a.status(EPOCH,sa)['render_ack_sources']==[0]
    assert not a.status(EPOCH,sa)['render_complete']


def test_confirmed_pixels_are_immutable_copy_and_hash_is_cached_for_same_final_set():
    s=complete_store(0);b=MeshBook36(0);pixels=render(s)
    ack=b.confirm_render(s,pixels,EPOCH)
    before=b.final_pixels.copy();pixels[:]=~pixels
    assert b.confirm_render(s,pixels,EPOCH+1)==ack
    np.testing.assert_array_equal(b.final_pixels,before)
    assert not b.final_pixels.flags.writeable
    assert b.stats['render_hash_computed']==1
    assert b.stats['render_confirmation_cache_hit']==1


def test_render_confirmation_rechecks_actual_bytes_even_if_cached_digest_metadata_lies():
    s=complete_store(0);s.delivery() # Prime digest caches with genuine source data.
    source,version,blob=s.snapshots[3]
    s.snapshots[3]=(source,version,bytes([blob[0]^1])+blob[1:])
    assert s.delivery()['final_complete'] # Metadata cache alone can still look complete.
    with pytest.raises(ValueError,match='bytes'):
        MeshBook36(0).confirm_render(s,render(s),EPOCH)


@pytest.mark.parametrize('bad',[
    (1,EPOCH+1,'a'*64,'b'*64,EPOCH+1),
    (1,EPOCH-1,'a'*64,'b'*64,EPOCH),
    (1,EPOCH,'x'*64,'b'*64,EPOCH),
    (1,EPOCH,'a'*64,'b'*64,EPOCH-1),
    (5,EPOCH,'a'*64,'b'*64,EPOCH),
    (True,EPOCH,'a'*64,'b'*64,EPOCH),
])
def test_invalid_future_other_epoch_or_unbounded_identity_ack_rejected(bad):
    s,_=store(0);b=MeshBook36(0)
    b.ingest(1,packet(acks=(bad,)),EPOCH,s)
    assert b.acks=={} and b.stats['invalid_render_ack']==1


@pytest.mark.parametrize('bad',[
    packet(edges=((1,2,EPOCH),)*21),
    packet(reports=((1,EPOCH,(-1,)*5),)*6),
    packet(manifests=((1,EPOCH,SEQUENCE,'a'*64,0),)*6),
    packet(acks=((1,EPOCH,'a'*64,'b'*64,EPOCH),)*6),
    ('mesh36-v1',[],(),None,(),()),
])
def test_wire_container_bounds_reject_oversized_messages(bad):
    s,_=store(0);b=MeshBook36(0)
    assert not b.ingest(1,bad,EPOCH,s)
    assert not b.edges and not b.acks


def test_own_snapshot_echo_cannot_replace_locally_published_source():
    s,_=store(0);b=MeshBook36(0);before=s.own_final
    b.ingest(1,packet(snapshot=(0,SEQUENCE+1,b'\x00'*len(before[2]))),EPOCH,s)
    assert s.snapshots[0]==before and b.stats['own_snapshot_echo_ignored']==1


def test_packet_payload_is_optional_and_metadata_stays_bounded():
    s=complete_store(0);b=MeshBook36(0)
    p=b.packet(EPOCH,s,include_payload=False)
    assert len(p)==6 and p[3] is None and len(p[4])==5
    assert p[2][0]==(0,EPOCH,(SEQUENCE,)*5)
