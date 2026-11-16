"""Deterministic component checks, not substitutes for full maps."""
import math
from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_link14 import (
    ReturnTrail, ReceiptBook, transport, opening_candidates, MyDroneRedLink14,
    MyDroneRedDelivery14, MyDroneRedOpening14, MyDroneRedRelative14)
from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
from swarm_rescue.solutions.my_drone_red_rn7 import LocalPassage
from swarm_rescue.solutions.devtools.test_red_lines11 import scan


def test_odometry_tracks_acceleration_braking_slip_and_wrapped_rotation():
    p=np.array([[70.,20.],[-40.,12.]])
    for odom in [(1.,0.,0.),(4.,.2,.04),(.6,math.pi/2,.12),(.1,-1.,-.2),(0.,0.,2*math.pi-.01)]:
        shift=odom[0]*np.array([math.cos(odom[1]),math.sin(odom[1])]);c,s=math.cos(-odom[2]),math.sin(-odom[2])
        expected=(p-shift)@np.array([[c,s],[-s,c]])
        p=transport(p,odom);np.testing.assert_allclose(p,expected,atol=1e-12)
    assert transport(p,None) is None


def test_receipts_require_actual_leader_version_and_reject_future_duplicate():
    b=ReceiptBook();assert not b.delivered(2,100)
    assert not b.accept((1,120,((2,100),)),130,5)
    assert not b.accept((0,150,((2,100),)),130,5)
    assert b.accept((0,120,((2,100),)),130,5)
    assert b.delivered(2,100) and not b.delivered(2,101)
    assert not b.accept((0,120,((2,110),)),130,5)
    assert not b.accept((0,125,((2,126),)),130,5)
    relay=ReceiptBook();assert relay.accept(b.packet(),140,5)
    assert relay.versions==b.versions and relay.step==120
    assert not relay.accept((0,130,((2,120),(2,121))),140,5)


def test_retreat_chain_requires_observed_links_and_bounds_memory():
    t=ReturnTrail();assert not t.valid;t.reset();t.update((32.,0.,0.))
    t.extend(SimpleNamespace(line_clear=lambda a,b:True))
    assert len(t.points)==2 and t.remaining()==pytest.approx(32)
    t.update((32.,0.,.2));t.extend(SimpleNamespace(line_clear=lambda a,b:False))
    assert not t.valid and not t.points and t.reason=='unobserved_trail_link'
    t.reset();t.update(None);assert not t.valid
    t.reset();t.MAX_POINTS=1;t.update((40.,0.,0.));t.extend(SimpleNamespace(line_clear=lambda a,b:True))
    assert t.reason=='trail_capacity'


@pytest.mark.parametrize('angle',[0.,.41,-math.pi+.02])
def test_observation_goals_require_complete_fresh_free_sweep(angle):
    walls=np.array([[[-220.,-55.],[220.,-55.]],[[-220.,55.],[220.,55.]]])
    d,a=scan(walls,np.zeros(2),angle)
    grid=_M19_SurveyGrid((600,600),lambda *args,**kw:None)
    options=opening_candidates(d,a,grid,np.zeros(2),angle)
    assert options
    fine=LocalPassage(d,a)
    assert all(fine.line_clear(np.zeros(2),v['body']) for v in options)
    assert not grid.free.any()  # detection never turns unknown into global free


def test_enclosed_space_does_not_manufacture_an_opening():
    walls=np.array([[[-28,-28],[28,-28]],[[28,-28],[28,28]],[[28,28],[-28,28]],[[-28,28],[-28,-28]]],float)
    d,a=scan(walls,np.zeros(2),0.)
    grid=_M19_SurveyGrid((600,600),lambda *args,**kw:None)
    assert not opening_candidates(d,a,grid,np.zeros(2),0.)


def actor(odom=(3.,0.,0.),peers=()):
    x=SimpleNamespace(odometer_values=lambda:np.asarray(odom),semantic_values=lambda:peers,_link_stats=Counter())
    x._brake14=lambda:MyDroneRedLink14._brake14(x)
    return x


def test_new_wall_and_dynamic_peer_are_not_overridden():
    x=actor();blocked=SimpleNamespace(line_clear=lambda a,b:False)
    assert MyDroneRedLink14._move14(x,np.array([50.,0.]),blocked) is None
    peer=SimpleNamespace(entity_type=SimpleNamespace(name='DRONE'),distance=40.,angle=0.)
    x=actor(peers=[peer]);clear=SimpleNamespace(line_clear=lambda a,b:True)
    cmd=MyDroneRedLink14._move14(x,np.array([70.,0.]),clear)
    assert cmd['forward']<0 and cmd['rotation']==0


def test_candidates_are_single_factor_and_do_not_override_mapping_or_pose():
    for cls,expected in [(MyDroneRedLink14,(False,False,False)),(MyDroneRedDelivery14,(True,False,False)),
                         (MyDroneRedOpening14,(False,True,False)),(MyDroneRedRelative14,(False,False,True))]:
        assert (cls.DELIVERY,cls.OPENING_TASKS,cls.RELATIVE_CORRIDOR)==expected
        assert '_submit' not in cls.__dict__


def test_short_opening_task_has_bounded_retry_and_keeps_history():
    x=SimpleNamespace(_opening=dict(id=4,entered=False,world=np.array([50.,0.])),_steps=100,
        _opening_history=[],_link_note=lambda *a,**k:None)
    MyDroneRedLink14._end_opening(x,'no_progress_or_budget')
    assert x._opening is None and x._opening_after==175 and x._opening_history[0][1]==300


def test_actual_message_acceptance_is_required_and_base_protocol_is_preserved():
    class Store:
        def __init__(self):self.snapshots={}
        def accept(self,snapshot):self.snapshots[snapshot[0]]=snapshot
    store=Store();book=ReceiptBook();grid=SimpleNamespace(shape=(1,1),shared_visited=np.zeros((1,1),bool))
    actor=SimpleNamespace(DELIVERY=True,_receipt=book,_link_stats=Counter(),_steps=100,
        elapsed_timestep=100,_n=5,identifier=2,_last_contact=-999,_status={},_peers={},_radio_maps=store,
        _explore=grid,_valid_peer=lambda p: isinstance(p,tuple) and len(p)==4,
        _bomb_claims={},_bomb_records={},communicator=SimpleNamespace(received_messages=[]))
    snap=(2,90,b'payload');base=('red-v4',('red-v3',('red-v1',((0,99,0),),snap),(0,99,(0.,0.),None),b'\x80'),(0,99,None),())
    MyDroneRedLink14._receive(actor)
    assert not book.delivered(2,90) and not store.snapshots
    actor.communicator.received_messages=[(None,('red-link14',base,(0,99,((2,90),))))]
    MyDroneRedLink14._receive(actor)
    assert book.delivered(2,90) and store.snapshots[2]==snap and actor._last_contact==100
    assert actor._status[0]==(99,0) and grid.shared_visited[0,0]


def test_offline_region_partition_respects_thin_wall_and_lists_disconnected_free():
    from swarm_rescue.solutions.devtools.red_link14_review import partition
    truth=np.zeros((256,256),np.uint8);truth[:,127:130]=1
    labels=partition(truth)
    assert labels[64,64]>0 and labels[64,192]>0
    assert labels[64,64]!=labels[64,192] and labels[64,128]==0
