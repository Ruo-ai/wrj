"""Independent real-message, election and override tests without simulation."""
from collections import Counter, deque
from types import SimpleNamespace
import numpy as np

from swarm_rescue.solutions.my_drone_red_dualmap32 import MyDroneRedDualMap32Clean
from swarm_rescue.solutions.my_drone_red_relay33 import MyDroneRedRelay33, choose_courier33, valid_report33
from swarm_rescue.solutions.devtools.test_red_dualmap32_controller import fixture as prior_fixture
from swarm_rescue.solutions.devtools.test_red_fusion27_delivery import fleet, deliver


def fixture(source=0, snapshots=None, progress=.9):
    prior = prior_fixture(source, snapshots=snapshots, messages=True)
    cls = type('RelayFixture33', (MyDroneRedRelay33,), {'identifier':source, 'drone_health':25})
    d = object.__new__(cls)
    d.__dict__.update(prior.__dict__)
    d._progress = lambda: progress
    d._home = np.zeros(2)
    d._heading = 0.
    d._velocity = np.zeros(2)
    d._gathering = False
    d._holding = False
    d._attempt_step = None
    d.carried_bombs_count = lambda: 0
    d._relay33_nav = SimpleNamespace(status=lambda: {}, command=lambda *a: dict(forward=.1,lateral=.2,rotation=0.))
    d._relay33_reports = {}
    d._relay33_previous = {}
    d._relay33_grant = None
    d._relay33_depth = 0.
    d._relay33_direct = {}
    d._relay33_serviced = {}
    d._relay33_searches = Counter()
    d._relay33_task = None
    d._relay33_task_since = 0
    d._relay33_contact_since = None
    d._relay33_events = deque(maxlen=160)
    d._d32_stats = Counter()
    d._relay33_refresh_report()
    return d


def exchange(drones, steps=1):
    for _ in range(steps):
        packets = [d.define_message_for_all() for d in drones]
        for i,d in enumerate(drones):
            d.communicator.received_messages = [(j,p) for j,p in enumerate(packets) if i!=j]
            d._receive()
        for d in drones:
            d._steps += 1
            d.elapsed_timestep += 1
            d._radio_maps._fixture_clock[0] = d.elapsed_timestep


def test_real_nested_message_exchange_delivers_payload_manifests_and_full_ack():
    stores = fleet()
    drones = [fixture(i, stores[i]) for i in range(5)]
    assert all(not d._exchange_state()['source_maps_complete'] for d in drones)
    exchange(drones, 20)
    for d in drones:
        state = d._exchange_state()
        assert state['source_maps_complete'] and state['final_complete']
        assert state['acknowledged_sources'] == list(range(5))
        assert state['received_versions'] == {str(i):4643 for i in range(5)}
        for i in range(5):
            assert d._radio_maps.snapshots[i] == stores[i].own_final
    drones[0]._submit()
    assert len(drones[0]._fixture_submissions) == 1
    assert drones[0]._s28_submission['final_complete']


def test_reports_relay_original_timestamp_without_entering_direct_peer_table():
    a,b,c = [fixture(i) for i in range(3)]
    a._steps = b._steps = c._steps = 4700
    a.elapsed_timestep = b.elapsed_timestep = c.elapsed_timestep = 4701
    a._position = np.array([-400., 300.])
    packet = a.define_message_for_all()
    b.communicator.received_messages = [(0,packet)]
    b._receive()
    b._steps = c._steps = 4720
    b.elapsed_timestep = c.elapsed_timestep = 4721
    c.communicator.received_messages = [(1,b.define_message_for_all())]
    c._receive()
    assert c._relay33_reports[0][1] == 4700
    assert c._relay33_reports[0][2] == (-400.,300.)
    assert 0 not in c._peers and 0 not in c._relay33_direct
    assert c._peers[1][0] == 4720


def test_direct_contact_timestamp_does_not_refresh_without_packet():
    a,b = fixture(0),fixture(1)
    origin = a._steps
    b.communicator.received_messages = [(0,a.define_message_for_all())]
    b._receive()
    assert b._relay33_direct[0] == origin
    b.communicator.received_messages = []
    for _ in range(3):
        b._steps += 1
        b.elapsed_timestep += 1
        b._receive()
    assert b._relay33_direct[0] == origin


def test_single_coordinator_grant_propagates_without_partition_election():
    drones = [fixture(i) for i in range(5)]
    exchange(drones,2)
    for d in drones[1:]:
        d._relay33_elect()
        assert d._relay33_grant is None
    drones[0]._relay33_elect()
    grant = drones[0]._relay33_grant
    assert grant is not None and grant[0] == 0
    exchange(drones,2)
    assert all(d._relay33_grant == grant for d in drones)
    assert sum(d._relay33_is_courier() for d in drones) == 1
    drones[2]._relay33_accept(((),(0,grant[1]+1,(grant[2]+1)%5)))
    assert drones[2]._relay33_grant == grant
    assert drones[2]._d32_stats['conflicting_grant_ignored'] == 1


def test_incomplete_source_finals_still_block_submission_after_relay_metadata():
    d = fixture()
    d._relay33_grant = (0,d._steps,0)
    d._relay33_refresh_report()
    state = d._exchange_state()
    d._s28_receipts = {i:(i,state['epoch_elapsed'],'a'*64) for i in range(5)}
    d._submit()
    assert not d._fixture_submissions and not d._exchange_state()['final_complete']


def test_invalid_or_future_metadata_does_not_claim_fresh_remote_position():
    d = fixture()
    row = fixture(2)._relay33_reports[2]
    assert valid_report33(row,5,d.elapsed_timestep)
    future = (row[0],d.elapsed_timestep+1,*row[2:])
    d._relay33_accept(((future,),None))
    assert 2 not in d._relay33_reports
    invalid = (9,*row[1:])
    d._relay33_accept(((invalid,),None))
    assert d._d32_stats['invalid_report'] == 2


def test_courier_eligibility_preserves_inventory_and_confirmation_priority():
    d = fixture(1)
    d._relay33_grant = (0,d._steps,1)
    assert d._relay33_is_courier()
    d.carried_bombs_count = lambda: 1
    assert not d._relay33_is_courier()
    d.carried_bombs_count = lambda: 0
    d._attempt_step = d._steps
    assert not d._relay33_is_courier()
    d._attempt_step = None
    d._exploration_submitted = True
    assert not d._relay33_is_courier()


def test_courier_choice_requires_fresh_healthy_finished_source():
    d = fixture()
    row = d._relay33_reports[0]
    reports = {
        0:(0,row[1]-61,*row[2:]),
        1:(1,row[1],(0.,0.),1,0.,25.,*row[6:]),
        2:(2,row[1],(0.,0.),0,0.,5.,*row[6:]),
        3:(3,row[1],(50.,0.),0,60.,25.,*row[6:]),
        4:(4,row[1],(70.,0.),0,100.,25.,*row[6:]),
    }
    assert choose_courier33(reports,d._steps,np.zeros(2)) == 3


def test_non_courier_navigation_keeps_original_motion_and_fields(monkeypatch):
    d = fixture(3,progress=.5)
    original = dict(forward=.8,lateral=-.2,rotation=.1,place_bomb=0)
    monkeypatch.setattr(MyDroneRedDualMap32Clean,'_navigation',lambda self: original.copy())
    assert d._navigation() == original
    assert d._d32_stats['courier_ticks'] == 0


def test_courier_motion_override_runs_after_original_updates_and_preserves_fields(monkeypatch):
    d = fixture(1)
    d._relay33_grant = (0,d._steps,1)
    d._relay33_task = dict(source=2,point=[100.,0.],purpose='collect')
    d._relay33_task_target = lambda: np.array([100.,0.])
    d.lidar_values = lambda: np.full(181,230.)
    d.lidar_rays_angles = lambda: np.linspace(-np.pi,np.pi,181)
    d.semantic_values = lambda: ()
    called = []
    def original(self):
        called.append('original')
        self._velocity = np.array([.2,.3])
        return dict(forward=.9,lateral=.9,rotation=.1,place_bomb=0)
    def relay(*args):
        called.append('relay')
        np.testing.assert_array_equal(args[2],[.2,.3])
        return dict(forward=.1,lateral=-.1,rotation=0.)
    monkeypatch.setattr(MyDroneRedDualMap32Clean,'_navigation',original)
    d._relay33_nav.command = relay
    assert d._navigation() == dict(forward=.1,lateral=-.1,rotation=0.,place_bomb=0)
    assert called == ['original','relay'] and d._d32_stats['courier_ticks'] == 1


def test_inherited_cutoff_and_mapping_observation_are_unchanged():
    for name in ('GATHER_AT','FREEZE_AT','DEADLINE_AT','_deployment','_update_observation'):
        assert getattr(MyDroneRedRelay33,name) is getattr(MyDroneRedDualMap32Clean,name)


def test_collect_confirmation_requires_actual_received_final_payload():
    d = fixture(1)
    peer = fixture(2)._relay33_reports[2]
    d._relay33_reports[2] = peer
    task = dict(source=2,purpose='collect',required_versions=[-1]*5)
    assert not d._relay33_task_confirmed(task)
    remote = fleet()[2]
    deliver(d._radio_maps,remote)
    assert d._relay33_task_confirmed(task)
    # A manifest alone cannot satisfy source final completeness.
    del d._radio_maps.snapshots[2]
    assert not d._relay33_task_confirmed(task)


def test_deliver_confirmation_needs_peer_versions_finals_and_acks():
    d = fixture(1)
    peer = list(fixture(0)._relay33_reports[0])
    task = dict(source=0,purpose='deliver',required_versions=[4643]*5,
                required_final_sources=list(range(5)),required_acks=[1,2])
    peer[6] = tuple([4643]*5)
    peer[7] = ()
    peer[8] = (1,)
    d._relay33_reports[0] = tuple(peer)
    assert not d._relay33_task_confirmed(task)
    peer[8] = (1,2)
    d._relay33_reports[0] = tuple(peer)
    assert d._relay33_task_confirmed(task)
    peer[7] = (4,)
    d._relay33_reports[0] = tuple(peer)
    assert not d._relay33_task_confirmed(task)
    peer[7] = ()
    peer[1] = d._steps-11
    d._relay33_reports[0] = tuple(peer)
    assert not d._relay33_task_confirmed(task)


def test_confirm_task_requires_exact_set_ack_in_inherited_barrier():
    stores = fleet()
    for remote in stores[1:]: deliver(stores[0],remote)
    d = fixture(0,snapshots=stores[0])
    d._relay33_reports[2] = fixture(2)._relay33_reports[2]
    task = dict(source=2,purpose='confirm')
    state = d._exchange_state()
    assert state['source_maps_complete'] and not d._relay33_task_confirmed(task)
    d._s28_receipts[2] = (2,state['epoch_elapsed'],'0'*64)
    assert not d._relay33_task_confirmed(task)
    d._s28_receipts[2] = (2,state['epoch_elapsed'],state['final_set_sha256'])
    assert d._relay33_task_confirmed(task)
