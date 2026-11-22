"""Complete-set acknowledgements must represent every source, not only leader."""
from collections import Counter
from types import SimpleNamespace
import numpy as np

from swarm_rescue.solutions.my_drone_red_structural28 import MyDroneRedStructural28
from swarm_rescue.solutions.devtools.test_red_fusion27_delivery import (
    controller_fixture, messaging_fixture, full_leader, fleet, deliver,
)


def fixture(source=0, progress=.90, snapshots=None, messages=False):
    prior = (messaging_fixture(source, snapshots) if messages else
             controller_fixture(source, progress, snapshots or full_leader()))
    cls = type('StructuralFixture28', (MyDroneRedStructural28,), {'identifier': source, 'drone_health': 25})
    drone = object.__new__(cls)
    drone.__dict__.update(prior.__dict__)
    drone._n = 5
    drone._s28_stats = Counter()
    drone._s28_delivery = {}
    drone._s28_submission = None
    drone._s28_receipts = {}
    drone._s28_return_plan = SimpleNamespace(status=lambda: {'fixture': True})
    drone._rn7 = SimpleNamespace(stats=Counter())
    return drone


def exchange_receipts(drones):
    for drone in drones:
        drone._exchange_state()
    for drone in drones:
        for other in drones:
            drone._s28_receipts.update(other._s28_receipts)


def test_all_payloads_at_leader_are_insufficient_without_other_acknowledgements():
    drone = fixture()
    state = drone._exchange_state()
    assert state['source_maps_complete']
    assert state['acknowledged_sources'] == [0]
    assert not state['final_complete']
    drone._submit()
    assert not drone._fixture_submissions


def test_every_source_must_acknowledge_exact_same_set():
    drones = [fixture(i) for i in range(5)]
    exchange_receipts(drones)
    assert drones[0]._exchange_state()['final_complete']
    drones[0]._s28_receipts[4] = (4, 4644, '0'*64)
    assert not drones[0]._exchange_state()['final_complete']
    assert drones[0]._exchange_state()['missing_acknowledgements'] == [4]


def test_matching_receipts_do_not_replace_actual_payloads():
    drone = fixture()
    state = drone._exchange_state()
    drone._s28_receipts = {i: (i, state['epoch_elapsed'], state['final_set_sha256']) for i in range(5)}
    del drone._radio_maps.snapshots[4]
    assert not drone._exchange_state()['final_complete']
    assert not drone._exchange_state()['source_maps_complete']


def test_different_epoch_receipt_does_not_satisfy_barrier():
    drones = [fixture(i) for i in range(5)]
    exchange_receipts(drones)
    ack = drones[0]._s28_receipts[3]
    drones[0]._s28_receipts[3] = (3, ack[1]-1, ack[2])
    assert drones[0]._exchange_state()['missing_acknowledgements'] == [3]


def test_mutual_exchange_submits_exact_received_sources_once():
    drones = [fixture(i) for i in range(5)]
    exchange_receipts(drones)
    leader = drones[0]
    leader._submit()
    leader._submit()
    assert len(leader._fixture_submissions) == 1
    assert leader._s28_submission['final_complete']
    assert leader._s28_submission['acknowledged_sources'] == list(range(5))
    assert leader._s28_submission['reason'] == 'all_final_maps_and_acknowledgements_received'
    assert leader._s28_submission is leader._f27_submission
    assert set(leader._fusion27.received) == set(range(5))


def test_deadline_without_mutual_receipts_is_explicitly_incomplete():
    drone = fixture(progress=.975)
    drone._submit()
    assert len(drone._fixture_submissions) == 1
    assert drone._s28_submission['source_maps_complete']
    assert not drone._s28_submission['final_complete']
    assert drone._s28_submission['reason'] == 'deadline_incomplete_fallback'


def test_real_wrapped_receive_relays_maps_and_acknowledgements():
    stores = fleet()
    for receiver in stores:
        for sender in stores:
            if sender is not receiver:
                deliver(receiver, sender)
    drones = [fixture(i, snapshots=stores[i], messages=True) for i in range(5)]
    packets = [d.define_message_for_all() for d in drones]
    for i, drone in enumerate(drones):
        drone.communicator.received_messages = [(j, packet) for j, packet in enumerate(packets) if j != i]
        drone._receive()
        assert drone._exchange_state()['final_complete']
        assert set(drone._radio_maps.snapshots) == set(range(5))


def test_invalid_acknowledgement_is_not_relayed():
    drone = fixture(messages=True)
    packet = drone.define_message_for_all()
    for bad in ((9, 4644, 'a'*64), (1, 999999, 'a'*64), (1, 4644, 'x'*64)):
        drone.communicator.received_messages = [(1, ('structural28-v1', packet[1], (bad,)))]
        drone._receive()
    assert set(drone._s28_receipts) <= {0}
    assert drone._s28_stats['invalid_receipt'] == 3


def test_return_bypasses_old_grid_and_preserves_command_fields(monkeypatch):
    drone = fixture()
    plan = SimpleNamespace(returning=True, velocity=np.array([.4, -.2]),
        should_return=lambda *args: True, start_return=lambda reason: None,
        command=lambda *args, **kwargs: dict(forward=.1, lateral=-.1, rotation=0.),
        status=lambda: dict(returning=True))
    drone._s28_return_plan = plan
    drone._s28_return_started = False
    drone._rn7.path = []
    drone._misc_data = SimpleNamespace(max_timestep_limit=5400, max_walltime_limit=180)
    drone._position = np.zeros(2)
    drone._heading = 0.
    drone._velocity = np.array([8., 8.])
    drone._peers = {}
    drone.carried_bombs_count = lambda: 0
    drone.lidar_values = lambda: np.ones(181)*200
    drone.lidar_rays_angles = lambda: np.linspace(-np.pi, np.pi, 181)
    drone.semantic_values = lambda: ()
    drone._empty_command = lambda: dict(forward=0., lateral=0., rotation=0., place_bomb=0)
    from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
    def forbidden(*args):
        raise AssertionError('Return must not enter old navigation planner')
    monkeypatch.setattr(MyDroneRedRepair17, '_navigation', forbidden)
    command = drone._navigation()
    assert np.array_equal(drone._velocity, plan.velocity)
    assert command == dict(forward=.1, lateral=-.1, rotation=0., place_bomb=0)
    assert drone._gathering and drone._route == [] and drone._bomb_target is None
