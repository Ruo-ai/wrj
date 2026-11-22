"""Paired-policy identity, radio barrier and navigation/submission isolation."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_fusion27 import MyDroneRedFusion27
from swarm_rescue.solutions.my_drone_red_dualmap32 import (
    MyDroneRedDualMap32, MyDroneRedDualMap32Clean,
)
from swarm_rescue.solutions.devtools.test_red_fusion27_delivery import (
    controller_fixture, messaging_fixture, full_leader, fleet, deliver,
)


def fixture(source=0, progress=.9, snapshots=None, messages=False):
    prior = (messaging_fixture(source, snapshots) if messages else
             controller_fixture(source, progress, snapshots or full_leader()))
    cls = type('DualMapFixture32', (MyDroneRedDualMap32,),
               {'identifier': source, 'drone_health': 25})
    drone = object.__new__(cls)
    drone.__dict__.update(prior.__dict__)
    drone._n = 5
    drone._s28_stats = Counter()
    drone._s28_delivery = {}
    drone._s28_submission = None
    drone._s28_receipts = {}
    if not messages:
        drone._explore = SimpleNamespace(
            free=np.eye(5, dtype=bool), obstacle=np.arange(25).reshape(5, 5).copy(),
            visits=np.zeros((5, 5), np.uint16))
    return drone


def test_control_retains_exact_fusion27_flight_methods_and_cutoffs():
    for cls in (MyDroneRedDualMap32, MyDroneRedDualMap32Clean):
        for name in ('_navigation', '_deployment', '_update_observation', '_try_hold'):
            assert getattr(cls, name) is getattr(MyDroneRedFusion27, name)
        for name in ('NARROW_ENABLED', 'GATHER_AT', 'FREEZE_AT', 'DEADLINE_AT'):
            assert getattr(cls, name) == getattr(MyDroneRedFusion27, name)
    assert MyDroneRedDualMap32Clean.CLEAN_NAVIGATION
    assert not MyDroneRedDualMap32.CLEAN_NAVIGATION


def test_barrier_waits_even_if_leader_has_all_five_maps():
    drone = fixture()
    assert drone._exchange_state()['source_maps_complete']
    drone._submit()
    assert not drone._fixture_submissions
    assert not drone._exchange_state()['final_complete']


def test_real_radio_round_trip_requires_all_matching_set_receipts():
    stores = fleet()
    for receiver in stores:
        for sender in stores:
            if sender is not receiver:
                deliver(receiver, sender)
    drones = [fixture(i, snapshots=stores[i], messages=True) for i in range(5)]
    packets = [drone.define_message_for_all() for drone in drones]
    for i, drone in enumerate(drones):
        drone.communicator.received_messages = [(j, packet) for j, packet in enumerate(packets) if i != j]
        drone._receive()
        assert drone._exchange_state()['final_complete']
    drones[0]._submit()
    drones[0]._submit()
    assert len(drones[0]._fixture_submissions) == 1
    assert drones[0]._f27_submission['acknowledged_sources'] == list(range(5))


def test_matching_ack_does_not_replace_missing_payload():
    drone = fixture()
    state = drone._exchange_state()
    drone._s28_receipts = {i: (i, state['epoch_elapsed'], state['final_set_sha256']) for i in range(5)}
    del drone._radio_maps.snapshots[4]
    drone._submit()
    assert not drone._fixture_submissions


@pytest.mark.parametrize('draw_all', [False, True])
def test_arbitrary_submission_pixels_cannot_change_navigation_evidence(draw_all):
    drone = fixture(progress=.975)
    before = {name: getattr(drone._explore, name).copy() for name in ('free', 'obstacle', 'visits')}
    nav_masks = {i: mask.copy() for i, mask in drone._radio_maps.masks.items()}
    class Renderer:
        last_stats = {}
        def fuse(self, masks):
            return np.full((60, 90), draw_all, bool)
    drone._fusion27 = Renderer()
    drone._submit()
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(drone._explore, name), value)
    for i, mask in nav_masks.items():
        np.testing.assert_array_equal(drone._radio_maps.masks[i], mask)
    assert np.all(drone._fixture_submissions[0] == draw_all)
    assert not drone._f27_submission['final_complete']
    assert drone._f27_submission['reason'] == 'deadline_incomplete_fallback'
    assert drone._f27_submission['navigation_policy'] == 'original_fusion27'
