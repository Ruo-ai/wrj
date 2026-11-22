"""Regression for unfinished deployment being preempted by return."""
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.devtools.test_red_structural28_delivery import fixture
from swarm_rescue.solutions.my_drone_red_structural29 import MyDroneRedStructural29
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.red_return29 import ReturnPlanner29


def make_drone(inventory, progress):
    old = fixture(progress=progress)
    cls = type('StructuralFixture29', (MyDroneRedStructural29,),
               {'identifier': 0, 'drone_health': 25})
    drone = object.__new__(cls)
    drone.__dict__.update(old.__dict__)
    plan = ReturnPlanner29(0)
    plan.trail = [np.array([0., 0.]), np.array([100., 0.])]
    plan.position = np.array([100., 0.])
    plan.step = 4000
    drone._s28_return_plan = plan
    drone._s28_return_started = False
    drone._misc_data = SimpleNamespace(max_timestep_limit=5400, max_walltime_limit=180)
    drone._position = np.array([100., 0.])
    drone._heading = 0.
    drone._velocity = np.zeros(2)
    drone._peers = {}
    drone._gathering = False
    drone._route = [(1, 2)]
    drone._bomb_target = np.array([110., 0.])
    drone._attempt_step = None
    drone._yield_until = 0
    drone._last_motion_angle = 0.
    drone._explore = object()
    drone._rn7 = SimpleNamespace(path=[], command=lambda *a: None, stats={})
    drone.carried_bombs_count = lambda: inventory
    drone.lidar_values = lambda: np.ones(181)*200
    drone.lidar_rays_angles = lambda: np.linspace(-np.pi, np.pi, 181)
    drone.semantic_values = lambda: ()
    drone._empty_command = lambda: dict(forward=0., lateral=0., rotation=0., place_bomb=0)
    return drone


@pytest.mark.parametrize('progress', [.4, .71, .85])
def test_pending_inventory_preserves_placement_route_and_target(monkeypatch, progress):
    drone = make_drone(1, progress)
    target = drone._bomb_target.copy()
    calls = []
    def original_navigation(self):
        calls.append(self._bomb_target.copy())
        return dict(forward=.2, lateral=.1, rotation=0., place_bomb=0)
    monkeypatch.setattr(MyDroneRedRepair17, '_navigation', original_navigation)
    drone._navigation()
    assert calls and np.array_equal(calls[0], target)
    assert not drone._s28_return_plan.returning
    assert not drone._gathering
    assert drone._route == [(1, 2)]
    assert np.array_equal(drone._bomb_target, target)


def test_zero_inventory_activates_return_without_entering_old_grid(monkeypatch):
    drone = make_drone(0, .71)
    def forbidden(*args):
        raise AssertionError('Return must use observed route planner')
    monkeypatch.setattr(MyDroneRedRepair17, '_navigation', forbidden)
    drone._s28_return_plan.command = lambda *a, **k: dict(forward=.1, lateral=0., rotation=0.)
    command = drone._navigation()
    assert command['forward'] == .1 and command['place_bomb'] == 0
    assert drone._s28_return_plan.returning and drone._gathering
    assert drone._bomb_target is None and not drone._route


def test_deadline_receipt_keeps_incomplete_exchange_explicit():
    old = fixture(progress=.975)
    cls = type('ReceiptFixture29', (MyDroneRedStructural29,), {'identifier': 0})
    drone = object.__new__(cls)
    drone.__dict__.update(old.__dict__)
    drone._submit()
    assert len(drone._fixture_submissions) == 1
    assert drone._s28_submission['source_maps_complete']
    assert not drone._s28_submission['final_complete']
    assert drone._s28_submission['missing_acknowledgements'] == [1, 2, 3, 4]
    assert 'complete deployment before return' in drone._s28_submission['return_policy']
