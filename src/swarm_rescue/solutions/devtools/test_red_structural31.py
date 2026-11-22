"""Current budget pressure must never bypass native deployment safeguards."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_structural31 import MyDroneRedStructural31
from swarm_rescue.solutions.my_drone_red_structural30 import MyDroneRedStructural30


def fixture(monkeypatch):
    cls = type('BudgetFixture31', (MyDroneRedStructural31,), {'identifier': 0})
    d = object.__new__(cls)
    d._s31_pressure_frames = 0
    d._s31_pressure_step = None
    d._steps = 2000
    d._s28_return_plan = SimpleNamespace(step=2000, returning=False, return_requested=True,
        last_budget={'required_steps': 3000., 'effective_remaining_steps': 2500.})
    d._s28_stats = Counter()
    d._position = np.array([200., 200.])
    d._initial_inventory = d._last_inventory = 2
    d._attempt_step = d._attempt_position = d._failed_place = None
    d._placements = []
    d._next_attempt = d._attempts = d._total_attempts = 0
    d._bomb_target = np.array([800., 800.])
    d._route = [(4, 8)]
    d._fixture_inventory = 2
    d._fixture_progress = .5
    d._fixture_objects = ()
    d._parent_calls = []
    d.carried_bombs_count = lambda: d._fixture_inventory
    d._progress = lambda: d._fixture_progress
    d.semantic_values = lambda: d._fixture_objects
    d._event = lambda *a, **k: None
    monkeypatch.setattr(MyDroneRedStructural30, '_deployment', lambda self, command: self._parent_calls.append(self._steps))
    return d


def tick(d):
    command = {'forward': .2, 'lateral': 0., 'rotation': 0., 'grasper': 0, 'place_bomb': 0}
    d._s28_return_plan.step = d._steps
    d._deployment(command)
    d._steps += 1
    return command


def arm(d):
    for _ in range(7):
        assert tick(d)['place_bomb'] == 0


def test_persistent_current_pressure_releases_with_original_fallback(monkeypatch):
    d = fixture(monkeypatch)
    arm(d)
    command = tick(d)
    assert command['place_bomb'] == 1 and d._total_attempts == 1
    assert len(d._parent_calls) == 7
    assert np.array_equal(d._bomb_target, [800., 800.])
    assert d._fixture_progress == .5


def test_latched_request_without_current_deficit_does_not_override(monkeypatch):
    d = fixture(monkeypatch)
    d._s28_return_plan.last_budget['effective_remaining_steps'] = 4000.
    for _ in range(12):
        assert tick(d)['place_bomb'] == 0
    assert len(d._parent_calls) == 12 and d._total_attempts == 0


def test_recovered_budget_resets_confirmation_streak(monkeypatch):
    d = fixture(monkeypatch)
    arm(d)
    d._s28_return_plan.last_budget['effective_remaining_steps'] = 4000.
    tick(d)
    d._s28_return_plan.last_budget['effective_remaining_steps'] = 2500.
    arm(d)
    assert tick(d)['place_bomb'] == 1


@pytest.mark.parametrize('case', ['release_time', 'own_bomb', 'seen_bomb', 'failed_position', 'retry_limit', 'retry_delay'])
def test_original_placement_safeguards_remain_authoritative(monkeypatch, case):
    d = fixture(monkeypatch)
    if case == 'release_time': d._fixture_progress = .1
    if case == 'own_bomb': d._placements = [(190., 200.)]
    if case == 'seen_bomb': d._fixture_objects = [SimpleNamespace(entity_type=SimpleNamespace(name='BOMB'), distance=30.)]
    if case == 'failed_position': d._failed_place = np.array([200., 210.])
    if case == 'retry_limit': d._total_attempts = 60
    if case == 'retry_delay': d._next_attempt = 9999
    arm(d)
    assert tick(d)['place_bomb'] == 0
    assert d._attempt_step is None


@pytest.mark.parametrize('case', ['inflight_attempt', 'inventory_confirmation', 'empty_inventory'])
def test_confirmation_and_empty_inventory_keep_parent_chain(monkeypatch, case):
    d = fixture(monkeypatch)
    arm(d)
    if case == 'inflight_attempt': d._attempt_step = d._steps-1
    if case == 'inventory_confirmation': d._fixture_inventory = 1
    if case == 'empty_inventory': d._fixture_inventory = d._last_inventory = 0
    assert tick(d)['place_bomb'] == 0
    assert len(d._parent_calls) == 8


def test_duplicate_same_step_cannot_accumulate_pressure(monkeypatch):
    d = fixture(monkeypatch)
    for _ in range(20):
        command = {'place_bomb': 0}
        d._deployment(command)
        assert command['place_bomb'] == 0
    assert d._s31_pressure_frames == 1


def test_stale_budget_never_triggers_fallback(monkeypatch):
    d = fixture(monkeypatch)
    d._s31_pressure_frames = 7
    d._s31_pressure_step = d._steps-1
    d._s28_return_plan.step -= 1
    command = {'place_bomb': 0}
    d._deployment(command)
    assert command['place_bomb'] == 0 and d._s31_pressure_frames == 0
