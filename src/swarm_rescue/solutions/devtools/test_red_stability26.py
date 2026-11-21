"""Failure-memory behavior through the original RL17 navigation MRO."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions import my_drone_red_advanced2 as old
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.my_drone_red_stability26 import (
    FailureRetirements26, MyDroneRedStability26,
)


def fixture(cls=MyDroneRedStability26, size=(400, 400)):
    # No simulation or truth access: instantiate only the Python policy state.
    kind = type('PolicyFixture26', (cls,), {'identifier': 0, 'drone_health': 25})
    drone = object.__new__(kind)
    events = []
    event = lambda *args, **kwargs: events.append((args, kwargs))
    grid = old._M19_SurveyGrid(size, event)
    grid.free[:] = True
    drone.__dict__.update(
        _position=np.zeros(2), _home=np.zeros(2), _heading=0., _steps=101,
        _launch_open=False, _velocity=np.zeros(2), _gathering=False,
        _loop_watch=SimpleNamespace(update=lambda *args: False),
        _slow_ticks=0, _last_motion_recovery=0, _direction_anchor=np.zeros(2),
        _peers={}, _blocked_goals={}, _route=[(5, 15)], _goal_step=101,
        _escape_until=0, _escape_heading=0., _last_route_step=101, _yield_until=0,
        _last_motion_angle=None, _bomb_target=None, _last_bomb_plan=-100,
        _n=5, _explore=grid, _exploration_submitted=False, _attempt_step=None,
        _motion_anchor=None, _stalled_steps=0, _replans=0, _yield_steps=0,
        _event=event, _fixture_events=events, _fixture_resets=[],
        _empty_command=lambda: dict(forward=0., lateral=0., rotation=0.),
        _progress=lambda: 0., carried_bombs_count=lambda: 0,
        semantic_values=lambda: [], lidar_values=lambda: np.full(181, 200.),
        lidar_rays_angles=lambda: np.linspace(-np.pi, np.pi, 181),
        odometer_values=lambda: np.zeros(3), _try_hold=lambda *args: False,
        _rn7=SimpleNamespace(path=[], command=lambda *args: None),
        _segment_motion=SimpleNamespace(state='NORMAL', PAUSED=(),
                                        pose=SimpleNamespace(step=0), since=0,
                                        step=lambda *args: None),
        _radio_maps=SimpleNamespace(publish=lambda *args: None),
        _map=SimpleNamespace(mask=lambda: np.zeros(grid.shape, bool)),
    )
    drone._waypoint_progress = SimpleNamespace(
        update=lambda *args: False, reset=lambda: drone._fixture_resets.append(drone._steps))
    drone._start_recovery = lambda *args: None
    if issubclass(cls, MyDroneRedStability26):
        drone._d26_stats = Counter()
        drone._failure_retirements26 = FailureRetirements26(drone._d26_stats)
    return drone


def release(drone, cell, step, reason='new_wall'):
    drone._steps = step
    drone._route = [cell]
    drone._release_stale_route(reason)


def expire(drone, step):
    drone._steps = step
    drone._failure_retirements26.expire(drone._explore.goal_visits, step)


def original_counts(size, cells):
    grid = old._M09_ExplorationGrid(size)
    for cell in cells:
        grid.retire_goal(cell)
    return grid.goal_visits


def test_candidate_has_exact_original_mro_and_inherited_safety_clock_mapping():
    assert MyDroneRedStability26.__bases__ == (MyDroneRedRepair17,)
    assert MyDroneRedStability26.__mro__[1:] == MyDroneRedRepair17.__mro__
    assert MyDroneRedRepair17._release_stale_route is old._M14_MyDroneRedV4Progress._release_stale_route
    for name in ('control', '_navigate_motion', '_update_observation', '_deployment',
                 '_progress', '_submit', '_start_recovery'):
        assert getattr(MyDroneRedStability26, name) is getattr(MyDroneRedRepair17, name)


def test_constructor_keeps_parent_initialization(monkeypatch):
    calls = []
    monkeypatch.setattr(MyDroneRedRepair17, '__init__', lambda self, **kw: calls.append(kw))
    drone = MyDroneRedStability26(sentinel=7)
    assert calls == [{'sentinel': 7}]
    assert drone._failure_retirements26.stats is drone._d26_stats
    assert not drone._failure_retirements26.entries


def test_release_preserves_original_penalty_and_all_original_side_effects():
    base, new = fixture(MyDroneRedRepair17), fixture()
    for drone in (base, new):
        drone._bomb_target = np.array([40., 50.])
        drone._release_stale_route('new_wall')
    np.testing.assert_array_equal(new._explore.goal_visits, base._explore.goal_visits)
    assert np.count_nonzero(new._explore.goal_visits) == 49
    for name in ('_route', '_blocked_goals', '_bomb_target', '_last_bomb_plan', '_last_route_step',
                 '_fixture_events', '_fixture_resets', '_steps'):
        assert getattr(new, name) == getattr(base, name)
    assert new._d26_stats['failure_events'] == 1
    assert new._d26_stats['first_failure_step'] == 101
    assert 'first_expiry_step' not in new._d26_stats


def test_true_mro_navigation_dispatches_original_waypoint_failure_to_candidate():
    drone = fixture()
    drone._waypoint_progress.update = lambda *args: True
    cmd = drone._navigation()
    assert set(cmd) == {'forward', 'lateral', 'rotation'}
    assert drone._d26_stats['failure_events'] == 1
    assert drone._blocked_goals == {(5, 15): 341}
    assert len(drone._explore.avoid_regions) == 1
    assert drone._explore.avoid_regions[0][1] == 341
    assert drone._route == []


def test_expiry_precedes_original_plan_and_blocked_goal_expiry():
    drone = fixture()
    release(drone, (5, 15), 101)
    seen = []

    def plan(position, home, identifier, number, claims, blocked, rendezvous):
        seen.append((drone._steps, drone._explore.goal_visits[5, 15], dict(blocked)))
        return [(5, 15)]

    drone._explore.plan = plan
    for step in (340, 341):
        drone._steps = step
        drone._route = []
        drone._last_route_step = 0
        drone._navigation()
    assert seen == [(340, 1., {(5, 15): 341}), (341, 0., {})]
    assert drone._d26_stats['first_expiry_step'] == 341


def test_no_failure_prefix_preserves_full_original_navigation_command_and_clock():
    base, new = fixture(MyDroneRedRepair17), fixture()
    for drone in (base, new):
        drone.elapsed_timestep = 777
        drone.elapsed_walltime = 31.25
    assert new._navigation() == base._navigation()
    np.testing.assert_array_equal(new._explore.goal_visits, base._explore.goal_visits)
    assert new._route == base._route and new._blocked_goals == base._blocked_goals
    assert new._steps == 101 and new.elapsed_timestep == 777 and new.elapsed_walltime == 31.25


def test_normal_arrival_stays_permanent_without_failure_entry():
    drone = fixture()
    cell = drone._explore.cell(drone._position)
    drone._route = [cell]
    drone._navigation()
    assert drone._d26_stats['failure_events'] == 0
    expire(drone, 2000)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts((400, 400), [cell]))


def test_goal_age_retirement_remains_permanent():
    drone = fixture()
    drone._steps = 701
    drone._last_route_step = 701
    drone._goal_step = 0
    drone._navigation()
    expire(drone, 2000)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts((400, 400), [(5, 15)]))
    assert drone._d26_stats['failure_events'] == 0


def test_success_before_after_failure_and_survey_overlap_are_preserved():
    drone = fixture()
    cell = (5, 15)
    drone._explore.retire_goal(cell)
    release(drone, cell, 100)
    drone._explore.survey = dict(cell=cell, start=100, unknown=0)
    drone._explore.step = 110
    drone._explore.finish_survey('observed')
    expire(drone, 339)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts((400, 400), [cell]*3))
    expire(drone, 340)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts((400, 400), [cell]*2))


@pytest.mark.parametrize('second', [(5, 15), (7, 13)])
def test_overlapping_failures_expire_independently_and_keep_success(second):
    drone = fixture()
    first = (5, 15)
    release(drone, first, 100)
    release(drone, second, 150)
    drone._explore.retire_goal(first)
    expire(drone, 340)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts((400, 400), [first, second]))
    expire(drone, 389)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts((400, 400), [first, second]))
    expire(drone, 390)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts((400, 400), [first]))
    assert drone._d26_stats['expired_events'] == 2


@pytest.mark.parametrize('size,cell', [
    ((400, 400), (0, 0)), ((400, 400), (0, 19)),
    ((400, 400), (19, 0)), ((400, 400), (19, 19)),
    ((400, 400), (0, 9)), ((400, 400), (19, 9)),
    ((20, 20), (0, 0)), ((140, 40), (1, 3)),
])
def test_clipped_disk_matches_original_even_at_zero_edges_and_thin_grids(size, cell):
    drone = fixture(size=size)
    release(drone, cell, 0)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts(size, [cell]))
    expire(drone, 239)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts(size, [cell]))
    expire(drone, 240)
    assert not drone._explore.goal_visits.any()
    assert drone._d26_stats['first_failure_step'] == 0


def test_empty_route_creates_no_expiry_debt():
    drone = fixture()
    drone._route = []
    drone._release_stale_route('new_wall')
    expire(drone, 1000)
    assert not drone._explore.goal_visits.any()
    assert drone._d26_stats['failure_events'] == 0


def test_actual_zero_increment_is_not_assumed_to_be_one():
    drone = fixture()
    drone._explore.retire_goal = lambda cell: None
    release(drone, (5, 15), 101)
    expire(drone, 1000)
    assert not drone._explore.goal_visits.any()
    assert drone._d26_stats['failure_events'] == 0


def test_unexpected_increment_is_preserved_not_authorized_for_subtraction():
    drone = fixture()
    original = drone._explore.retire_goal
    drone._explore.retire_goal = lambda cell: (original(cell), original(cell))
    release(drone, (5, 15), 101)
    expire(drone, 1000)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts((400, 400), [(5, 15)]*2))
    assert drone._d26_stats['unexpected_delta'] == 1
    assert drone._d26_stats['expired_events'] == 0


def test_array_replacement_does_not_apply_old_debt_to_new_counter():
    drone = fixture()
    release(drone, (5, 15), 101)
    replacement = np.ones(drone._explore.shape, np.float32)
    drone._explore.goal_visits = replacement
    expire(drone, 1000)
    np.testing.assert_array_equal(replacement, np.ones_like(replacement))
    assert drone._d26_stats['array_replacements'] == 1
    assert drone._d26_stats['discarded_events'] == 1


def test_two_overlapping_debts_and_reset_to_permanent_one_never_erase_success():
    drone = fixture()
    release(drone, (5, 15), 100)
    release(drone, (5, 15), 150)
    # Counter replacement in-place makes the old provenance unknowable.
    drone._explore.goal_visits.fill(1)
    expire(drone, 340)
    expire(drone, 390)
    np.testing.assert_array_equal(drone._explore.goal_visits, np.ones(drone._explore.shape))
    assert drone._d26_stats['underflow_guard_events'] == 1
    assert drone._d26_stats['discarded_events'] == 2
    assert drone._d26_stats['expired_events'] == 0
    assert not drone._failure_retirements26.owned.any()
    assert not drone._failure_retirements26.entries


def test_corrupt_owned_ledger_never_changes_goal_memory():
    drone = fixture()
    release(drone, (5, 15), 101)
    before = drone._explore.goal_visits.copy()
    drone._failure_retirements26.owned.fill(0)
    expire(drone, 341)
    np.testing.assert_array_equal(drone._explore.goal_visits, before)
    assert drone._d26_stats['ledger_guard_resets'] == 1


def test_backward_steps_and_repeated_expiry_do_not_advance_clock_or_subtract_twice():
    drone = fixture()
    release(drone, (5, 15), 100)
    expire(drone, 200)
    expire(drone, 99)
    np.testing.assert_array_equal(drone._explore.goal_visits, original_counts((400, 400), [(5, 15)]))
    expire(drone, 340)
    expire(drone, 340)
    expire(drone, 900)
    assert not drone._explore.goal_visits.any()
    assert drone._d26_stats['expired_events'] == 1
    assert drone._d26_stats['clock_regressions'] == 1


def test_jumped_real_step_expires_all_due_and_records_actual_first_expiry_step():
    drone = fixture()
    release(drone, (5, 15), 100)
    release(drone, (7, 13), 150)
    expire(drone, 500)
    assert not drone._explore.goal_visits.any()
    assert drone._d26_stats['expired_events'] == 2
    assert drone._d26_stats['first_failure_step'] == 100
    assert drone._d26_stats['first_expiry_step'] == 500


def test_unrelated_grid_observations_and_avoidance_memory_are_unchanged():
    drone = fixture()
    drone._explore.visits[1, 2] = 7
    drone._explore.obstacle[3, 4] = 2
    drone._explore.shared_visited[7, 9] = True
    drone._explore.avoid_region(np.array([20., 30.]), 400)
    before = {n:getattr(drone._explore, n).copy() for n in ('free', 'visits', 'obstacle', 'shared_visited')}
    release(drone, (5, 15), 100)
    expire(drone, 340)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(drone._explore, name), value)
    assert drone._explore.avoid_regions[0][1] == 400
    assert drone._blocked_goals == {(5, 15): 340}
