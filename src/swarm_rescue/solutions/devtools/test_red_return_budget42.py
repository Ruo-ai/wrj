"""Only budget changes; prove real navigation defers and still preempts."""
import numpy as np
import pytest
from swarm_rescue.solutions.devtools.test_red_schedule40 import drone
from swarm_rescue.solutions.my_drone_red_schedule40r1 import MyDroneRedSchedule40R1 as Old
from swarm_rescue.solutions.my_drone_red_return_budget42 import MyDroneRedReturnBudget42 as New


def ready(step=3200, speed=6., route=3500.):
    d = drone(New, step)
    d._d39_route = [0]; d._d39_cost = route; d._d39_plan_step = step
    d._rb42_motion.extend([(speed, True)]*239)
    d._rb42_sample_step = step-1
    d.odometer_values = lambda: np.array([speed, 0., 0.])
    return d


def test_real_navigation_keeps_exploring_when_observed_speed_affords_it():
    old = drone(Old, 3200)
    old._d39_route = [0]; old._d39_cost = 3500.; old._d39_plan_step = 3200
    new = ready()
    old._navigation(); new._navigation()
    assert old._d39_mode != 'explore'
    assert new._d39_mode == 'explore' and new._explore.extension_enabled
    assert new._rb42_stats['early_return_deferred_ticks'] == 1
    assert new._d39_budget['effective_speed'] == 3.8


def test_hard_cutoff_still_preempts_even_fast_drone():
    d = ready(step=4540, route=500.)
    d._navigation()
    assert d._d39_mode != 'explore'
    assert not d._explore.extension_enabled


def test_wall_clock_budget_still_preempts_before_step_limit():
    d = ready(step=3200)
    d.elapsed_walltime = 150.
    d._navigation()
    assert d._d39_mode != 'explore'


@pytest.mark.parametrize('case', ['sparse', 'missing', 'stopped', 'unknown_route', 'sampling_gap'])
def test_unreliable_or_slow_samples_keep_original_conservative_budget(case):
    d = ready()
    if case == 'sparse': d._rb42_motion.clear()
    if case == 'missing':
        d._rb42_motion.clear(); d._rb42_motion.extend([(0., False)]*239)
        d.odometer_values = lambda: None
    if case == 'stopped':
        d._rb42_motion.clear(); d._rb42_motion.extend([(0., True)]*239)
        d.odometer_values = lambda: np.zeros(3)
    if case == 'unknown_route': d._d39_route = []
    if case == 'sampling_gap': d._rb42_sample_step -= 5
    assert d._budget39()
    assert d._d39_budget['effective_speed'] == 2.1
    assert d._d39_budget['needed'] == d._d39_budget['legacy_needed']


def test_recent_stalls_count_and_cannot_be_hidden_by_fast_history():
    d = ready()
    d._rb42_motion.clear()
    d._rb42_motion.extend([(6., True)]*180+[(0., True)]*59)
    d.odometer_values = lambda: np.zeros(3)
    d._budget39()
    assert d._d39_budget['effective_speed'] == 2.1
    assert d._d39_budget['mean_odometry_long'] > 4.


def test_non_budget_methods_and_margins_preserved():
    for name in ('_update_observation', '_navigation', '_navigate_motion', '_path39',
                 '_return_motion39', '_submit', '_deployment', '_receive',
                 'define_message_for_all', '_try_hold', 'control'):
        assert getattr(New, name) is getattr(Old, name)
    d = ready(); d._budget39()
    assert d._d39_budget['recovery_reserve'] == 180
    assert d._d39_budget['exchange_reserve'] == 100
    assert (d.FREEZE_AT, d.DEADLINE_AT, d.GATHER_AT) == (Old.FREEZE_AT, Old.DEADLINE_AT, Old.GATHER_AT)
    assert {name for name, value in New.__dict__.items() if callable(value)} == {'__init__', '_budget39'}


def test_duplicate_budget_queries_do_not_duplicate_observations():
    d = ready(); d._rb42_motion.clear()
    d._budget39(); d._budget39()
    assert len(d._rb42_motion) == 1


def test_before_budget_divergence_control_is_identical():
    a, b = drone(Old, 1800), drone(New, 1800)
    a._d39_cost = b._d39_cost = 500.
    a._d39_plan_step = b._d39_plan_step = 1800
    assert a._navigation() == b._navigation()
    np.testing.assert_array_equal(a._explore.free, b._explore.free)
    np.testing.assert_array_equal(a._explore.obstacle, b._explore.obstacle)
