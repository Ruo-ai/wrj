"""Integration checks: frontier selection must not replace Clean delivery."""
import numpy as np
from swarm_rescue.solutions.devtools.test_red_schedule40 import drone
from swarm_rescue.solutions.my_drone_red_clean_frontier41 import MyDroneRedCleanFrontier41 as New
from swarm_rescue.solutions.my_drone_red_frontier34 import MyDroneRedCleanBoundary34 as Old
from swarm_rescue.solutions.red_delivery39 import FrontierQueue39


def test_mapping_radio_deployment_and_deadlines_unchanged():
    for method in ('_update_observation', '_submit', '_deployment', '_receive',
                   'define_message_for_all', '_try_hold', '_navigate_motion'):
        assert getattr(New, method) is getattr(Old, method)
    a, b = drone(Old, 1800), drone(New, 1800)
    assert type(a._submission_mapper) is type(b._submission_mapper)
    assert type(a._fusion27) is type(b._fusion27)
    assert (a.GATHER_AT, a.FREEZE_AT, a.DEADLINE_AT) == (b.GATHER_AT, b.FREEZE_AT, b.DEADLINE_AT)
    assert isinstance(b._explore, FrontierQueue39)
    assert not hasattr(b, '_d39_mode') and not hasattr(b, '_walk39')


def test_full_navigation_enables_frontier_before_sixty_percent():
    d = drone(New, 1800)
    # Create a legitimate observed-free/unknown boundary and force planning.
    d._explore.free[:, 16:] = False
    d._route = []
    d._last_route_step = -100
    d._navigation()
    assert d._explore.extension_enabled
    assert d._explore.stats39['frontier_selected'] > 0
    assert d._route


def test_bomb_tasks_keep_original_motion():
    a, b = drone(Old, 1800, 2), drone(New, 1800, 2)
    assert a._navigation() == b._navigation()
    assert not b._explore.extension_enabled


def test_actual_gathering_chain_disables_frontier_and_preserves_motion():
    a, b = drone(Old, 3600), drone(New, 3600)
    assert a._navigation() == b._navigation()
    assert a._gathering and b._gathering
    assert not b._explore.extension_enabled
    assert not b._explore.stats39.get('frontier_selected', 0)


def test_grid_observation_is_identical_to_clean():
    a, b = drone(Old, 1800), drone(New, 1800)
    for d in (a, b):
        d._explore.observe(np.zeros(2), 0., np.full(181, 120.),
                           np.linspace(-np.pi, np.pi, 181))
    for key in ('free', 'obstacle', 'visits'):
        np.testing.assert_array_equal(getattr(a._explore, key), getattr(b._explore, key))
