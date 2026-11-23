import numpy as np
import pytest
from types import SimpleNamespace

from swarm_rescue.solutions.red_deep_bomb45 import (
    placement_due, difficulty_features, rank, walked_costs, choose_deep_placement,
)
from swarm_rescue.solutions.my_drone_red_startup44_delivery import MyDroneRedStartup44Delivery as Old
from swarm_rescue.solutions.my_drone_red_startup44_deep_bomb import MyDroneRedStartup44DeepBomb as New
from swarm_rescue.solutions.devtools.test_red_schedule40 import drone


def test_reserve_second_charge_but_leave_fallback_budget():
    for i in range(5):
        assert .28 <= placement_due(2, 2, i) < .32
        assert .53 <= placement_due(2, 1, i) < .57


def test_long_wait_without_route_evidence_is_not_difficulty():
    f = difficulty_features((100, 0), None, (0, 0), (500, 500), 5000)
    assert f == dict(depth=0., detour=0., late=0.)


def test_detour_and_depth_can_outweigh_small_wall_loss():
    near = difficulty_features((80, 0), 80, (0, 0), (1000, 1000))
    deep = difficulty_features((500, 0), 1000, (0, 0), (1000, 1000))
    assert rank(80, 8, deep) > rank(100, 5, near)


def test_walked_graph_uses_shortest_route_not_accumulated_loops():
    g = SimpleNamespace(points=[(0, 0), (30, 0), (60, 0)],
                        edges=[{1:30, 2:60}, {0:30, 2:100}, {0:60, 1:100}], blocked={})
    assert walked_costs(g, 1)[2] == 60
    g.blocked[(0,2)] = 100
    assert walked_costs(g, 1)[2] == 130


def test_protocol_map_and_release_safety_unchanged():
    for name in ('_submit', '_deployment', '_receive', 'define_message_for_all',
                 '_budget39', '_return_motion39', '_coverage_field'):
        assert getattr(New, name) is getattr(Old, name)


@pytest.mark.parametrize('step', [500, 1000, 4100])
def test_empty_inventory_keeps_existing_navigation(step):
    a, b = drone(Old, step), drone(New, step)
    for d in (a,b):
        d._st44_done = True
        d._d39_cost = 200.; d._d39_route = [0]; d._d39_plan_step = step
    x, y = a._navigation(), b._navigation()
    assert a._d39_mode == b._d39_mode
    for k in x: assert x[k] == pytest.approx(y[k])


def test_real_navigation_dispatch_reaches_new_planner(monkeypatch):
    d = drone(New, 1800); d._st44_done = True
    d.carried_bombs_count = lambda: 2
    d._initial_inventory = 2
    d._route = []; d._last_bomb_plan = -100
    called = []
    import swarm_rescue.solutions.my_drone_red_startup44_deep_bomb as module
    monkeypatch.setattr(module, 'choose_deep_placement', lambda *a: called.append(True))
    d._navigation()
    assert called == [True]


def test_reserving_inventory_keeps_frontier_enabled(monkeypatch):
    d = drone(New, 2300); d._st44_done = True
    d.carried_bombs_count = lambda: 1
    d._initial_inventory = 2
    d._bomb_target = None; d._d39_mode = 'explore'; d._gathering = False
    d._d39_budget = dict(budget_due=False)
    d._navigate_motion()
    assert d._explore.extension_enabled


def test_empty_wall_gain_never_attracts_a_charge():
    d = drone(New, 2000)
    field = SimpleNamespace(placed=[], sampled_value=lambda p:0., value=lambda p:0., radius=50)
    assert choose_deep_placement(d._explore, d._position, field, None, d._walk39,
                                 {}, 2000, 5400) is None
