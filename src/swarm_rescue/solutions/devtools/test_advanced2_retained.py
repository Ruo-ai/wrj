"""Behavioural checks for the three standalone strategy files."""

# Original behavioural checks: test_red_advanced2
import numpy as _test_red_advanced2_np
from swarm_rescue.solutions.my_drone_red_advanced2 import _M09_ExplorationGrid as _test_red_advanced2_ExplorationGrid
from swarm_rescue.solutions.my_drone_red_advanced2 import _M16_CoverageField as _test_red_advanced2_CoverageField, _M16_choose_placement as _test_red_advanced2_original
from swarm_rescue.solutions.my_drone_red_advanced2 import _M17_transport_costs as _test_red_advanced2_transport_costs, _M17_difficulty as _test_red_advanced2_difficulty, _M17_choose_placement as _test_red_advanced2_choose_placement

def test_test_red_advanced2_transport_does_not_cross_unknown_or_disconnected_room():
    g = _test_red_advanced2_ExplorationGrid((400, 400))
    g.free[:] = True
    g.free[:, 10] = False
    distances = _test_red_advanced2_transport_costs(g, g.point((5, 3)))
    assert (5, 8) in distances and (5, 12) not in distances
    assert _test_red_advanced2_transport_costs(g, None) == {}

def test_test_red_advanced2_transport_route_detects_detour_around_observed_wall():
    g = _test_red_advanced2_ExplorationGrid((400, 400))
    g.free[:] = True
    g.obstacle[3:17, 10] = 3
    center = g.point((10, 5))
    target = g.point((10, 15))
    d = _test_red_advanced2_transport_costs(g, center)
    assert d[10, 15] > _test_red_advanced2_np.linalg.norm(target - center) + 100
    assert _test_red_advanced2_difficulty(target, d[10, 15], center, (400, 400)) > _test_red_advanced2_difficulty(target, 200, center, (400, 400))
    assert 0 <= _test_red_advanced2_difficulty(target, 1000000.0, center, (400, 400)) <= 1

def test_test_red_advanced2_no_center_retains_original_plan_and_wall_floor_is_respected():
    g = _test_red_advanced2_ExplorationGrid((400, 400))
    g.free[:] = True
    mask = _test_red_advanced2_np.zeros((100, 100), bool)
    mask[1, :] = True
    mask[70, 5:95] = True
    f = _test_red_advanced2_CoverageField(mask, 4, (400, 400), [], [], 48)
    pos = g.point((10, 10))
    old = _test_red_advanced2_original(g, pos, f)
    neutral = _test_red_advanced2_choose_placement(g, pos, f)
    assert _test_red_advanced2_np.array_equal(neutral.target, old.target)
    events = []
    plan = _test_red_advanced2_choose_placement(g, pos, f, g.point((18, 10)), lambda *a, **kw: events.append(kw))
    assert plan.gain >= 0.9 * old.gain
    assert events and events[0]['selected_gain'] == plan.gain
    assert all((g.free[cell] for cell in plan.route))

# Original behavioural checks: test_red_advanced2_bounds
import numpy as _test_red_advanced2_bounds_np
from swarm_rescue.solutions import my_drone_red_advanced2 as _test_red_advanced2_bounds_planner
from swarm_rescue.solutions.my_drone_red_advanced2 import _M09_ExplorationGrid as _test_red_advanced2_bounds_ExplorationGrid

def test_test_red_advanced2_bounds_partial_last_grid_cell_cannot_place_outside_map(monkeypatch):
    grid = _test_red_advanced2_bounds_ExplorationGrid((410, 400))
    grid.free[:] = True
    position = _test_red_advanced2_bounds_np.array([197.0, 10.0])
    base = _test_red_advanced2_bounds_planner._M16_PlacementPlan([grid.cell(position)], position, 100.0)
    monkeypatch.setattr(_test_red_advanced2_bounds_planner, '_M16_choose_placement', lambda *args: base)

    class Field:
        placed = []
        radius = 48

        def sampled_value(self, p):
            return 150.0 if p[0] >= 205 else 0.0

        def value(self, p):
            return 150.0 if p[0] >= 205 else 100.0
    chosen = _test_red_advanced2_bounds_planner._M17_choose_placement(grid, position, Field(), _test_red_advanced2_bounds_np.array([-150.0, 10.0]))
    assert abs(chosen.target[0]) < 205 and abs(chosen.target[1]) < 200


