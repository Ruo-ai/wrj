"""Rotation-coordinate steering regressions; no simulator or truth-map access."""
import ast
import inspect
import math
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions import my_drone_red_advanced2 as baseline
from swarm_rescue.solutions import my_drone_red_stability24 as candidate_module
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.my_drone_red_stability24 import (
    MyDroneRedStability24, TargetMotion24, safe_velocity24,
)


def rotate(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]]) @ np.asarray(vector)


def observed_room():
    """One fixed set of legal range endpoints in an asymmetric empty room."""
    angles = np.linspace(-math.pi, math.pi, 181, endpoint=False)
    x, y = np.cos(angles), np.sin(angles)
    tx = np.divide(np.where(x >= 0, 57., -163.), x,
                   out=np.full_like(x, np.inf), where=np.abs(x) > 1e-12)
    ty = np.divide(np.where(y >= 0, 93., -71.), y,
                   out=np.full_like(y, np.inf), where=np.abs(y) > 1e-12)
    return np.minimum(tx, ty), angles


def world_result(helper, yaw, target=.13, previous=.55, target_distance=160.):
    distances, world_angles = observed_room()
    world_velocity = np.array([.4, -.15])
    force, selected = helper(
        distances, world_angles - yaw, target - yaw,
        rotate(world_velocity, -yaw),
        None if previous is None else previous - yaw, target_distance,
    )
    return rotate(force, yaw), rotate([math.cos(selected), math.sin(selected)], yaw)


@pytest.mark.parametrize('yaw', [-math.pi, -2.31, -1.11, -.37, .19, .73, 1.62, 2.84, math.pi])
@pytest.mark.parametrize('target,previous,distance', [
    (.13, .55, 160.), (-1.12, None, 90.), (1.81, 1.5, float('inf')),
])
def test_same_observed_points_give_same_world_force_and_direction(yaw, target, previous, distance):
    # Only the coordinate representation changes; no scan resampling or new RNG.
    expected = world_result(safe_velocity24, 0., target, previous, distance)
    actual = world_result(safe_velocity24, yaw, target, previous, distance)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0.)


def test_old_body_fixed_grid_has_a_rotation_counterexample():
    expected_force, expected_direction = world_result(baseline._M11_safe_velocity, 0.)
    results = [world_result(baseline._M11_safe_velocity, yaw)
               for yaw in (.03, .07, .19, .37, .73)]
    assert max(np.linalg.norm(direction - expected_direction)
               for _, direction in results) > 1e-3
    assert max(np.linalg.norm(force - expected_force)
               for force, _ in results) > 1e-3


@pytest.mark.parametrize('count', [0, 3, 11])
def test_insufficient_valid_rays_preserve_original_brake(count):
    distances = np.r_[np.full(count, 100.), [np.nan, np.inf, -1., 0.]]
    angles = np.linspace(-math.pi, math.pi, len(distances), endpoint=False)
    velocity = np.array([2., -.5])
    force, direction = safe_velocity24(distances, angles, .4, velocity, .1, 80.)
    old_force, old_direction = baseline._M11_safe_velocity(
        distances, angles, .4, velocity, .1, 80.)
    np.testing.assert_array_equal(force, old_force)
    np.testing.assert_array_equal(force, [-1., .35])
    assert direction is old_direction is None


def test_all_invalid_angles_preserve_original_brake():
    force, direction = safe_velocity24(np.full(20, 100.), np.full(20, np.nan),
                                       .4, np.array([.3, -.2]))
    np.testing.assert_allclose(force, [-.21, .14], atol=1e-15)
    assert direction is None


def test_arrived_target_and_force_bound_are_preserved():
    distances, angles = observed_room()
    velocity = np.array([.4, -.15])
    force, _ = safe_velocity24(distances, angles, .13, velocity, .55, 6.)
    np.testing.assert_allclose(force, -.65 * velocity, atol=1e-15)
    for target in np.linspace(-math.pi, math.pi, 17):
        force, direction = safe_velocity24(distances, angles, target, velocity)
        assert np.isfinite(force).all() and math.isfinite(direction)
        assert np.linalg.norm(force) <= 1. + 1e-12


def function_ast(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def test_helper_diff_is_only_target_relative_grid_phase():
    original = function_ast(baseline._M11_safe_velocity)
    candidate = function_ast(safe_velocity24)
    candidate.name = original.name
    changed = 0
    expected = ast.parse(
        'target_angle + np.linspace(-math.pi, math.pi, 48, endpoint=False)',
        mode='eval').body
    for node in ast.walk(candidate):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'headings'
                and ast.dump(node.value) == ast.dump(expected)):
            node.value = node.value.right
            changed += 1
    assert changed == 1
    assert ast.dump(candidate) == ast.dump(original)


def test_inherited_navigation_copy_changes_only_exploration_helper():
    active = baseline._M14_MyDroneRedV4Progress._navigate_motion
    assert MyDroneRedRepair17._navigate_motion is active
    original = function_ast(active)
    candidate = function_ast(TargetMotion24._navigate_motion)
    changed = 0
    for node in ast.walk(candidate):
        if isinstance(node, ast.Name) and node.id == 'safe_velocity24':
            node.id = '_M11_safe_velocity'
            changed += 1
    assert changed == 1
    assert ast.dump(candidate) == ast.dump(original)
    without_hook = tuple(cls for cls in MyDroneRedStability24.__mro__[1:]
                         if cls is not TargetMotion24)
    assert without_hook == MyDroneRedRepair17.__mro__
    for name in ('__init__', 'control', '_navigation', '_update_observation', '_deployment'):
        assert getattr(MyDroneRedStability24, name) is getattr(MyDroneRedRepair17, name)


def navigation_fixture(base):
    """Minimal legal-observation state for Advanced2's unchanged dispatch."""
    fixture_class = type('NavigationFixture', (base,), {'identifier': 0})
    drone = object.__new__(fixture_class)
    distances, angles = observed_room()
    drone.__dict__.update(
        _position=np.zeros(2), _heading=0., _home=np.zeros(2), _steps=2,
        _launch_open=False, _velocity=np.zeros(2), _gathering=False,
        _loop_watch=SimpleNamespace(update=lambda *args: False),
        _slow_ticks=0, _last_motion_recovery=0, _direction_anchor=np.zeros(2),
        _peers={}, _blocked_goals={}, _route=[(0, 0)], _goal_step=0,
        _escape_until=0, _last_route_step=0, _yield_until=0,
        _last_motion_angle=None, _bomb_target=None, _n=5,
        _waypoint_progress=SimpleNamespace(update=lambda *args: False,
                                           reset=lambda: None),
        _explore=SimpleNamespace(survey=None, free=np.ones((1, 1), bool),
                                 point=lambda cell: np.array([100., 0.]),
                                 line_clear=lambda *args: True),
        _empty_command=lambda: dict(forward=0., lateral=0., rotation=0.),
        _progress=lambda: 0., carried_bombs_count=lambda: 0,
        semantic_values=lambda: [], lidar_values=lambda: distances,
        lidar_rays_angles=lambda: angles, odometer_values=lambda: np.zeros(3),
        _try_hold=lambda *args: False,
    )
    return drone


def test_actual_advanced2_dispatch_uses_candidate_and_preserves_near_bomb(monkeypatch):
    calls = []
    original = baseline._M11_safe_velocity

    def old_spy(*args, **kwargs):
        calls.append('old')
        return original(*args, **kwargs)

    def new_spy(*args, **kwargs):
        calls.append('new')
        return safe_velocity24(*args, **kwargs)

    monkeypatch.setattr(baseline, '_M11_safe_velocity', old_spy)
    monkeypatch.setattr(candidate_module, 'safe_velocity24', new_spy)
    old = navigation_fixture(MyDroneRedRepair17)
    new = navigation_fixture(MyDroneRedStability24)
    assert super(baseline.MyDroneRedAdvanced2, new)._navigate_motion.__func__ is TargetMotion24._navigate_motion
    baseline.MyDroneRedAdvanced2._navigation(old)
    assert calls == ['old']
    calls.clear()
    baseline.MyDroneRedAdvanced2._navigation(new)
    assert calls == ['new']
    calls.clear()
    new._bomb_target = np.array([40., 0.])
    baseline.MyDroneRedAdvanced2._navigation(new)
    assert calls == ['new', 'old']
