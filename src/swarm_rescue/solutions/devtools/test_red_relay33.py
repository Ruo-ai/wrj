"""Relay routing connectivity and fresh-observation safety regressions."""
import math
import numpy as np

from swarm_rescue.solutions.red_relay33 import RelayNavigator33
from swarm_rescue.solutions.red_return30 import ReturnLocal30
from swarm_rescue.solutions.devtools.test_red_return30 import Corner, TinyCorner


D = np.full(181, 230.)
A = np.linspace(-math.pi, math.pi, 181)


class Grid:
    CELL = 20.
    def __init__(self):
        self.free = np.ones((21, 31), bool)
        self.obstacle = np.zeros((21, 31))
    def cell(self, point):
        x, y = np.asarray(point)
        return int(np.clip(round(10-y/20), 0, 20)), int(np.clip(round(15+x/20), 0, 30))
    def point(self, cell):
        return np.array([(cell[1]-15)*20., (10-cell[0])*20.])


def call(p, grid=None, target=(100., 0.), position=(0., 0.), velocity=(0., 0.), d=D, a=A, tick=1):
    return p.command(np.array(position), 0., np.array(velocity), d, a, (),
                     Grid() if grid is None else grid, np.array(target), tick)


def test_open_target_uses_original_inflated_route_and_preserves_all_inputs():
    p = RelayNavigator33()
    g = Grid()
    free, obstacle, target = g.free.copy(), g.obstacle.copy(), np.array([100., 0.])
    result = call(p, g, target)
    assert result['forward'] > 0 and result['lateral'] == 0 and result['rotation'] == 0
    assert p.inflated_target_reachable and p.planning_mode == 'inflated_known_free'
    np.testing.assert_array_equal(g.free, free)
    np.testing.assert_array_equal(g.obstacle, obstacle)
    np.testing.assert_array_equal(target, [100., 0.])


def corridor():
    g = Grid()
    g.obstacle[9, :] = 3.
    g.obstacle[11, :] = 3.
    return g


def test_narrow_coarse_connection_requires_fresh_hull_then_can_move():
    p = RelayNavigator33()
    # Independent legal scan supplies a 50px corridor that fits the hull;
    # coarse raster endpoints can occupy adjacent 20px rows.
    d = np.minimum(D, 25./np.maximum(1e-9, np.abs(np.sin(A))))
    result = call(p, corridor(), d=d)
    assert not p.inflated_target_reachable and p.target_reachable
    assert p.planning_mode == 'unexpanded_known_free_fresh_hull_required'
    assert result['forward'] > 0
    force = np.array([result['forward'], result['lateral']])
    v = .6*force
    assert p._last_local.line_clear(np.zeros(2), v*3+v*np.linalg.norm(v)/1.2)


def test_same_coarse_connection_cannot_authorize_too_narrow_actual_hull():
    p = RelayNavigator33()
    d = np.minimum(D, 10./np.maximum(1e-9, np.abs(np.sin(A))))
    result = call(p, corridor(), d=d)
    assert not any(result.values())
    assert p.reason == 'current_hull_not_observed_clear'
    assert p.stats['motion_ticks'] == 0


def test_unknown_strip_remains_disconnected_in_both_coarse_modes():
    p = RelayNavigator33()
    g = Grid()
    g.free[:, 18] = False
    call(p, g, target=(200., 0.))
    assert not p.target_reachable
    assert all(point[0] < 60 for point in p.route)
    assert p.proxy_distance >= 160


def test_clipped_outside_target_is_not_declared_known_or_appended_to_route():
    p = RelayNavigator33()
    g = Grid()
    target = (9000., 0.)
    call(p, g, target=target)
    assert not p.inflated_target_reachable and not p.target_reachable
    assert p.proxy_distance == 8700.
    assert all(abs(point[0]) <= 300 and abs(point[1]) <= 200 for point in p.route)


def test_current_wall_invalidates_previous_route_without_replanning_permission():
    p = RelayNavigator33()
    assert call(p)['forward'] > 0
    result = call(p, d=np.full_like(D, 10.), tick=2)
    assert not any(result.values())
    assert p.reason == 'current_hull_not_observed_clear'


def test_missing_scan_brakes_measured_velocity_and_never_reuses_local_free():
    p = RelayNavigator33()
    call(p)
    result = call(p, d=None, velocity=(1., -.5), tick=2)
    assert result == dict(forward=-.65, lateral=.325, rotation=0.)
    assert p.reason == 'scan_unavailable'


class StopsBlocked:
    CELL = 2.
    PLAN_RADIUS = 20.
    TRACK_RADIUS = 18.5
    def __init__(self, *args): pass
    def line_clear(self, a, b):
        length = np.linalg.norm(np.asarray(b)-a)
        return length < .01 or length >= 30.
    def sample(self, points): return np.full(len(points), 50.)
    def route(self, *args, **kwargs): return []


def test_destination_clear_is_insufficient_when_next_stopping_sweep_is_blocked():
    p = RelayNavigator33(local_factory=StopsBlocked)
    result = call(p)
    assert not any(result.values())
    assert p.reason == 'stopping_sweep_blocked' and p.blocked_ticks == 1


def test_arrival_requires_observed_clear_connection_to_target():
    p = RelayNavigator33()
    assert not any(call(p, target=(20., 0.)).values()) and p.arrived
    q = RelayNavigator33()
    call(q, target=(20., 0.), d=np.full_like(D, 10.))
    assert not q.arrived


def test_target_array_never_aliased_into_corner_repair_or_normalized():
    p = RelayNavigator33(local_factory=Corner)
    p.ARRIVAL_RADIUS = 1.
    target = np.array([5., 21.])
    before = target.copy()
    result = call(p, target=target)
    np.testing.assert_array_equal(target, before)
    np.testing.assert_array_equal(p.target, before)
    assert result['forward'] > 0 and result['lateral'] > 0
    np.testing.assert_allclose(p.waypoint, [3., 5.])


def test_two_pixel_repair_corner_has_nonzero_safe_command():
    p = RelayNavigator33(local_factory=TinyCorner)
    p.ARRIVAL_RADIUS = 1.
    result = call(p, target=(2., 30.))
    assert result['forward'] > 0 and result['lateral'] == 0
    np.testing.assert_allclose(p.waypoint, [2., 0.])


def test_wrong_repair_terminal_is_rejected():
    class Wrong(Corner):
        def route(self, *args, **kwargs): return [np.zeros(2), np.array([1., 1.])]
    p = RelayNavigator33(local_factory=Wrong)
    p.ARRIVAL_RADIUS = 1.
    result = call(p, target=(5., 21.))
    assert not any(result.values())
    assert p.stats['repair_wrong_terminal_rejected'] > 0


def test_heading_rotation_settles_without_new_forward_force():
    p = RelayNavigator33()
    call(p)
    result = p.command(np.zeros(2), .5, np.zeros(2), D, A, (), Grid(), np.array([100., 0.]), 2)
    assert not any(result.values()) and p.reason == 'heading_settle'


def test_reset_discards_old_route_and_target_without_mutating_caller():
    p = RelayNavigator33()
    call(p)
    p.reset()
    assert p.target is None and not p.route and not p.arrived
    result = call(p, target=(-100., 0.), tick=2)
    assert result['forward'] < 0 and p.target[0] == -100.


def test_step_alias_is_identical_public_interface():
    assert RelayNavigator33.step is RelayNavigator33.command
