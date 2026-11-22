"""Observed-space return budgets and finite current-scan recovery regressions."""
import math

import numpy as np

from swarm_rescue.solutions.red_frontier34 import FrontierNavigator34
from swarm_rescue.solutions.red_passage28 import HullPassage28
from swarm_rescue.solutions.devtools.test_red_relay33 import (
    A, D, Grid, call, corridor, StopsBlocked,
)


def near_wall():
    d = D.copy()
    below = np.sin(A) < 0
    d[below] = np.minimum(d[below], 18. / -np.sin(A[below]))
    return d


def test_open_navigation_retains_verified_original_motion_and_hull():
    p = FrontierNavigator34()
    result = call(p)
    assert result['forward'] > 0 and result['lateral'] == 0
    assert not p.failed and p.reason == 'following_verified_relay_route'
    np.testing.assert_array_equal(p._last_local.footprint(), HullPassage28.footprint())


def test_margin_overlap_can_retreat_using_same_current_scan_hull():
    p = FrontierNavigator34()
    d = near_wall()
    before = d.copy()
    result = call(p, d=d)
    assert p.reason == 'current_scan_monotone_retreat'
    assert result['lateral'] > 0 and result['rotation'] == 0
    assert p.status()['recovery_details']['same_hull_kernel']
    np.testing.assert_array_equal(d, before)
    # Independently validate every swept footprint against the initial unsafe
    # set; this checks geometry rather than merely mirroring the chosen vector.
    fine = p._last_local
    known = fine.free & ~fine.static
    kernel = fine.footprint().astype(bool)
    offsets = np.argwhere(kernel) - np.asarray(kernel.shape) // 2
    initial = np.asarray(fine.cell(np.zeros(2))) + offsets
    unsafe = set(map(tuple, initial[~known[initial[:, 0], initial[:, 1]]]))
    velocity = .6 * np.array([result['forward'], result['lateral']])
    stopping = 3 * velocity + velocity * np.linalg.norm(velocity) / 1.2
    previous = len(unsafe)
    for point in np.linspace(np.zeros(2), stopping, 30):
        cells = np.asarray(fine.cell(point)) + offsets
        bad = set(map(tuple, cells[~known[cells[:, 0], cells[:, 1]]]))
        assert bad <= unsafe and len(bad) <= previous
        previous = len(bad)


def test_missing_ray_does_not_authorize_margin_recovery_or_reuse_previous_scan():
    p = FrontierNavigator34()
    call(p)
    d = near_wall()
    d[0] = np.nan
    result = call(p, d=d, tick=2)
    assert not any(result.values())
    assert p.reason == 'current_hull_not_observed_clear'
    assert p.stats['recovery_incomplete_observation'] == 1
    result = call(p, d=None, tick=3)
    assert not any(result.values()) and p._last_local is None


def test_margin_recovery_first_brakes_high_measured_speed():
    p = FrontierNavigator34()
    result = call(p, d=near_wall(), velocity=(0., -2.))
    assert result == dict(forward=0., lateral=1., rotation=0.)
    assert p.stats['recovery_wait_velocity'] == 1


def test_physically_too_narrow_corridor_never_gets_a_recovery_command():
    p = FrontierNavigator34()
    d = np.minimum(D, 10. / np.maximum(1e-9, np.abs(np.sin(A))))
    for tick in range(1, p.BLOCKED_LIMIT + 1):
        result = call(p, corridor(), d=d, tick=tick)
        assert not any(result.values())
    assert p.failed and p.failure_reason == 'blocked_budget'
    assert p.status()['failure_tick'] == p.BLOCKED_LIMIT


def test_failed_same_target_stays_explicit_until_reset_or_target_change():
    p = FrontierNavigator34()
    p.BLOCKED_LIMIT = 2
    for tick in (1, 2):
        call(p, d=np.full_like(D, 10.), tick=tick)
    assert p.failed
    assert not any(call(p, tick=3).values())
    assert p.reason == 'frontier_navigation_failed'
    result = call(p, target=(-100., 0.), tick=4)
    assert not p.failed and result['forward'] < 0
    p.reset()
    assert not p.failed and p.failure_reason is None and p.target is None


def test_command_attempts_without_physical_displacement_have_finite_budget():
    p = FrontierNavigator34()
    for tick in range(1, p.STAGNATION_LIMIT + 2):
        call(p, tick=tick)
    assert p.failed and p.failure_reason == 'no_displacement_budget'


def test_actual_displacement_resets_stagnation_and_inactive_time_is_not_counted():
    p = FrontierNavigator34()
    p.STAGNATION_LIMIT = 5
    for tick in range(1, 20):
        call(p, position=(float(tick * 4), 0.), target=(220., 0.), tick=tick)
        assert not p.failed
    call(p, position=(80., 0.), target=(220., 0.), tick=1000)
    assert not p.failed and p.no_progress_ticks == 0


def test_destination_clear_does_not_bypass_stopping_sweep():
    p = FrontierNavigator34(local_factory=StopsBlocked)
    result = call(p)
    assert not any(result.values())
    assert p.reason == 'stopping_sweep_blocked'
    assert p.consecutive_blocked_ticks == 1


def test_route_estimate_includes_detour_and_never_mutates_evidence():
    g = Grid()
    g.obstacle[4:18, 17] = 3
    free, obstacle = g.free.copy(), g.obstacle.copy()
    estimate = FrontierNavigator34.estimate_route(g, [0., 0.], [120., 0.])
    assert estimate['exact'] and estimate['reachable']
    assert estimate['length'] > 120 and estimate['travel_steps'] > math.ceil(120 / 1.8)
    points = np.asarray(estimate['route_points'])
    np.testing.assert_allclose(points[0], [0., 0.])
    np.testing.assert_allclose(points[-1], [120., 0.])
    np.testing.assert_array_equal(g.free, free)
    np.testing.assert_array_equal(g.obstacle, obstacle)


def test_unknown_disconnection_has_no_finite_complete_route_budget():
    g = Grid()
    g.free[:, 18] = False
    estimate = FrontierNavigator34.estimate_route(g, [0., 0.], [200., 0.])
    assert not estimate['exact'] and estimate['steps'] is None
    assert estimate['proxy_gap_px'] >= 160
    assert all(point[0] < 60 for point in estimate['route_points'])


def test_outside_target_and_invalid_input_cannot_be_costed_as_reachable():
    p = FrontierNavigator34.estimate_route(Grid(), [0., 0.], [9000., 0.])
    assert not p['exact'] and p['steps'] is None and p['proxy_gap_px'] == 8700
    q = FrontierNavigator34.estimate_route(Grid(), [float('nan'), 0.], [10., 0.])
    assert not q['exact'] and q['length'] is None and q['route_points'] == []


def test_narrow_observed_route_reports_need_for_fresh_hull_execution():
    estimate = FrontierNavigator34.estimate_route(corridor(), [0., 0.], [100., 0.])
    assert estimate['exact'] and estimate['mode'] == 'unexpanded_known_free_fresh_hull_required'
    p = FrontierNavigator34()
    d = np.minimum(D, 25. / np.maximum(1e-9, np.abs(np.sin(A))))
    assert call(p, corridor(), d=d)['forward'] > 0


def test_step_alias_and_status_are_json_serializable():
    import json
    p = FrontierNavigator34()
    call(p, d=near_wall())
    assert FrontierNavigator34.step is FrontierNavigator34.command
    json.dumps(p.status(), allow_nan=False)
