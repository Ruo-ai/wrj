"""Native lidar shape, sector geometry and recorded-route return regressions."""
import math

import numpy as np

from swarm_rescue.simulation.ray_sensors.distance_sensor import compute_ray_angles
from swarm_rescue.simulation.utils.constants import (
    FOV_LIDAR_SENSOR, RESOLUTION_LIDAR_SENSOR, MAX_RANGE_LIDAR_SENSOR,
)
from swarm_rescue.solutions.red_rendezvous35 import Navigator35
from swarm_rescue.solutions.red_passage28 import HullPassage28
from swarm_rescue.solutions.devtools.test_red_relay33 import Grid, StopsBlocked, corridor
from swarm_rescue.solutions.devtools.test_red_return30 import Corner


A = compute_ray_angles(math.radians(FOV_LIDAR_SENSOR), RESOLUTION_LIDAR_SENSOR)
D = np.full(RESOLUTION_LIDAR_SENSOR, float(MAX_RANGE_LIDAR_SENSOR))


def call(p, tick=1, d=D, target=(100.,0.), position=(0.,0.), velocity=(0.,0.), grid=None):
    return p.command(np.asarray(position,float),0.,np.asarray(velocity,float),d,A,(),
                     Grid() if grid is None else grid,np.asarray(target,float),tick)


def test_native_scan_dimensions_are_used_without_range_normalization():
    assert D.shape == A.shape == (181,)
    assert A[0] == -math.pi and A[-1] == math.pi
    p = Navigator35()
    result = call(p)
    assert result['forward'] > 0 and result['rotation'] == 0
    assert p.status()['scan_status']['long_ray_count'] == 181
    np.testing.assert_array_equal(p._last_local.footprint(),HullPassage28.footprint())


def test_column_sensor_values_are_flattened_but_inputs_remain_unmodified():
    p = Navigator35()
    d = D[:,None].copy()
    before = d.copy()
    assert call(p,d=d)['forward'] > 0
    np.testing.assert_array_equal(d,before)


def test_visible_sector_can_retreat_when_unrelated_bearings_are_invalid():
    p = Navigator35()
    d = D.copy()
    d[np.abs(A) > math.pi-.08] = np.nan  # Rear unobserved sector remains closed.
    for tick in range(1,6):
        result = call(p,tick=tick,d=d)
    assert p.stats['directional_reobserve_hold'] == p.REOBSERVE_TICKS
    assert p.reason == 'current_sector_monotone_retreat'
    assert result['forward'] > 0 and result['rotation'] == 0
    assert p.status()['scan_status']['partial']
    details = p.status()['recovery_details']
    assert details['new_unobserved_pixels'] == 0 and details['same_hull_kernel']
    # Independently check the complete predicted swept hull against the initial
    # unsafe pixel set. Invalid bearings must not be silently filled with free.
    fine = p._last_local
    known = fine.free & ~fine.static
    kernel = fine.footprint().astype(bool)
    offsets = np.argwhere(kernel)-np.asarray(kernel.shape)//2
    initial = np.asarray(fine.cell(np.zeros(2)))+offsets
    unsafe = set(map(tuple,initial[~known[initial[:,0],initial[:,1]]]))
    predicted = .6*np.array([result['forward'],result['lateral']])
    stopping = predicted*3+predicted*np.linalg.norm(predicted)/1.2
    previous = len(unsafe)
    for point in np.linspace(np.zeros(2),stopping,40):
        cells = np.asarray(fine.cell(point))+offsets
        bad = set(map(tuple,cells[~known[cells[:,0],cells[:,1]]]))
        assert bad <= unsafe and len(bad) <= previous
        previous = len(bad)


def test_all_near_zero_native_noise_never_becomes_a_safe_escape_sector():
    # Observed failure format: finite signed Gaussian-like values around zero,
    # with scattered rays >3 but no rays beyond the physical hull.
    d = 7.5*np.sin(np.arange(181)*2.399)
    p = Navigator35()
    for tick in range(1,p.BLOCKED_LIMIT+1):
        result = call(p,tick=tick,d=d)
        assert not any(result.values())
    assert p.failed and p.stats.get('current_sector_monotone_retreat',0) == 0
    assert p.status()['scan_status']['long_ray_count'] == 0


def test_missing_or_nan_scan_cannot_reuse_previous_clear_frame():
    for invalid in (None,np.full(181,np.nan)):
        p = Navigator35()
        call(p)
        result = call(p,tick=2,d=invalid,velocity=(1.,-.5))
        assert result == dict(forward=-.65,lateral=.325,rotation=0.)
        assert p.reason == 'scan_unavailable' and p._last_local is None


def test_actual_too_narrow_hull_never_passes_despite_known_coarse_route():
    p = Navigator35()
    d = np.minimum(D,10./np.maximum(1e-9,np.abs(np.sin(A))))
    for tick in range(1,8):
        assert not any(call(p,tick=tick,d=d,grid=corridor()).values())
    assert p.stats.get('current_sector_monotone_retreat',0) == 0


def test_stopping_sweep_is_required_even_when_distant_destination_is_clear():
    p = Navigator35(local_factory=StopsBlocked)
    assert not any(call(p).values())
    assert p.reason == 'stopping_sweep_blocked'


def test_no_progress_failure_survives_target_jitter_until_explicit_reset():
    p = Navigator35()
    p.BLOCKED_LIMIT = 6
    blocked = np.full(181,10.)
    for tick in range(1,7):
        call(p,tick=tick,d=blocked,target=(100.+12*tick,0.))
    assert p.failed and p.failure_reason == 'blocked_budget'
    assert not any(call(p,tick=7,target=(-200.,0.)).values())
    assert p.reason == 'rendezvous_navigation_failed'
    p.reset()
    assert call(p,tick=8,target=(-200.,0.))['forward'] < 0


def test_station_wait_resets_progress_clock_instead_of_failing_after_small_departure():
    p = Navigator35()
    p.STAGNATION_LIMIT = 10
    for tick in range(1,30):
        call(p,tick=tick,target=(20.,0.))
        assert p.arrived and not p.failed
    call(p,tick=30,target=(40.,0.))
    assert not p.failed and p.no_progress_ticks == 1


def test_observed_route_copies_points_and_is_idempotent_without_failure_reset():
    p = Navigator35()
    points = np.array([[0.,0.],[50.,0.],[100.,0.]])
    p.set_observed_route(points)
    points[1] = [1000.,1000.]
    np.testing.assert_array_equal(p._observed_route35[1],[50.,0.])
    p._observed_cursor35 = 1
    p.failed = True
    p.set_observed_route([[0.,0.],[50.,0.],[100.,0.]])
    assert p._observed_cursor35 == 1 and p.failed
    remaining = p.remaining_observed_route()
    np.testing.assert_array_equal(remaining,[[50.,0.],[100.,0.]])
    remaining[0][:] = -1000.
    np.testing.assert_array_equal(p._observed_route35[1],[50.,0.])
    p.reset()
    assert not p.failed and not p._observed_route35


def test_recorded_route_does_not_call_global_search_every_control_tick():
    p = Navigator35()
    p.set_observed_route([[0.,0.],[40.,0.],[80.,0.],[120.,0.]])
    def forbidden(*args,**kwargs):
        raise AssertionError('Recorded-route execution must not run global BFS')
    p._search = forbidden
    result = call(p,target=(120.,0.))
    assert result['forward'] > 0
    assert p.planning_mode == 'own_recorded_route_fresh_hull_required'


def test_near_recorded_corner_cannot_be_removed_before_successor_is_visible():
    p = Navigator35(local_factory=Corner)
    p.ARRIVAL_RADIUS = 1.
    p.set_observed_route([[3.,5.],[5.,21.]])
    result = call(p,target=(5.,21.))
    assert p._observed_cursor35 == 0
    assert result['forward'] > 0 and result['lateral'] > 0
    np.testing.assert_allclose(p.waypoint,[3.,5.])


def test_reached_recorded_terminal_switches_to_observed_grid_rendezvous_leg():
    p = Navigator35(local_factory=Corner)
    p.ARRIVAL_RADIUS = 1.
    p.set_observed_route([[0.,0.]])
    result = call(p,target=(5.,21.))
    assert p.stats['observed_terminal_reached_grid_leg'] == 1
    assert not p.remaining_observed_route()
    assert p.planning_mode == 'inflated_known_free'
    assert result['forward'] > 0 and result['lateral'] > 0
    np.testing.assert_allclose(p.waypoint,[3.,5.])
    # Subsequent periodic replanning after cursor exhaustion remains valid.
    call(p,tick=20,target=(5.,21.))
    assert p.planning_mode == 'inflated_known_free'


def test_geometrically_near_unobserved_terminal_does_not_skip_to_grid_leg():
    p = Navigator35(local_factory=Corner)
    p.ARRIVAL_RADIUS = 1.
    p.set_observed_route([[0.,8.]])
    assert not Corner().line_clear(np.zeros(2),np.array([0.,8.]))
    call(p,target=(5.,21.))
    assert p.stats.get('observed_terminal_reached_grid_leg',0) == 0
    assert p._observed_cursor35 == 0 and len(p.remaining_observed_route()) == 1
    assert p.planning_mode == 'own_recorded_route_fresh_hull_required'


def test_unknown_final_gap_is_not_given_an_observed_route_complete_budget():
    p = Navigator35()
    p.set_observed_route([[0.,0.],[40.,0.]])
    estimate = p.estimate_observed_route([0.,0.],[120.,0.])
    assert estimate['unknown_gap'] and estimate['steps'] is None
    assert not estimate['exact'] and estimate['terminal_gap_px'] == 80.


def test_recorded_budget_covers_path_and_corners_and_labels_historical_scope():
    p = Navigator35()
    p.set_observed_route([[0.,0.],[0.,100.],[100.,100.]])
    estimate = p.estimate_observed_route([0.,0.],[100.,100.],speed=2.,reserve_steps=120)
    assert estimate['terminal_covered'] and estimate['length'] == 200.
    assert estimate['corner_count'] == 1 and estimate['steps'] == 226
    assert not estimate['exact'] and estimate['budget_scope']=='own_recorded_route_not_current_clearance'


def test_grid_budget_adds_corners_without_costing_unknown_target_as_exact():
    g = Grid()
    g.obstacle[4:18,17] = 3
    estimate = Navigator35.estimate_route(g,[0.,0.],[120.,0.])
    assert estimate['exact'] and estimate['corner_count'] > 0
    assert estimate['steps'] >= math.ceil(estimate['length']/1.8)+120
    g.free[:,18] = False
    unknown = Navigator35.estimate_route(g,[0.,0.],[200.,0.])
    assert not unknown['exact'] and unknown['steps'] is None and unknown['unknown_gap']


def test_status_and_step_alias_are_stable_json_interface():
    import json
    p = Navigator35()
    call(p)
    assert Navigator35.step is Navigator35.command
    json.dumps(p.status(),allow_nan=False)
