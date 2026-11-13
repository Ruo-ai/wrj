import pytest as _retired_pytest
_retired_pytest.importorskip('swarm_rescue.solutions.my_drone_red_advanced3_verified', reason='Historical strategy retired by user; source archived')
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

# Original behavioural checks: test_red_advanced3_verified
"""Only a semantic contradiction may replace the stage2 target."""
from types import SimpleNamespace as _test_red_advanced3_verified_SimpleNamespace
import numpy as _test_red_advanced3_verified_np
from swarm_rescue.solutions import my_drone_red_advanced3_verified as _test_red_advanced3_verified_module
from swarm_rescue.solutions.my_drone_red_advanced3_verified import _M16_PlacementPlan as _test_red_advanced3_verified_PlacementPlan, _M16_CoverageField as _test_red_advanced3_verified_CoverageField
from swarm_rescue.solutions.my_drone_red_advanced3_verified import _M20_SurveyGrid as _test_red_advanced3_verified_SurveyGrid

def _test_red_advanced3_verified_make_field():
    mask = _test_red_advanced3_verified_np.zeros((100, 100), bool)
    mask[30, 10:40] = True
    mask[70, 60:90] = True
    nonwall = _test_red_advanced3_verified_np.zeros_like(mask, dtype=_test_red_advanced3_verified_np.uint8)
    return _test_red_advanced3_verified_module._M18_VerifiedField(mask, 4, (400, 400), [], [], 48, nonwall)

def test_test_red_advanced3_verified_valid_plan_returns_same_object_without_second_search(monkeypatch):
    field = _test_red_advanced3_verified_make_field()
    plan = _test_red_advanced3_verified_PlacementPlan([(10, 10)], _test_red_advanced3_verified_np.zeros(2), 100.0)
    calls = []
    monkeypatch.setattr(_test_red_advanced3_verified_module, '_M17_choose_placement', lambda *a: calls.append(1) or plan)
    assert _test_red_advanced3_verified_module._M18_choose_placement(None, _test_red_advanced3_verified_np.zeros(2), field) is plan
    assert calls == [1]

def test_test_red_advanced3_verified_semantic_false_wall_forces_new_measured_wall_target():
    field = _test_red_advanced3_verified_make_field()
    field.observed_nonwall[30, 10:40] = 2
    grid = _test_red_advanced3_verified_SurveyGrid((400, 400), None)
    grid.free[:] = True
    position = _test_red_advanced3_verified_np.array([-100.0, 80.0])
    old = _test_red_advanced3_verified_module._M17_choose_placement(grid, position, field)
    new = _test_red_advanced3_verified_module._M18_choose_placement(grid, position, field)
    assert new is not None and _test_red_advanced3_verified_np.linalg.norm(new.target - old.target) > 80
    assert new.gain == field.value(new.target)
    assert field.observed_walls[30, 20]

def test_test_red_advanced3_verified_release_field_is_identical_to_original_even_with_semantic_metadata():
    field = _test_red_advanced3_verified_make_field()
    field.observed_nonwall[30, 10:40] = 2
    old = _test_red_advanced3_verified_CoverageField(field.observed_walls, 4, (400, 400), [], [], 48)
    _test_red_advanced3_verified_np.testing.assert_array_equal(field.weight, old.weight)
    _test_red_advanced3_verified_np.testing.assert_array_equal(field.gain, old.gain)

# Original behavioural checks: test_red_advanced4
"""Behavioural regression tests for bounded coordination and passage geometry."""
import numpy as _test_red_advanced4_np
import pytest as _test_red_advanced4_pytest
from swarm_rescue.solutions.my_drone_red_advanced4 import _M22_TaskBoard as _test_red_advanced4_TaskBoard, _M22_CooperativeGrid as _test_red_advanced4_CooperativeGrid
from swarm_rescue.solutions.my_drone_red_advanced4 import _M23_FinePassages as _test_red_advanced4_FinePassages, _M23_passage_velocity as _test_red_advanced4_passage_velocity

def test_test_red_advanced4_task_logging_uses_parent_event_signature_and_does_not_repeat():
    from types import SimpleNamespace
    from swarm_rescue.solutions.my_drone_red_advanced4 import MyDroneRedAdvanced4Explore as MyDroneRedAdvanced4Explore
    events = []
    drone = SimpleNamespace(_passage_route=[], _bomb_target=None, _route=[], _gathering=False, _last_task=None, _task_changes=0, _in_narrow=False, carried_bombs_count=lambda : 2, _event=lambda kind, **data: events.append((kind, data)))
    MyDroneRedAdvanced4Explore._record_task(drone)
    MyDroneRedAdvanced4Explore._record_task(drone)
    assert len(events) == 1 and events[0][1]['task_kind'] == 'idle'

def test_test_red_advanced4_relay_does_not_renew_expired_ownership():
    owner = _test_red_advanced4_TaskBoard(0, 3, (600, 400))
    owner.publish(10, 'explore', (120.0, 0.0), (0.0, 0.0), 2)
    relay = _test_red_advanced4_TaskBoard(1, 3, (600, 400))
    receiver = _test_red_advanced4_TaskBoard(2, 3, (600, 400))
    relay.accept(owner.snapshot(), 20)
    receiver.accept(relay.snapshot(), 80)
    assert len(receiver.active(80)) == 1
    receiver.accept(relay.snapshot(), 200)
    assert receiver.active(200) == []
    assert receiver.records[0][1] == 10

@_test_red_advanced4_pytest.mark.parametrize('change', [{0: 8}, {1: 101}, {3: (float('nan'), 0.0)}, {3: (10000.0, 0.0)}, {5: -1}, {6: 10000}, {7: (((0.0, 0.0), float('inf')),)}, {2: 'unknown-task'}])
def test_test_red_advanced4_malformed_task_does_not_replace_good_status(change):
    origin = _test_red_advanced4_TaskBoard(0, 2, (400, 400))
    good = origin.publish(20, 'explore', (80.0, 0.0), (0.0, 0.0), 2)
    receiver = _test_red_advanced4_TaskBoard(1, 2, (400, 400))
    receiver.accept((good,), 20)
    bad = list(good)
    bad[1] = 21
    for (index, value) in change.items():
        bad[index] = value
    receiver.accept((tuple(bad),), 100)
    assert receiver.records[0] == good

def test_test_red_advanced4_no_inventory_teammate_can_explore_but_cannot_win_placement():
    board = _test_red_advanced4_TaskBoard(0, 2, (400, 400))
    peer = _test_red_advanced4_TaskBoard(1, 2, (400, 400))
    peer.publish(10, 'explore', (100.0, 0.0), (90.0, 0.0), 0)
    board.accept(peer.snapshot(), 10)
    assert board.owner(_test_red_advanced4_np.array([100.0, 0.0]), _test_red_advanced4_np.array([0.0, 0.0]), 10) == 1
    assert board.owner(_test_red_advanced4_np.array([100.0, 0.0]), _test_red_advanced4_np.array([0.0, 0.0]), 10, True) == 0

def test_test_red_advanced4_radio_free_evidence_is_not_republished_as_local_observation():
    sender = _test_red_advanced4_CooperativeGrid((400, 400), lambda *a, **kw: None, _test_red_advanced4_TaskBoard(0, 2, (400, 400)))
    receiver = _test_red_advanced4_CooperativeGrid((400, 400), lambda *a, **kw: None, _test_red_advanced4_TaskBoard(1, 2, (400, 400)))
    sender.local_free[2, 3] = True
    receiver.accept(sender.local_snapshot(0, 10), 10)
    assert receiver.free[2, 3]
    receiver.local_snapshot(1, 11)
    own = receiver.sources[1]
    bits = _test_red_advanced4_np.unpackbits(_test_red_advanced4_np.frombuffer(own[2], _test_red_advanced4_np.uint8))[:400].reshape(20, 20)
    assert not bits[2, 3]
    assert receiver.sources[0][1] == 10

def test_test_red_advanced4_local_wall_needs_repeated_evidence_and_blocks_shared_free():
    grid = _test_red_advanced4_CooperativeGrid((200, 200), lambda *a, **kw: None, _test_red_advanced4_TaskBoard(0, 2, (200, 200)))
    for _ in range(3):
        grid.observe(_test_red_advanced4_np.zeros(2), 0.0, _test_red_advanced4_np.array([50.0]), _test_red_advanced4_np.array([0.0]))
    wall = grid.cell((50.0, 0.0))
    assert grid.local_wall[wall]
    assert not grid.allowed()[wall]

@_test_red_advanced4_pytest.mark.parametrize('width,expected', [(15, False), (20, False), (30, True), (44, True), (60, True)])
def test_test_red_advanced4_fine_route_respects_body_clearance(width, expected):
    fine = _test_red_advanced4_FinePassages((400, 200))
    (rows, cols) = _test_red_advanced4_np.indices(fine.shape)
    y = fine.height / 2 - (rows + 0.5) * fine.CELL
    fine.free[_test_red_advanced4_np.abs(y) < width / 2] = True
    path = fine.route(_test_red_advanced4_np.array([-80.0, 0.0]), _test_red_advanced4_np.array([80.0, 0.0]))
    assert bool(path) == expected
    if path:
        assert all(fine.valid_points(_test_red_advanced4_np.asarray(path)))
        assert _test_red_advanced4_np.linalg.norm(path[-1] - [80.0, 0.0]) < 5

def test_test_red_advanced4_visible_wall_breaks_fine_corridor_route():
    fine = _test_red_advanced4_FinePassages((400, 200))
    fine.free[16:24, :] = True
    fine.hits[:, 40] = 3
    assert fine.route(_test_red_advanced4_np.array([-80.0, 0.0]), _test_red_advanced4_np.array([80.0, 0.0])) == []

def test_test_red_advanced4_passage_steering_brakes_when_lidar_unavailable():
    (force, allowed) = _test_red_advanced4_passage_velocity(_test_red_advanced4_np.array([_test_red_advanced4_np.nan]), _test_red_advanced4_np.array([0.0]), 0, _test_red_advanced4_np.array([2.0, 0.5]))
    assert not allowed and force[0] < 0 and (force[1] < 0)

def test_test_red_advanced4_narrow_mode_requires_two_observed_sides():
    from swarm_rescue.solutions.my_drone_red_advanced4 import _M23_corridor_width as corridor_width
    angles = _test_red_advanced4_np.linspace(-_test_red_advanced4_np.pi, _test_red_advanced4_np.pi, 180, endpoint=False)
    distances = _test_red_advanced4_np.minimum(300, 22 / _test_red_advanced4_np.maximum(1e-06, _test_red_advanced4_np.abs(_test_red_advanced4_np.sin(angles))))
    assert abs(corridor_width(distances, angles, 0) - 44) < 1e-06
    distances[_test_red_advanced4_np.sin(angles) < 0] = 300
    assert corridor_width(distances, angles, 0) == float('inf')

def test_test_red_advanced4_narrow_recovery_follows_corridor_instead_of_turning_into_wall():
    from swarm_rescue.solutions.my_drone_red_advanced4 import _M23_corridor_heading as corridor_heading
    angles = _test_red_advanced4_np.linspace(-_test_red_advanced4_np.pi, _test_red_advanced4_np.pi, 180, endpoint=False)
    distances = _test_red_advanced4_np.minimum(300, 22 / _test_red_advanced4_np.maximum(1e-06, _test_red_advanced4_np.abs(_test_red_advanced4_np.sin(angles))))
    heading = corridor_heading(distances, angles, _test_red_advanced4_np.pi / 2, 0.0)
    assert heading is not None and abs(heading) < 0.2

def test_test_red_advanced4_fully_observed_narrow_corridor_can_be_used_for_return():
    fine = _test_red_advanced4_FinePassages((400, 200))
    fine.free[16:24, :] = True
    fine.free[:, 5:15] = True
    path = fine.route(_test_red_advanced4_np.array([50.0, 0.0]), exit_toward=_test_red_advanced4_np.array([-160.0, 0.0]))
    assert path and path[-1][0] < 0

@_test_red_advanced4_pytest.mark.parametrize('distance', [15, 20, 22, 30])
def test_test_red_advanced4_right_angle_corner_is_not_a_parallel_narrow_corridor(distance):
    from swarm_rescue.solutions.my_drone_red_advanced4 import _M23_corridor_heading as corridor_heading
    angles = _test_red_advanced4_np.linspace(-_test_red_advanced4_np.pi, _test_red_advanced4_np.pi, 180, endpoint=False)
    x = _test_red_advanced4_np.where(_test_red_advanced4_np.cos(angles) > 0, distance / _test_red_advanced4_np.maximum(_test_red_advanced4_np.cos(angles), 1e-09), 300)
    y = _test_red_advanced4_np.where(_test_red_advanced4_np.sin(angles) > 0, distance / _test_red_advanced4_np.maximum(_test_red_advanced4_np.sin(angles), 1e-09), 300)
    assert corridor_heading(_test_red_advanced4_np.minimum(300, _test_red_advanced4_np.minimum(x, y)), angles, _test_red_advanced4_np.pi, None) is None

def test_test_red_advanced4_narrow_movement_does_not_block_inventory_confirmation_or_late_deployment(monkeypatch):
    from types import SimpleNamespace
    from swarm_rescue.solutions.my_drone_red_advanced4 import _M20_MyDroneRedAdvanced2 as MyDroneRedAdvanced2
    from swarm_rescue.solutions.my_drone_red_advanced4 import MyDroneRedAdvanced4Explore as MyDroneRedAdvanced4Explore
    calls = []
    monkeypatch.setattr(MyDroneRedAdvanced2, '_deployment', lambda self, command: calls.append(command))
    drone = object.__new__(MyDroneRedAdvanced4Explore)
    drone._passage_route = []
    drone._in_narrow = True
    drone._attempt_step = None
    drone._last_inventory = 2
    drone.carried_bombs_count = lambda : 2
    drone._wall_progress = lambda : 0.4
    drone._progress = lambda : 0.4
    drone._deployment({})
    assert not calls
    drone.carried_bombs_count = lambda : 1
    drone._deployment({})
    assert len(calls) == 1
    drone._last_inventory = 1
    drone._progress = lambda : 0.69
    drone._deployment({})
    assert len(calls) == 2

def test_test_red_advanced4_old_candidate_age_is_not_reset_by_status_republication():
    source = _test_red_advanced4_TaskBoard(0, 2, (400, 400))
    target = _test_red_advanced4_TaskBoard(1, 2, (400, 400))
    source.publish(1000, 'explore', (80.0, 0.0), (0.0, 0.0), 2, [((50.0, 0.0), 10.0, 1)])
    target.accept(source.snapshot(), 1000)
    assert target.candidates(1000) == []

def test_test_red_advanced4_corridor_commitment_survives_one_missing_scan_then_expires():
    from swarm_rescue.solutions.my_drone_red_advanced4 import _M23_CorridorCommitment as CorridorCommitment
    state = CorridorCommitment()
    assert state.update(None, 0.0, 1) is None
    assert state.update(0.0, 0.0, 2) == 0.0
    assert state.update(None, 0.1, 3) == _test_red_advanced4_pytest.approx(-0.1)
    assert state.update(None, 0.1, 15) is None

def test_test_red_advanced4_corridor_does_not_flip_direction_on_one_frame_goal_change():
    from swarm_rescue.solutions.my_drone_red_advanced4 import _M23_CorridorCommitment as CorridorCommitment
    state = CorridorCommitment()
    assert state.update(0.0, 0.0, 2) == 0.0
    assert state.update(_test_red_advanced4_np.pi, 0.0, 3) == 0.0
    assert abs(state.update(_test_red_advanced4_np.pi, 0.0, 20)) == _test_red_advanced4_pytest.approx(_test_red_advanced4_np.pi)

def test_test_red_advanced4_peer_claim_reassigns_destination_without_blocking_common_trunk():
    size = (800, 400)
    board = _test_red_advanced4_TaskBoard(0, 2, size)
    grid = _test_red_advanced4_CooperativeGrid(size, lambda *a, **kw: None, board)
    grid.free[10:, :] = True
    position = _test_red_advanced4_np.array([0.0, -100.0])
    first = grid.plan(position, position, 0, 2, [])
    target = grid.point(first[-1])
    peer = _test_red_advanced4_TaskBoard(1, 2, size)
    peer.publish(100, 'explore', tuple(target), tuple(target), 2)
    board.accept(peer.snapshot(), 100)
    grid.step = 100
    second = grid.plan(position, position, 0, 2, [])
    assert _test_red_advanced4_np.linalg.norm(grid.point(second[-1]) - target) >= 100
    assert grid.allowed()[grid.cell(position)]

