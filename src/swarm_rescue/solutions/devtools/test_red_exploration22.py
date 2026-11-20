"""Endpoint alias regression, historical route cases and unchanged safety."""
import numpy as np
import pytest

from swarm_rescue.solutions import my_drone_red_exploration22 as candidate
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.my_drone_red_rn7 import LocalPassage, PassageMotion
from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor
from swarm_rescue.solutions.devtools.red_exploration22_replay import load_cases, replay_case, restore


def grid():
    return _M19_SurveyGrid((500, 500), lambda *a, **k: None)


@pytest.fixture(scope='module')
def historical():
    return load_cases()


@pytest.mark.parametrize('index', [0, 1, 2])
def test_real_repair_requests_keep_far_endpoint(historical, index):
    row = replay_case(historical[index])
    assert row['passed']
    assert np.linalg.norm(row['old_target_after']) == pytest.approx(1.)
    assert row['fixed_endpoint_error_px'] <= np.sqrt(2.)


@pytest.mark.parametrize('kind', ['same_array', 'shared_view', 'read_only'])
def test_route_owns_preferred_without_mutating_callers(kind):
    d, a = corridor(55)
    fine = candidate.LocalPassage22(d, a)
    target = np.array([100., 0.])
    preferred = target.view() if kind == 'shared_view' else target
    if kind == 'read_only':
        target.setflags(write=False)
    path = fine.route(np.zeros(2), preferred, None, target=target, limit=165)
    np.testing.assert_array_equal(target, [100., 0.])
    assert path and fine.cell(path[-1]) == fine.cell(target)
    assert all(fine.line_clear(x, y) for x, y in zip(path, path[1:]))


@pytest.mark.parametrize('scenario', ['27px', 'unknown', 'front_wall'])
def test_impossible_target_is_rejected_without_relaxing_clearance(scenario):
    d, a = corridor(27 if scenario == '27px' else 55, end=24 if scenario == 'front_wall' else 280)
    if scenario == 'unknown':
        d[:] = np.nan
    target = np.array([100., 0.])
    for cls in (LocalPassage, candidate.LocalPassage22):
        fine = cls(d, a)
        # Separate old preferred/target to exercise the unchanged safety check.
        assert not fine.route(np.zeros(2), target.copy(), None, target=target.copy(), limit=165)
        assert fine.PLAN_RADIUS == LocalPassage.PLAN_RADIUS
        assert fine.TRACK_RADIUS == LocalPassage.TRACK_RADIUS


def test_unrepaired_commands_and_paths_match_baseline():
    d, a = corridor(55)
    motions = [PassageMotion(), candidate.PassageMotion22()]
    states = []
    for motion in motions:
        motion.tick = 100
        motion.valid_odom = True
        commands = []
        for _ in range(3):
            commands.append(motion.command(d, a, grid(), np.zeros(2), 0., 0.))
            motion.update(np.array([1., 0., 0.]))
        states.append((commands, [p.tolist() for p in motion.path], dict(motion.stats), motion.cooldown))
    assert states[0] == states[1]


def test_no_false_completion_at_near_side_but_actual_goal_still_completes(historical, monkeypatch):
    snapshot = historical[0]
    row = replay_case(snapshot)
    fixed = restore(snapshot, candidate.LocalPassage22)
    monkeypatch.setattr(candidate, 'LocalPassage22', lambda *a, **k: fixed)
    d, a = corridor(55)
    old = PassageMotion()
    new = candidate.PassageMotion22()
    for motion, path in ((old, row['old_path']), (new, row['fixed_path'])):
        motion.path = [np.array(p) for p in path[1:]]
        motion.tick = motion.since = motion.last_progress = 100
        motion.valid_odom = True
        motion.command(d, a, grid(), np.zeros(2), 0., 0.)
    assert old.stats['completed'] == 1
    assert new.stats['completed'] == 0 and new.path
    new.path = [np.array([0.5, 0.5])]
    new.command(d, a, grid(), np.zeros(2), 0., 0.)
    assert new.stats['completed'] == 1 and not new.path


def test_parent_mission_gate_and_simulated_payload_are_inherited():
    # Health, deadline, payload proximity, yielding and submission gating are
    # in inherited navigation, never bypassed by the new local executor.
    assert candidate.MyDroneRedExploration22._navigation is MyDroneRedRepair17._navigation
    assert candidate.MyDroneRedExploration22._deployment is MyDroneRedRepair17._deployment
    d, a = corridor(55)
    motion = candidate.PassageMotion22()
    motion.path = [np.array([80., 0.])]
    motion.valid_odom = True
    assert motion.command(d, a, grid(), np.zeros(2), 0., 0., allow=False) is None
    assert not motion.path and motion.stats['mission_preempted'] == 1
