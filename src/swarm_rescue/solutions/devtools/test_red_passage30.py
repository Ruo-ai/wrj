"""Target fidelity and retained-corner regressions; no native flight or truth."""
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.red_passage28 import HullPassage28, PassageMotion28
from swarm_rescue.solutions.red_passage30 import HullPassage30, PassageMotion30


A = np.linspace(-np.pi, np.pi, 181)
D = np.full(181, 230.)
COARSE = SimpleNamespace(visits=np.zeros((1, 1), int), cell=lambda p: (0, 0))


@pytest.mark.parametrize('goal', [[60., 30.], [-65., 40.], [50., -60.]])
def test_route_target_fidelity_with_shared_direction_and_target(goal):
    fine = HullPassage30(D, A)
    shared = np.array(goal)
    before = shared.copy()
    path = fine.route(np.zeros(2), shared, None, target=shared, limit=165)
    np.testing.assert_array_equal(shared, before)
    assert path
    assert np.linalg.norm(path[-1]-before) <= np.sqrt(2)*fine.CELL
    assert all(fine.line_clear(a, b) for a, b in zip(path[:-1], path[1:]))


def test_route_keeps_distinct_inputs_and_shared_base_views_unchanged():
    fine = HullPassage30(D, A)
    base = np.array([[120., 0.], [60., 40.]])
    before = base.copy()
    path = fine.route(np.zeros(2), base[0], None, target=base[1], limit=165)
    np.testing.assert_array_equal(base, before)
    assert np.linalg.norm(path[-1]-before[1]) <= np.sqrt(2)*fine.CELL


def test_readonly_shared_target_is_supported():
    fine = HullPassage30(D, A)
    shared = np.array([75., 25.])
    shared.flags.writeable = False
    path = fine.route(np.zeros(2), shared, None, target=shared, limit=165)
    assert np.linalg.norm(path[-1]-shared) <= np.sqrt(2)*fine.CELL


def test_untargeted_probe_does_not_mutate_preferred_direction():
    fine = HullPassage30(D, A)
    preferred = np.array([80., 20.])
    before = preferred.copy()
    fine.route(np.zeros(2), preferred, COARSE, limit=10)
    np.testing.assert_array_equal(preferred, before)


@pytest.mark.parametrize('scene', ['open', 'narrow', 'wall', 'missing'])
def test_local_geometry_and_hull_clearance_are_unchanged(scene):
    d = D.copy()
    if scene == 'narrow':
        d = np.minimum(d, 21./np.maximum(np.abs(np.sin(A)), 1e-9))
    elif scene == 'wall':
        forward = np.cos(A) > 0
        d[forward] = np.minimum(d[forward], 45./np.cos(A[forward]))
    elif scene == 'missing':
        d[np.abs(A) < .3] = np.nan
    old, new = HullPassage28(d, A), HullPassage30(d, A)
    for attribute in ('free', 'static', 'clearance', 'squeeze'):
        np.testing.assert_array_equal(getattr(old, attribute), getattr(new, attribute))
    np.testing.assert_array_equal(old.footprint(), new.footprint())
    assert old.PLAN_RADIUS == new.PLAN_RADIUS
    assert old.SAFETY_MARGIN == new.SAFETY_MARGIN


class CornerCentreSpace(HullPassage30):
    """Already-inflated synthetic centre-safe L; native route still executes."""
    def __init__(self, *args):
        super().__init__(D, A)
        yy, xx = np.indices(self.shape)
        x = (xx+.5)*self.CELL-self.width/2
        y = self.height/2-(yy+.5)*self.CELL
        safe = (((x >= -6) & (x <= 8) & (y >= -3) & (y <= 3)) |
                ((x >= 3) & (x <= 8) & (y >= -3) & (y <= 50)))
        self.free[:] = safe
        self.static[:] = False
        self.hits[:] = 0
        self._clearance = np.where(safe, 25., 0.)
        self._squeeze = np.zeros(self.shape, bool)


def test_passage_repair_retains_destination_then_moves_toward_required_corner():
    motion = PassageMotion30()
    motion.LOCAL_FACTORY = CornerCentreSpace
    goal = np.array([5., 41.])
    motion.path = [goal.copy()]
    motion.update(np.zeros(3))
    motion.command(D, A, COARSE, np.zeros(2), 0., 0.)
    assert motion.stats['local_replan'] == 1
    np.testing.assert_allclose(motion.path[-1], goal)
    corner = motion.path[0].copy()
    assert 3 < np.linalg.norm(corner) < 6
    fine = CornerCentreSpace()
    assert fine.line_clear(np.zeros(2), corner)
    assert fine.line_clear(corner, goal)
    assert not fine.line_clear(np.zeros(2), goal)
    motion.update(np.zeros(3))
    command = motion.command(D, A, COARSE, np.zeros(2), 0., 0.)
    force = np.array([command['forward'], command['lateral']])
    assert force @ corner > 0
    np.testing.assert_allclose(motion.path[0], corner)
    predicted = .6*force
    stopping = predicted*3+predicted*np.linalg.norm(predicted)/1.2
    assert fine.line_clear(np.zeros(2), stopping)
    assert motion.stats['active_ticks'] == 1


class PreciseSmallCorner:
    """Analytic centre geometry isolates a necessary sub-3px waypoint."""
    PLAN_RADIUS = 20.
    TRACK_RADIUS = 18.5
    def __init__(self, *args):
        pass

    @staticmethod
    def allowed(points):
        x, y = np.asarray(points).T
        return ((x >= -1) & (x <= 2) & (np.abs(y) <= .25)) | ((np.abs(x-1) <= .25) & (y >= -.25) & (y <= 30))

    def sample(self, points):
        return np.where(self.allowed(np.atleast_2d(points)), 25., 0.)

    def line_clear(self, a, b):
        points = np.linspace(a, b, max(2, int(np.linalg.norm(np.asarray(b)-a)/.02)+1))
        return bool(self.allowed(points).all())


def test_required_short_corner_has_positive_motion_and_safe_stopping_sweep():
    motion = PassageMotion30()
    motion.LOCAL_FACTORY = PreciseSmallCorner
    motion.path = [np.array([1., 0.]), np.array([1., 25.])]
    motion.update(np.zeros(3))
    command = motion.command(D, A, COARSE, np.zeros(2), 0., 0.)
    assert command['forward'] > 0 and command['lateral'] == 0
    assert len(motion.path) == 2
    np.testing.assert_array_equal(motion.path[0], [1., 0.])
    predicted = np.array([command['forward'], command['lateral']])*.6
    stopping = predicted*3+predicted*np.linalg.norm(predicted)/1.2
    assert PreciseSmallCorner().line_clear(np.zeros(2), stopping)


def test_visible_successor_allows_near_corner_skip():
    motion = PassageMotion30()
    motion.path = [np.array([1., 0.]), np.array([40., 0.])]
    motion.update(np.zeros(3))
    command = motion.command(D, A, COARSE, np.zeros(2), 0., 0.)
    assert len(motion.path) == 1
    np.testing.assert_array_equal(motion.path[0], [40., 0.])
    assert command['forward'] > 0


def test_changed_obstacle_still_prevents_positive_motion():
    motion = PassageMotion30()
    motion.path = [np.array([40., 0.])]
    motion.update(np.array([1., 0., 0.]))
    motion.repair_tick = motion.tick
    command = motion.command(np.full(len(A), 18.), A, COARSE, np.zeros(2), 0., 0.)
    assert command['forward'] < 0
    assert motion.stats['active_ticks'] == 0


def test_angular_brake_and_existing_exploration_parameters_are_retained():
    old, new = PassageMotion28(), PassageMotion30()
    for motion in (old, new):
        motion.path = [np.array([40., 0.])]
        motion.update(np.array([1., 0., .05]))
    args = (D, A, COARSE, np.zeros(2), 0., 0.)
    assert old.command(*args) == new.command(*args)
    assert new.stats['angular_brake'] == 1
