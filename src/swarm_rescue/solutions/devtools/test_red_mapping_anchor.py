"""A5 boundary behavior and rollback; synthetic inputs, no map-name tuning."""
import json
import math
import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper, MyDroneRedMapping, transform, wrap
from swarm_rescue.solutions.my_drone_red_mapping_anchor import (
    StableAnchorMapper, MyDroneRedMappingAnchor, MyDroneRedMappingAnchorOff,
    fit_recovery, absolute_pose_available,
)
from swarm_rescue.solutions.devtools.red_anchor_fixture import inputs, run_case


def tick(m, gps=None, compass=None):
    m.update(gps, compass, [0., 0., 0.], [100.], [0.])


def prepared():
    m = StableAnchorMapper((500, 500)); m.RECORD_ANCHORS = True
    tick(m, [0., 0.], 0.)
    for _ in range(80): tick(m)
    return m


def test_motion_and_turn_compensated_fit_rejects_first_outlier():
    origin = np.array([7., -9.]); translation = np.array([5., -3.]); angle = .16
    rows = []
    for i in range(12):
        position = origin+np.array([2*i, .2*i*i]); heading = wrap(3.05+i*.04)
        gps = origin+transform((position-origin)[None], np.zeros(2), angle)[0]+translation
        compass = wrap(heading+angle)
        if i == 0: gps += [35, -40]; compass = wrap(compass+.4)
        rows.append((position, heading, gps, compass))
    correction, diag = fit_recovery(rows, origin)
    np.testing.assert_allclose(correction[0], translation, atol=1e-9)
    assert correction[1] == pytest.approx(angle) and diag['inliers'] == 11
    assert np.linalg.norm(np.mean([r[2] for r in rows], axis=0)-rows[-1][2]) > 10


def test_circular_fit_does_not_average_across_pi():
    rows = [(np.zeros(2), 0., np.ones(2), math.pi+s*.01) for s in [-1, 1]*6]
    correction, diag = fit_recovery(rows, np.zeros(2))
    assert abs(abs(correction[1])-math.pi) < 1e-9 and diag['inliers'] == 12


def test_consistent_bias_is_not_misrepresented_as_absolute_accuracy():
    correction, diag = fit_recovery([(np.zeros(2), 0., np.array([15., 0.]), .08)]*12, np.zeros(2))
    assert diag['reason'] == 'accepted'
    np.testing.assert_allclose(correction[0], [15., 0.])  # cannot discover hidden bias
    assert correction[1] == pytest.approx(.08)


def test_first_return_is_provisional_then_clean_window_commits_once():
    m = prepared(); unrelated = m._indices(np.array([[-180., -180.]]))[0]
    m.hits.ravel()[unrelated] = 9.
    before = m.hits.copy(); initial_step = m.step
    tick(m, [30., -12.], .2)
    assert m.pending and m.missing and m.stats['relocks'] == 0
    np.testing.assert_array_equal(m.hits, before)
    for _ in range(11): tick(m, [0., 0.], 0.)
    assert m.step == initial_step+12 and not m.pending and m.missing == 0
    assert m.stats['anchor_commits'] == m.stats['relocks'] == 1
    assert m.stats['anchored'] == 23 and m.stats['max_recovery_samples'] == 12
    np.testing.assert_allclose(m.position, [0., 0.], atol=1e-9)
    assert m.hits.ravel()[unrelated] == 9.
    json.dumps(list(m.anchor_events), allow_nan=False)


def test_interrupted_window_restarts_without_discarding_scans():
    m = prepared()
    for _ in range(7): tick(m, [0., 0.], 0.)
    saved = len(m.pending); tick(m)
    assert not m.recovery_samples and len(m.pending) >= saved
    assert m.stats['anchor_interruptions'] == 1
    for _ in range(11): tick(m, [0., 0.], 0.)
    assert m.missing and not m.stats['anchor_commits']
    tick(m, [0., 0.], 0.)
    assert m.stats['anchor_commits'] == 1 and not m.pending


def test_inconsistent_recovery_is_bounded_then_original_a3_fallback():
    m = prepared()
    for i in range(48): tick(m, [40.*(1 if i % 2 else -1), 0.], .3 if i % 2 else -.3)
    assert m.stats['anchor_rejections'] > 0 and m.missing and m.pending
    tick(m, [7., 0.], .01)
    assert m.stats['anchor_fallbacks'] == 1 and m.stats['anchor_commits'] == 0
    np.testing.assert_array_equal(m.position, [7., 0.])
    assert not m.pending and not m.missing and m.stats['relocks'] == 1


def test_long_outage_keeps_mapping_and_buffer_is_bounded():
    m = prepared(); old = m.hits.copy()
    for _ in range(280): tick(m)
    assert len(m.pending) == 90 and m.mask().any()
    np.testing.assert_array_equal(m.hits, old)
    # Lower the existing cap only in the fixture to cheaply exercise eviction.
    m.MAX_PENDING = 91
    for _ in range(40): tick(m)
    assert len(m.pending) == 91 and m.stats['dropped'] == 9
    assert StableAnchorMapper.MAX_PENDING == SubmissionMapper.MAX_PENDING == 1400


def test_event_logging_is_optional_and_bounded():
    m = StableAnchorMapper((30, 30))
    m._record('disabled'); assert not m.anchor_events
    m.RECORD_ANCHORS = True
    for i in range(100): m._record('fixture', sample=i)
    assert len(m.anchor_events) == 32 and m.anchor_events[0]['sample'] == 68


def test_disabled_is_exact_a3_on_identical_complete_sensor_inputs():
    a3, off = SubmissionMapper((400, 400)), StableAnchorMapper((400, 400))
    off.STABLE_ANCHOR = False
    rows, _, _ = inputs('turn_first_outlier')
    for row in rows:
        a3.update(*row); off.update(*row)
        np.testing.assert_array_equal(a3.hits, off.hits)
        np.testing.assert_array_equal(a3.mask(), off.mask())
        np.testing.assert_array_equal(a3.position, off.position)
        assert a3.heading == off.heading and a3.step == off.step
        assert a3.stats == {k: off.stats[k] for k in a3.stats}


def test_no_outage_has_identical_map_and_pose():
    a3, on = SubmissionMapper((400, 400)), StableAnchorMapper((400, 400))
    for step in range(80):
        for m in (a3, on): m.update([step*.1, 0.], .01, [.1, 0., 0.], [100.], [0.])
    np.testing.assert_array_equal(a3.hits, on.hits)
    np.testing.assert_array_equal(a3.position, on.position)
    assert a3.heading == on.heading and on.stats['anchor_windows'] == 0


@pytest.mark.parametrize('gps,compass', [(None, 0.), ([0., 0.], None), ([np.nan, 0.], 0.),
                                       ([0., 0.], np.inf), ([1.], 0.), ([0., 0.], [0.])])
def test_invalid_absolute_observation_cannot_start_recovery(gps, compass):
    assert not absolute_pose_available(gps, compass)
    m = prepared(); tick(m, gps, compass)
    assert not m.recovery_samples and m.stats['anchor_windows'] == 0


@pytest.mark.parametrize('case', ['first_outlier', 'turn_first_outlier'])
def test_identical_fixture_reduces_first_frame_error_and_false_map(case):
    from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer
    a3, pred3, r3 = run_case(case, SubmissionMapper)
    a5, pred5, r5 = run_case(case, StableAnchorMapper)
    assert r3['input_sha256'] == r5['input_sha256']
    assert r5['endpoint_error_px'] < r3['endpoint_error_px']
    truth = np.zeros((400, 400), np.uint8); truth[130:271, 290:296] = 1
    s3, s5 = [MapExplorationScorer().score(truth, pred, 0.) for pred in (pred3, pred5)]
    assert s5.charged_pixels < s3.charged_pixels


@pytest.mark.xfail(strict=True, reason='Unmet performance gate: thin endpoint raster loses wall-thickness hits despite lower pose error')
@pytest.mark.parametrize('case', ['first_outlier', 'turn_first_outlier'])
def test_unmet_map_credit_adoption_gate(case):
    # Preserve the original failed hypothesis explicitly, rather than calling
    # lower pose error a map-score success. This gate MUST block adoption.
    from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer
    truth = np.zeros((400, 400), np.uint8); truth[130:271, 290:296] = 1
    scores = [MapExplorationScorer().score(truth, run_case(case, cls)[1], 0.)
              for cls in (SubmissionMapper, StableAnchorMapper)]
    s3, s5 = scores
    assert s5.wall.hits >= s3.wall.hits


def test_motion_placement_radio_and_submission_stay_inherited():
    assert MyDroneRedMappingAnchor.STABLE_ANCHOR and not MyDroneRedMappingAnchorOff.STABLE_ANCHOR
    for name in ('control', '_navigation', '_deployment', '_submit', '_update_observation'):
        assert getattr(MyDroneRedMappingAnchor, name) is getattr(MyDroneRedMapping, name)
