"""A3 behavioral tests: honest observations, bounded outage repair, radio epochs."""
import json
from types import SimpleNamespace
import numpy as np
from swarm_rescue.solutions.my_drone_red_mapping import (
    SubmissionMapper, MappingSnapshots, wall_points, anchor_frames, free_points,
)


def test_max_range_invalid_and_dynamic_hits_never_become_walls():
    assert not len(wall_points([230, 230, np.nan], [0, 1, 2]))
    entity = SimpleNamespace(distance=100., angle=0.)
    assert not len(wall_points([100., 100., 100.], [-.05, 0., .05], [entity]))
    assert not len(wall_points([100.], [0., 1.]))


def test_interpolation_never_bridges_missing_rays_or_doorway():
    points = wall_points([100, 230, 100], [-.1, 0., .1])
    assert len(points) == 2  # no fabricated wall across the middle unknown ray
    points = wall_points([100, 160], [0., .01])
    assert len(points) == 2  # depth discontinuity stays open


def test_anchor_fixes_only_buffered_segment_and_wraps_heading():
    local = np.array([[10., 0.]])
    frames = [(0, local, np.array([0., 0.]), 3.1, .2),
              (5, local, np.array([50., 10.]), 3.1, .2),
              (10, local, np.array([100., 20.]), 3.1, .2)]
    out = anchor_frames(frames, (0, np.zeros(2)), frames[-1][2], 3.1,
                        np.array([100., 0.]), -3.1, 10)
    np.testing.assert_allclose(out[0][2], [0, 0])
    np.testing.assert_allclose(out[-1][2], [100, 0], atol=1e-9)
    assert abs(out[-1][3]+3.1) < 1e-9
    np.testing.assert_array_equal(frames[-1][2], [100, 20])


def test_outage_scans_are_kept_then_reanchored_without_shifting_old_map():
    mapper = SubmissionMapper((600, 600))
    d = np.full(24, 80.); a = np.linspace(-.8, .8, 24)
    for _ in range(12):
        mapper.update([0., 0.], 0., [0., 0., 0.], d, a)
    unrelated = mapper._indices(np.array([[-200., -200.]]))[0]
    mapper.hits.ravel()[unrelated] = 9.
    before = mapper.hits.copy()
    for _ in range(280):
        mapper.update(None, None, [0., 0., 0.], d, a)
    assert len(mapper.pending) == 70 and mapper.stats['buffered'] == 70
    np.testing.assert_array_equal(before, mapper.hits)
    mapper.update([3., 0.], 0., [0., 0., 0.], d, a)
    assert not mapper.pending and mapper.stats['anchored'] == 70
    # Old unrelated evidence is not globally transformed by a recovered pose.
    assert mapper.hits.ravel()[unrelated] == 9. and mapper.stats['relocks'] == 1
    json.dumps(mapper.stats, allow_nan=False)


def test_bad_reference_can_restart_without_claiming_absolute_lock():
    mapper = SubmissionMapper((400, 400))
    mapper.update([0., 0.], 0., [0., 0., 0.], None, None)
    for _ in range(80):
        mapper.update(None, None, [0., 0., 0.], [230.], [0.])
    assert mapper.stats['reference_resets'] >= 2
    assert mapper.missing == 80 and mapper.last_match == -100
    assert not mapper.mask().any()


def test_radio_replaces_corrected_mapping_and_preserves_navigation():
    mapper = SubmissionMapper((30, 30))
    sender = MappingSnapshots(mapper.shape, 2, mapper)
    receiver = MappingSnapshots(mapper.shape, 2, SubmissionMapper((30, 30)))
    nav = np.zeros(mapper.shape, bool); nav[1, 1] = True
    mapper.hits[3, 4] = 5
    sender.publish(0, 40, nav)
    assert receiver.accept(sender.snapshots[0])
    np.testing.assert_array_equal(receiver.merged(), nav)
    assert receiver.submission()[3, 4]
    mapper.hits[3, 4] = 0; mapper.hits[5, 6] = 5
    sender.publish(0, 80, nav)
    assert receiver.accept(sender.snapshots[0])
    assert not receiver.submission()[3, 4] and receiver.submission()[5, 6]
    assert not receiver.accept(sender.snapshots[0])
    assert not receiver.accept((0, 81, b'invalid'))
    assert not receiver.accept((9, 81, sender.snapshots[0][2]))


def test_map_requires_multiple_scan_support_and_stays_within_shape():
    mapper = SubmissionMapper((31, 29))
    p = np.array([[0., 0.], [0., 0.], [10000., 0.]])
    mapper._insert(mapper.hits, p, 1.)
    assert not mapper.mask().any()  # repeated rays in one scan are one vote
    for _ in range(2): mapper._insert(mapper.hits, p, 1.)
    assert mapper.mask().sum() == 1
    assert mapper.to_pixels(mapper.mask()).shape == (29, 31)


def test_free_beams_remove_repeatedly_contradicted_wall_but_not_endpoint():
    mapper = SubmissionMapper((400, 400))
    obstacles = np.array([[48., 0.], [100., 0.]])
    for _ in range(4): mapper._insert(mapper.hits, obstacles, 1.)
    free = free_points([100.], [0.])
    assert np.max(np.linalg.norm(free, axis=1)) < 82
    for _ in range(15): mapper._erase(mapper.hits, free, 1.)
    false_wall, real_wall = mapper._indices(obstacles)
    assert not mapper.mask().ravel()[false_wall]
    assert mapper.mask().ravel()[real_wall]


def test_unanchored_free_rays_do_not_erase_confirmed_geometry():
    mapper = SubmissionMapper((400, 400))
    point = np.array([[48., 0.]])
    for _ in range(4): mapper._insert(mapper.hits, point, 1.)
    empty = np.empty((0, 2), np.float32)
    mapper.pending = [(i, empty, np.zeros(2), 0., .45, free_points([100.], [0.])) for i in range(40)]
    assert mapper.mask().ravel()[mapper._indices(point)[0]]
