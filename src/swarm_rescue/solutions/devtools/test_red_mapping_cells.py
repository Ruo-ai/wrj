"""Red-only A4 behavior: radial blind spots, endpoint safety and A3 rollback."""
import json
import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper, MappingSnapshots, free_points
from swarm_rescue.solutions.my_drone_red_mapping_cells import (
    CellSubmissionMapper, cell_free_points, MyDroneRedMappingCells, MyDroneRedMappingCellsOff,
)


def seed_wall(mapper, x):
    index = mapper._indices(np.array([[x, 0.]]))[0]
    mapper.hits.ravel()[index] = 4.
    return index


def test_a3_radial_gap_reproduced_and_a4_clears_with_repeated_observation():
    old, new = SubmissionMapper((400, 400)), CellSubmissionMapper((400, 400))
    gap = [seed_wall(m, 51.) for m in (old, new)]
    endpoint = [seed_wall(m, 100.) for m in (old, new)]
    for _ in range(64):
        for m in (old, new):
            m.update([0., 0.], 0., [0., 0., 0.], [100.], [0.])
    assert old.hits.ravel()[gap[0]] == 4.  # 51px is between 48/57px samples
    assert new.hits.ravel()[gap[1]] == 0.
    assert all(m.mask().ravel()[i] for m, i in zip((old, new), endpoint))


@pytest.mark.parametrize('distance', [31., 100., 230., 500.])
def test_endpoint_guard_range_and_original_samples_are_preserved(distance):
    sparse = free_points([distance], [0.])
    dense = cell_free_points([distance], [0.])
    assert set(map(tuple, sparse)) <= set(map(tuple, dense))
    if len(dense):
        assert np.linalg.norm(dense, axis=1).max() < min(distance, 230)-18
        assert dense[:, 0].min() >= 12


@pytest.mark.parametrize('d,a', [(None, [0]), ([100], None), ([100, 120], [0]),
                                   ([np.nan, -1, np.inf], [0, 0, 0])])
def test_invalid_scan_does_not_clear_unknown_space(d, a):
    assert len(cell_free_points(d, a)) == 0


def test_no_filled_sector_or_negative_votes_multiplied_by_ray_count():
    # Only ray 0 is selected; unmeasured angles between beams stay unknown.
    p = cell_free_points([100., 100., 100.], [0., .4, -.4])
    assert np.all(p[:, 1] == 0.)
    mapper = CellSubmissionMapper((400, 400))
    i = seed_wall(mapper, 51)
    mapper._erase(mapper.hits, np.tile(p, (10, 1)), 1.)
    assert mapper.hits.ravel()[i] == pytest.approx(3.65)


def test_unclosed_outage_cannot_erase_anchored_map_then_relock_can():
    m = CellSubmissionMapper((400, 400))
    m.update([0., 0.], 0., [0., 0., 0.], [100.], [0.])
    i = seed_wall(m, 51.)
    unrelated = seed_wall(m, -150.)
    for _ in range(80):
        m.update(None, None, [0., 0., 0.], [100.], [0.])
    assert m.hits.ravel()[i] == 4 and m.mask().ravel()[i]
    assert len(m.pending) == 20
    m.update([0., 0.], 0., [0., 0., 0.], [100.], [0.])
    assert not m.pending and m.stats['anchored'] == 20
    assert m.hits.ravel()[i] == 0 and m.hits.ravel()[unrelated] == 4
    json.dumps(m.stats, allow_nan=False)


def test_disabled_candidate_identical_and_enabled_pose_does_not_change():
    a3, off, on = (SubmissionMapper((500, 500)), CellSubmissionMapper((500, 500)),
                   CellSubmissionMapper((500, 500)))
    off.DENSE_FREE_RAYS = False
    rays = np.linspace(-.8, .8, 32)
    for step in range(100):
        gps, heading = (None, None) if 15 <= step < 80 else ([step*.1, 0.], .01)
        for m in (a3, off, on):
            m.update(gps, heading, [.1, 0., 0.], np.full(32, 100.), rays)
        np.testing.assert_array_equal(a3.hits, off.hits)
        np.testing.assert_array_equal(a3.mask(), off.mask())
        for m in (off, on):
            np.testing.assert_array_equal(a3.position, m.position)
            assert a3.heading == m.heading and a3.stats == m.stats


def test_corrected_radio_replaces_old_false_wall_with_navigation_unchanged():
    m = CellSubmissionMapper((400, 400))
    tx = MappingSnapshots(m.shape, 2, m)
    rx = MappingSnapshots(m.shape, 2, CellSubmissionMapper((400, 400)))
    i = seed_wall(m, 51.)
    nav = np.zeros(m.shape, bool); nav[10, 10] = True
    tx.publish(0, 1, nav); assert rx.accept(tx.snapshots[0])
    assert rx.submission().ravel()[i]
    for _ in range(64): m.update([0., 0.], 0., [0., 0., 0.], [100.], [0.])
    tx.publish(0, 65, nav); assert rx.accept(tx.snapshots[0])
    assert not rx.submission().ravel()[i]
    np.testing.assert_array_equal(rx.merged(), nav)
    assert not rx.accept((0, 1, tx.snapshots[0][2]))


def test_runtime_switch_is_explicit_and_controller_inherits_a3_motion():
    from swarm_rescue.solutions.my_drone_red_mapping import MyDroneRedMapping
    assert MyDroneRedMappingCells.DENSE_FREE_RAYS is True
    assert MyDroneRedMappingCellsOff.DENSE_FREE_RAYS is False
    for name in ('control', '_navigation', '_deployment', '_submit'):
        assert getattr(MyDroneRedMappingCells, name) is getattr(MyDroneRedMapping, name)
