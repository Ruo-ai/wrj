"""Observed-mask renderer fixtures; no map names, seeds, scores or truth files."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from swarm_rescue.solutions.red_dualmap32 import FinalRenderer32
from swarm_rescue.solutions.my_drone_red_mapping import MappingSnapshots


SIZE = (420, 360)
SHAPE = (120, 140)


def blank():
    return np.zeros(SHAPE, bool)


def wall(row=44):
    mask = blank()
    mask[row-4:row+5, 10:120] = True
    return mask


@pytest.mark.parametrize('width', [7, 9, 11])
@pytest.mark.parametrize('vertical', [False, True])
def test_ordinary_long_wall_has_predeclared_pixel_width(width, vertical):
    mask = wall()
    if vertical:
        mask = np.zeros(SHAPE, bool)
        mask[10:110, 40:49] = True
    renderer = FinalRenderer32(SIZE, wall_width=width)
    pixels = renderer.fuse({0: mask})
    section = pixels[180, :] if vertical else pixels[:, 180]
    assert section.sum() == width
    assert np.all(np.diff(np.flatnonzero(section)) == 1)
    assert renderer.last_stats['ordinary_axial_line_width'] == width
    assert renderer.last_stats['modeled_lines'] > 0
    assert not renderer.last_stats['navigation_map_modified']


def test_multisource_offsets_choose_one_center_without_enveloping_width():
    renderer = FinalRenderer32(SIZE)
    pixels = renderer.fuse({0: wall(42), 1: wall(44), 2: wall(46)})
    section = np.flatnonzero(pixels[:, 180])
    assert len(section) == 11
    assert abs(section.mean()-(44*3+1)) <= .5
    assert any(len(line['sources']) == 3 for line in renderer.last_lines)
    assert all(not line['participating_bands_enveloped'] for line in renderer.last_lines)


@pytest.mark.parametrize('gap', [1, 3, 10])
def test_observed_opening_is_not_bridged_by_hough_or_round_endcaps(gap):
    mask = wall()
    mask[:, 60:60+gap] = False
    renderer = FinalRenderer32(SIZE)
    pixels = renderer.fuse({0: mask, 1: mask.copy()})
    assert pixels[:, :180].any() and pixels[:, 3*(60+gap):].any()
    assert not pixels[:, 180:3*(60+gap)].any()


def test_sparse_near_trunk_brush_is_removed_without_shifting_center():
    mask = wall()
    for col in (23, 39, 55, 71, 87, 103):
        mask[37:40, col] = True
    pixels = FinalRenderer32(SIZE).fuse({0: mask})
    assert not pixels[111:120, 60:330].any()
    assert pixels[:, 180].sum() == 11
    assert abs(np.flatnonzero(pixels[:, 180]).mean()-(44*3+1)) <= .5


def test_isolated_short_wall_and_compact_box_are_preserved_exactly():
    mask = wall()
    mask[75:78, 15:23] = True
    mask[80:91, 60:72] = True
    renderer = FinalRenderer32(SIZE)
    pixels = renderer.fuse({0: mask})
    raw = np.repeat(np.repeat(mask, 3, 0), 3, 1)
    np.testing.assert_array_equal(pixels[210:, :], raw[210:, :])
    assert pixels[:, 180].sum() > 11  # The box is deliberately not thinned.


def test_single_origin_model_does_not_erase_another_origins_short_obstacle():
    main = wall()
    independent = blank()
    independent[35:41, 70:73] = True
    pixels = FinalRenderer32(SIZE).fuse({0: main, 1: independent})
    expanded = np.repeat(np.repeat(independent, 3, 0), 3, 1)
    assert np.all(pixels[expanded])


def test_separate_short_column_near_same_origins_wall_is_not_erased():
    mask = wall()
    mask[34:38, 70:72] = True
    pixels = FinalRenderer32(SIZE).fuse({0: mask})
    assert pixels[102:114, 210:216].all()


def test_duplicate_relays_do_not_add_an_origin_vote():
    mapper = SimpleNamespace(value=wall(42))
    mapper.mask = lambda: mapper.value.copy()
    store = MappingSnapshots(SHAPE, 2, mapper)
    store.publish(0, 10, blank())
    first_snapshot = store.snapshots[0]
    mapper.value = wall(46)
    store.publish(1, 10, blank())
    for _ in range(8):
        assert not store.accept(first_snapshot)
    renderer = FinalRenderer32(SIZE)
    actual = renderer.fuse(store.mapping_masks)
    reference = FinalRenderer32(SIZE).fuse({0: wall(42), 1: wall(46)})
    np.testing.assert_array_equal(actual, reference)
    assert renderer.last_stats['source_count'] == 2
    assert all(len(line['sources']) == len(set(line['sources'])) for line in renderer.last_lines)


def test_near_parallel_walls_of_one_source_remain_distinct_with_open_corridor():
    mask = blank()
    mask[38:41, 10:120] = True
    mask[48:51, 10:120] = True
    renderer = FinalRenderer32(SIZE)
    pixels = renderer.fuse({0: mask})
    section = pixels[:, 180]
    assert not section[126:140].any()
    groups = np.split(np.flatnonzero(section), np.flatnonzero(np.diff(np.flatnonzero(section)) > 1)+1)
    assert len(groups) == 2
    assert [len(g) for g in groups] == [11, 11]


def test_rotated_observed_wall_is_not_forced_to_world_axes():
    mask = blank().astype(np.uint8)
    cv2.line(mask, (20, 25), (110, 85), 1, 7)
    renderer = FinalRenderer32(SIZE)
    pixels = renderer.fuse({0: mask.astype(bool)})
    assert pixels.any() and renderer.last_lines
    expected = np.array([90., 60.])/np.linalg.norm([90., 60.])
    assert max(abs(np.asarray(line['direction'])@expected) for line in renderer.last_lines) > .995


def test_actual_received_masks_are_immutable_and_packet_order_does_not_matter():
    first, second = wall(43), wall(45)
    first.flags.writeable = False
    second.flags.writeable = False
    before = (first.tobytes(), second.tobytes())
    renderer = FinalRenderer32(SIZE)
    a = renderer.fuse({9: second, 2: first})
    first_stats = json.dumps(renderer.last_stats, sort_keys=True)
    b = renderer.fuse({2: first, 9: second})
    np.testing.assert_array_equal(a, b)
    assert before == (first.tobytes(), second.tobytes())
    assert first_stats == json.dumps(renderer.last_stats, sort_keys=True)
    json.dumps(renderer.last_lines)


def test_empty_input_and_invalid_masks_are_explicit():
    renderer = FinalRenderer32(SIZE)
    assert not renderer.fuse({}).any()
    with pytest.raises(ValueError):
        renderer.fuse({0: blank().astype(np.uint8)})
    with pytest.raises(ValueError):
        renderer.fuse({0: np.zeros((1, 1), bool)})
    with pytest.raises(ValueError):
        renderer.fuse({0: blank()}, {0: 4})
    with pytest.raises(ValueError):
        FinalRenderer32(SIZE, wall_width=13)
