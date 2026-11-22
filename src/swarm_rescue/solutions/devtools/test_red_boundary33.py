"""Observed-geometry outline fixtures; no map identity, seed or truth files."""
import json

import numpy as np
import pytest

from swarm_rescue.solutions.red_boundary33 import FinalBoundary33
from swarm_rescue.solutions.red_dualmap32 import FinalRenderer32


SIZE = (1200, 900)


def observed_layout():
    mask = np.zeros((300, 400), bool)
    # A translated, partly observed enclosure far from every canvas edge.
    mask[30:33, 40:300] = True
    mask[30:33, 155:215] = False
    mask[250:253, 40:165] = True
    mask[30:253, 40:43] = True
    mask[30:150, 297:300] = True
    # Actual interior evidence distinguishes a whole observed layout from an
    # isolated room outline. These walls are not modified by completion.
    mask[90:93, 80:260] = True
    mask[150:153, 80:200] = True
    mask[60:210, 137:140] = True
    mask[160:225, 217:220] = True
    return mask


def test_empty_and_single_wall_never_produce_a_boundary_from_canvas_size():
    renderer = FinalBoundary33(SIZE)
    assert not renderer.fuse({}).any()
    mask = np.zeros((300, 400), bool)
    mask[90:93, 60:320] = True
    actual = renderer.fuse({0: mask})
    np.testing.assert_array_equal(actual, FinalRenderer32(SIZE).fuse({0: mask}))
    assert not renderer.last_stats['boundary33']['diagnostic']['inferred']
    assert not renderer.last_added_boundary_pixels.any()


def test_four_observed_sides_support_missing_segments_and_only_outline_changes():
    mask = observed_layout()
    renderer = FinalBoundary33(SIZE)
    result = renderer.fuse({0: mask})
    diagnostic = renderer.last_stats['boundary33']['diagnostic']
    assert diagnostic['inferred']
    bounds = diagnostic['chosen']['bounds_pixels']
    np.testing.assert_allclose(bounds, [124, 94, 895, 754], atol=1)
    assert result[94, 480:630].all()  # observed top's missing centre
    assert result[754, 540:860].all()  # partial bottom extended to observed right
    assert result[500:730, 895].all()  # partial right extended to observed bottom
    assert not result[0].any() and not result[-1].any()
    assert not result[:, 0].any() and not result[:, -1].any()
    original = FinalRenderer32(SIZE).fuse({0: mask})
    assert np.all(result[original])
    np.testing.assert_array_equal(result[115:725, 150:870], original[115:725, 150:870])
    assert renderer.last_stats['boundary33']['required_observed_sides'] == 4
    assert not renderer.last_stats['boundary33']['guesses_unobserved_side_position']


def test_absent_fourth_side_is_not_guessed_from_canvas_or_other_edges():
    mask = observed_layout()
    mask[:, 297:300] = False
    renderer = FinalBoundary33(SIZE)
    actual = renderer.fuse({0: mask})
    np.testing.assert_array_equal(actual, FinalRenderer32(SIZE).fuse({0: mask}))
    assert not renderer.last_stats['boundary33']['diagnostic']['inferred']


def test_isolated_room_outline_is_not_declared_whole_map_boundary():
    mask = np.zeros((300, 400), bool)
    mask[80:83, 100:220] = True
    mask[180:183, 100:220] = True
    mask[80:183, 100:103] = True
    mask[80:183, 217:220] = True
    mask[120:140, 217:220] = False
    renderer = FinalBoundary33(SIZE)
    actual = renderer.fuse({0: mask})
    np.testing.assert_array_equal(actual, FinalRenderer32(SIZE).fuse({0: mask}))
    assert not renderer.last_stats['boundary33']['diagnostic']['inferred']
    assert not actual[370:410, 650:660].any()


def test_inner_room_is_rejected_when_other_observed_walls_extend_outside_it():
    mask = observed_layout()
    # Remove the true right side, add a plausible interior rectangle. Its
    # enclosure fails because many other observed wall runs remain outside.
    mask[:, 297:300] = False
    mask[95:98, 170:235] = True
    mask[205:208, 170:235] = True
    mask[95:208, 170:173] = True
    mask[95:208, 232:235] = True
    renderer = FinalBoundary33(SIZE)
    actual = renderer.fuse({0: mask})
    np.testing.assert_array_equal(actual, FinalRenderer32(SIZE).fuse({0: mask}))
    assert not renderer.last_added_boundary_pixels.any()


def test_translation_and_larger_canvas_change_no_inferred_geometry_except_offset():
    original = observed_layout()
    first = FinalBoundary33(SIZE)
    a = first.fuse({0: original})
    shifted = np.pad(original, ((20, 70), (50, 90)))
    second = FinalBoundary33((shifted.shape[1]*3, shifted.shape[0]*3))
    b = second.fuse({0: shifted})
    left = np.asarray(first.last_stats['boundary33']['diagnostic']['chosen']['bounds_pixels'])
    right = np.asarray(second.last_stats['boundary33']['diagnostic']['chosen']['bounds_pixels'])
    np.testing.assert_allclose(right-left, [150, 60, 150, 60], atol=1)
    np.testing.assert_array_equal(b[60:960, 150:1350], a)
    assert not b[:55].any() and not b[:, :145].any()
    assert not second.last_stats['boundary33']['canvas_used_for_geometry_inference']


def test_received_sources_immutable_and_preboundary_ablation_exact():
    mask = observed_layout()
    mask.flags.writeable = False
    before = mask.tobytes()
    renderer = FinalBoundary33(SIZE)
    result = renderer.fuse({4: mask})
    baseline = FinalRenderer32(SIZE).fuse({4: mask})
    np.testing.assert_array_equal(renderer.last_pre_boundary_pixels, baseline)
    np.testing.assert_array_equal(result, baseline | renderer.last_boundary_pixels)
    np.testing.assert_array_equal(renderer.last_added_boundary_pixels, result & ~baseline)
    assert before == mask.tobytes()
    assert not np.shares_memory(renderer.last_pre_boundary_pixels, result)
    assert renderer.last_stats['boundary33']['existing_pixels_removed'] == 0
    assert not renderer.last_stats['navigation_map_modified']
    assert renderer.last_stats['boundary33']['added_pixels'] == int((result & ~baseline).sum())
    json.dumps(renderer.last_stats)


def test_order_independence_and_previous_inference_never_leaks_into_next_map():
    mask = observed_layout()
    second = mask.copy()
    second[190:205, 260:263] = True
    renderer = FinalBoundary33(SIZE)
    a = renderer.fuse({4: second, 0: mask})
    stats = json.dumps(renderer.last_stats, sort_keys=True)
    b = renderer.fuse({0: mask, 4: second})
    np.testing.assert_array_equal(a, b)
    assert stats == json.dumps(renderer.last_stats, sort_keys=True)
    assert not renderer.fuse({}).any()
    assert not renderer.last_boundary_pixels.any()
    assert not renderer.last_stats['boundary33']['diagnostic']['inferred']


def test_invalid_sources_remain_errors():
    renderer = FinalBoundary33(SIZE)
    with pytest.raises(ValueError):
        renderer.fuse({0: observed_layout().astype(np.uint8)})
    with pytest.raises(ValueError):
        renderer.fuse({0: np.zeros((1, 1), bool)})
    with pytest.raises(ValueError):
        renderer.fuse({0: observed_layout()}, {0: 2})
