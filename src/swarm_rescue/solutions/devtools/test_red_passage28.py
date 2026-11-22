"""Analytic wall/unknown-space tests for fixed-hull passage safety."""
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_rescue.solutions.red_passage28 import HullPassage28, PassageMotion28
from swarm_rescue.solutions.my_drone_red_rn7 import LocalPassage, HULL


ANGLES = np.linspace(-np.pi, np.pi, 181)


def corridor(width, orientation=0.):
    return np.minimum(230., width/2 / np.maximum(np.abs(np.sin(ANGLES-orientation)), 1e-9))


@pytest.mark.parametrize('width', [40., 42.])
def test_observed_safe_gap_rejected_by_old_circle_is_passable(width):
    distances = corridor(width)
    assert not LocalPassage(distances, ANGLES).line_clear([0, 0], [80, 0])
    assert HullPassage28(distances, ANGLES).line_clear([0, 0], [80, 0])


@pytest.mark.parametrize('orientation', [0., np.pi/4, np.pi/2, 3*np.pi/4])
def test_gap_smaller_than_actual_projected_hull_plus_margin_rejected(orientation):
    normal = np.array([-np.sin(orientation), np.cos(orientation)])
    projected_width = np.ptp(HULL@normal)
    fine = HullPassage28(corridor(projected_width+3., orientation), ANGLES)
    target = 60*np.array([np.cos(orientation), np.sin(orientation)])
    assert not fine.line_clear([0, 0], target)


@pytest.mark.parametrize('orientation', [0., np.pi/4, np.pi/2, 3*np.pi/4])
def test_wide_gap_in_all_directions_passes(orientation):
    target = 60*np.array([np.cos(orientation), np.sin(orientation)])
    assert HullPassage28(corridor(70., orientation), ANGLES).line_clear([0, 0], target)


def test_missing_forward_rays_never_create_free_passage():
    d = corridor(50.)
    d[np.abs(ANGLES) < .3] = np.nan
    assert not HullPassage28(d, ANGLES).line_clear([0, 0], [80, 0])


def test_occluding_endpoint_blocks_swept_body_before_target():
    d = corridor(50.)
    forward = np.cos(ANGLES) > 0
    d[forward] = np.minimum(d[forward], 45/np.cos(ANGLES[forward]))
    fine = HullPassage28(d, ANGLES)
    assert not fine.line_clear([0, 0], [55, 0])


def test_gap_visibility_does_not_extend_past_lidar_range():
    assert not HullPassage28(corridor(100.), ANGLES).line_clear([0, 0], [260, 0])


def test_motion_does_not_use_fixed_hull_during_rotation():
    motion = PassageMotion28()
    motion.path = [np.array([40., 0.])]
    motion.update(np.array([1., 0., .05]))
    command = motion.command(corridor(50.), ANGLES, None, np.zeros(2), 0., 0.)
    assert command['rotation'] == 0.
    assert command['forward'] < 0.
    assert motion.stats['angular_brake'] == 1


def test_new_obstacle_brakes_existing_passage():
    motion = PassageMotion28()
    motion.path = [np.array([40., 0.])]
    motion.update(np.array([1., 0., 0.]))
    motion.repair_tick = motion.tick
    command = motion.command(np.full(len(ANGLES), 18.), ANGLES, None, np.zeros(2), 0., 0.)
    assert command['forward'] < 0.
    assert motion.stats['active_ticks'] == 0


def test_remote_inputs_are_not_mutated():
    d = corridor(50.)
    a = ANGLES.copy()
    before = (d.copy(), a.copy())
    fine = HullPassage28(d, a)
    fine.line_clear([0, 0], [80, 0])
    assert np.array_equal(d, before[0]) and np.array_equal(a, before[1])
