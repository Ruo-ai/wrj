"""Fixed startup deadline, unchanged match clock and exact launch-window tests."""
import ast
import inspect
import math
import textwrap

import numpy as np
import pytest

from swarm_rescue.solutions import my_drone_red_stability25 as candidate
from swarm_rescue.solutions.my_drone_red_stability25 import (
    MyDroneRedStability25, StartupAlignment25, TargetMotion25,
)
from swarm_rescue.solutions import my_drone_red_advanced2 as baseline
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.devtools.test_red_stability24 import navigation_fixture


ANGLES = np.linspace(-math.pi, math.pi, 181)
RAYS = np.full(181, 200.)


@pytest.mark.parametrize('yaw', np.linspace(-math.pi, math.pi, 9)[:-1])
def test_all_yaws_wait_until_step96_even_when_alignment_finishes_early(yaw):
    gate = StartupAlignment25()
    delta = 0.
    for step in range(1, gate.RELEASE_STEP + 1):
        cmd = gate.command(step, yaw, yaw, [0., 0., delta], RAYS, ANGLES)
        if step < gate.RELEASE_STEP:
            assert cmd is not None and not gate.released
            assert cmd['forward'] == cmd['lateral'] == 0.
            delta = cmd['rotation'] * .18
            yaw = baseline._M01_wrap_angle(yaw + delta)
        else:
            assert cmd is None and gate.released
    assert gate.reason == 'aligned'
    assert gate.release_step == 96 and gate.ticks == 96
    assert gate.stats['preparation_ticks'] == 95
    assert gate.aligned_step < 96 and abs(yaw) < gate.HOLD_TOLERANCE
    assert gate.command(97, yaw, yaw, [0., 0., 0.], RAYS, ANGLES) is None
    assert gate.ticks == 96


@pytest.mark.parametrize('heading,compass,odom,rays,reason', [
    (None, 0., [0., 0., 0.], RAYS, 'sensor_fallback'),
    (0., None, [0., 0., 0.], RAYS, 'sensor_fallback'),
    (0., 0., None, RAYS, 'sensor_fallback'),
    (1., 1., [0., 0., 0.], np.full(181, 20.), 'clearance_fallback'),
    (1., 1., [0., 0., 0.], np.full(181, np.nan), 'clearance_fallback'),
])
def test_safety_fallback_cannot_release_early(heading, compass, odom, rays, reason):
    gate = StartupAlignment25()
    for step in range(1, 96):
        cmd = gate.command(step, heading, compass, odom, rays, ANGLES)
        assert cmd is not None and cmd['rotation'] == 0.
        assert not gate.released
    assert gate.command(96, heading, compass, odom, rays, ANGLES) is None
    assert gate.reason == reason and gate.release_step == 96


def test_moving_drone_brakes_before_turning_and_budget_is_fixed():
    gate = StartupAlignment25()
    for step in range(1, 96):
        cmd = gate.command(step, 2., 2., [2., 0., 0.], RAYS, ANGLES)
        assert cmd['forward'] < 0. and cmd['rotation'] == 0.
    assert gate.command(96, 2., 2., [2., 0., 0.], RAYS, ANGLES) is None
    assert gate.reason == 'budget_fallback'


def test_alignment_hysteresis_does_not_turn_for_small_hold_noise():
    gate = StartupAlignment25()
    for step in range(1, 5):
        gate.command(step, 0., 0., [0., 0., 0.], RAYS, ANGLES)
    assert gate.settled and gate.aligned_step == 4
    assert gate.command(5, .07, .07, [0., 0., 0.], RAYS, ANGLES)['rotation'] == 0.
    assert gate.command(6, .11, .11, [0., 0., 0.], RAYS, ANGLES)['rotation'] < 0.
    assert not gate.settled and gate.stats['alignment_rechecks'] == 1


def test_transient_blockage_rechecks_new_scan_without_lowering_clearance():
    gate = StartupAlignment25()
    for step in range(1, 8):
        cmd = gate.command(step, 1., 1., [0., 0., 0.], np.full(181, 21.9), ANGLES)
        assert cmd['rotation'] == 0.
    cmd = gate.command(8, 1., 1., [0., 0., 0.], RAYS, ANGLES)
    assert cmd['rotation'] < 0. and gate.fallback_reason is None
    assert not gate.released
    assert gate.stats['fresh_observation_rechecks'] == 7


def test_transient_sensor_loss_can_recover_before_the_fixed_deadline():
    gate = StartupAlignment25()
    assert gate.command(1, None, None, None, RAYS, ANGLES)['rotation'] == 0.
    assert gate.command(2, 1., 1., [0., 0., 0.], RAYS, ANGLES)['rotation'] < 0.
    assert gate.fallback_reason is None and not gate.released


def test_velocity_filter_rotates_previous_body_frame_before_blending():
    gate = StartupAlignment25()
    first = gate.command(1, 0., 0., [2., 0., 0.], RAYS, ANGLES)
    assert first['forward'] < 0. and first['rotation'] == 0.
    second = gate.command(2, 0., 0., [0., 0., math.pi/2], RAYS, ANGLES)
    np.testing.assert_allclose(gate.velocity, [0., -.5], atol=1e-12)
    assert abs(second['forward']) < 1e-12 and second['lateral'] > 0.


def test_launch_windows_keep_original_44_and_64_control_ticks():
    drone = object.__new__(MyDroneRedStability25)
    active = {45: [], 65: []}
    for step in range(96, 200):
        drone._steps = step
        age = drone._launch_age25()
        assert age == step - 96 + 1
        for boundary in active:
            if age < boundary:
                active[boundary].append(step)
    assert active[45] == list(range(96, 140))
    assert active[65] == list(range(96, 160))
    assert 96 % 6 == 0  # First inherited navigation observes before planning.


def test_motion_copy_changes_exactly_two_launch_ages_and_uses_old_helper():
    original = ast.parse(textwrap.dedent(inspect.getsource(
        baseline._M14_MyDroneRedV4Progress._navigate_motion)))
    copied = ast.parse(textwrap.dedent(inspect.getsource(TargetMotion25._navigate_motion)))

    class RestoreAge(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == '_launch_age25'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'):
                assert not node.args and not node.keywords
                self.count += 1
                return ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                     attr='_steps', ctx=ast.Load())
            return self.generic_visit(node)

    restore = RestoreAge()
    restored = restore.visit(copied)
    assert restore.count == 2
    assert ast.dump(restored) == ast.dump(original)
    assert candidate._M11_safe_velocity is baseline._M11_safe_velocity
    assert tuple(cls for cls in MyDroneRedStability25.__mro__[1:] if cls is not TargetMotion25) == MyDroneRedRepair17.__mro__
    for name in ('control', '_update_observation', '_deployment', '_progress', '_submit'):
        assert getattr(MyDroneRedStability25, name) is getattr(MyDroneRedRepair17, name)


def test_gate_releases_to_original_wrappers_without_modifying_match_time(monkeypatch):
    drone = navigation_fixture(MyDroneRedStability25)
    drone._startup25 = StartupAlignment25()
    drone._startup25_reported = False
    drone.measured_compass_angle = lambda: 0.
    events = []
    drone._event = lambda *args, **kwargs: events.append((args, kwargs))
    inherited_calls = []

    def inherited_navigation(self):
        inherited_calls.append(self._steps)
        return {'sentinel': 'original_navigation'}

    monkeypatch.setattr(MyDroneRedRepair17, '_navigation', inherited_navigation)
    for step in range(1, 98):
        drone._steps = step
        drone.elapsed_timestep = 1000 + step
        drone.elapsed_walltime = 17.25 + step
        cmd = MyDroneRedStability25._navigation(drone)
        assert drone._steps == step
        assert drone.elapsed_timestep == 1000 + step
        assert drone.elapsed_walltime == 17.25 + step
        if step < 96:
            assert 'sentinel' not in cmd
        else:
            assert cmd == {'sentinel': 'original_navigation'}
    assert inherited_calls == [96, 97]
    assert len(events) == 1 and events[0][1]['release_step'] == 96


def test_real_advanced2_super_dispatch_reaches_epoch_hook_and_old_helper(monkeypatch):
    drone = navigation_fixture(MyDroneRedStability25)
    drone._steps = 97  # Avoid unrelated periodic grid refresh in this minimal fixture.
    calls = []
    old_helper = candidate._M11_safe_velocity

    def spy(*args, **kwargs):
        calls.append(args[2])
        return old_helper(*args, **kwargs)

    monkeypatch.setattr(candidate, '_M11_safe_velocity', spy)
    assert super(baseline.MyDroneRedAdvanced2, drone)._navigate_motion.__func__ is TargetMotion25._navigate_motion
    baseline.MyDroneRedAdvanced2._navigation(drone)
    assert len(calls) == 1
