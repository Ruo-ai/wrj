"""Independent mission, native-message and frozen-baseline contracts for 34."""
import ast
from collections import deque
import inspect
import textwrap
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_advanced2 import _M14_MyDroneRedV4Progress
from swarm_rescue.solutions.my_drone_red_dualmap32 import MyDroneRedDualMap32Clean
from swarm_rescue.solutions.my_drone_red_frontier34 import (
    FrontierSurvey34, MyDroneRedCleanBoundary34, MyDroneRedFrontier34, valid_report34,
)
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
from swarm_rescue.solutions.red_boundary33 import FinalBoundary33
from swarm_rescue.solutions.devtools.test_red_dualmap32_controller import fixture as prior_fixture
from swarm_rescue.solutions.devtools.test_red_fusion27_delivery import fleet, store, deliver


def fixture(source=0, snapshots=None, progress=.9, sealed=True, policy=False):
    snapshots = snapshots if snapshots is not None else store(source, frozen=sealed)[0]
    prior = prior_fixture(source, snapshots=snapshots, messages=True)
    cls = type('FrontierFixture34', (MyDroneRedFrontier34,),
               {'identifier': source, 'drone_health': 25})
    d = object.__new__(cls)
    d.__dict__.update(prior.__dict__)
    d._progress = lambda: progress
    d._home = np.zeros(2)
    d._heading = 0.
    d._velocity = np.zeros(2)
    d._gathering = False
    d._holding = False
    d._attempt_step = None
    d._last_route_step = 0
    d._misc_data = SimpleNamespace(max_timestep_limit=5400, max_walltime_limit=180.)
    d.carried_bombs_count = lambda: 0
    d._f34_init()
    d._explore.last_frontier34 = None
    d._explore.support_mode34 = False
    d._explore.radio_claims34 = []
    if policy:
        d._explore = FrontierSurvey34((600, 600), d._event)
        d._steps = 3001
        d.elapsed_timestep = 3002
        d.elapsed_walltime = 80.
        d._position = np.array([160., 100.])
        d._f34_budget = dict(required_steps=400., arrival_remaining_steps=1800.)
        d._f34_budget_update = lambda: None
        d._f34_samples = deque([(2820+6*i, 100, 50) for i in range(31)], maxlen=41)
        d._route = [(8, 8), (8, 9)]
    d._f34_report()
    return d


def exchange(drones, rounds=1, edges=None):
    for _ in range(rounds):
        packets = [d.define_message_for_all() for d in drones]
        for i, d in enumerate(drones):
            d.communicator.received_messages = [
                (j, packet) for j, packet in enumerate(packets)
                if i != j and (edges is None or (j, i) in edges)]
            d._receive()
        for d in drones:
            d._steps += 1
            d.elapsed_timestep += 1
            d._radio_maps._fixture_clock[0] = d.elapsed_timestep


def test_control_arm_retains_clean_flight_radio_observation_and_clock():
    for name in ('_navigation', '_navigate_motion', '_deployment', '_progress',
                 '_update_observation', '_try_hold', 'control', '_receive',
                 'define_message_for_all', '_exchange_state'):
        assert getattr(MyDroneRedCleanBoundary34, name) is getattr(MyDroneRedDualMap32Clean, name)
    for name in ('GATHER_AT', 'FREEZE_AT', 'DEADLINE_AT', 'NARROW_ENABLED'):
        assert getattr(MyDroneRedCleanBoundary34, name) == getattr(MyDroneRedDualMap32Clean, name)


def test_control_constructor_only_replaces_submission_renderer(monkeypatch):
    marker = object()
    def old_init(self, **kwargs):
        self._fixture_marker = marker
        self._fusion27 = object()
    monkeypatch.setattr(MyDroneRedDualMap32Clean, '__init__', old_init)
    cls = type('BoundaryConstructorFixture', (MyDroneRedCleanBoundary34,), {'size_area': (90, 60)})
    d = cls()
    assert d._fixture_marker is marker and isinstance(d._fusion27, FinalBoundary33)
    assert set(d.__dict__) == {'_fixture_marker', '_fusion27'}


def test_motion_copy_removes_only_fixed_72_percent_gather_gate():
    original = ast.parse(textwrap.dedent(inspect.getsource(_M14_MyDroneRedV4Progress._navigate_motion)))
    candidate = ast.parse(textwrap.dedent(inspect.getsource(MyDroneRedFrontier34._navigate_motion)))
    body = original.body[0].body
    gates = [node for node in body if isinstance(node, ast.If)
             and '_progress' in ast.unparse(node.test) and '0.72' in ast.unparse(node.test)]
    assert len(gates) == 1
    body.remove(gates[0])
    assert ast.dump(original, include_attributes=False) == ast.dump(candidate, include_attributes=False)
    assert MyDroneRedFrontier34._progress is MyDroneRedDualMap32Clean._progress


def test_productive_deep_exploration_keeps_current_mission():
    d = fixture(sealed=False, policy=True)
    d._f34_samples[-1] = (3000, 128, 61)
    before = list(d._route)
    d._f34_policy()
    assert d._f34_mode == 'explore' and d._route == before
    assert not d._gathering and not d._explore.support_mode34
    assert d._d32_stats['local_complete_support'] == 0


def test_local_stagnation_enables_support_without_fixed_55_percent_election():
    d = fixture(sealed=False, policy=True, progress=.4)
    d._f34_policy()
    assert d._f34_mode == 'support' and d._explore.support_mode34
    assert not d._gathering and d._route == []
    assert d._d32_stats['local_complete_support'] == 1


def test_support_returns_to_normal_exploration_when_observations_resume():
    d = fixture(sealed=False, policy=True)
    d._f34_mode = 'support'
    d._explore.support_mode34 = True
    d._f34_samples[-1] = (3000, 120, 54)
    d._f34_policy()
    assert d._f34_mode == 'explore' and not d._explore.support_mode34


@pytest.mark.parametrize('pending,inventory', [(None, 1), (2999, 0)])
def test_deployment_and_confirmation_are_not_preempted_by_support(pending, inventory):
    d = fixture(sealed=False, policy=True)
    d._attempt_step = pending
    d.carried_bombs_count = lambda: inventory
    d._f34_policy()
    assert d._f34_mode == 'explore'


def test_final_epoch_preempts_support_without_waiting_for_old_task_budget():
    d = fixture(policy=True)
    d._f34_mode = 'support'
    d._explore.support_mode34 = True
    d._f34_policy()
    assert d._f34_mode == 'return' and d._gathering
    assert d._f34_return_reason == 'final_epoch_preemption'
    assert not d._explore.support_mode34 and not d._route


def test_route_exchange_budget_preempts_before_sealing():
    d = fixture(sealed=False, policy=True)
    d._f34_budget.update(arrival_remaining_steps=399., required_steps=400.)
    d._f34_policy()
    assert d._f34_mode == 'return'
    assert d._f34_return_reason == 'route_and_exchange_budget'


def test_remaining_budget_obeys_both_native_step_and_wall_limits():
    d = fixture()
    d.elapsed_timestep, d.elapsed_walltime = 3000, 160.
    left, values = d._f34_remaining(.94)
    assert values['step_remaining'] == pytest.approx(2076.)
    assert left == pytest.approx((.94*180-160)*25)
    d.elapsed_timestep, d.elapsed_walltime = 5000, 100.
    assert d._f34_remaining(.94)[0] == pytest.approx(76.)
    d.elapsed_walltime = 171.
    assert d._f34_remaining(.94)[0] < 0


def test_budget_includes_travel_recovery_exchange_and_is_cached():
    d = fixture(sealed=False)
    d._position = np.array([100., 0.])
    d._home = np.zeros(2)
    d._f34_crumbs = [np.zeros(2), np.array([50., 0.])]
    calls = []
    d._f34_nav = SimpleNamespace(estimate_route=lambda *a, **kw:
        calls.append(kw) or dict(exact=True, length=100.))
    d._f34_budget_update()
    budget = dict(d._f34_budget)
    assert budget['required_steps'] == pytest.approx(100/1.8 + 160 + 180)
    d._f34_budget_update()
    assert len(calls) == 1


def test_support_goal_is_observed_reachable_and_avoids_fresh_claim():
    grid = FrontierSurvey34((600, 600), lambda *a, **kw: None)
    grid.support_mode34 = True
    grid.free[7:24, 7:24] = True
    start = np.asarray(grid.point((15, 15)))
    route = grid.plan(start, start, 0, 5, [])
    assert route and grid.last_frontier34['unknown'] >= 3
    first_goal = grid.point(route[-1]).copy()
    grid.radio_claims34 = [first_goal]
    alternate = grid.plan(start, start, 0, 5, [])
    assert alternate and np.linalg.norm(grid.point(alternate[-1])-first_goal) >= 110
    assert all(grid.free[cell] and grid.obstacle[cell] < 2 for cell in alternate)
    assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(alternate, alternate[1:]))
    wall = cv2.dilate((grid.obstacle >= 2).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))).astype(bool)
    unknown = cv2.filter2D((~grid.free & ~wall).astype(np.float32), -1,
                          np.ones((5, 5), np.float32), borderType=cv2.BORDER_CONSTANT)
    assert unknown[alternate[-1]] >= 3


def test_real_five_drone_nested_radio_delivers_bytes_manifests_and_full_ack():
    stores = fleet()
    drones = [fixture(i, snapshots=stores[i]) for i in range(5)]
    exchange(drones, 24)
    for d in drones:
        state = d._exchange_state()
        assert state['source_maps_complete'] and state['final_complete']
        assert state['acknowledged_sources'] == list(range(5))
        assert state['received_versions'] == {str(i): 4643 for i in range(5)}
        for i in range(5):
            assert d._radio_maps.snapshots[i] == stores[i].own_final
    drones[0]._submit()
    assert len(drones[0]._fixture_submissions) == 1
    assert drones[0]._s28_submission['final_complete']


def test_store_and_forward_across_actual_chain_delivers_final_set():
    drones = [fixture(i) for i in range(5)]
    edges = {(i,j) for i in range(5) for j in range(5) if abs(i-j) == 1}
    exchange(drones, 90, edges)
    assert all(d._exchange_state()['final_complete'] for d in drones)
    assert 4 not in drones[0]._peers and 4 not in drones[0]._f34_direct
    assert drones[0]._radio_maps.snapshots[4] == drones[4]._radio_maps.own_final


def test_partition_metadata_and_forged_ack_cannot_replace_missing_payload():
    d = fixture(0)
    remote = fixture(1)
    packet = remote.define_message_for_all()
    claim = list(remote._f34_reports[1])
    claim[4] = (4643,)*5
    claim[5] = tuple(range(5))
    claim[6] = tuple(range(5))
    d.communicator.received_messages = [(1, ('frontier34-v1', packet[1], (tuple(claim),)))]
    d._receive()
    assert not d._exchange_state()['source_maps_complete']
    d._s28_receipts = {i:(i,4644,'a'*64) for i in range(5)}
    d._submit()
    assert not d._fixture_submissions
    assert not d._exchange_state()['final_complete']


def test_forwarded_report_retains_origin_age_and_does_not_create_direct_peer():
    a, b, c = [fixture(i) for i in range(3)]
    a._steps = b._steps = c._steps = 4700
    a.elapsed_timestep = b.elapsed_timestep = c.elapsed_timestep = 4701
    b.communicator.received_messages = [(0,a.define_message_for_all())]
    b._receive()
    b._steps = c._steps = 4720
    b.elapsed_timestep = c.elapsed_timestep = 4721
    c.communicator.received_messages = [(1,b.define_message_for_all())]
    c._receive()
    assert c._f34_reports[0][1] == 4700
    assert 0 not in c._peers and 0 not in c._f34_direct
    assert c._peers[1][0] == 4720


def test_invalid_or_future_frontier_report_is_rejected():
    d = fixture()
    row = d._f34_reports[0]
    assert valid_report34(row, 5, d.elapsed_timestep)
    assert not valid_report34((0,d.elapsed_timestep+1,*row[2:]), 5, d.elapsed_timestep)
    assert not valid_report34((0,row[1],(float('nan'),0.),*row[3:]), 5, d.elapsed_timestep)
    assert not valid_report34((9,*row[1:]), 5, d.elapsed_timestep)


def test_normal_navigation_preserves_parent_command_and_does_not_force_gather(monkeypatch):
    d = fixture(sealed=False, policy=True, progress=.75)
    d._f34_samples[-1] = (3000, 130, 65)
    original = dict(forward=.5,lateral=-.3,rotation=.1,place_bomb=0)
    monkeypatch.setattr(MyDroneRedRepair17,'_navigation',lambda self: original.copy())
    assert d._navigation() == original
    assert d._f34_mode == 'explore' and not d._gathering


def test_return_override_runs_after_parent_observation_updates(monkeypatch):
    d = fixture(policy=True)
    calls = []
    def parent(self):
        calls.append('parent')
        self._velocity = np.array([.2,.3])
        return dict(forward=.5,lateral=.5,rotation=.1,place_bomb=0)
    def return_motion():
        calls.append('return')
        np.testing.assert_array_equal(d._velocity,[.2,.3])
        return dict(forward=-.1,lateral=.2,rotation=0.)
    monkeypatch.setattr(MyDroneRedRepair17,'_navigation',parent)
    d._f34_return_motion = return_motion
    assert d._navigation() == dict(forward=-.1,lateral=.2,rotation=0.,place_bomb=0)
    assert calls == ['parent','return']


def test_received_complete_sources_still_wait_for_matching_all_source_ack():
    stores = fleet()
    for remote in stores[1:]:
        deliver(stores[0], remote)
    d = fixture(snapshots=stores[0])
    assert d._exchange_state()['source_maps_complete']
    assert not d._exchange_state()['final_complete']
    d._submit()
    assert not d._fixture_submissions


def test_candidate_does_not_change_sealing_deployment_or_native_deadlines():
    for name in ('_update_observation','_deployment','FREEZE_AT','DEADLINE_AT','_exchange_state'):
        assert getattr(MyDroneRedFrontier34,name) is getattr(MyDroneRedDualMap32Clean,name)


def test_exact_route_budget_does_not_use_shorter_unverified_breadcrumb_shortcut():
    d = fixture(sealed=False)
    d._position = np.array([100., 0.])
    d._home = np.zeros(2)
    d._f34_crumbs = [np.zeros(2)]
    d._f34_nav = SimpleNamespace(estimate_route=lambda *a, **kw: dict(exact=True, length=500.))
    d._f34_budget_update()
    assert d._f34_budget['length_px'] == 500.
    assert d._f34_budget['required_steps'] == pytest.approx(500/1.8 + 160 + 180)


def test_breadcrumb_loop_removal_requires_current_observed_clear_connection():
    d = fixture(sealed=False, policy=True)
    points = [np.array([0.,0.]), np.array([100.,0.]), np.array([100.,100.]),
              np.array([0.,100.]), np.array([-100.,0.])]
    d._position = np.array([10.,0.])
    d._f34_crumbs = [p.copy() for p in points]
    d._explore.line_clear = lambda *a: False
    d._f34_trace()
    assert len(d._f34_crumbs) == 6
    d._f34_crumbs = [p.copy() for p in points]
    d._explore.line_clear = lambda *a: True
    d._f34_trace()
    assert len(d._f34_crumbs) == 1


def motion_fixture():
    d = fixture(policy=True)
    d._f34_mode = 'return'
    d._position = np.array([50.,0.])
    d._home = np.zeros(2)
    d._f34_safe_station = lambda *a, **kw: None
    d.semantic_values = lambda: ()
    d.lidar_values = lambda: np.full(181,200.)
    d.lidar_rays_angles = lambda: np.linspace(-np.pi,np.pi,181)
    calls = []
    def reset():
        calls.append('reset')
        d._f34_nav.failed = False
    d._f34_nav = SimpleNamespace(failed=False,target=np.zeros(2),
        estimate_route=lambda *a, **kw: calls.append('estimate') or dict(exact=True,length=50.),
        command=lambda *a: calls.append('command') or dict(forward=.2,lateral=0.,rotation=0.),
        reset=reset,status=lambda: {})
    return d,calls


def test_exchange_station_requires_fresh_clearance_and_peer_separation():
    d,calls = motion_fixture()
    assert d._f34_return_motion() == dict(forward=0.,lateral=0.,rotation=0.)
    assert d._f34_mode == 'exchange' and calls == []
    d.lidar_values = lambda: np.full(181,20.)
    assert d._f34_return_motion()['forward'] == .2
    assert d._f34_mode == 'return' and 'command' in calls
    d.lidar_values = lambda: np.full(181,200.)
    d._peers = {1:(d._steps,np.array([55.,0.]),None)}
    assert d._f34_return_motion()['forward'] == .2
    assert d._f34_mode == 'return'


def test_return_estimate_cache_avoids_per_step_full_grid_search():
    d,calls = motion_fixture()
    d._position = np.array([250.,0.])
    for _ in range(3):
        d._f34_return_motion()
        d._steps += 1
    assert calls.count('estimate') == 1
    assert calls.count('command') == 3


def test_exhausted_breadcrumb_failure_has_bounded_retry_not_permanent_latch():
    d,calls = motion_fixture()
    d._position = np.array([250.,0.])
    d._f34_nav.failed = True
    d._f34_return_motion()
    assert calls.count('reset') == 1
    assert d._f34_retry_after == d._steps+60
    assert d._d32_stats['exhausted_trace_rescan_retry'] == 1
    d._steps += 1
    d._f34_nav.failed = True
    d._f34_return_motion()
    assert calls.count('reset') == 1


def test_failed_station_search_is_cached_during_near_home_congestion():
    d,calls = motion_fixture()
    d.lidar_values = lambda: np.full(181,20.)
    probes = []
    d._f34_safe_station = lambda *a, **kw: probes.append(d._steps) or None
    for _ in range(30):
        d._f34_return_motion()
        d._steps += 1
    assert len(probes) == 1
    d._f34_return_motion()
    assert len(probes) == 2
