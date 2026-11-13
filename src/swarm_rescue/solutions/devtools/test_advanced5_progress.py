import pytest as _retired_pytest
_retired_pytest.importorskip('swarm_rescue.solutions.my_drone_red_advanced5_progress', reason='Historical strategy retired by user; source archived')
"""Second-pass behavior checks, using real monitor time series."""
from types import SimpleNamespace
import numpy as np
from swarm_rescue.solutions.my_drone_red_advanced5_progress import PersistentProgressMonitor, MyDroneRedAdvanced5Progress
from swarm_rescue.solutions import my_drone_red_advanced5_narrow as baseline


def test_intermittent_wait_cannot_erase_oscillation():
    monitor = PersistentProgressMonitor()
    for step in range(181):
        state = monitor.update(step, np.array([10*(-1)**(step//6), 0.]), [(100., 0.)], 0,
                               'existing_yield' if step % 20 < 8 else None)
    assert len(monitor.samples) == 21
    assert state['wait_budget_used'] > 30
    assert state['confirmed']


def test_short_wait_keeps_history_and_protects_motion():
    monitor = PersistentProgressMonitor()
    for step in range(60):
        monitor.update(step, np.zeros(2), [(100., 0.)], 0)
    count = len(monitor.samples)
    state = monitor.update(60, np.zeros(2), [(100., 0.)], 0, 'existing_yield')
    assert len(monitor.samples) >= count
    assert state['kind'] == 'waiting' and not state['confirmed']


def test_genuine_displacement_releases_wait_budget():
    monitor = PersistentProgressMonitor()
    for step in range(100):
        monitor.update(step, np.zeros(2), [(200., 0.)], 0, 'existing_yield')
    monitor.update(100, np.array([70., 0.]), [(200., 0.)], 0)
    state = monitor.update(101, np.array([70., 0.]), [(200., 0.)], 0, 'existing_yield')
    assert state['kind'] == 'waiting' and state['wait_budget_used'] <= 1


def test_information_alone_does_not_prove_recovery():
    events = []
    drone = SimpleNamespace(_steps=90, _position=np.zeros(2), _loop_information=100,
        _loop_failures=[], _event=lambda kind, **data: events.append(data),
        _loop_recovery=dict(start=0, position=np.zeros(2), information=0,
                            reason='test', direction=0., attempt=1, goal=None))
    MyDroneRedAdvanced5Progress._verify_loop_recovery(drone)
    assert not events[0]['successful'] and len(drone._loop_failures) == 1


def test_useful_stationary_observation_is_still_allowed():
    monitor = PersistentProgressMonitor()
    for step in range(180):
        state = monitor.update(step, np.zeros(2), [(100., 0.)], step)
    assert state['kind'] == 'normal'


def test_gathering_preserves_original_recovery_gate_and_restores_monitor():
    drone = object.__new__(MyDroneRedAdvanced5Progress)
    drone._gathering = True
    drone._bomb_target = None
    drone._attempt_step = None
    drone._holding = False
    drone._steps = 120
    drone._escape_until = 0
    drone._loop_suppressed_at = 120
    current = PersistentProgressMonitor()
    current.last.update(kind='stationary', confirmed=True, age=120)
    drone._progress_monitor = current
    drone._protected_progress = baseline.ProgressMonitor()
    drone._protected_progress.last.update(kind='waiting')
    assert not drone._allow_recovery('no_progress')
    assert drone._progress_monitor is current
    drone._gathering = False
    assert drone._allow_recovery('no_progress')

