"""Exercise real A5 mapper wiring after the controller constructor assigns flags."""
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_mapping import MyDroneRedMapping
from swarm_rescue.solutions.my_drone_red_mapping_anchor import MyDroneRedMappingAnchor
from swarm_rescue.solutions.devtools.red_anchor_event_batch import anchor_event_recording
from swarm_rescue.solutions.devtools.red_anchor_fixture import inputs


def light_parent(self, **kwargs):
    # Avoid engine graphics; run the actual A5 constructor and mapper unchanged.
    self._size_area = (400, 400)
    self._map = SimpleNamespace(shape=(134, 134))
    self._n = 1


def test_controller_construction_preserves_requested_logging_and_mapping(monkeypatch):
    monkeypatch.setattr(MyDroneRedMapping, '__init__', light_parent)
    maps = []
    for enabled in (False, True):
        with anchor_event_recording(enabled):
            mapper = MyDroneRedMappingAnchor()._submission_mapper
        assert mapper.RECORD_ANCHORS is enabled
        for row in inputs('first_outlier')[0]: mapper.update(*row)
        assert mapper.stats['anchor_commits'] == 1
        assert len(mapper.anchor_events) == int(enabled)
        if enabled: assert mapper.anchor_events[0]['event'] == 'stable_anchor'
        maps.append(mapper)
    np.testing.assert_array_equal(maps[0].hits, maps[1].hits)
    np.testing.assert_array_equal(maps[0].mask(), maps[1].mask())
    np.testing.assert_array_equal(maps[0].position, maps[1].position)
    assert maps[0].stats == maps[1].stats


def test_flag_restored_even_after_worker_error():
    previous = MyDroneRedMappingAnchor.RECORD_ANCHORS
    with pytest.raises(RuntimeError):
        with anchor_event_recording(True):
            assert MyDroneRedMappingAnchor.RECORD_ANCHORS
            raise RuntimeError('fixture')
    assert MyDroneRedMappingAnchor.RECORD_ANCHORS == previous


def test_worker_defers_controller_flag_until_guarded_run(monkeypatch, tmp_path):
    from swarm_rescue.solutions.devtools import red_anchor_event_batch as entry
    from swarm_rescue.solutions.devtools import red_mapping_anchor_batch as driver
    from swarm_rescue.solutions.devtools.optimization.evaluation import worker
    request = tmp_path/'run'/'request.json'
    request.parent.mkdir()
    (tmp_path/'red_scope.json').write_text('{"anchor_diagnostics": true}', encoding='utf-8')
    monkeypatch.setattr(entry.sys, 'argv', [entry.MODULE, '--worker', str(request)])
    def original_run(path):
        assert MyDroneRedMappingAnchor.RECORD_ANCHORS is True
        assert path == request
        return 7
    monkeypatch.setattr(worker, 'run', original_run)
    def guarded_main():
        assert not MyDroneRedMappingAnchor.RECORD_ANCHORS
        assert driver.MODULE == entry.MODULE
        assert worker.run(request) == 7
        assert not MyDroneRedMappingAnchor.RECORD_ANCHORS
    monkeypatch.setattr(driver, 'main', guarded_main)
    previous = driver.MODULE
    entry.main()
    assert worker.run is original_run and driver.MODULE == previous
