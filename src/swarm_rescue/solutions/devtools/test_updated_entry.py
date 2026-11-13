"""Ensure isolated evaluation selects the requested class on the new launcher."""
import json
from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev


def test_worker_patches_actual_launcher_entry(tmp_path, monkeypatch):
    import swarm_rescue.launcher as launcher
    class Candidate:
        pass
    class Default:
        pass
    monkeypatch.setattr(launcher.drone_entry, 'drone_class_for_mode', lambda mode: Default)
    monkeypatch.setattr(ev, 'load_controller', lambda request: Candidate)
    seen = []
    def run(path):
        seen.append(launcher.drone_entry.drone_class_for_mode('place'))
        assert launcher.drone_entry.drone_class_for_mode('rescue') is Default
        return 0
    monkeypatch.setattr(ev.red, 'run_worker', run)
    monkeypatch.setattr(ev.red, 'write_json', ev.red.write_json)
    path = tmp_path/'request.json'
    path.write_text(json.dumps(ev.controller_identity('advanced5_narrow')), encoding='utf-8')
    assert ev.worker(path, 'red') == 0
    assert seen == [Candidate]
