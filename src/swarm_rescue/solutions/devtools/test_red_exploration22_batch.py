"""Necessary pairing, provenance and no-implicit-flight checks; no simulation."""
from copy import deepcopy
from collections import Counter
import json

import pytest

from swarm_rescue.solutions.devtools import red_exploration22_batch as batch


def test_exact_official_schedule_pairs_both_seeds_for_three_distinct_sources():
    rows = batch.official_rows(batch.MAPS)
    schedule = batch.schedule_for(rows)
    assert len(schedule) == 18
    groups = Counter((x['name'], x['round'], x['seed']) for x in schedule)
    assert len(groups) == 6 and set(groups.values()) == {3}
    for item in schedule:
        assert item['official_map'] == item['map'] == item['name']
        assert item['seed'] == batch.SEEDS[item['round'] - 1]
        assert item['blue_variant'] == 'adapted'
        assert item['map_file'] == 'map_data/map_' + item['name'][-2:] + '.json'
    assert schedule[0]['controller'] == 'candidate_rl22'
    assert schedule[3]['controller'] == 'archived_rl17r'
    expected_zones = {'Map06': ['NO_COM_ZONE'], 'Map08': ['NO_GPS_ZONE'],
                      'Map09': ['NO_COM_ZONE', 'NO_GPS_ZONE']}
    assert all(item['scenario']['zones'] == expected_zones[item['name']] for item in schedule)


def test_schedule_refuses_same_number_generated_map():
    rows = deepcopy(batch.official_rows(['Map06']))
    rows[0]['name'] = 'yh_map_fs_6'
    with pytest.raises(ValueError, match='official'):
        batch.schedule_for(rows)


def test_run_requires_prepared_and_never_creates_output(tmp_path):
    out = tmp_path / 'forbidden'
    with pytest.raises(SystemExit) as error:
        batch.main(['--output', str(out), '--run'])
    assert error.value.code == 2
    assert not out.exists()


def test_changed_seed_and_map_fail_frozen_schedule_validation():
    rows = batch.official_rows(batch.MAPS)
    plan = {'configuration': {'smoke': False}, 'map_rows': rows,
            'schedule': batch.schedule_for(rows)}
    batch.validate_schedule(plan)
    plan['schedule'][1]['seed'] += 1
    with pytest.raises(ValueError, match='schedule'):
        batch.validate_schedule(plan)


def test_current_archive_hash_is_exact_audited_old_zip():
    assert batch.digest(batch.ARCHIVE) == batch.EXPECTED_ARCHIVE_SHA256


def test_exit_zero_callback_error_and_diagnostic_failure_are_retained(tmp_path):
    (tmp_path / 'loaded_identity.json').write_text(json.dumps({'observed_instances': {'0': {}}}))
    (tmp_path / 'motion_diagnostics.json').write_text('{}')
    (tmp_path / 'local13_diagnostics.json').write_text('{"errors": ["bad observer"]}')
    (tmp_path / 'exploration_diagnostics.json').write_text('{"errors": []}')
    row = {'status': 'ok', 'exit_code': 0, 'engine_callback_errors': 1,
           'submitted': True, 'valid_shape': True, 'phase1': {'score_map': 85}}
    issues = batch.phase_issues(row, tmp_path, 'red', False)
    assert 'engine_callback_error' in issues
    assert 'diagnostic_invalid:local13_diagnostics' in issues
    assert 'phase_invalid' in issues


def test_actual_loaded_class_mapper_and_cpu_are_checked_not_just_request_labels(tmp_path):
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    request = dict(map='Map06', round=1, seed=2026111901, phase='red', smoke=False,
        zones=['NO_COM_ZONE'], controller_selector='candidate_rl22',
        controller_module='swarm_rescue.solutions.my_drone_red_exploration22',
        controller_class='MyDroneRedExploration22')
    row = dict(request, actual_conditions=dict(drones=5, zones=['NO_COM_ZONE'],
        instantiated_zone_types=['NO_COM_ZONE'], max_steps=5400, max_walltime=180))
    plan = dict(configuration={}, source_sha256={'src/example.py': 'frozen'})
    actual = dict(observed_instances={str(i): dict(
        controller=request['controller_module'] + '.' + request['controller_class'],
        mapper='swarm_rescue.solutions.my_drone_red_repair17.RepairMapper17') for i in range(5)},
        loaded_sources={'src/example.py': 'frozen'})
    resources = dict(cpu_count=1, memory_limit_bytes=4_000_000_000, status='exited')
    for path, doc in [(tmp_path / 'plan.json', plan), (attempt / 'request.json', request),
                      (attempt / 'loaded_identity.json', actual), (attempt / 'resources.json', resources)]:
        path.write_text(json.dumps(doc))
    assert batch.loaded_issues(row, attempt, 'red', False) == []
    actual['observed_instances']['0']['controller'] = 'workspace.same_filename.WrongClass'
    actual['observed_instances']['1']['mapper'] = 'wrong.Mapper'
    actual['loaded_sources']['src/example.py'] = 'changed'
    resources['cpu_count'] = 2
    (attempt / 'loaded_identity.json').write_text(json.dumps(actual))
    (attempt / 'resources.json').write_text(json.dumps(resources))
    issues = batch.loaded_issues(row, attempt, 'red', False)
    assert 'actual_controller_mismatch:0' in issues
    assert 'actual_mapper_mismatch:1' in issues
    assert 'loaded_source_not_frozen:src/example.py' in issues
    assert 'resource_envelope_mismatch' in issues
