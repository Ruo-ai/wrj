"""Real native event emitter compatibility; no permissive event mock."""
import ast
import inspect
import json
import textwrap

import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_frontier34 import (
    FrontierSurvey34, MyDroneRedFrontier34,
)
from swarm_rescue.solutions.my_drone_red_frontier34r1 import MyDroneRedFrontier34R1
from swarm_rescue.solutions.devtools.test_red_frontier34_controller import (
    fixture as original_fixture, exchange,
)
from swarm_rescue.solutions.devtools.test_red_fusion27_delivery import fleet


def fixture(source=0, snapshots=None, policy=False, revised=True, sealed=True):
    previous = original_fixture(source, snapshots=snapshots, policy=policy, sealed=sealed)
    cls = type('NativeEventFixture34R1',
               (MyDroneRedFrontier34R1 if revised else MyDroneRedFrontier34,),
               {'identifier': source, 'drone_health': 25})
    d = object.__new__(cls)
    d.__dict__.update(previous.__dict__)
    # The actual M06 inherited emitter prints V3_EVENT with reserved kind,
    # source and native elapsed clock. Tests must exercise that exact callable.
    del d.__dict__['_event']
    if hasattr(d._explore, 'event'):
        d._explore.event = d._event
    return d


def emitted(capsys):
    return [json.loads(line.split('V3_EVENT ', 1)[1])
            for line in capsys.readouterr().out.splitlines() if 'V3_EVENT ' in line]


def test_original_kind_collision_is_reproduced_by_actual_native_emitter():
    d = fixture(revised=False)
    assert 'kind' in inspect.signature(d._event).parameters
    with pytest.raises(TypeError, match='kind'):
        d._f34_event('own_final_at_leader', kind='leader_reported_payload_manifest_receipt',
                     received_step=d._steps, leader_report_step=d._steps-1, version=4643)


def test_original_frontier_step_collision_is_reproduced_by_actual_plan():
    d = fixture(policy=True, sealed=False, revised=False)
    d._explore.free[7:24,7:24] = True
    d._explore.support_mode34 = True
    start = d._explore.point((15,15))
    with pytest.raises(TypeError, match='step'):
        d._explore.plan(start, start, 0, 5, [])


def test_r1_changes_only_two_diagnostic_adapters_and_no_policy_parameters():
    assert MyDroneRedFrontier34R1.__bases__ == (MyDroneRedFrontier34,)
    methods = {name for name,value in MyDroneRedFrontier34R1.__dict__.items() if callable(value)}
    assert methods == {'_f34_event','_event'}
    for name in ('__init__','_f34_init','_navigation','_navigate_motion','_f34_policy',
                 '_f34_budget_update','_f34_remaining','_f34_trace','_f34_start_return',
                 '_f34_return_motion','_f34_safe_station','_receive','define_message_for_all',
                 '_f34_report','_submit','_exchange_state','_update_observation','_deployment',
                 '_progress','_try_hold','control'):
        assert getattr(MyDroneRedFrontier34R1,name) is getattr(MyDroneRedFrontier34,name)
    for name in ('FREEZE_AT','DEADLINE_AT','ARRIVAL_PROGRESS34','EXCHANGE_RESERVE34','RETURN_SPEED34'):
        assert getattr(MyDroneRedFrontier34R1,name) == getattr(MyDroneRedFrontier34,name)


EVENT_PAYLOADS = {
    'own_final_at_leader': dict(received_step=4700, leader_report_step=4699,
                              version=4643, kind='leader_reported_payload_manifest_receipt'),
    'return_started': dict(trigger='route_and_exchange_budget',
                          budget=dict(required_steps=600.,arrival_remaining_steps=599.)),
    'local_complete_support': dict(gain=dict(window_steps=240,new_free=2,new_visited=1)),
    'productive_exploration_resumed': dict(gain=dict(window_steps=240,new_free=15,new_visited=7)),
    'exchange_station_reached': {},
    'separated_exchange_station': dict(point=[10.,20.]),
    'return_target_blocked': dict(navigator=dict(failed=True,reason='frontier_navigation_failed')),
    'exhausted_trace_detour': dict(point=[30.,40.]),
    'exhausted_trace_rescan_retry': {},
}


def test_native_emitter_cases_cover_every_frontier34_event_call_site():
    tree = ast.parse(textwrap.dedent(inspect.getsource(MyDroneRedFrontier34)))
    calls = {node.args[0].value for node in ast.walk(tree) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Attribute) and node.func.attr=='_f34_event'
             and node.args and isinstance(node.args[0],ast.Constant)}
    assert calls == set(EVENT_PAYLOADS)
    grid_tree = ast.parse(textwrap.dedent(inspect.getsource(FrontierSurvey34)))
    direct = {node.args[0].value for node in ast.walk(grid_tree) if isinstance(node,ast.Call)
              and isinstance(node.func,ast.Attribute) and node.func.attr=='event'
              and node.args and isinstance(node.args[0],ast.Constant)}
    assert direct == {'frontier34_target'}


@pytest.mark.parametrize('reason',sorted(EVENT_PAYLOADS))
def test_every_frontier_event_serializes_through_actual_native_emitter(reason,capsys):
    d = fixture()
    payload = EVENT_PAYLOADS[reason]
    before = dict(payload)
    d._f34_event(reason,**payload)
    row, = emitted(capsys)
    assert row['event'] == 'frontier34_'+reason
    assert row['drone'] == 0 and row['step'] == d.elapsed_timestep
    assert d._d32_stats[reason] == 1
    assert d._f34_events[-1]['step'] == d._steps
    if 'kind' in payload:
        assert row['receipt_kind'] == payload['kind']
        assert d._f34_events[-1]['receipt_kind'] == payload['kind']
    assert payload == before


def test_actual_frontier_selection_keeps_native_elapsed_and_diagnostic_control_step(capsys):
    d = fixture(policy=True,sealed=False)
    grid = d._explore
    grid.free[7:24,7:24] = True
    grid.support_mode34 = True
    grid.step = d._steps
    start = grid.point((15,15))
    route = grid.plan(start,start,0,5,[])
    assert route and grid.last_frontier34['step'] == d._steps
    row, = emitted(capsys)
    assert row['event'] == 'frontier34_target'
    assert row['step'] == d.elapsed_timestep
    assert row['control_step'] == d._steps
    assert row['unknown'] >= 3 and row['point'] == grid.point(route[-1]).tolist()


def test_actual_nested_network_upload_receipts_reach_native_event_sink(capsys):
    stores = fleet()
    drones = [fixture(i,snapshots=stores[i]) for i in range(5)]
    exchange(drones,24)
    rows = emitted(capsys)
    uploads = [r for r in rows if r['event']=='frontier34_own_final_at_leader']
    assert {r['drone'] for r in uploads} == set(range(5))
    assert all(r['version']==4643 and r['receipt_kind']=='leader_reported_payload_manifest_receipt'
               for r in uploads)
    for d in drones:
        assert d._exchange_state()['final_complete']
        assert d._f34_upload_acks[int(d.identifier)]['kind'] == 'leader_reported_payload_manifest_receipt'
        assert all(d._radio_maps.snapshots[i] == stores[i].own_final for i in range(5))


def test_policy_transition_uses_real_event_emitter_without_loosening_signature(capsys):
    d = fixture(policy=True,sealed=False)
    d._f34_policy()
    assert d._f34_mode == 'support'
    rows = emitted(capsys)
    assert [r['event'] for r in rows] == ['frontier34_local_complete_support']
    d._radio_maps.own_final = (0,4643,b'fixture-final-marker')
    d._f34_policy()
    assert d._f34_mode == 'return'
    rows = emitted(capsys)
    assert rows[-1]['event'] == 'frontier34_return_started'
    assert rows[-1]['trigger'] == 'final_epoch_preemption'


def test_nonconflicting_native_event_payload_is_unchanged(capsys):
    old, new = fixture(revised=False),fixture()
    payload = dict(inventory=0,attempt=2,point=[1.,2.])
    old._event('diagnostic_contract_probe',**payload)
    new._event('diagnostic_contract_probe',**payload)
    rows = emitted(capsys)
    assert rows[0] == rows[1]


def test_report_and_wire_packet_unchanged_before_receipt_emission():
    old,new = fixture(revised=False),fixture()
    assert old.define_message_for_all() == new.define_message_for_all()
