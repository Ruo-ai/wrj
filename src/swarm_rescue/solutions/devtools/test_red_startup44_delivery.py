"""Delivery regression: actual messages, immutable payloads and motion gating."""
import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_startup44 import MyDroneRedStartup44 as Old
from swarm_rescue.solutions.my_drone_red_startup44_delivery import MyDroneRedStartup44Delivery as New
from swarm_rescue.solutions.red_startup44_delivery import delivered, elect_carrier, relay_source
from swarm_rescue.solutions.devtools.test_red_schedule40 import drone
from swarm_rescue.simulation.utils.misc_data import MiscData


def fleet():
    result = [New(identifier=i, misc_data=MiscData(size_area=(400, 400), number_drones=5,
        max_timestep_limit=5400, max_walltime_limit=180)) for i in range(5)]
    for d in result:
        d._steps = 10; d.elapsed_timestep = 11
        d._position = np.array([20.*d.identifier, 0.]); d._heading = 0.
        d.lidar_values = lambda: np.full(181, 230.)
        d.lidar_rays_angles = lambda: np.linspace(-np.pi, np.pi, 181)
        d._walk39.observe(d._position, 0., d.lidar_values(), d.lidar_rays_angles(), 10)
        mask = np.zeros(d._map.shape, bool); mask[d.identifier+1, 3] = True
        d._radio_maps.publish(d.identifier, 10, mask)
    return result


def radio(f, start, stop, edges):
    for t in range(start, stop):
        for d in f: d._steps = t-1; d.elapsed_timestep = t
        packets = [d.define_message_for_all() for d in f]
        for i, d in enumerate(f):
            d.communicator._received_messages = [(None, packets[j]) for j in range(5) if (i,j) in edges]
            d._receive()


def test_exact_final_confirmation_survives_link_loss_but_not_new_bytes():
    snap = (2, 4643, b'final')
    receipt = (4670, ((2, 4643, 'a'*64),))
    assert delivered(receipt, 2, snap, 'a'*64, 5200, True)
    assert not delivered(receipt, 2, snap, 'b'*64, 5200, True)
    assert not delivered(receipt, 2, (2, 4700, b'new'), 'b'*64, 5200, False)
    assert not delivered(receipt, 2, (2, 4700, b'final'), 'a'*64, 5200, True)


def test_actual_four_hop_forwarding_then_disconnection_keeps_final_proof():
    f = fleet(); edges = {(i,j) for i in range(5) for j in range(5) if abs(i-j)==1}
    radio(f, 12, 62, edges)
    assert all(d._link_state43() for d in f)
    for d in f: d._radio_maps.own_final = d._radio_maps.snapshots[d.identifier]
    for d in f[1:]:
        d.elapsed_timestep = 500
        assert d._link_state43()
    assert len(f[-1]._radio_maps.snapshots) == 5


def test_a_peer_copy_does_not_count_as_submitter_delivery():
    f = fleet(); edges = {(i,j) for i in range(1,5) for j in range(1,5) if i!=j}
    radio(f, 12, 62, edges)
    assert len(f[2]._radio_maps.snapshots) == 4
    assert not f[2]._link_state43()
    assert list(f[0]._radio_maps.snapshots) == [0]


def test_contact_ticket_requires_real_round_trip_and_is_relayed():
    f = fleet(); edges = {(0,2),(2,0),(2,4),(4,2)}
    radio(f, 12, 30, edges)
    assert 2 in f[0]._dl44.tickets and 2 in f[2]._dl44.tickets
    assert 2 in f[4]._dl44.tickets
    assert (2, f[0]._dl44.tickets[2][1]) in f[0]._dl44.local_nodes
    assert (2, f[2]._dl44.tickets[2][1]) in f[2]._dl44.local_nodes
    assert 4 not in f[0]._dl44.tickets  # Forwarded presence is not direct contact.


def test_startup_assignment_callback_and_scalar_plan_survive_message_envelope():
    f = fleet()
    plan = (40, tuple((i, i*3, 140.) for i in range(5)))
    f[0]._st44_plan = plan
    radio(f, 41, 45, {(i,j) for i in range(5) for j in range(5) if i!=j})
    assert all(d._st44_plan == plan for d in f)
    assert all(any(e['kind']=='received_assignment' for e in d._st44_events) for d in f[1:])


def test_source_priority_is_fair_even_when_one_final_is_missing():
    snapshots = {i:(i,10,b'x') for i in range(5)}
    receipt = (11, tuple((i,10,'a'*64) for i in range(4)))
    chosen = [relay_source(snapshots, receipt, lambda i:'a'*64, t) for t in range(30)]
    assert chosen.count(4) >= 20
    assert set(chosen) == set(range(5))


def test_active_deep_explorer_cannot_be_elected_as_courier():
    ticket = (2,100,(0.,0.),(30.,0.))
    def status(i, sealed, mode):
        return (i,200,(0.,0.),sealed,mode,((1,100,'a'*64),), 'explore',None)
    assert elect_carrier([status(2,False,'explore'),status(3,True,'return')],ticket,[1]) == 3


@pytest.mark.parametrize('step', [500, 1500, 3000, 4100])
def test_no_new_early_return_gate_and_unchanged_exploration_command(step):
    a, b = drone(Old, step), drone(New, step)
    for d in (a,b):
        d._st44_done = True
        d._d39_cost = 200.; d._d39_route = [0]; d._d39_plan_step = step
    ca, cb = a._navigation(), b._navigation()
    assert a._d39_mode == b._d39_mode == 'explore'
    for k in ca: assert ca[k] == pytest.approx(cb[k])


def test_navigation_at_empty_launch_has_a_bounded_exit():
    d = drone(New, 4700); d._st44_done = True; d._d39_mode = 'return'
    d._dl44.route_node = 0; d._dl44.started = 4600; d._dl44.arrived = 4600
    d._dl44.stage = 2; d._d39_budget = dict(remaining=300)
    d._return_motion39()
    assert d._dl44.stats['contact_no_delivery'] == 1


def test_unsealed_stage_keeps_original_receipt_expiry_until_final_freeze():
    f=fleet();d=f[1]
    digest=d._radio_maps.payload_digest(1)
    d._lp43_receipt=(20,((1,10,digest),));d.elapsed_timestep=200
    assert not d._link_state43()
    d._radio_maps.own_final=d._radio_maps.snapshots[1]
    assert d._link_state43()


@pytest.mark.parametrize('step',[4250,4400,4600])
def test_pre_freeze_return_executor_commands_remain_original(step):
    a,b=drone(Old,step),drone(New,step)
    for d in (a,b):
        d._st44_done=True;d._d39_mode='return';d._path39()
    ca,cb=a._return_motion39(),b._return_motion39()
    assert b._dl44.route_node is None
    assert a._lp43_target_node==b._lp43_target_node
    for key in ca:assert ca[key]==pytest.approx(cb[key])


def test_pose_startup_renderer_freeze_and_submission_implementations_inherited():
    for name in ('_update_observation','_submit','_deployment','_exchange_state','_finish44','_budget39'):
        assert getattr(New,name) is getattr(Old,name)
    assert (New.FREEZE_AT,New.DEADLINE_AT) == (Old.FREEZE_AT,Old.DEADLINE_AT)
    a,b = drone(Old,1000),drone(New,1000)
    assert type(a._submission_mapper) is type(b._submission_mapper)
    assert type(a._fusion27) is type(b._fusion27)
    assert type(a._st44_initial) is type(b._st44_initial)
