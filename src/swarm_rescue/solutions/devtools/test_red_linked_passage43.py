"""End-to-end receipt scheduling and measured passage regression checks."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from swarm_rescue.solutions.devtools.test_red_schedule40 import drone
from swarm_rescue.solutions.my_drone_red_linked_passage43 import MyDroneRedLinkedPassage43 as New
from swarm_rescue.solutions.my_drone_red_return_budget42 import MyDroneRedReturnBudget42 as Old
from swarm_rescue.solutions.red_linked_passage43 import valid_receipt, receipt_delivered, observed_approach
from swarm_rescue.solutions.red_delivery39 import FrontierQueue39
from swarm_rescue.simulation.utils.misc_data import MiscData

ROOT=Path(__file__).resolve().parents[4]

def ready(step=4550,connected=True):
    d=drone(New,step);d._d39_cost=3500.;d._d39_route=[0];d._d39_plan_step=step
    d._rb42_motion.extend([(6.,True)]*239);d._rb42_sample_step=step-1
    d.odometer_values=lambda:np.array([6.,0.,0.])
    if connected:
        for i in range(5):d._radio_maps.publish(i,step,np.zeros(d._map.shape,bool))
    return d

def test_real_navigation_connected_past_old_forced_return_keeps_exploring():
    d=ready();d._navigation()
    assert d._d39_mode=='explore' and not d._gathering and d._explore.extension_enabled
    assert d._lp43_stats['connected_budget_deferred']>0

def test_disconnected_budget_preempts():
    d=ready(4550,False);d._navigation()
    assert d._d39_mode!='explore' and not d._explore.extension_enabled

def test_disconnected_with_remaining_budget_can_explore():
    d=ready(1500,False);d._navigation()
    assert d._d39_mode=='explore'

def test_wall_clock_deadline_still_applies_when_disconnected():
    d=ready(2500,False);d.elapsed_walltime=166.;d._navigation()
    assert d._d39_mode!='explore'

def test_mapping_renderer_source_freeze_and_submission_unchanged():
    d=ready()
    for name in ('_submit','_deployment','_exchange_state'):
        assert getattr(New,name) is getattr(Old,name)
    assert type(d._submission_mapper).__name__=='ContinuousMapper38'
    assert type(d._fusion27).__name__=='FinalBoundary33'
    assert (New.FREEZE_AT,New.DEADLINE_AT)==(Old.FREEZE_AT,Old.DEADLINE_AT)

@pytest.mark.parametrize('receipt', [None,(100,((0,101,'a'*64),)),(100,((0,90,'a'*64),(0,90,'a'*64))),(999,())])
def test_invalid_receipts_do_not_establish_connection(receipt):
    assert not valid_receipt(receipt,5,120)

def test_stale_or_wrong_digest_receipts_cannot_confirm_final():
    s=(2,4643,b'payload');digest='a'*64
    assert receipt_delivered((4660,((2,4643,digest),)),2,s,digest,4670,True)
    assert not receipt_delivered((4660,((2,4643,'b'*64),)),2,s,digest,4670,True)
    assert not receipt_delivered((4660,((2,4610,digest),)),2,s,digest,4670,True)
    assert not receipt_delivered((4660,((2,4643,digest),)),2,s,digest,4800,True)

def test_actual_candidate_messages_forward_leader_receipt_over_four_hops():
    fleet=[New(identifier=i,misc_data=MiscData(size_area=(400,400),number_drones=5,max_timestep_limit=5400,max_walltime_limit=180)) for i in range(5)]
    for d in fleet:
        d._steps=10;d.elapsed_timestep=11
        d._radio_maps.publish(d.identifier,10,np.zeros(d._map.shape,bool))
    for t in range(12,52):
        for d in fleet:d._steps=t-1;d.elapsed_timestep=t
        packets=[d.define_message_for_all() for d in fleet]
        for i,d in enumerate(fleet):
            d.communicator._received_messages=[(None,packets[j]) for j in range(5) if abs(j-i)==1]
            d._receive()
    assert all(d._link_state43() for d in fleet)
    for d in fleet[1:]:
        assert next(x for x in d._lp43_receipt[1] if x[0]==d.identifier)[2]==d._radio_maps.payload_digest(d.identifier)
    d=fleet[4];d._radio_maps.publish(4,51,np.eye(*d._map.shape,dtype=bool))
    d._radio_maps.own_final=d._radio_maps.snapshots[4]
    assert not d._link_state43()  # A new final cannot be satisfied by an old receipt.

@pytest.mark.parametrize('half_width,allowed',[(27.,True),(12.,False)])
def test_radar_corridor_uses_hull_clearance_not_coarse_cell_width(half_width,allowed):
    a=np.linspace(-np.pi,np.pi,181)
    d=np.minimum(230.,half_width/np.maximum(np.abs(np.sin(a)),1e-9))
    fine,path=observed_approach(d,a,(),np.zeros(2),0.,np.array([100.,0.]))
    assert bool(path)==allowed
    if path:assert all(fine.line_clear(x,y) for x,y in zip(path,path[1:]))

def test_recorded_failed_goal_gets_safe_approach_through_real_dispatch():
    spec=importlib.util.spec_from_file_location('input43',ROOT/'output/sf13/continuous38_replay_20261122.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    run=ROOT/'output/sf13/return_budget42_run_20261122/runs/Map08_r02/candidate_return_budget42/red/attempt_001'
    with np.load(run/'mapper_legal_inputs.npz') as tape:
        args=next(a for t,a in helper.inputs(tape,3) if t==3666)
    with np.load(ROOT/'output/sf13/map08_southeast_diagnosis_20261122/navigation_replay.npz') as z:
        pose=z['d3_t3666_pose'];free=z['d3_t3666_free'];obstacle=z['d3_t3666_obstacle']
    d=drone(New,3666);d._position=pose[:2].copy();d._heading=float(pose[2])
    d._explore=FrontierQueue39((1660,1122),d._event);d._explore.free[:]=free;d._explore.obstacle[:]=obstacle
    d._route=[(48,64),(51,66)];d._explore.step=3666
    d.lidar_values=lambda:args[3];d.lidar_rays_angles=lambda:args[4];d.semantic_values=lambda:args[5]
    d._d39_cost=300.;d._d39_route=[0];d._d39_plan_step=3666
    d._release_stale_route('new_wall')
    assert d._fine43_goal is not None and d._lp43_stats['coarse_rejection_rescued']==1
    np.testing.assert_array_equal(d._explore.obstacle,obstacle)
    def forbidden(*args):raise AssertionError('Legacy passage must not compete with fine task')
    d._rn7.command=forbidden
    command=d._navigation()
    assert d._lp43_stats['fine_task_ticks']==1
    assert all(np.isfinite(command[k]) for k in ('forward','lateral','rotation'))
