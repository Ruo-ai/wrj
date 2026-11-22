"""Delivery regressions using legal observations and actual candidate dispatch."""
import math
import numpy as np
import pytest
from swarm_rescue.solutions.red_delivery39 import WalkGraph39,FrontierQueue39
from swarm_rescue.solutions.my_drone_red_delivery39 import MyDroneRedDelivery39
from swarm_rescue.solutions.my_drone_red_continuous38 import MyDroneRedContinuous38
from swarm_rescue.simulation.utils.misc_data import MiscData

A=np.linspace(-math.pi,math.pi,181);D=np.full(181,230.)

def drone(i=0):
    d=MyDroneRedDelivery39(identifier=i,misc_data=MiscData(size_area=(400,400),number_drones=5,max_timestep_limit=5400,max_walltime_limit=180))
    d.measured_gps_position=lambda:np.array([float(i*30),0.])
    d.measured_compass_angle=lambda:0.
    d.odometer_values=lambda:np.zeros(3)
    d.lidar_values=lambda:D.copy()
    d.lidar_rays_angles=lambda:A.copy()
    d.semantic_values=lambda:()
    d.carried_bombs_count=lambda:0
    return d

def test_mapping_submission_and_deployment_preserved():
    for name in ('_submit','_deployment'):
        assert getattr(MyDroneRedDelivery39,name) is getattr(MyDroneRedContinuous38,name)
    d=drone(); assert type(d._submission_mapper).__name__=='ContinuousMapper38'
    assert type(d._fusion27).__name__=='FinalBoundary33'

def test_pause_does_not_break_actually_walked_route():
    g=WalkGraph39()
    for tick in range(1,101):g.observe(np.zeros(2),0,D,A,tick)
    for tick in range(101,151):g.observe(np.array([2*(tick-100),0.]),0,D,A,tick)
    path,cost=g.path(np.array([100.,0.]),0,D,A,151)
    assert path[-1]==0 and cost==pytest.approx(100)
    assert g.stats['discontinuities']==0

def test_no_shortcut_through_unobserved_wall():
    g=WalkGraph39();g.observe(np.zeros(2),0,D,A,1)
    # A sensor/pose discontinuity is not a physically walked edge.
    g.observe(np.array([150.,0.]),0,np.full(181,10.),A,2)
    path,cost=g.path(np.array([150.,0.]),0,np.full(181,10.),A,3)
    assert path==[] and math.isinf(cost)
    assert g.stats['discontinuities']==1

def test_return_keeps_winding_route_without_grid_connectivity():
    g=WalkGraph39();points=[]
    for x in range(0,201,4):points.append((x,0))
    for y in range(4,201,4):points.append((200,y))
    for x in range(196,-1,-4):points.append((x,200))
    for tick,p in enumerate(points,1):g.observe(np.array(p,float),0,np.full(181,15.),A,tick)
    route,cost=g.path(np.array(points[-1],float),0,np.full(181,15.),A,len(points)+1)
    assert route[-1]==0 and cost>500 # Euclidean shortcut home is only 200px.

def test_unfinished_frontier_retained_and_eventual_completion():
    g=FrontierQueue39((400,400),lambda *a,**k:None);g.extension_enabled=True;g.step=100
    c=(10,10);g.retire_goal(c)
    assert g.goal_visits[c]==0 and g.retry39[c]==220
    g.free[:]=True;g.retire_goal(c)
    assert g.goal_visits[c]>0

def test_candidate_radio_dispatch_and_real_source_bytes():
    fleet=[drone(i) for i in range(5)]
    for t in range(1,15):
        packets=[d.define_message_for_all() for d in fleet]
        for i,d in enumerate(fleet):
            d._steps=t;d.elapsed_timestep=t+1
            d.communicator._received_messages=[(None,p) for j,p in enumerate(packets) if j!=i]
            d._receive();d._update_observation()
    assert all(len(d._d39_accepts)==5 for d in fleet)
    # Only an actual received packet may move another source's final bytes.
    for d in fleet:
        d._steps=4643;d.elapsed_timestep=4644
        d._radio_maps.freeze(int(d.identifier),4643,4644,d._map.mask(),0)
    assert not fleet[0]._exchange_state()['source_maps_complete']
    for t in range(4644,4665):
        packets=[d.define_message_for_all() for d in fleet]
        for i,d in enumerate(fleet):
            d._steps=t;d.elapsed_timestep=t+1
            d.communicator._received_messages=[(None,p) for j,p in enumerate(packets) if j!=i]
            d._receive()
    assert all(d._exchange_state()['final_complete'] for d in fleet)
    for i in range(5):assert fleet[0]._radio_maps.snapshots[i]==fleet[i]._radio_maps.own_final

def test_delivery_has_precedence_and_entry_state_can_be_released():
    d=drone();d._steps=4600;d.elapsed_timestep=4601;d.elapsed_walltime=120
    d._update_observation();d._entry37.state='BRAKE'
    command=d._navigation()
    assert d._d39_mode in ('return','exchange') and not d._entry37.active
    assert all(math.isfinite(command[k]) for k in ('forward','lateral','rotation'))

def test_legacy_gathering_cannot_override_budget():
    d=drone();d._gathering=True;d._holding=True
    assert not d._try_hold(D,np.zeros(181,bool),{})
    assert not d._gathering and not d._holding

def test_radio_report_validation_rejects_future_or_malformed_contract():
    assert not MyDroneRedDelivery39._valid_contract39((30,(0.,0.),.86,.95),10)
    assert not MyDroneRedDelivery39._valid_contract39((2,(float('nan'),0.),.86,.95),10)
