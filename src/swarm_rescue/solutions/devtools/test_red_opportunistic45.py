import numpy as np
import pytest
from types import SimpleNamespace
from swarm_rescue.solutions.my_drone_red_opportunistic45 import MyDroneRedOpportunistic45 as New
from swarm_rescue.solutions.my_drone_red_startup44_delivery import MyDroneRedStartup44Delivery as Old
from swarm_rescue.solutions.red_opportunistic45 import post_exploration_allowed, movement_snapshot
from swarm_rescue.solutions.devtools.test_red_schedule40 import drone
from swarm_rescue.solutions.devtools import test_red_startup44_delivery as radio_test


@pytest.mark.parametrize('step',[100,500,1500,2000,3000,4500])
def test_inventory_does_not_change_exploration_command_or_return_gate(step):
    a,b=drone(New,step,0),drone(New,step,2)
    for d in (a,b):
        d._st44_done=True;d._d39_cost=200.;d._d39_route=[0];d._d39_plan_step=step
    x,y=a._navigation(),b._navigation()
    assert a._route==b._route
    assert a._d39_mode==b._d39_mode
    assert a._explore.extension_enabled==b._explore.extension_enabled
    for k in ('forward','lateral','rotation'):assert x[k]==pytest.approx(y[k])


def test_no_premature_frontier_launch():
    a,b=drone(Old,100,2),drone(New,100,2)
    for d in (a,b):d._st44_done=True
    x,y=a._navigation(),b._navigation()
    assert not b._explore.extension_enabled
    for k in x:assert x[k]==pytest.approx(y[k])


def test_release_at_current_position_preserves_route_and_all_motion():
    d=drone(New,3000,2);d._st44_done=True;d._heading=.5
    d._coverage_field=lambda **k:SimpleNamespace(value=lambda p:100.)
    d._walk39.points=[np.array([0.,0.])];d._walk39.edges=[{}];d._walk39.last=0
    cmd=dict(forward=.7,lateral=-.3,rotation=.2,grasper=0,place_bomb=0)
    before=movement_snapshot(d,cmd)
    d._deployment(cmd)
    assert cmd['place_bomb']==1 and movement_snapshot(d,cmd)==before
    assert d._op45_violations==0


def test_failed_drop_does_not_trigger_recovery_or_route_change():
    d=drone(New,3000,2);d._st44_done=True
    d._initial_inventory=d._last_inventory=2
    d._attempt_step=2990;d._attempt_position=d._position.copy();d._attempts=1
    d._start_recovery=lambda *a:pytest.fail('failed opportunistic drop cannot steer')
    cmd=dict(forward=.7,lateral=-.3,rotation=.2,grasper=0,place_bomb=0)
    before=movement_snapshot(d,cmd);d._deployment(cmd)
    assert movement_snapshot(d,cmd)==before and cmd['place_bomb']==0


@pytest.mark.parametrize('all_explored,map_closed,fresh,remaining,allowed',[
    (False,True,True,500,False),(True,False,True,500,False),
    (True,True,False,500,False),(True,True,True,100,False),(True,True,True,500,True)])
def test_active_route_requires_real_complete_exploration_and_delivery(all_explored,map_closed,fresh,remaining,allowed):
    reports={i:(i,490 if fresh else 100,all_explored,0 if all_explored else 2,1) for i in range(5)}
    assert post_exploration_allowed(reports,500,5,dict(final_complete=map_closed),remaining)==allowed


def test_real_messages_acknowledge_all_map_bytes_with_ammunition_remaining(monkeypatch):
    monkeypatch.setattr(radio_test,'New',New)
    f=radio_test.fleet();edges={(i,j) for i in range(5) for j in range(5) if i!=j}
    for d in f:
        d.elapsed_timestep=100;d._steps=99
        d._radio_maps.freeze(d.identifier,99,100,d._map.mask(),2)
    radio_test.radio(f,101,135,edges)
    for d in f:
        s=d._exchange_state()
        assert s['source_maps_complete'] and len(s['acknowledged_sources'])==5
        assert set(s['inventory_at_cutoff'].values())=={2}


def test_startup_assignment_survives_outer_completion_envelope(monkeypatch):
    monkeypatch.setattr(radio_test,'New',New)
    f=radio_test.fleet();plan=(40,tuple((i,i*3,140.) for i in range(5)));f[0]._st44_plan=plan
    radio_test.radio(f,41,45,{(i,j) for i in range(5) for j in range(5) if i!=j})
    assert all(d._st44_plan==plan for d in f)


def test_uncleared_unknown_is_not_completed_at_freeze():
    d=drone(New,4644,1);d._op45_frontiers=99;d._op45_explored=False
    d._radio_maps.own_final=(0,4643,b'payload')
    d._completion_evidence45()
    assert d._op45_reports[0][2] is False and d._op45_reports[0][3]==99
