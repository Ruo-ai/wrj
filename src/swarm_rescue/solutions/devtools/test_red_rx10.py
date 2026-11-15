from collections import Counter
from types import SimpleNamespace, MethodType
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_branches10 import (
    BranchGrid,observed_gain,safe_reachable,finish_budget,MyDroneRedBranches10,
    MyDroneRedRX10Branch,MyDroneRedRX10Finish,MyDroneRedRX10Combined)
from swarm_rescue.solutions.my_drone_red_rn7 import MyDroneRedRN7B
from swarm_rescue.solutions.my_drone_red_advanced2 import _M06_MyDroneRedV3


def fixture():
    grid=BranchGrid((600,400),lambda *a,**k:None,None)
    grid.free[7:14,2:27]=True
    grid.free[2:8,8:15]=True
    grid.obstacle[6,2:8]=3;grid.obstacle[6,15:28]=3
    grid.obstacle[14,2:28]=3
    grid.obstacle[1:7,7]=3;grid.obstacle[1:7,15]=3
    grid.branch_enabled=True;grid.step=100
    return grid,grid.point((10,10))


def test_ray_gain_stops_at_first_unknown_and_never_sees_through_wall():
    free=np.zeros((30,30),bool);free[8:23,8:23]=True
    walls=np.zeros_like(free)
    gain=observed_gain((15,15),free,walls)
    assert 0<gain<=24
    walls[7:24,7]=True;walls[7:24,23]=True;walls[7,7:24]=True;walls[23,7:24]=True
    assert observed_gain((15,15),free,walls)==0


def test_received_observations_discount_but_do_not_invent_free_space():
    grid,pos=fixture();before=grid.free.copy()
    a=observed_gain((3,10),grid.free,grid.obstacle>=2)
    b=observed_gain((3,10),grid.free,grid.obstacle>=2,np.ones(grid.shape,bool))
    assert 0<b<a
    grid.remote_seen[:]=True
    grid.plan(pos,pos,0,5,[])
    np.testing.assert_array_equal(grid.free,before)


def test_branch_route_only_uses_original_observed_inflated_free_space():
    grid,pos=fixture();path=grid.plan(pos,pos,0,5,[])
    allowed,_,_,_=safe_reachable(grid,pos)
    assert len(path)>3 and grid.task is not None
    assert all(allowed[cell] for cell in path)
    assert all(grid.free[cell] for cell in path)
    assert grid.stats['discovered'] and grid.stats['selected']==1


def test_unknown_separated_room_cannot_supply_route():
    grid,pos=fixture();grid.free[16:19,20:28]=True
    _,_,reachable,_=safe_reachable(grid,pos)
    assert (17,25) not in reachable
    path=grid.plan(pos,pos,0,5,[])
    assert all(cell in reachable for cell in path)


def test_observation_waypoint_keeps_task_then_no_progress_expires():
    grid,pos=fixture();path=grid.plan(pos,pos,0,5,[]);ident=grid.task['id']
    goal=grid.point(path[-1]);grid.step+=30;grid.update_task(goal)
    assert grid.task['entered']
    grid.retire_goal(path[-1]);assert grid.task['id']==ident
    assert grid.goal_visits.sum()<1
    grid.step+=121;grid.update_task(pos)
    assert grid.task is None and grid.cooling


def test_peer_claim_reduces_duplicate_branch_assignment():
    grid,pos=fixture();first=grid.plan(pos,pos,0,5,[])[-1]
    other,pos=fixture();second=other.plan(pos,pos,1,5,[grid.point(first)])[-1]
    assert np.linalg.norm(grid.point(first)-grid.point(second))>60


@pytest.mark.parametrize('change',[dict(inventory=1),dict(health=5),dict(gain=0),
    dict(return_steps=float('inf')),dict(remaining_steps=200),dict(progress=.84),dict(progress=.69)])
def test_finish_reserves_bombs_return_path_and_hard_deadline(change):
    args=dict(progress=.74,inventory=0,health=25,gain=8,return_steps=120,remaining_steps=500)
    assert finish_budget(**args)
    args.update(change);assert not finish_budget(**args)


def test_finish_only_uses_retained_exploration_planner():
    grid,pos=fixture();grid.branch_enabled=False;grid.continue_after_gather=True
    path=grid.plan(pos,pos,0,5,[],rendezvous=True)
    assert path and grid.task is None


def receiver():
    grid,pos=fixture()
    actor=SimpleNamespace(_rx_enabled=True,_rx_peers={},_explore=grid,_n=5,identifier=0,
        elapsed_timestep=100,_steps=100,_status={},_peers={},_radio_maps=SimpleNamespace(accept=lambda x:True),
        _bomb_claims={},_bomb_records={},_rx_stats=Counter(),communicator=SimpleNamespace(received_messages=[]))
    actor._valid_peer=MethodType(_M06_MyDroneRedV3._valid_peer,actor)
    return actor


def test_protocol_accepts_only_received_packets_and_rejects_stale_or_invalid():
    actor=receiver();data=np.packbits(actor._explore.free).tobytes()
    old=('red-v4',('red-v3',('red-v1',((1,90,0),),None),(1,90,(0.,0.),None),None),(1,90,None),())
    packet=(1,90,data,(10.,20.),True,80,True)
    actor.communicator.received_messages=[(1,('red-rx10',old,packet))]
    MyDroneRedBranches10._receive(actor)
    assert actor._rx_peers[1]['seq']==90 and actor._status[1]==(90,0)
    assert set(actor._rx_peers)=={1}
    stale=(1,80,data,None,False,0,True)
    future=(2,120,data,None,False,0,True)
    malformed=(3,95,b'bad',None,False,0,True)
    actor.communicator.received_messages=[(1,('red-rx10',old,p)) for p in (stale,future,malformed)]
    MyDroneRedBranches10._receive(actor)
    assert actor._rx_peers[1]['wants'] and set(actor._rx_peers)=={1}


def test_hard_submission_always_bypasses_defer(monkeypatch):
    calls=[];monkeypatch.setattr(MyDroneRedRN7B,'_submit',lambda self:calls.append('submitted'))
    actor=object.__new__(MyDroneRedRX10Finish)
    actor._progress=lambda:.84;actor._rx_continue=True
    actor._submit();assert calls==['submitted']


def test_off_and_inherited_mechanisms_unchanged(monkeypatch):
    token=object();monkeypatch.setattr(MyDroneRedRN7B,'_navigation',lambda self:token)
    actor=object.__new__(MyDroneRedBranches10)
    assert actor._navigation() is token
    for cls in (MyDroneRedRX10Branch,MyDroneRedRX10Finish,MyDroneRedRX10Combined):
        assert cls._update_observation is MyDroneRedRN7B._update_observation
        assert cls._deployment is MyDroneRedRN7B._deployment
    assert MyDroneRedRX10Branch.BRANCH_TASKS and not MyDroneRedRX10Branch.ADAPTIVE_FINISH
    assert MyDroneRedRX10Finish.ADAPTIVE_FINISH and not MyDroneRedRX10Finish.BRANCH_TASKS
