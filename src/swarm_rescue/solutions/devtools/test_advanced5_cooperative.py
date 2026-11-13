import pytest as _retired_pytest
_retired_pytest.importorskip('swarm_rescue.solutions.my_drone_red_advanced5_cooperative', reason='Historical strategy retired by user; source archived')
"""Block3 radio convergence, progress leases and safe task alternatives."""
from types import SimpleNamespace
import numpy as np
from swarm_rescue.solutions import my_drone_red_advanced5_cooperative as s


def board(identifier=0):
    return s.CooperativeTaskBoard(identifier, 5, (800, 800))


def publish(b, step=60, target=(200., 0.), cost=200., unknown=15, position=(0., 0.)):
    return b.update(step, 'explore', target, position, 2, cost, unknown)


def test_looping_positions_and_heartbeats_do_not_extend_lease():
    b=board();publish(b)
    for step in range(72, 241, 12):
        publish(b, step, position=((step%24)*3., 0.))
    assert b.own[8]==60 and b.active(241)==[]


def test_remaining_path_progress_renews_but_not_returning_oscillation():
    b=board();publish(b)
    publish(b, 90, cost=170.)
    assert b.own[8]==90
    publish(b, 120, cost=200.)
    publish(b, 150, cost=170.)
    assert b.own[8]==90


def test_local_information_counts_as_task_progress():
    b=board();publish(b)
    publish(b, 100, unknown=11)
    assert b.own[8]==100


def test_newer_release_overrides_and_old_relay_cannot_resurrect():
    a,b=board(),board(1);publish(a)
    old=a.snapshot();a.finish(90,'failed')
    b.accept(a.snapshot(),90);b.accept(old,100)
    assert not b.active(100) and b.records[0][2]=='released'


def test_relay_does_not_retimestamp_expired_origin():
    a,b,c=board(),board(1),board(2);publish(a)
    b.accept(a.snapshot(),70);c.accept(b.snapshot(),140)
    assert c.records[0][1]==60 and not c.active(140)


def test_stalled_task_cannot_immediately_reclaim_same_place():
    b=board();publish(b)
    b.finish(220,'no_progress_no_alternative')
    publish(b,232)
    assert b.own[2]=='released'
    publish(b,244,cost=160.)
    assert b.own[2]=='active'


def test_reordered_simultaneous_bids_have_one_deterministic_winner():
    a,b=board(),board(1)
    publish(a,cost=190.);publish(b,cost=150.)
    a.accept(b.snapshot(),60);b.accept(a.snapshot(),60)
    grid=SimpleNamespace(line_clear=lambda a,b:True)
    assert a.winner((200.,0.),190.,60,grid)==b.winner((200.,0.),150.,60,grid)==1


def test_established_owner_not_replaced_by_new_close_drone():
    a,b=board(),board(1);publish(a,60)
    publish(a,120,cost=150.)
    publish(b,120,cost=50.)
    b.accept(a.snapshot(),120)
    grid=SimpleNamespace(line_clear=lambda a,b:True)
    assert b.winner((200.,0.),50.,120,grid)==0


def test_wall_separates_nearby_tasks():
    a,b=board(),board(1);publish(a)
    b.accept(a.snapshot(),60)
    grid=SimpleNamespace(line_clear=lambda a,b:False)
    assert b.winner((200.,30.),50.,60,grid)==1


def test_malformed_future_oversized_and_out_of_bounds_messages_ignored():
    a,b=board(),board(1);publish(a)
    row=a.own
    for index,value in [(0,-1),(1,99),(4,(10000.,0.)),(8,1000),(9,10000),(10,float('nan'))]:
        broken=list(row);broken[index]=value
        b.accept((tuple(broken),),60)
    b.accept((row,)*6,60)
    assert not b.records


def test_bomb_task_does_not_steal_exploration_ownership():
    a,b=board(),board(1)
    a.update(60,'place',(200.,0.),(0.,0.),2,200.,10)
    b.accept(a.snapshot(),60)
    grid=SimpleNamespace(line_clear=lambda a,b:True)
    assert b.winner((200.,0.),200.,60,grid)==1


def test_offers_and_records_remain_bounded():
    b=board()
    for i in range(100):b.observe_offer((float(i*80),0.),i*4,5)
    assert len(b.catalog)<=20 and len(b.offers(400))<=5


def test_alternative_never_routes_through_unknown_or_declared_wall():
    events=[];owner=SimpleNamespace(_team=board(),_bomb_target=None,_progress=lambda:0.2)
    grid=s.CooperativeExplorationGrid((800,800),lambda *a,**k:events.append((a,k)),owner)
    grid.free[10:30,10:30]=True
    grid.obstacle[10:30,20]=3
    grid.step=100
    candidates,parents=grid.candidates((-100.,0.))
    for _,cell,_,_ in candidates:
        route=s._M16_path_to(parents,cell)
        assert all(grid.free[c] and grid.obstacle[c]<2 for c in route)
        assert all(c[1]<20 for c in route)


def test_single_available_route_is_not_blocked_to_enforce_diversity():
    owner=SimpleNamespace(_team=board(1),_bomb_target=None,_progress=lambda:0.2)
    grid=s.CooperativeExplorationGrid((800,800),lambda *a,**k:None,owner)
    grid.free[17:24,15:30]=True;grid.step=100
    base=s._M19_SurveyGrid.plan(grid,(-60.,0.),(-60.,0.),1,5,[])
    assert base
    other=board();publish(other,90,target=tuple(grid.point(base[-1])),cost=60.)
    owner._team.accept(other.snapshot(),100)
    route=grid.plan((-60.,0.),(-60.,0.),1,5,[])
    assert route and all(grid.free[c] for c in route)


def test_gather_plan_unchanged_by_task_claims():
    owner=SimpleNamespace(_team=board(1),_bomb_target=None,_progress=lambda:0.2)
    grid=s.CooperativeExplorationGrid((800,800),lambda *a,**k:None,owner)
    grid.free[10:30,10:30]=True;grid.step=100
    baseline=s._M19_SurveyGrid.plan(grid,(100.,0.),(0.,0.),1,5,[],rendezvous=True)
    assert grid.plan((100.,0.),(0.,0.),1,5,[],rendezvous=True)==baseline


def test_controller_update_and_release_use_original_event_signature(capsys):
    grid=SimpleNamespace(point=lambda c:np.array([200.,0.]), cell=lambda p:(1,1), unknown_near=lambda c:15)
    d=SimpleNamespace(_position=np.zeros(2),_steps=60,elapsed_timestep=60,identifier=0,
        _team_last_update=-100,_team_status_bucket=-1,_team=board(),_explore=grid,_route=[(1,1)],
        carried_bombs_count=lambda:2,_team_kind=lambda:'explore')
    d._event=lambda *a,**k:s._M06_MyDroneRedV3._event(d,*a,**k)
    d._team_release=lambda reason:s.MyDroneRedAdvanced5Cooperative._team_release(d,reason)
    s.MyDroneRedAdvanced5Cooperative._team_update(d)
    d._team_release('failed')
    output=capsys.readouterr().out
    assert 'team_task_start' in output and 'team_task_end' in output and 'task_kind' in output


def test_known_nearby_goal_can_be_redundant_without_explicit_visit():
    owner=SimpleNamespace(_team=board(),_bomb_target=None,_progress=lambda:0.2)
    grid=s.CooperativeExplorationGrid((800,800),lambda *a,**k:None,owner)
    grid.free[:]=True;cell=grid.cell((40.,0.))
    assert grid.redundant_goal(np.zeros(2),cell)
    assert grid.redundant_goal(np.array([-300.,0.]),cell)
    grid.free[cell[0]-3:cell[0]+4,cell[1]+1:cell[1]+4]=False
    assert not grid.redundant_goal(np.zeros(2),cell)


def test_busy_peer_offer_requires_local_information_and_reachable_path():
    owner=SimpleNamespace(_team=board(1),_bomb_target=None,_progress=lambda:0.2)
    grid=s.CooperativeExplorationGrid((800,800),lambda *a,**k:None,owner)
    grid.free[15:25,10:25]=True;grid.step=100
    peer=board(0);point=tuple(grid.point((18,24)))
    peer.observe_offer(point,90,10)
    peer.update(90,'place',(-250.,-250.),(-200.,0.),2,100.,10)
    owner._team.accept(peer.snapshot(),100)
    original=[grid.cell((0.,0.))]*12
    route=grid.handoff(np.array([0.,0.]),original,())
    assert route and route[-1]==grid.cell(point)
    grid.obstacle[15:25,22]=3
    assert grid.handoff(np.array([0.,0.]),original,()) is None


def test_following_requires_fresh_same_direction_radio_and_actual_detection():
    from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor
    d,a=corridor(width=80.)
    peer={4:(100,np.array([70.,0.]),np.array([200.,0.]))}
    detected=[np.array([70.,0.])]
    assert s.cooperative_following(np.zeros(2),0.,0.,d,a,peer,detected,100)==(70.,4)
    assert s.cooperative_following(np.zeros(2),0.,0.,d,a,peer,[],100) is None
    assert s.cooperative_following(np.zeros(2),0.,0.,d,a,peer,detected,112) is None
    peer[4]=(100,np.array([70.,0.]),np.array([-200.,0.]))
    assert s.cooperative_following(np.zeros(2),0.,0.,d,a,peer,detected,100) is None


def test_open_area_does_not_force_a_following_queue():
    from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor
    d,a=corridor(width=200.)
    peer={4:(100,np.array([70.,0.]),np.array([200.,0.]))}
    assert s.cooperative_following(np.zeros(2),0.,0.,d,a,peer,[np.array([70.,0.])],100) is None


def test_stationary_following_expires_and_enters_cooldown():
    from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor
    d,a=corridor(width=80.);events=[]
    drone=SimpleNamespace(_steps=100,_escape_until=0,_team_traffic_after=0,
        _team_traffic=None,_team_waiting_until=0,_team_kind=lambda:'explore',
        _position=np.zeros(2),_heading=0.,_velocity=np.zeros(2),_last_motion_angle=None,_yield_until=120,
        _peers={4:(100,np.array([70.,0.]),np.array([200.,0.]))},
        semantic_values=lambda:[SimpleNamespace(entity_type=SimpleNamespace(name='DRONE'),distance=70.,angle=0.)],
        _event=lambda *a,**k:events.append((a,k)),_empty_command=lambda:{})
    assert s.MyDroneRedAdvanced5Cooperative._team_follow_command(drone,d,a,0.,150.) is not None
    drone._steps=136;drone._peers[4]=(136,np.array([70.,0.]),np.array([200.,0.]))
    assert s.MyDroneRedAdvanced5Cooperative._team_follow_command(drone,d,a,0.,150.) is None
    assert drone._team_traffic is None and drone._team_traffic_after==226
    assert events[-1][1]['reason']=='no_progress'


def test_recently_progressing_task_is_not_preempted_by_older_claim():
    a,b=board(),board(1)
    publish(a,100,cost=250.);publish(a,190,cost=180.)
    publish(b,150,cost=220.);publish(b,190,cost=160.)
    b.accept(a.snapshot(),200)
    calls=[]
    grid=SimpleNamespace(point=lambda c:np.array([200.,0.]),line_clear=lambda a,b:True,
                         redundant_goal=lambda p,c:True,alternative=lambda *a:calls.append(a))
    actor=SimpleNamespace(_position=np.zeros(2),_steps=200,_team_next_switch=65,
        _team_kind=lambda:'explore',_progress=lambda:.2,_yield_until=0,_goal_step=150,
        _team=b,_route=[(1,1)],_explore=grid)
    s.MyDroneRedAdvanced5Cooperative._team_reassign(actor)
    assert calls==[] and actor._route==[(1,1)]

