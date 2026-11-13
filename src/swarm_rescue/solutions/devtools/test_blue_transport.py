"""Observed-space reachability and payload delivery behavior, without truth online."""
from collections import deque
from types import SimpleNamespace
import math
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_blue_transport import TransportGrid, terminal_command,trace_waypoint,SteeringTable,MyDroneBlueTransport
from swarm_rescue.solutions.my_drone_eval_blue import RescueGrid,steering_choice,MyDroneBlueAdapted


@pytest.mark.parametrize('seed', [11, 28, 742])
def test_native_shortest_paths_equal_reference_reachability_and_distances(seed):
    grid=TransportGrid((320,256));rng=np.random.default_rng(seed)
    allowed=rng.random(grid.shape)>.25;start=(6,6);allowed[start]=True
    expected={start:0};queue=deque([start])
    while queue:
        r,c=queue.popleft()
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nxt=r+dr,c+dc
            if 0<=nxt[0]<grid.shape[0] and 0<=nxt[1]<grid.shape[1] and allowed[nxt] and nxt not in expected:
                expected[nxt]=expected[(r,c)]+1;queue.append(nxt)
    _,distance,parent=grid._shortest(allowed,grid.point(start))
    assert int(np.isfinite(distance).sum())==len(expected)
    for cell,length in expected.items():
        flat=np.ravel_multi_index(cell,grid.shape)
        assert distance[flat]==length
        if length:assert distance[parent[flat]]==length-1


def test_disconnected_goal_is_partial_and_never_routes_through_wall():
    grid=TransportGrid((640,320));grid.free[:]=True;grid.evidence[:,20]=8
    position,target=grid.point((10,4)),grid.point((10,35))
    path=grid.route(position,target)
    assert path and not grid.last_complete
    assert all(grid.cell(p)[1]<20 for p in path)
    assert np.linalg.norm(path[-1]-target)>64


def test_path_goes_through_observed_doorway_instead_of_cutting_wall():
    grid=TransportGrid((640,480));grid.free[:]=True
    grid.evidence[:20,20]=8
    position,target=grid.point((10,4)),grid.point((10,35))
    path=grid.route(position,target)
    assert grid.last_complete
    assert max(grid.cell(p)[0] for p in path)>=21
    allowed,_=grid.layers()
    assert all(allowed[grid.cell(p)] for p in path)


def test_unknown_space_is_never_a_return_path_even_in_narrow_fallback():
    grid=TransportGrid((320,256));grid.free[4:9,2:6]=True;grid.payload=True
    path=grid.route(grid.point((6,3)),grid.point((6,17)))
    assert not grid.last_complete
    assert all(grid.free[grid.cell(p)] for p in path)


def test_semantic_approach_respects_a_nearer_obstacle():
    hit=SimpleNamespace(distance=70.,angle=0.)
    cmd=terminal_command(hit,np.array([25.,70.]),np.array([0.,.1]),0.,1)
    assert cmd['forward']==cmd['lateral']==0 and cmd['grasper']==1


def test_delivery_requires_actual_turn_and_close_surface_before_release():
    d,a=np.array([23.]),np.array([0.])
    hit=SimpleNamespace(distance=23.,angle=0.)
    assert terminal_command(hit,d,a,0.,1)['grasper']==1
    assert terminal_command(hit,d,a,math.pi,-1)['grasper']==0
    hit.distance=30.
    assert terminal_command(hit,d,a,2*math.pi,1)['grasper']==1


def test_partial_endpoint_avoidance_selects_a_different_reachable_approach():
    grid=TransportGrid((640,480));grid.free[:]=True;grid.evidence[:,20]=8
    p,t=grid.point((14,6)),grid.point((14,35))
    grid.route(p,t);old=grid.last_endpoint.copy();grid.avoid=[old]
    path=grid.route(p,t)
    assert np.linalg.norm(grid.last_endpoint-old)>=48
    assert all(grid.cell(q)[1]<20 for q in path)


@pytest.mark.parametrize('targeted', [False,True])
def test_native_bfs_preserves_B0_search_and_connected_route_tie_breaks(targeted):
    old=RescueGrid((640,480));new=TransportGrid((640,480))
    for g in (old,new):
        g.free[:]=True;g.evidence[:20,20]=8;g.visits[4:9,3:6]=2
    start=old.point((10,4));target=old.point((10,35)) if targeted else None
    np.testing.assert_array_equal(old.route(start,target,2,10),new.route(start,target,2,10))


def test_obstructed_disposal_center_keeps_checked_navigation_action():
    hit=SimpleNamespace(distance=60.,angle=0.)
    fallback=dict(forward=0.,lateral=.2,rotation=.3,grasper=1)
    cmd=terminal_command(hit,np.array([20.,60.]),np.array([0.,.1]),0.,1,fallback)
    assert cmd==fallback


def test_return_trace_requires_current_clearance_and_never_uses_future_nodes():
    trail=[np.array([0.,0.]),np.array([40.,0.]),np.array([80.,0.]),np.array([120.,0.])]
    pos=np.array([120.,0.]);angles=np.linspace(-math.pi,math.pi,180,endpoint=False)
    clear=np.full(180,230.)
    i,p=trace_waypoint(trail,2,pos,0.,clear,angles)
    assert i==0 and np.array_equal(p,trail[0])
    blocked=clear.copy();blocked[np.abs(angles)>2.5]=25.
    assert trace_waypoint(trail,2,pos,0.,blocked,angles) is None


@pytest.mark.parametrize('carrying',[False,True])
def test_cached_steering_matches_B0_including_invalid_rays_and_shape_changes(carrying):
    table=SteeringTable();rng=np.random.default_rng(742)
    for length in (180,180,90,180):
        angles=np.linspace(-math.pi,math.pi,length,endpoint=False)
        for _ in range(10):
            d=rng.uniform(4,240,length);d[rng.integers(length)]=np.nan
            desired=rng.uniform(-math.pi,math.pi)
            assert table.choose(d,angles,desired,carrying)==steering_choice(d,angles,desired,carrying)
    assert table.choose(np.array([np.nan]),np.array([0.]),0.,carrying) is None


def test_semantic_optimization_keeps_B0_obstacle_exclusion_and_landmark_updates():
    from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
    enum=DroneSemanticSensor.TypeEntity;Data=DroneSemanticSensor.Data
    objects=[Data(90.,.2,enum.WALL,False),Data(80.,.4,enum.DRONE,False),
             Data(55.,-.3,enum.BOMB,False),Data(60.,.1,enum.BOMB,True),
             Data(90.,1.,enum.DISPOSAL_CENTER,False),Data(82.,.9,enum.DISPOSAL_CENTER,False)]
    angles=np.linspace(-math.pi,math.pi,180,endpoint=False);distances=np.full(180,80.)
    def stub():
        events=[];observed=[]
        g=SimpleNamespace(observe=lambda *args:observed.append(args[-1].copy()),snapshot=lambda:('snapshot',))
        return SimpleNamespace(_center_hit=None,_center=np.array([1.,2.]),_center_step=1,_snapshot=None,
            _position=np.array([30.,40.]),_heading=.3,_tick=18,_grid=g,identifier=0,
            semantic_values=lambda:objects,_record_bomb=lambda *a,**k:events.append((a,k))),events,observed
    a,ea,oa=stub();b,eb,ob=stub()
    ha=MyDroneBlueAdapted._sense(a,distances,angles);hb=MyDroneBlueTransport._sense(b,distances,angles)
    np.testing.assert_array_equal(a._center,b._center);np.testing.assert_array_equal(oa,ob)
    assert a._center_hit==b._center_hit and a._snapshot==b._snapshot and len(ha)==len(hb)
    for x,y in zip(ea,eb):
        np.testing.assert_array_equal(x[0][0],y[0][0]);assert x[0][1:]==y[0][1:] and x[1]==y[1]
