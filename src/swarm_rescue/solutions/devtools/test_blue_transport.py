"""Observed-space reachability and payload delivery behavior, without truth online."""
from collections import deque
from types import SimpleNamespace
import math
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_blue_transport import TransportGrid, terminal_command


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
