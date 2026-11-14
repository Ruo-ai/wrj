import math
import cv2
import numpy as np
import pytest
from swarm_rescue.solutions.red_pixel_faces import PixelFaceMapper,PixelSnapshots,enclosed_fill
from swarm_rescue.solutions.red_footprint8 import FootprintMap,FootprintMotion,hull_kernel
from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor


def rectangle_evidence(inward=False,open_end=False):
    wall=np.zeros((50,70),bool);wall[15:36,15]=True;wall[15:36,54]=True
    wall[15,15:55]=True;wall[35,15:55]=True
    bits=np.zeros(wall.shape,np.uint8)
    bits[15:36,15]=1<<(0 if inward else 4)
    bits[15:36,54]=1<<(4 if inward else 0)
    bits[15,15:55]=1<<(6 if inward else 2)
    bits[35,15:55]=1<<(2 if inward else 6)
    if open_end:wall[15,20:50]=False
    return wall,bits


def test_closed_two_sided_solid_can_fill_but_room_cannot():
    wall,bits=rectangle_evidence();free=np.zeros_like(wall)
    filled=enclosed_fill(wall,free,bits)
    assert filled[20:30,25:45].all()
    wall,bits=rectangle_evidence(inward=True)
    assert not enclosed_fill(wall,free,bits).any()


def test_one_side_open_ends_missing_direction_or_any_free_veto_fill():
    wall,bits=rectangle_evidence(open_end=True)
    assert not enclosed_fill(wall,np.zeros_like(wall),bits).any()
    wall,bits=rectangle_evidence();free=np.zeros_like(wall);free[25,30]=True
    assert not enclosed_fill(wall,free,bits).any()
    assert not enclosed_fill(wall,np.zeros_like(wall),np.zeros_like(bits)).any()
    wall[:]=False;wall[15,15:55]=True
    assert not enclosed_fill(wall,np.zeros_like(wall),bits).any()


def test_no_free_pixel_is_ever_filled_in_random_contradictions():
    wall,bits=rectangle_evidence();rng=np.random.default_rng(742)
    for _ in range(12):
        free=rng.random(wall.shape)<.02
        assert not (enclosed_fill(wall,free,bits)&free).any()


def test_pixel_resolution_unique_per_scan_and_no_closing():
    m=PixelFaceMapper((70,50));m.position=np.array([0.,-20.]);m._trusted=True
    points=np.column_stack((np.arange(-12.,13.),np.zeros(25)))
    many=np.repeat(points,20,axis=0)
    m._insert(m.hits,many,1.);assert m.hits.max()==1
    m._insert(m.hits,many,1.);m._insert(m.hits,many,1.)
    mask=m.mask();assert mask.shape==(50,70) and mask.sum()==25
    pixels=m.to_pixels(mask);assert pixels.sum()==25
    assert not pixels[24].any() and not pixels[26].any()


def test_snapshot_navigation_prefix_versioning_and_remote_free_veto():
    m=PixelFaceMapper((70,50));a=PixelSnapshots((17,24),2,m);b=PixelSnapshots((17,24),2,m)
    nav=np.zeros((17,24),bool);nav[3,4]=True
    wall,bits=rectangle_evidence();fill=enclosed_fill(wall,np.zeros_like(wall),bits)
    m._cached_step=m.step;m._cached=(wall,fill,np.zeros_like(wall))
    a.publish(0,30,nav);assert b.accept(a.snapshots[0])
    assert not b.accept(a.snapshots[0]);np.testing.assert_array_equal(b.merged(),nav)
    free=np.zeros_like(wall);free[25,30]=True
    m._cached=(np.zeros_like(wall),np.zeros_like(wall),free)
    b.publish(1,31,np.zeros_like(nav));assert not b.submission()[25,30]
    assert b.submission()[25,31]


def test_pose_update_and_outage_buffer_are_a3_not_a_new_localizer():
    from swarm_rescue.solutions.my_drone_red_mapping import SubmissionMapper
    a=SubmissionMapper((300,200));b=PixelFaceMapper((300,200));d,angles=corridor(80,end=90)
    for tick in range(48):
        gps=np.array([tick*.5,0.]) if tick<12 or tick>=32 else None
        compass=0. if gps is not None else None;odo=np.array([.5,0.,0.])
        a.update(gps,compass,odo,d,angles);b.update(gps,compass,odo,d,angles)
        np.testing.assert_allclose(a.position,b.position);assert a.heading==b.heading
        assert len(a.pending)==len(b.pending)


@pytest.mark.parametrize('width',[46.,50.,80.])
def test_observed_passage_uses_full_hull_once(width):
    d,a=corridor(width);f=FootprintMap(d,a)
    assert f.line_clear(np.zeros(2),np.array([60.,0.]))
    assert not f.line_clear(np.zeros(2),np.array([60.,width]))


def test_orientation_changes_passability_without_doubling_radius():
    d,a=corridor(36);axis=FootprintMap(d,a,margin=2.)
    assert axis.line_clear(np.zeros(2),np.array([50.,0.]))
    d,a=corridor(36,angle=math.pi/4);diagonal=FootprintMap(d,a,margin=2.)
    assert not diagonal.line_clear(np.zeros(2),50*np.array([2**-.5,2**-.5]))


def test_too_narrow_unknown_and_margin_bounds():
    d,a=corridor(25);assert not FootprintMap(d,a).line_clear(np.zeros(2),np.array([50.,0.]))
    d[:]=np.nan;assert not FootprintMap(d,a).configuration.any()
    with pytest.raises(ValueError):hull_kernel(1.)
    with pytest.raises(ValueError):hull_kernel(5.)


def test_late_entrance_budget_recovers_but_active_budget_stays_bounded():
    m=FootprintMotion();m.episodes=8;m.entries=list(range(8));m.tick=100
    assert not m.budget_available()
    m.tick=700;assert m.budget_available()
    m.active_budget=1400;assert not m.budget_available()


def test_actual_velocity_predictive_braking_in_front_of_wall():
    m=FootprintMotion();m.tick=100;m.since=100;m.last_progress=100
    m.valid_odom=True;m.velocity=np.array([5.,0.]);m.path=[np.array([80.,0.])]
    d,a=corridor(50,end=24)
    class Coarse:
        visits=np.zeros((1,1),int)
        def cell(self,p):return (0,0)
    command=m.command(d,a,Coarse(),np.zeros(2),0.,0.)
    assert command['forward']<0 and command['rotation']==0


def test_local_waypoint_completion_inside_corridor_continues_same_task():
    m=FootprintMotion();m.tick=100;m.since=40;m.last_progress=100
    m.valid_odom=True;m.episodes=1;m.path=[np.array([5.,0.])]
    d,a=corridor(50)
    class Coarse:
        visits=np.zeros((1,1),int)
        def cell(self,p):return (0,0)
    cmd=m.command(d,a,Coarse(),np.zeros(2),0.,0.)
    assert m.path and m.episodes==1 and m.stats['continue_inside_passage']==1
    assert cmd['forward']>0


def test_spatial_scan_fusion_does_not_paint_neighbourhood_or_count_one_scan_many_times():
    m=PixelFaceMapper((70,50));m.position=np.array([0.,-20.]);m._trusted=True
    for yy in (0.,1.,-1.):
        pts=np.column_stack((np.arange(-12.,13.),np.full(25,yy)))
        m._insert(m.hits,np.repeat(pts,10,axis=0),1.)
    assert m.support.max()==3
    result=m.mask()
    assert result.sum()==75  # exactly the three observed rows, never the 5px support neighbourhood
