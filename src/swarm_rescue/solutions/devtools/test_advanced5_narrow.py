"""Block2 geometry, motion and bounded ownership; no hidden map online."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions import my_drone_red_advanced5_narrow as s


def corridor(width=44., end=280., angle=0., offset=0.):
    angles=np.linspace(-math.pi,math.pi,181)
    a=angles-angle;dx,dy=np.cos(a),np.sin(a)
    with np.errstate(divide='ignore',invalid='ignore'):
        top=(width/2-offset)/dy;bottom=(-width/2-offset)/dy
        front=end/dx;back=-280/dx
    rays=np.stack((top,bottom,front,back))
    rays=np.where(rays>0,rays,s._SF_LIDAR_RANGE)
    return np.minimum(s._SF_LIDAR_RANGE,rays.min(axis=0)),angles


def filled_corridor(width=44., angle=0.):
    fine=s.NarrowLocalMap((500,500))
    d,a=corridor(width,angle=angle)
    for _ in range(2):fine.observe(np.zeros(2),0.,d,a,np.zeros(len(d),bool))
    return fine


@pytest.mark.parametrize('angle',[0.,math.pi/4,-math.pi/3])
def test_straight_and_diagonal_known_clearance(angle):
    fine=filled_corridor(angle=angle)
    goal=110*np.array([math.cos(angle),math.sin(angle)])
    assert fine.line_clear(np.zeros(2),goal)
    assert not fine.line_clear(np.zeros(2),goal+50*np.array([-math.sin(angle),math.cos(angle)]))


def test_unobserved_is_not_free_and_too_narrow_is_rejected():
    fine=s.NarrowLocalMap((300,300))
    assert not fine.line_clear(np.zeros(2),np.array([100.,0.]))
    fine=filled_corridor(width=21.)
    assert not fine.line_clear(np.zeros(2),np.array([60.,0.]))


def test_unknown_boundary_cannot_supply_two_observed_walls():
    fine=s.NarrowLocalMap((300,300))
    fine.free[40:60,10:90]=True
    assert not fine.squeeze.any()


def test_one_wall_is_not_a_two_sided_passage():
    fine=s.NarrowLocalMap((300,300));fine.free[:]=True
    fine.hits[50,:]=3;fine.free[50,:]=False
    assert not fine.squeeze.any()


@pytest.mark.parametrize('angle',[0.,math.pi/4])
def test_opposite_observed_walls_supply_a_passage(angle):
    fine=filled_corridor(angle=angle)
    assert fine.squeeze[fine.cell((0.,0.))]


def test_full_swept_segment_not_only_endpoints():
    fine=s.NarrowLocalMap((300,300));fine.free[:]=True
    cell=fine.cell((0.,0.));fine.free[cell[0]-2:cell[0]+3,cell[1]-2:cell[1]+3]=False
    assert fine.sample([[-90.,0.],[90.,0.]]).min()>fine.PLAN_RADIUS
    assert not fine.line_clear(np.array([-90.,0.]),np.array([90.,0.]))


def test_straight_speed_is_not_old_fixed_2_3_cap():
    d,a=corridor()
    force,metrics=s.narrow_velocity(d,a,0.,np.zeros(2),120.)
    assert metrics['speed']>=5.5 and force[0]>0 and abs(force[1])<.01


def test_turn_brakes_before_corner():
    d,a=corridor();v=np.array([5.,0.])
    straight,fast=s.narrow_velocity(d,a,0.,v,14.)
    turning,slow=s.narrow_velocity(d,a,0.,v,14.,turn=math.pi/2)
    assert slow['speed']<fast['speed'] and turning[0]<0


def test_front_wall_braking_and_wraparound():
    d,a=corridor(end=22.)
    force,metrics=s.narrow_velocity(d,a,2*math.pi,np.array([4.,0.]),100.)
    assert force[0]<0 and metrics['clearance']<13
    np.testing.assert_allclose(s.narrow_scan(d,a,0.)['clearance'],s.narrow_scan(d,a,2*math.pi)['clearance'])


def test_missing_sensor_brakes():
    force,metrics=s.narrow_velocity(np.array([np.nan]),np.zeros(1),0.,np.array([2.,1.]),100.)
    assert np.all(force<0) and metrics['mode']=='blind'


def test_center_control_opposes_sideways_drift():
    d,a=corridor(offset=3.)
    force,_=s.narrow_velocity(d,a,0.,np.array([3.,1.]),100.,center_error=3.)
    assert force[1]<0


@pytest.mark.parametrize('angle',[0.,math.pi/4,-math.pi/3])
def test_axis_uses_both_parallel_walls_and_oblique_heading(angle):
    d,a=corridor(angle=angle)
    axis=s.narrow_axis(d,a,angle+.3)
    assert axis is not None and math.cos(axis-angle)>.99


def test_front_wall_is_not_mistaken_for_two_side_walls():
    d,a=corridor(width=120,end=20)
    assert s.narrow_axis(d,a,0.) is None


def test_fine_route_handles_l_bend_without_corner_cut():
    fine=s.NarrowLocalMap((300,300))
    for r in range(fine.shape[0]):
        for c in range(fine.shape[1]):
            x,y=fine.point((r,c))
            fine.free[r,c]=(-120<x<22 and abs(y)<22) or (abs(x)<22 and -22<y<120)
    coarse=s._M19_SurveyGrid((300,300),lambda *a,**k:None)
    route=fine.route(np.array([-80.,0.]),np.array([1.,1.]),coarse,target=np.array([0.,80.]))
    assert len(route)>2
    assert all(fine.line_clear(a,b) for a,b in zip(route,route[1:]))


@pytest.mark.parametrize('angle',[0.,math.pi/4])
def test_fine_route_through_measured_narrow_corridor(angle):
    fine=filled_corridor(angle=angle)
    coarse=s._M19_SurveyGrid((500,500),lambda *a,**k:None)
    path=fine.route(np.zeros(2),np.array([math.cos(angle),math.sin(angle)]),coarse)
    assert len(path)>1 and np.linalg.norm(path[-1])>60
    assert all(fine.line_clear(a,b) for a,b in zip(path,path[1:]))


def test_physics_hull_enclosed_by_design_radius():
    from swarm_rescue.simulation.drone.drone_base import DroneBase
    body=DroneBase(radius=10)
    assert max(v.length for p in body._pm_shapes for v in p.get_vertices())<=s.NarrowLocalMap.BODY_RADIUS+1e-8


def traffic_drone():
    obj=object.__new__(s.MyDroneRedAdvanced5Narrow)
    obj.identifier=1;obj._steps=100;obj._position=np.array([25.,0.]);obj._heading=0.
    obj._narrow_task=dict(entry=np.zeros(2),start=50)
    obj._narrow_map=filled_corridor()
    obj.semantic_values=lambda:[]
    return obj


def test_stale_reservation_does_not_lock_corridor():
    obj=traffic_drone()
    obj._narrow_peers={0:(0,70,(100.,0.),(0.,0.),(60.,0.),'TRAVERSE',40,70)}
    assert obj._narrow_traffic(np.array([1.,0.])) is None


def test_existing_opposite_occupant_has_right_of_way():
    obj=traffic_drone()
    obj._narrow_peers={0:(0,100,(100.,0.),(0.,0.),(60.,0.),'TRAVERSE',40,100)}
    assert obj._narrow_traffic(np.array([1.,0.]))=='retreat'


def test_following_keeps_spacing():
    obj=traffic_drone()
    obj._narrow_peers={0:(0,100,(0.,0.),(150.,0.),(50.,0.),'TRAVERSE',40,100)}
    assert obj._narrow_traffic(np.array([1.,0.]))=='wait'


def test_retreat_is_bounded_and_uses_recorded_breadcrumbs():
    obj=traffic_drone();obj._event=lambda *a,**k:None
    obj._narrow_task.update(state='TRAVERSE',breadcrumbs=[np.zeros(2),np.array([12.,0.])],deadline=200)
    obj._narrow_retreat('blocked')
    task=obj._narrow_task
    assert task['state']=='RETREAT' and task['deadline']<=obj._steps+100
    np.testing.assert_array_equal(task['path'][-1],np.zeros(2))
    deadline=task['deadline'];obj._steps+=10;obj._narrow_retreat('blocked')
    assert task['deadline']==deadline


def test_side_motion_stays_inside_physical_corridor():
    # A multi-step trajectory test, not a mirror of the force implementation.
    position=np.array([0.,3.]);velocity=np.array([3.,.5])
    for _ in range(40):
        d,a=corridor(offset=position[1])
        force,_=s.narrow_velocity(d,a,0.,velocity,120.,center_error=position[1])
        velocity=.95*velocity+.6*force;position+=velocity
        assert abs(position[1])+10<22
    assert position[0]>100 and abs(position[1])<1.
