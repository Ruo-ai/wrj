"""Startup safety, radio-only coordination, bounded handoff and preservation."""
import math
import numpy as np
from swarm_rescue.solutions.devtools.test_red_schedule40 import drone
from swarm_rescue.solutions.my_drone_red_startup44 import MyDroneRedStartup44 as New
from swarm_rescue.solutions.my_drone_red_linked_passage43 import MyDroneRedLinkedPassage43 as Old
from swarm_rescue.solutions.red_startup44 import InitialMap44,scan_lengths,assignment,valid_bid,valid_plan,ANGLES
from swarm_rescue.simulation.utils.misc_data import MiscData


def test_rotating_sensor_frame_does_not_rotate_world_exit():
    angles=np.linspace(-math.pi,math.pi,181)
    profiles=[]
    for heading in (0.,math.pi/2,math.pi):
        world=angles+heading
        d=np.minimum(230.,40/np.maximum(np.abs(np.sin(world)),1e-9))
        profiles.append(scan_lengths(d,angles,heading))
    for p in profiles:
        assert p[0]>130 and p[12]>130 and p[6]<30 and p[18]<30
    np.testing.assert_allclose(profiles[0],profiles[1],atol=3)


def test_open_space_fleet_assigns_separated_exits():
    bids={i:(i,18,(float(i*10),0.),(160.,)*24) for i in range(5)}
    plan=assignment(bids)
    assert len(plan)==5 and len({p[1] for p in plan})==5
    headings=[ANGLES[p[1]] for p in plan]
    assert min(abs((a-b+math.pi)%(2*math.pi)-math.pi) for i,a in enumerate(headings) for b in headings[i+1:])>=math.pi/4-.001
    assert plan==assignment(dict(reversed(list(bids.items()))))


def test_confined_exits_not_invented_to_make_five_roles():
    p=[0.]*24;p[0]=160;p[12]=140
    plan=assignment({i:(i,18,(float(i*5),0.),tuple(p)) for i in range(5)})
    assert len(plan)==5 and {r[1] for r in plan}=={0,12}
    assert assignment({0:(0,18,(0.,0.),(0.,)*24)})==()


def test_invalid_packets_do_not_allocate():
    assert not valid_bid((0,18,(0.,0.),(float('nan'),)*24),5,30)
    assert not valid_bid((0,99,(0.,0.),(160.,)*24),5,30)
    assert not valid_plan((24,((0,1,100.),(0,2,100.))),5,30)
    assert not valid_plan((24,((5,1,100.),)),5,30)


def fleet():
    ds=[]
    for i in range(5):
        d=New(identifier=i,misc_data=MiscData(size_area=(600,600),number_drones=5,max_timestep_limit=5400,max_walltime_limit=180))
        d.measured_gps_position=lambda i=i:np.array([i*12.,0.])
        d.measured_compass_angle=lambda i=i:i*.6
        d.odometer_values=lambda:np.zeros(3)
        d.lidar_values=lambda:np.full(181,230.)
        d.lidar_rays_angles=lambda:np.linspace(-np.pi,np.pi,181)
        d.semantic_values=lambda:()
        d.carried_bombs_count=lambda:2
        ds.append(d)
    return ds


def test_real_control_and_radio_chain_coordinate_and_preserve_maps():
    ds=fleet()
    for t in range(1,130):
        packets=[d.define_message_for_all() for d in ds]
        for i,d in enumerate(ds):
            d.elapsed_timestep=t+1;d.elapsed_walltime=t*.03
            d.communicator._received_messages=[(None,packets[j]) for j in range(5) if abs(i-j)==1]
            command=d.control()
            assert all(np.isfinite(command[k]) for k in ('forward','lateral','rotation'))
    assert len(ds[0]._st44_bids)==5
    assert all(d._st44_goal is not None and d._st44_plan==ds[0]._st44_plan for d in ds)
    assert all(d._radio_maps.snapshots for d in ds)
    assert len({next(e['assigned_sector'] for e in d._st44_events if e['kind']=='depart') for d in ds})==5
    assert all(d._lp43_receipt is not None for d in ds)


def test_disconnection_falls_back_without_fabricating_peer_knowledge():
    d=fleet()[3]
    for t in range(1,170):
        d.elapsed_timestep=t+1;d.elapsed_walltime=t*.03;d.control()
    assert set(d._st44_bids)=={3}
    assert d._st44_done
    event=next(e for e in d._st44_events if e['kind']=='depart')
    assert not event['coordinated']


def test_startup_expires_and_later_navigation_is_identical():
    new,old=drone(New,1500),drone(Old,1500)
    a=new._navigation();b=old._navigation()
    assert new._st44_done
    # Initial expired instance has no startup route: compare normal continued execution.
    new,old=drone(New,1500),drone(Old,1500);new._st44_done=True
    assert new._navigation()==old._navigation()
    for name in ('_update_observation','_submit','_deployment','_budget39','_return_motion39','_navigate_motion','_link_state43'):
        assert getattr(New,name) is getattr(Old,name)
    assert type(new._submission_mapper).__name__=='ContinuousMapper38'
    assert type(new._fusion27).__name__=='FinalBoundary33'


def test_independent_map_median_rejects_transient_false_wall_and_motion():
    m=InitialMap44();angles=np.linspace(-np.pi,np.pi,181)
    for j in range(13):
        d=np.full(181,230.);d[30:60]=100.+(j%3-1)*2
        if j==5:d[80:110]=40. # A single faulty frame cannot permanently add a wall.
        m.add(d,angles,np.zeros(2),.02*(j%3-1),0.)
    m.add(np.ones(181)*20,angles,np.zeros(2),0.,3.)
    p=m.finish()
    assert m.rejected==1 and np.min(m.ranges[80:110])>225
    assert np.max(np.abs(m.ranges[32:58]-100))<3
    assert m.grid.shape==(121,121) and p.shape==(24,)
    assert abs(m.heading)<.01


def test_dynamic_occlusion_stays_unknown_in_initial_map():
    from types import SimpleNamespace as NS
    m=InitialMap44();a=np.linspace(-np.pi,np.pi,181)
    robot=NS(entity_type=NS(name='DRONE'),angle=0.,distance=40.)
    for j in range(13):
        d=np.full(181,230.);d[np.abs(a)<.19]=40.
        m.add(d,a,np.zeros(2),0.,0.,[robot])
    profile=m.finish()
    assert np.isnan(m.ranges[90]) and profile[0]==0
    assert m.grid[60,80]==-1


def test_small_uniform_bid_noise_does_not_permute_fleet_roles():
    variants=[]
    for jitter in (-1.5,0.,1.5):
        bids={i:(i,32,(float(i*10),0.),tuple(np.full(24,150.)+jitter)) for i in range(5)}
        variants.append([(i,k) for i,k,length in assignment(bids)])
    assert variants[0]==variants[1]==variants[2]


def test_no_yaw_command_when_current_turn_clearance_is_unsafe():
    d=fleet()[0];d._steps=80;d._position=np.zeros(2);d._heading=0.
    d._st44_goal=np.array([0.,140.]);d._st44_angle=np.pi/2
    d.lidar_values=lambda:np.full(181,15.)
    assert d._navigation()['rotation']==0
    assert not d._st44_started


def test_planning_map_cannot_write_navigation_or_submission_map():
    # Initial map owns only arrays/scalars; no reference to the controller maps.
    d=fleet()[0];initial=d._st44_initial
    assert all(v is not d._map and v is not d._submission_mapper and v is not d._explore for v in initial.__dict__.values())
    assert New._update_observation is Old._update_observation
