"""Schedule conflicts: exercise inherited navigation and delivery with legal sensors."""
import numpy as np
import pytest
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.solutions.my_drone_red_delivery39r2 import MyDroneRedDelivery39R2
from swarm_rescue.solutions.my_drone_red_schedule40 import MyDroneRedSchedule40, ScheduledNarrow40
from swarm_rescue.solutions.my_drone_red_rn7 import NarrowMixin


def drone(cls=MyDroneRedSchedule40, step=4000, inventory=0):
    d = cls(identifier=0, misc_data=MiscData(size_area=(400,400),
        number_drones=5,max_timestep_limit=5400,max_walltime_limit=180))
    d.measured_gps_position=lambda:np.zeros(2)
    d.measured_compass_angle=lambda:0.
    d.odometer_values=lambda:np.zeros(3)
    d.lidar_values=lambda:np.full(181,230.)
    d.lidar_rays_angles=lambda:np.linspace(-np.pi,np.pi,181)
    d.semantic_values=lambda:()
    d.carried_bombs_count=lambda:inventory
    d._steps=step;d.elapsed_timestep=step+1;d.elapsed_walltime=step*.028
    d._update_observation()
    d._explore.free[:]=True
    d._route=[(10,13),(10,16)]
    d._goal_step=d._last_route_step=step
    return d


def test_route_survives_legacy_cutoff_until_actual_delivery_preemption():
    old,new=drone(MyDroneRedDelivery39R2),drone()
    old._navigate_motion();new._navigate_motion()
    assert old._route==[]  # Reproduce the diagnosed historical failure.
    assert new._route==[(10,13),(10,16)]
    assert new._d39_mode=='explore' and not new._gathering
    assert new._schedule40_stats['legacy_route_clear_prevented']==1


def test_pre_cutoff_motion_and_collision_braking_preserved():
    for distance in (230.,15.):
        old,new=drone(MyDroneRedDelivery39R2,2000),drone(step=2000)
        for d in (old,new):d.lidar_values=lambda:np.full(181,distance)
        a,b=old._navigate_motion(),new._navigate_motion()
        assert a.keys()==b.keys()
        for key in a:assert a[key]==pytest.approx(b[key])


def test_frontiers_available_after_bombs_before_old_sixty_percent_gate():
    old,new=drone(MyDroneRedDelivery39R2,1800),drone(step=1800)
    old._navigation();new._navigation()
    assert not old._explore.extension_enabled
    assert new._explore.extension_enabled and new._d39_mode=='explore'
    assert new._schedule40_stats['early_frontier_enabled_ticks']==1


def test_bomb_placement_keeps_its_existing_priority():
    d=drone(step=1800,inventory=2);d._navigation()
    assert not d._explore.extension_enabled and d._d39_mode=='explore'


def test_narrow_executor_available_late_and_dispatched_once():
    old,new=drone(MyDroneRedDelivery39R2),drone()
    calls=[[],[]]
    for i,d in enumerate((old,new)):
        d._rn7.command=lambda *args,i=i:calls[i].append(args[-1])
        d._navigation()
    assert calls==[[False],[True]]
    assert new.__class__.__mro__.index(ScheduledNarrow40)<new.__class__.__mro__.index(NarrowMixin)


def test_budget_still_preempts_search_and_narrow_actions():
    d=drone(step=5100);calls=[]
    d._rn7.command=lambda *args:calls.append(args)
    command=d._navigation()
    assert d._d39_mode in ('return','exchange')
    assert not d._explore.extension_enabled and not calls
    assert all(np.isfinite(command[k]) for k in ('forward','lateral','rotation'))


def test_entry_guard_still_owns_navigation():
    d=drone();d._entry37.state='BRAKE'
    route=list(d._route);d._navigate_motion()
    assert d._route==route and not d._schedule40_stats['legacy_route_clear_prevented']


def test_mapping_protocol_deadlines_and_return_budget_unchanged():
    for name in ('_update_observation','_submit','_deployment','_budget39',
                 '_return_motion39','_receive','define_message_for_all'):
        assert getattr(MyDroneRedSchedule40,name) is getattr(MyDroneRedDelivery39R2,name)
    a,b=drone(MyDroneRedDelivery39R2),drone()
    assert type(a._submission_mapper) is type(b._submission_mapper)
    assert type(a._fusion27) is type(b._fusion27)
    assert (a.FREEZE_AT,a.DEADLINE_AT)==(b.FREEZE_AT,b.DEADLINE_AT)
