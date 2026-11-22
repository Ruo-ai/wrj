"""Full dispatch regression: testing the override alone missed the integration."""
import numpy as np
from swarm_rescue.solutions.devtools.test_red_schedule40 import drone
from swarm_rescue.solutions.my_drone_red_schedule40 import MyDroneRedSchedule40
from swarm_rescue.solutions.my_drone_red_schedule40r1 import MyDroneRedSchedule40R1
from swarm_rescue.solutions.my_drone_red_delivery39r2 import MyDroneRedDelivery39R2


def test_full_navigation_reaches_fixed_executor_and_retains_route():
    failed=drone(MyDroneRedSchedule40);fixed=drone(MyDroneRedSchedule40R1)
    failed._navigation();fixed._navigation()
    assert failed._route==[]
    assert fixed._route==[(10,13),(10,16)]
    assert fixed._schedule40_stats['legacy_route_clear_prevented']==1
    assert fixed._d39_mode=='explore' and not fixed._gathering


def test_full_navigation_runs_narrow_once_and_budget_still_preempts():
    d=drone(MyDroneRedSchedule40R1);calls=[]
    d._rn7.command=lambda *args:calls.append(args[-1])
    d._navigation();assert calls==[True]
    d._steps=5100;d.elapsed_timestep=5101;d.elapsed_walltime=150
    d._navigation();assert calls==[True] and d._d39_mode!='explore'


def test_pre_cutoff_full_motion_is_unchanged_with_committed_route():
    a=drone(MyDroneRedDelivery39R2,2000);b=drone(MyDroneRedSchedule40R1,2000)
    x,y=a._navigation(),b._navigation()
    for k in x:assert np.isclose(x[k],y[k])


def test_final_protocol_and_localization_are_still_inherited():
    for name in ('_update_observation','_submit','_deployment','_receive',
                 'define_message_for_all','_return_motion39','_budget39'):
        assert getattr(MyDroneRedSchedule40R1,name) is getattr(MyDroneRedDelivery39R2,name)
