"""Regression for a reached breadcrumb immediately before a blind corner."""
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from swarm_rescue.solutions.devtools.test_red_delivery39 import drone
from swarm_rescue.solutions.my_drone_red_delivery39r1 import MyDroneRedDelivery39R1


def test_reached_node_cannot_remain_target_when_next_corner_is_not_visible():
    d=drone(2);d.__class__=MyDroneRedDelivery39R1
    d._steps=4100;d.elapsed_timestep=4101;d._position=np.array([0.,0.]);d._heading=0.
    d._walk39.points=[np.array([150.,150.]),np.array([30.,0.]),np.array([0.,0.])]
    d._d39_route=[2,1,0];d._d39_plan_step=4100;d._d39_advance=4100
    d._walk39.visible=lambda *args:False
    with patch('swarm_rescue.solutions.my_drone_red_delivery39r1._M11_safe_velocity',return_value=(np.array([.1,0.]),0.)) as safe:
        command=d._return_motion39()
    assert d._d39_target==1 and d._d39_route==[1,0]
    assert safe.call_args.args[-1]==30. # actual next edge, never zero-distance current node
    assert command['forward']==.1 and d._d39_stats['reached_nodes_consumed']==1
