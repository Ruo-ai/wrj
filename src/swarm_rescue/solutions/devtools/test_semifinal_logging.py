"""Regression: NumPy diagnostic scalars must not crash periodic live logging."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.semifinal_localization import LocalMatcher
from swarm_rescue.solutions.my_drone_red_semifinal import MyDroneRedSemifinal
from swarm_rescue.solutions.my_drone_blue_semifinal import MyDroneBlueSemifinal
from .test_semifinal_joint import corner


def diagnostic(kind='matched'):
    points=corner()
    if kind=='flat':points=np.column_stack((np.linspace(-120,120,40),np.full(40,100)))
    matcher=LocalMatcher((600,600));matcher.add(points,np.zeros(2),0)
    query=points+[180,140] if kind=='rejected' else points[:5] if kind=='insufficient' else points
    return matcher.match(query,np.array([8.,-6.]),0)[1]


@pytest.mark.parametrize('kind',['matched','flat','rejected','insufficient'])
def test_real_match_diagnostics_are_strict_json(kind):
    payload=diagnostic(kind)
    parsed=json.loads(json.dumps(payload,allow_nan=False))
    assert parsed==payload
    if 'observable' in parsed:
        assert all(type(flag) is bool for flag in payload['observable'])


@pytest.mark.parametrize('team',['red','blue'])
def test_actual_periodic_controller_logging_after_matching(team,monkeypatch,capsys):
    cls=MyDroneRedSemifinal if team=='red' else MyDroneBlueSemifinal
    drone=object.__new__(cls);drone.identifier=0
    payload=diagnostic()
    guard=SimpleNamespace(trusted=True,absolute_missing_steps=32,relocked=False,diagnostic=payload)
    for sensor in ('lidar_values','lidar_rays_angles','measured_gps_position','measured_compass_angle','odometer_values'):
        monkeypatch.setattr(cls,sensor,lambda _:None)
    monkeypatch.setattr(cls,'semantic_values',lambda _:[])
    if team=='red':
        from swarm_rescue.solutions.my_drone_red_fs1 import MyDroneRedFS1Gain
        monkeypatch.setattr(MyDroneRedFS1Gain,'_update_observation',lambda _:None)
        drone._pose_filter=guard;drone._steps=240;drone.elapsed_timestep=241
        drone._update_observation()
        prefix='V3_EVENT '
    else:
        guard.update=lambda *args:(np.zeros(2),0.)
        drone._pose_guard=guard;drone._home=None;drone._breadcrumbs=[]
        drone._tick=380;drone._last_region_log=140
        assert drone._pose()
        prefix='REGION_POSE '
    line=next(line for line in capsys.readouterr().out.splitlines() if line.startswith(prefix))
    event=json.loads(line[len(prefix):])
    actual=event['diagnostic'] if team=='red' else event
    assert actual['observable']==payload['observable']
