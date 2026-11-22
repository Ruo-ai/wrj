from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from swarm_rescue.solutions.my_drone_red_entry37 import MyDroneRedEntry37
from swarm_rescue.solutions.my_drone_red_entry37r1 import MyDroneRedEntry37R1
from swarm_rescue.solutions.my_drone_red_frontier34 import MyDroneRedCleanBoundary34
from swarm_rescue.solutions.devtools.test_red_entry37 import lost, observed, D, A


def fixture():
    cls=type('EntryFixture37',(MyDroneRedEntry37R1,),{'identifier':0,'drone_health':25})
    d=object.__new__(cls)
    d._entry37=lost();d._entry37.step(D,A,True,.1)
    d._position=np.array([90.,0.]);d._heading=0.;d._steps=60
    d._velocity=np.zeros(2);d._explore=SimpleNamespace(step=0,observe=Mock(),
        shared_visited=np.zeros((3,3),bool),visits=np.zeros((3,3)))
    d._radio_maps=SimpleNamespace(publish=Mock());d._map=SimpleNamespace(mask=lambda:np.zeros((3,3),bool))
    d._waypoint_progress=SimpleNamespace(reset=Mock(),last_advance=0)
    d._loop_watch=SimpleNamespace(samples=[])
    d._escape_until=d._yield_until=0;d._attempt_step=None;d._bomb_target=None
    d._gathering=d._exploration_submitted=False;d._route=[]
    d._rn7=SimpleNamespace(path=[]);d._segment_motion=SimpleNamespace(release=Mock())
    d._progress=lambda:.1;d.lidar_values=lambda:D;d.lidar_rays_angles=lambda:A
    d.semantic_values=lambda:();d.odometer_values=lambda:np.zeros(3)
    return d


def test_active_motion_preserves_observation_and_publish_without_stall_planner():
    d=fixture()
    with patch.object(MyDroneRedEntry37,'_navigate_motion',side_effect=AssertionError('Competing planner called')):
        for t in range(60,180):
            d._steps=t;d._navigate_motion()
    assert d._explore.observe.call_count==20
    assert d._radio_maps.publish.call_count==4
    assert d._escape_until==0 and d._motion_anchor is None


def test_outer_navigation_can_cross_multiple_60step_boundaries_and_calibrate():
    d=fixture();g=d._entry37;g.transition('CALIBRATE','regression')
    with patch.object(MyDroneRedCleanBoundary34,'_navigation',lambda self:self._navigate_motion()):
        for t in range(44,100):
            d._steps=t;observed(g,t,90,True);d._navigation()
            if g.state=='REENTER':break
    assert g.calibrated and g.state=='REENTER'
    assert not g.stats['finish_task_preempted']
    assert d._escape_until==0


def test_inactive_motion_delegates_and_submission_radio_remain_identical():
    d=fixture();d._entry37.release('test')
    with patch.object(MyDroneRedEntry37,'_navigate_motion',return_value={'unchanged':True}) as original:
        assert d._navigate_motion()=={'unchanged':True}
        original.assert_called_once()
    for name in ('_submit','_receive','define_message_for_all','_exchange_state'):
        assert getattr(MyDroneRedEntry37R1,name) is getattr(MyDroneRedCleanBoundary34,name)
    assert MyDroneRedEntry37R1.FREEZE_AT==MyDroneRedCleanBoundary34.FREEZE_AT


def test_real_deadline_still_releases_entry_ownership():
    d=fixture();d._progress=lambda:.61
    with patch.object(MyDroneRedCleanBoundary34,'_navigation',return_value={'forward':0.}):
        d._navigation()
    assert not d._entry37.active
    assert d._entry37.stats['finish_task_preempted']==1
