from unittest.mock import patch

from swarm_rescue.solutions.my_drone_red_entry37r2 import MyDroneRedEntry37R2
from swarm_rescue.solutions.my_drone_red_frontier34 import MyDroneRedCleanBoundary34
from swarm_rescue.solutions.devtools.test_red_entry37r1 import fixture
from swarm_rescue.solutions.devtools.test_red_entry37 import observed


def controller():
    old=fixture()
    cls=type('EntryFixture37R2',(MyDroneRedEntry37R2,),{'identifier':0,'drone_health':25})
    d=object.__new__(cls);d.__dict__.update(old.__dict__)
    return d


def test_real_navigation_entry_never_calls_legacy_static_dispatch_while_owned():
    d=controller();g=d._entry37;g.transition('CALIBRATE','regression')
    with patch.object(MyDroneRedCleanBoundary34,'_navigation',side_effect=AssertionError('Legacy planner entered')):
        for t in range(44,100):
            d._steps=t;observed(g,t,90,True);command=d._navigation()
            assert set(command)=={'forward','lateral','rotation','grasper','place_bomb'}
            if g.state=='REENTER':break
    assert g.calibrated and g.state=='REENTER'
    assert g.stats['owned_navigation_ticks']>24
    assert not g.stats['finish_task_preempted']


def test_original_controller_is_used_when_idle_or_preempted():
    d=controller();d._entry37.release('test');d._entry37.missing=0
    with patch.object(MyDroneRedCleanBoundary34,'_navigation',return_value={'original':True}) as original:
        assert d._navigation()=={'original':True}
        original.assert_called_once()
    d=controller();d._progress=lambda:.61
    with patch.object(MyDroneRedCleanBoundary34,'_navigation',return_value={'original':True}) as original:
        assert d._navigation()=={'original':True}
        assert not d._entry37.active
        original.assert_called_once()
