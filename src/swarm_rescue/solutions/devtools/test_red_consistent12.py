"""RL12 behavioural checks; synthetic truth belongs only to these fixtures."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_consistent12 import (
    ConsistentMapper, ContinuousHeadingMapper, ContinuousHeadingMatcher,
    MyDroneRedConsistent12, MyDroneRedHeading12)
from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMapper, MyDroneRedLines11
from swarm_rescue.solutions.devtools.red_lines11_fixture import inputs
from swarm_rescue.solutions.devtools.test_red_lines11 import ROOM
from swarm_rescue.solutions.semifinal_localization import transform, wrap


@pytest.mark.parametrize('case',['ideal_oblique','noisy_room','noisy_corridor'])
def test_lag_changes_only_uncommitted_scans_not_integrated_current_pose(case):
    old=WallLineMapper((500,500));new=ConsistentMapper((500,500));old_hits=None
    for k,row in enumerate(inputs(case)):
        old.update(*row[:5]);new.update(*row[:5])
        np.testing.assert_array_equal(old.position,new.position)
        assert old.heading==new.heading
        if k==63:old_hits=new.hits.copy()
        if 64<=k<360:np.testing.assert_array_equal(new.hits,old_hits)
    assert new.stats['lag_max_scans']<=new.MAX_LAG//4
    assert new.stats['lag_corrections']>0


def test_disabled_options_exactly_reproduce_rl11():
    class HeadingOff(ContinuousHeadingMapper): CONTINUOUS_HEADING=False
    baseline=WallLineMapper((500,500));lag=ConsistentMapper((500,500));lag.SMOOTH_PENDING=False
    heading=HeadingOff((500,500))
    for row in inputs('noisy_room'):
        for m in (baseline,lag,heading):m.update(*row[:5])
    for m in (lag,heading):
        np.testing.assert_array_equal(m.mask(),baseline.mask())
        np.testing.assert_array_equal(m.position,baseline.position)
        assert m.heading==baseline.heading


@pytest.mark.parametrize('angle',[.06,math.pi-.02,-math.pi+.02])
def test_compass_available_heading_uses_prior_walls_before_inserting(angle):
    owner=SimpleNamespace(heading=angle)
    matcher=ContinuousHeadingMatcher((600,600),owner)
    actual=wrap(angle-.06)
    refs=[transform(s,np.zeros(2),actual) for s in ROOM]
    matcher.references=[s.copy() for s in refs]
    matcher.frames.append(np.vstack(refs));matcher.absolute=True;matcher.lines=list(ROOM)
    # The resulting angle must move halfway toward the historical geometry.
    matcher.add(np.repeat(np.vstack(ROOM),2,axis=0),np.zeros(2),angle)
    assert wrap(owner.heading-angle)==pytest.approx(-.03)
    assert matcher.stats['absolute_heading_accepted']==1


def test_single_wall_cannot_correct_absolute_heading():
    owner=SimpleNamespace(heading=.07);matcher=ContinuousHeadingMatcher((600,600),owner)
    matcher.references=list(ROOM[:1]);matcher.lines=list(ROOM[:1]);matcher.frames.append(np.vstack(ROOM))
    matcher.absolute=True;matcher.add(np.repeat(ROOM[0],7,axis=0),np.zeros(2),owner.heading)
    assert owner.heading==.07 and matcher.stats['absolute_heading_accepted']==0


def test_navigation_placement_and_sensor_access_unchanged():
    for cls in (MyDroneRedConsistent12,MyDroneRedHeading12):
        assert cls._navigation is MyDroneRedLines11._navigation
        assert cls._deployment is MyDroneRedLines11._deployment
        assert cls._update_observation is MyDroneRedLines11._update_observation
        assert not cls.SLOW_OUTAGE


def test_navigation_disabled_and_normal_zone_exact_baseline():
    from swarm_rescue.solutions.my_drone_red_navlines12 import NavigationLinePose
    from swarm_rescue.solutions.my_drone_red_advanced2 import _M01_PoseFilter
    base=_M01_PoseFilter();on=NavigationLinePose((500,500));off=NavigationLinePose((500,500));off.ENABLED=False
    for row in inputs('noisy_room')[:64]:
        gps,h,odom,d,a=row[:5]
        for p in (on,off):p.prepare(d,a)
        baseline=base.update(gps,h,odom)
        for p in (on,off):
            position,heading=p.update(gps,h,odom)
            np.testing.assert_array_equal(position,baseline[0]);assert heading==baseline[1]


def test_navigation_integrates_every_missing_step_and_wraps_without_lidar():
    from swarm_rescue.solutions.my_drone_red_navlines12 import NavigationLinePose
    p=NavigationLinePose((500,500));p.update(np.zeros(2),math.pi-.01,[0.,0.,0.])
    position,heading=p.update(None,None,[2.,0.,.03])
    np.testing.assert_allclose(position,2*np.array([math.cos(math.pi-.01),math.sin(math.pi-.01)]))
    assert heading==pytest.approx(-math.pi+.02)
    for _ in range(24):p.update(None,None,[0.,0.,.001])
    assert p.stats['accepted']==0 and p.stats['local_reference_reset']==0


def test_stable_line_reuses_a5_without_lag_or_absolute_heading():
    from swarm_rescue.solutions.my_drone_red_consistent12 import StableLineMapper
    from swarm_rescue.solutions.my_drone_red_lines11 import WallLineMatcher
    m=StableLineMapper((500,500))
    assert type(m.matcher) is WallLineMatcher
    m.STABLE_ANCHOR=False;baseline=WallLineMapper((500,500))
    for row in inputs('noisy_corridor'):
        m.update(*row[:5]);baseline.update(*row[:5])
    np.testing.assert_array_equal(m.mask(),baseline.mask())
