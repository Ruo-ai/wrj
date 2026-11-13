import pytest as _retired_pytest
_retired_pytest.importorskip('swarm_rescue.solutions.my_drone_red_advanced5', reason='Historical strategy retired by user; source archived')
"""Block1: distinguish actual progress, finite waiting, oscillation and recovery."""
import numpy as np
import pytest
from types import SimpleNamespace
from swarm_rescue.solutions import my_drone_red_advanced5 as strategy


def simulate(path, route=None, gains=None, waiting=None):
    monitor=strategy.ProgressMonitor()
    for i,position in enumerate(path):
        steps=i*6
        current_route=route(i) if callable(route) else route or []
        state=monitor.update(steps,np.array(position,float),current_route,
                             gains(i) if gains else 0,waiting)
    return monitor,state


def test_straight_progress_is_not_stall():
    _,state=simulate([(i*6,0) for i in range(30)],[(180.,0.)])
    assert state['kind']=='normal' and not state['confirmed']


def test_u_turn_uses_along_route_progress_not_goal_distance():
    route=[(0.,0.),(0.,100.),(100.,100.),(100.,0.)]
    a=strategy.ProgressMonitor.projected_progress(np.array([0.,60.]),route)
    b=strategy.ProgressMonitor.projected_progress(np.array([80.,100.]),route)
    assert b>a and a==pytest.approx(60.) and b==pytest.approx(180.)
    _,state=simulate([(0,i*4) for i in range(25)],route)
    assert state['kind']=='normal'


def test_stationary_with_new_observations_is_not_failure():
    _,state=simulate([(0,0)]*30,gains=lambda i:i*2)
    assert state['kind']=='normal'


def test_small_oscillation_confirmed_with_no_information():
    _,state=simulate([(10*(-1)**i,0) for i in range(30)],[(100.,0.)])
    assert state['confirmed'] and state['kind']=='oscillation'
    assert state['reversals']>=3


def test_changing_waypoints_does_not_hide_cycle():
    _,state=simulate([(10*(-1)**i,0) for i in range(30)],
                     lambda i:[(100.*(-1)**i,0.)])
    assert state['confirmed'] and state['kind']=='goal_churn'


def test_short_yield_does_not_trigger_escape():
    _,state=simulate([(0,0)]*5,waiting='existing_yield')
    assert state['kind']=='waiting' and not state['confirmed']


def test_waiting_does_not_mask_indefinite_blockage():
    _,state=simulate([(0,0)]*35,waiting='existing_yield')
    assert state['kind']=='stationary' and state['confirmed']


def test_stationary_requires_persistence():
    _,short=simulate([(0,0)]*8)
    _,long=simulate([(0,0)]*20)
    assert not short['confirmed'] and long['confirmed']


def test_small_position_noise_is_not_counted_as_direction_reversals():
    _,state=simulate([(0.1*(-1)**i,0) for i in range(25)])
    assert state['reversals']==0 and state['kind']=='stationary'


def test_progress_gate_does_not_cancel_bomb_confirmation():
    dummy=SimpleNamespace(_holding=False,_attempt_step=100,_steps=103,
                          _yield_until=0,_bomb_target=None)
    assert strategy.MyDroneRedAdvanced5._waiting_reason(dummy)=='placement_confirmation'


def test_existing_escape_is_not_restarted_by_no_progress(monkeypatch):
    cls=strategy.MyDroneRedAdvanced5
    drone=object.__new__(cls)
    drone._progress_monitor=strategy.ProgressMonitor()
    drone._steps=100;drone._escape_until=120
    drone._allow_recovery=lambda reason:True
    calls=[]
    monkeypatch.setattr(strategy._Advanced2Baseline,'_start_recovery',
                        lambda *a,**kw:calls.append(1))
    drone._start_recovery(np.ones(180)*300,np.linspace(-np.pi,np.pi,180),'no_progress')
    assert not calls and drone._escape_until==120


@pytest.mark.parametrize('good',[False,True])
def test_recovery_is_verified_from_displacement_or_information(good):
    drone=object.__new__(strategy.MyDroneRedAdvanced5)
    drone._steps=100;drone._position=np.array([50. if good else 2.,0.])
    drone._loop_information=0;drone._loop_failures=[];events=[]
    drone._event=lambda kind,**data:events.append((kind,data))
    drone._loop_recovery=dict(start=10,position=np.zeros(2),information=0,
                             reason='test',direction=0.,attempt=1,goal=None)
    drone._verify_loop_recovery()
    assert drone._loop_recovery is None
    assert events[0][1]['successful']==good
    assert len(drone._loop_failures)==(0 if good else 1)


def test_prediction_brakes_when_measurements_missing():
    force=strategy.predictive_correction(np.array([np.nan]),np.zeros(1),
                                        np.array([2.,1.]),0.,np.ones(2))
    assert np.all(force<0)


def test_prediction_does_not_accelerate_into_close_wall():
    angles=np.linspace(-np.pi,np.pi,180,endpoint=False)
    d=np.where(np.cos(angles)>.05,35/np.maximum(np.cos(angles),.05),300.)
    force=strategy.predictive_correction(d,angles,np.array([4.,0.]),0.,np.array([1.,0.]))
    assert force[0]<0


def test_prediction_keeps_clear_straight_motion():
    angles=np.linspace(-np.pi,np.pi,180,endpoint=False)
    force=strategy.predictive_correction(np.full(180,300.),angles,np.zeros(2),0.,np.array([1.,0.]))
    assert force[0]>0 and abs(force[1])<1e-9


def test_returning_along_route_does_not_retain_previous_forward_credit():
    path=[(0.,0.)]+[(40.,0.),(0.,0.)]*10
    _,state=simulate(path,[(100.,0.)])
    assert state['advance']==0 and state['net']==0


def test_wait_resets_stale_progress_metrics():
    monitor,_=simulate([(i*10.,0.) for i in range(30)],[(400.,0.)],gains=lambda i:10*i)
    state=monitor.update(180,np.array([300.,0.]),[(400.,0.)],300,'existing_yield')
    assert state['information']==state['advance']==state['net']==state['age']==0


def test_failed_direction_changes_escape_without_changing_map():
    drone=object.__new__(strategy.MyDroneRedAdvanced5)
    drone._explore=strategy._M05_ExplorationGrid((600,600))
    drone._position=np.zeros(2);drone._heading=0.;drone._launch_angle=0.;drone._escape_heading=0.
    drone._event=lambda *args,**kwargs:None
    before=drone._explore.obstacle.copy()
    a=np.linspace(-np.pi,np.pi,180,endpoint=False)
    heading=drone._failure_aware_heading(np.full(180,300.),a,[dict(direction=0.)])
    assert abs(heading)>1.4
    np.testing.assert_array_equal(drone._explore.obstacle,before)


def test_goal_repair_keeps_goal_only_with_valid_alternative(monkeypatch):
    drone=object.__new__(strategy.MyDroneRedAdvanced5)
    grid=strategy._M05_ExplorationGrid((300,300));grid.free[:]=True
    grid.obstacle[7,7]=3
    drone._explore=grid;drone._position=grid.point((7,4))
    drone._route=[(7,6),(7,10)];drone._goal_step=100;drone._steps=130
    drone._bomb_target=None;drone._gathering=False;drone._loop_goal_repairs={}
    drone._progress_monitor=strategy.ProgressMonitor()
    drone._progress_monitor.last.update(switches=4,net=5,information=0)
    drone._waypoint_progress=strategy._M13_WaypointProgress()
    drone._event=lambda *args,**kwargs:None
    releases=[]
    monkeypatch.setattr(strategy._Advanced2Baseline,'_release_stale_route',lambda *args:releases.append(1))
    drone._release_stale_route('new_wall')
    assert not releases and drone._route[-1]==(7,10)
    assert (7,7) not in drone._route and not strategy._M13_obstructed_route(grid,drone._position,drone._route)
    # The original commitment deadline is not renewed by repairs.
    drone._steps=191
    drone._release_stale_route('new_wall')
    assert releases and drone._goal_step==100


def test_old_progress_does_not_suppress_current_blockage():
    path=[(-200+20*i,0.) for i in range(11)]+[(0.,10*(-1)**i) for i in range(10)]
    monitor,state=simulate(path,[(200.,0.)])
    assert state['net']>65 and state['recent_net']<25
    drone=object.__new__(strategy.MyDroneRedAdvanced5)
    drone._steps=121;drone._escape_until=0;drone._progress_monitor=monitor
    assert drone._allow_recovery('no_progress')


def test_recent_information_protects_useful_observation():
    monitor,state=simulate([(0.,0.)]*25,gains=lambda i:i*2)
    drone=object.__new__(strategy.MyDroneRedAdvanced5)
    drone._steps=145;drone._escape_until=0;drone._progress_monitor=monitor
    drone._loop_suppressed_at=145
    assert not drone._allow_recovery('no_progress')


def test_strong_loop_escalates_but_weak_stall_cannot_restart_escape():
    drone=object.__new__(strategy.MyDroneRedAdvanced5)
    drone._progress_monitor=strategy.ProgressMonitor()
    drone._steps=110;drone._escape_until=120
    assert not drone._allow_recovery('no_progress')
    assert drone._allow_recovery('local_loop')

