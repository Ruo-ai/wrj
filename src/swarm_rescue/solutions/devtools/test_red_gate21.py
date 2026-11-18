"""Observer invariance and historical endpoint transport, not new flight claims."""
import math
import numpy as np
import pytest

from swarm_rescue.solutions.my_drone_red_rn7 import PassageMotion,LocalPassage,rotate,HULL
from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor
from swarm_rescue.solutions.devtools.red_gate21_diagnostics import recording,verify
from swarm_rescue.solutions.devtools.red_gate21_audit import recovered_endpoint,task_state


@pytest.mark.parametrize('width,angle,allow,velocity',[
    (55,0.,True,(0.,0.)),(55,math.pi/4,True,(0.,0.)),
    (25,0.,True,(0.,0.)),(55,0.,False,(0.,0.)),(46,0.,True,(0.,6.))])
def test_observer_preserves_route_command_and_grid(tmp_path,width,angle,allow,velocity):
    d,a=corridor(width,angle=angle)
    def run():
        grid=_M19_SurveyGrid((500,500),lambda *a,**k:None);motion=PassageMotion()
        motion.tick=100;motion.valid_odom=True;motion.velocity=np.array(velocity)
        state=(grid.free.copy(),grid.obstacle.copy(),grid.visits.copy())
        cmd=motion.command(d,a,grid,np.zeros(2),0.,angle,allow=allow)
        for before,after in zip(state,(grid.free,grid.obstacle,grid.visits)):np.testing.assert_array_equal(before,after)
        return cmd,[p.tolist() for p in motion.path],dict(motion.stats),motion.cooldown
    expected=run();path=tmp_path/'probe.json'
    with recording(path):actual=run()
    assert actual==expected
    result=verify(path)
    assert not result['errors'] and not result['rank_mismatches']
    assert LocalPassage.TRACK_RADIUS>np.linalg.norm(HULL,axis=1).max()


def test_inverse_endpoint_includes_odometric_turn_and_slip():
    point=np.array([85.,-30.]);rows=np.zeros((4,11));rows[:,3]=1
    current=point.copy()
    for t in (2,3,4):
        rows[t-1,8:11]=[2.5,.3,.2]
        current=rotate(current-2.5*np.array([math.cos(.3),math.sin(.3)]),-.2)
    recovered,step,error=recovered_endpoint(0,1,[dict(tick=1,reason='entered')],
        [dict(step=4,rn7_path=[current.tolist()])],{'drone_0':rows})
    np.testing.assert_allclose(recovered,point,atol=1e-12)
    assert step==4 and error is None


def test_replanned_endpoint_cannot_represent_original_selection():
    result=recovered_endpoint(0,1,[dict(tick=1,reason='entered'),dict(tick=2,reason='local_replan')],
        [dict(step=4,rn7_path=[[80.,0.]])],{'drone_0':np.zeros((4,11))})
    assert result[0] is None


def test_existing_cooldown_and_active_task_not_ranking_misses():
    assert task_state(4,105,[],[])['certainly_in_cooldown']
    state=task_state(2,162,[dict(tick=143,reason='entered'),dict(tick=202,reason='completed')],[])
    assert state['active_by_events']
    assert task_state(2,250,[dict(tick=202,reason='completed')],[])['certainly_in_cooldown']
