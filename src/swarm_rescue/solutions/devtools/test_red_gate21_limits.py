"""The pre-existing RN7 timeout/fallback must survive passive recording."""
import numpy as np
import pytest
from swarm_rescue.solutions.my_drone_red_rn7 import PassageMotion
from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
from swarm_rescue.solutions.devtools.test_advanced5_narrow import corridor
from swarm_rescue.solutions.devtools.red_gate21_diagnostics import recording,verify


@pytest.mark.parametrize('reason,since,last_progress',[
    ('task_timeout',0,300),('no_progress',200,250)])
def test_existing_limit_exits_preserved(tmp_path,reason,since,last_progress):
    d,a=corridor(55)
    def run():
        motion=PassageMotion();motion.tick=300;motion.since=since;motion.last_progress=last_progress
        motion.best_remaining=0.;motion.valid_odom=True;motion.velocity=np.array([3.,0.]);motion.path=[np.array([80.,0.])]
        grid=_M19_SurveyGrid((500,500),lambda *a,**k:None)
        command=motion.command(d,a,grid,np.zeros(2),0.,0.)
        assert not motion.path and motion.stats[reason]==1 and command['forward']<0
        return command,dict(motion.stats),motion.cooldown
    baseline=run();path=tmp_path/'limits.json'
    with recording(path):observed=run()
    assert baseline==observed and not verify(path)['errors']
