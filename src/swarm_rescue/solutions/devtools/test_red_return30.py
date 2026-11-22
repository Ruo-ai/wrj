"""Target alias and necessary-corner regressions, without native simulation."""
import math
import numpy as np
from types import SimpleNamespace
from swarm_rescue.solutions.red_return30 import ReturnLocal30, ReturnPlanner30
from swarm_rescue.solutions.red_return29 import ReturnPlanner29
from swarm_rescue.solutions.devtools.test_red_return29 import D,A,obj


class Corner(ReturnLocal30):
    def __init__(self,*args,shift=(0.,0.)):
        super().__init__(D,A)
        yy,xx=np.indices(self.shape)
        x=(xx+.5)*self.CELL-self.width/2+shift[0]
        y=self.height/2-(yy+.5)*self.CELL+shift[1]
        safe=(((x>=-6)&(x<=8)&(y>=-3)&(y<=3))|((x>=3)&(x<=8)&(y>=-3)&(y<=30)))
        self.free[:]=safe;self.static[:]=False;self.hits[:]=0
        self._clearance=np.where(safe,25.,0.);self._squeeze=np.zeros(self.shape,bool)


def planner(factory=Corner,goal=(5.,21.)):
    p=ReturnPlanner30(0,factory);p.observe(1,np.zeros(2),0.,np.zeros(3),D,A)
    p.trail=[np.asarray(goal,float)];p.trail_steps=[1];p.start_return();p.velocity[:]=0
    return p


def command(p,semantic=(),peers=None):
    return p.command(p.position,p.heading,p.velocity,D,A,semantic,peers)


def test_same_array_preferred_target_remains_unchanged_on_real_corner_route():
    fine=Corner();target=np.array([5.,21.]);before=target.copy()
    route=fine.route(np.zeros(2),target,None,target=target,limit=165)
    np.testing.assert_array_equal(target,before)
    np.testing.assert_allclose(route[-1],before)
    assert len(route)==3 and all(fine.line_clear(a,b) for a,b in zip(route,route[1:]))


def test_long_observed_direct_target_is_not_normalized_or_searched():
    fine=ReturnLocal30(D,A);target=np.array([153.,0.]);before=target.copy()
    route=fine.route(np.zeros(2),target,None,target=target,limit=165)
    np.testing.assert_array_equal(target,before)
    np.testing.assert_array_equal(route[-1],before)
    assert len(route)==2 and fine.last_reason=='observed_direct_target'


def test_direct_shortcut_respects_public_route_limit():
    fine=ReturnLocal30(D,A);target=np.array([150.,0.])
    assert fine.route(np.zeros(2),target,None,target=target,limit=20)==[]
    np.testing.assert_array_equal(target,[150,0])


def test_necessary_near_corner_is_retained_and_produces_safe_motion():
    p=planner();result=command(p)
    np.testing.assert_allclose(p.last_target,[3.,5.])
    assert result['forward']>0 and result['lateral']>0 and result['rotation']==0
    assert p._repair_goal_index30==0 and len(p._repair)==2
    assert p.stats['necessary_near_corner_retained']==1
    predicted=.6*np.array([result['forward'],result['lateral']])
    stopping=predicted*3+predicted*np.linalg.norm(predicted)/1.2
    assert p._local.line_clear(np.zeros(2),stopping)


def test_actual_observed_progress_to_corner_then_goal_changes_target_correctly():
    state={'position':np.zeros(2)}
    factory=lambda *args:Corner(*args,shift=state['position'])
    p=planner(factory);first=command(p);assert first['lateral']>0
    # Prescribed legal own motion along each verified leg, not a native flight.
    previous=np.zeros(2)
    for step,position in enumerate(([1.,1.6],[2.,3.3],[3.,5.]),2):
        position=np.asarray(position);delta=position-previous
        assert Corner(shift=previous).line_clear(np.zeros(2),delta)
        state['position']=position.copy()
        p.observe(step,position,0.,np.array([np.linalg.norm(delta),math.atan2(delta[1],delta[0]),0.]),D,A)
        command(p);previous=position
    # Hold for measured velocity to settle, while fresh geometry remains valid.
    for step in range(5,13):p.observe(step,previous,0.,np.zeros(3),D,A);second=command(p)
    assert second['lateral']>0 and second['rotation']==0
    np.testing.assert_allclose(p.last_target,[5.,21.])
    assert p._target_index==0


class TinyCorner:
    CELL=2.;TRACK_RADIUS=18.5;PLAN_RADIUS=19.914
    def __init__(self,*args):pass
    def sample(self,points):return np.full(len(np.atleast_2d(points)),30.)
    def line_clear(self,a,b):
        a,b=np.asarray(a),np.asarray(b)
        return bool((abs(a[1])<.01 and abs(b[1])<.01 and min(a[0],b[0])>=0 and max(a[0],b[0])<=2.01)
                    or (abs(a[0]-2)<.01 and abs(b[0]-2)<.01))
    def route(self,*args,**kwargs):return [np.zeros(2),np.array([2.,0.]),np.array([2.,30.])]


def test_necessary_two_pixel_corner_does_not_receive_zero_speed_cap():
    p=planner(TinyCorner,(2.,30.));result=command(p)
    assert result['forward']>0 and result['lateral']==0
    assert p.last_target==[2.,0.]
    predicted=.6*np.array([result['forward'],result['lateral']])
    assert p._local.line_clear(np.zeros(2),predicted*3+predicted*np.linalg.norm(predicted)/1.2)


def test_successful_repair_requires_unchanged_requested_terminal():
    class WrongTerminal(Corner):
        def route(self,*args,**kwargs):return [np.zeros(2),np.array([1.,1.])]
    p=planner(WrongTerminal);result=command(p)
    assert not any(result.values()) and p.stats['repair_target_mismatch_rejected']==1


def test_home_peer_yield_keeps_29_arbitration_before_holding():
    p=ReturnPlanner30(3);p.set_home([0,0]);p.observe(1,np.zeros(2),0.,np.zeros(3),D,A)
    p.start_return();p.velocity[:]=0
    result=command(p,[obj('DRONE',35,math.pi)],{1:(1,[-40.,0.],None)})
    assert result['forward']>0 and p.last_reason=='priority_peer_yield_motion'


def test_inventory_gate_and_bomb_memory_remain_inherited():
    p=ReturnPlanner30();p.observe(1,np.zeros(2),0.,np.zeros(3),D,A)
    assert not p.should_return(10,1,2) and p.return_requested and not p.returning
    assert ReturnPlanner30._bomb_rays is ReturnPlanner29._bomb_rays
    assert ReturnPlanner30._cache_affine is ReturnPlanner29._cache_affine


def test_ordinary_long_direct_return_command_keeps_original_speed_rule():
    def ready(cls):
        p=cls();p.observe(1,np.array([200.,0.]),0.,np.zeros(3),D,A)
        p.trail=[np.array([0.,0.]),np.array([100.,0.]),np.array([200.,0.])];p.trail_steps=[1,1,1]
        p.start_return();p.velocity[:]=0;return p
    p=ready(ReturnPlanner30);old=ready(ReturnPlanner29)
    assert command(p)==command(old)
