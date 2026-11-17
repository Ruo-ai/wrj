"""Passive R20 plan decisions/snapshots and motion attribution. No policy edits."""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np

from swarm_rescue.solutions.devtools.red_pair19_batch import write


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x, (tuple,list,set)): return [plain(v) for v in x]
    return x


@contextmanager
def recording(destination):
    from swarm_rescue.simulation.gui_map import gui_sr
    from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
    old_control,old_plan=gui_sr.run_control,_M19_SurveyGrid.plan
    plans=[];samples=defaultdict(list);arrays={};errors=[];seconds=0.;truncated=0;identities={};refs={}
    def plan(grid,*args,**kwargs):
        nonlocal seconds,truncated
        before=time.perf_counter();record=None
        try:
            if len(plans)>=2500:truncated+=1
            else:
                index=len(plans);key=f'plan_{index}'
                state={}
                for k,v in vars(grid).items():
                    if callable(v):continue
                    if isinstance(v,np.ndarray):
                        arrays[key+'_'+k]=v.copy();state[k]={'array':key+'_'+k}
                    else:state[k]=plain(v)
                captured_args=list(args);captured_kwargs=dict(kwargs)
                # The planner consumes blocked-goal KEYS, not their expiry
                # values. Preserve tuple cells as arrays instead of JSON keys.
                if len(captured_args)>5:captured_args[5]=list(captured_args[5])
                if 'blocked_goals' in captured_kwargs:captured_kwargs['blocked_goals']=list(captured_kwargs['blocked_goals'])
                record=dict(index=index,owner=identities.get(id(grid)),step=grid.step,state=state,
                            args=plain(captured_args),kwargs=plain(captured_kwargs))
        except Exception as e:errors.append(repr(e))
        seconds+=time.perf_counter()-before
        route=old_plan(grid,*args,**kwargs)
        if record is not None:
            record['selected_route']=plain(route);plans.append(record)
        return route
    def control(drone):
        nonlocal seconds
        identities[id(drone._explore)]=int(drone.identifier)
        refs[str(drone.identifier)]=drone
        command=old_control(drone);before=time.perf_counter()
        try:
            tick=drone._steps
            if tick%24==0:
                rn=drone._rn7;grid=drone._explore
                samples[str(drone.identifier)].append(plain(dict(step=tick,truth=drone.true_position(),
                    position=drone._position,heading=drone._heading,route=drone._route,
                    goal=None if not drone._route else grid.point(drone._route[-1]),bomb_target=drone._bomb_target,
                    inventory=drone.carried_bombs_count(),gathering=drone._gathering,holding=drone._holding,
                    escape_until=drone._escape_until,yield_until=drone._yield_until,slow_ticks=drone._slow_ticks,
                    stalled_steps=drone._stalled_steps,goal_step=drone._goal_step,peers=drone._peers,
                    received_message_count=len(drone.communicator.received_messages),
                    communication_disabled=drone.communicator_is_disabled(),command=command,
                    free_cells=int(grid.free.sum()),rn7_path=rn.path,rn7_stats=dict(rn.stats),
                    rn7_episodes=rn.episodes,lifetime_episodes=getattr(rn,'lifetime_episodes',rn.episodes),
                    renewals=getattr(rn,'renewals',0),cooldown=rn.cooldown)))
        except Exception as e:errors.append(repr(e))
        seconds+=time.perf_counter()-before
        return command
    gui_sr.run_control=control;_M19_SurveyGrid.plan=plan
    try:yield
    finally:
        gui_sr.run_control=old_control;_M19_SurveyGrid.plan=old_plan
        np.savez_compressed(Path(destination).with_suffix('.npz'),**arrays)
        final={k:dict(stalled_steps=d._stalled_steps,yield_steps=d._yield_steps,replans=d._replans,
                      rn7_stats=dict(d._rn7.stats),lifetime_episodes=getattr(d._rn7,'lifetime_episodes',d._rn7.episodes),
                      inventory=d.carried_bombs_count(),step=d._steps) for k,d in refs.items()}
        write(destination,dict(schema='r20-plan-trace-v1',plans=plans,drones=dict(samples),final=final,errors=errors,
                               truncated_plans=truncated,diagnostic_seconds=seconds,
                               scope='legal pre-plan state snapshots; truth only after guarded control'))


def verify_decisions(path):
    """Recompute every captured planning decision without flight or future data."""
    from swarm_rescue.solutions.my_drone_red_advanced2 import _M19_SurveyGrid
    from swarm_rescue.solutions.devtools.red_pair19_batch import read
    doc=read(path);bad=[]
    with np.load(Path(path).with_suffix('.npz'),allow_pickle=False) as arrays:
        for p in doc['plans']:
            state=p['state'];grid=_M19_SurveyGrid((state['width'],state['height']),lambda *a,**k:None)
            for k,v in state.items():
                setattr(grid,k,arrays[v['array']].copy() if isinstance(v,dict) and 'array' in v else v)
            grid.avoid_regions=[(np.asarray(point),t) for point,t in grid.avoid_regions]
            grid.survey_history=[np.asarray(point) for point in grid.survey_history]
            if grid.survey is not None:grid.survey['cell']=tuple(grid.survey['cell'])
            args=p['args'];args[0]=np.asarray(args[0]);args[1]=np.asarray(args[1]);args[4]=[np.asarray(v) for v in args[4]]
            if len(args)>5:args[5]={tuple(v) for v in args[5]}
            if 'blocked_goals' in p['kwargs']:p['kwargs']['blocked_goals']={tuple(v) for v in p['kwargs']['blocked_goals']}
            route=grid.plan(*args,**p['kwargs'])
            if plain(route)!=p['selected_route']:bad.append(p['index'])
    return dict(plans=len(doc['plans']),mismatches=bad,errors=doc['errors'],truncated=doc['truncated_plans'])
