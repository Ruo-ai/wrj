"""Single generated-map worker, with passive telemetry outside controller calls."""
import json
import math
import os
from pathlib import Path
import sys


def install_telemetry(destination,lifecycle=False):
    from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
    original = GuiSR.on_update
    state = dict(submission_step=None,submission_walltime=None,last_bomb_step=None,
                 last_bomb_walltime=None,per_drone=[])
    previous = {}
    observer=None
    if lifecycle:
        from ...blue_lifecycle import LifecycleObserver
        observer=LifecycleObserver()
    carried={};pending={};bomb_ids={}
    state.update(transport=dict(discovered=None,semantic_sightings=0,pickups=0,verified_disposals=0,unconfirmed_losses=0,
                                disposed_transport_steps=[],events=[]),pose_samples=[],mapping_samples=[])
    def bomb_id(bomb):
        key=id(bomb)
        if key not in bomb_ids:bomb_ids[key]=len(bomb_ids)+1
        return bomb_ids[key]
    def observed(gui,delta):
        for drone in gui._drones:
            key = int(drone.identifier)
            if key not in previous:
                position = tuple(map(float,drone.true_position()))
                previous[key] = (position,drone.carried_bombs_count())
                state['per_drone'].append(dict(id=key,distance_px=0.,placed_count=0,
                    initial_inventory=drone.carried_bombs_count(),remaining_inventory=drone.carried_bombs_count()))
        if observer:observer.begin(gui)
        result = original(gui,delta)
        if observer:
            observer.end(gui)
            state['lifecycle']=observer.export(gui)
        step=int(gui._elapsed_timestep)
        for drone in gui._drones:
            key = int(drone.identifier)
            position = tuple(map(float,drone.true_position()))
            inventory = drone.carried_bombs_count()
            old_pos,old_inventory = previous[key]
            row = next(r for r in state['per_drone'] if r['id']==key)
            row['distance_px'] += math.dist(old_pos,position)
            row['remaining_inventory'] = inventory
            if inventory<old_inventory:
                row['placed_count'] += old_inventory-inventory
                state['last_bomb_step'] = int(gui._elapsed_timestep)
                state['last_bomb_walltime'] = float(gui._elapsed_walltime)
            if getattr(drone,'_exploration_map_submission',None) is not None and state['submission_step'] is None:
                state['submission_step'] = int(gui._elapsed_timestep)
                state['submission_walltime'] = float(gui._elapsed_walltime)
            previous[key] = position,inventory
            # Passive offline truth diagnostics. These values are never made
            # available to controller calls or radio messages.
            if step%30==0 and getattr(drone,'_position',None) is not None:
                estimate=list(map(float,drone._position))
                state['pose_samples'].append(dict(drone=key,step=step,truth=position,estimate=estimate,
                    error=math.dist(position,estimate),gps_missing=drone.measured_gps_position() is None))
                mapper=getattr(drone,'_submission_mapper',None)
                if mapper is not None and mapper.position is not None:
                    mapped=list(map(float,mapper.position))
                    state['mapping_samples'].append(dict(drone=key,step=step,truth=position,estimate=mapped,
                        error=math.dist(position,mapped),missing=mapper.missing,pending=len(mapper.pending),
                        stats=dict(mapper.stats)))
            now={bomb_id(b):b for b in drone.grasped_bombs()}
            before=carried.get(key,{})
            t=state['transport']
            for ident,bomb in now.items():
                if ident not in before:
                    t['pickups']+=1;pending[(key,ident)]=(bomb,step)
                    t['events'].append(dict(event='pickup',drone=key,bomb=ident,step=step))
            for ident in before.keys()-now.keys():
                bomb,started=pending.pop((key,ident),(before[ident],step))
                disposed=bool(getattr(bomb,'disposed',False))
                t['verified_disposals' if disposed else 'unconfirmed_losses']+=1
                if disposed:t['disposed_transport_steps'].append(step-started)
                t['events'].append(dict(event='disposed' if disposed else 'grasp_lost',drone=key,
                    bomb=ident,step=step,transport_steps=step-started))
            carried[key]=now
            for obj in drone.semantic_values() or ():
                if getattr(obj.entity_type,'name','')=='BOMB':
                    # The semantic API has no persistent bomb ID. Count sightings
                    # separately; a unique discovery count would be invented.
                    t['semantic_sightings']+=1
        return result
    GuiSR.on_update = observed
    return state


def run(request_path):
    from dataclasses import asdict
    from ... import evaluate_verified_revision as ev
    from ... import evaluate_red as red
    from ..map_generator.native_map import register
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreConfig
    request = ev.read(request_path)
    phase = request['phase']
    run_dir = Path(request['run_dir'])
    from .scenarios import resolve_map,scenario,engine_profile,verify_recording
    conditions=scenario(request['map_file'],request.get('official_map'))
    expected_count=10 if phase=='blue' and conditions['stage']=='semifinal' else 5
    if request['drones']!=expected_count or request['zones']!=conditions['zones']:
        raise ValueError('Requested drones/zones differ from scenario contract')
    profile=engine_profile()
    cls = resolve_map(request)
    if cls.__name__!=request['map']:
        raise ValueError('Registered map identity mismatch')
    red.MAPS = (request['map'],)
    if request.get('smoke'):
        init = cls.__init__
        def limited(self,*args,**kwargs):
            init(self,*args,**kwargs)
            self._max_timestep_limit = request.get('diagnostic_steps',8)
            if not 1<=self._max_timestep_limit<=128:raise ValueError('Diagnostic step budget outside 1..128')
            self._max_walltime_limit = 20
        cls.__init__ = limited
    original_init=cls.__init__
    actual={}
    def checked_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        from swarm_rescue.simulation.elements.sensor_disablers import NoComZone,NoGpsZone
        actual.update(drones=len(self.drones),zones=sorted(z.name for z in self._zones_config),
            max_steps=self._max_timestep_limit,max_walltime=self._max_walltime_limit,
            instantiated_zone_types=sorted({('NO_COM_ZONE' if isinstance(e,NoComZone) else 'NO_GPS_ZONE')
                for e in self.playground.elements if isinstance(e,(NoComZone,NoGpsZone))}))
        if actual['drones']!=expected_count or actual['zones']!=request['zones'] or actual['instantiated_zone_types']!=request['zones']:
            raise ValueError('Native map conditions differ from requested scenario')
        if not request.get('smoke') and conditions['stage']=='semifinal' and request.get('official_map') in (None,'Map06','Map07t4','Map08','Map09'):
            if (actual['max_steps'],actual['max_walltime'])!=(5400,180):raise ValueError('Unexpected semifinal map limits')
    cls.__init__=checked_init
    telemetry = install_telemetry(run_dir/'telemetry.json',request.get('lifecycle_diagnostics',False))
    try:
        if request.get('diagnostic_profile'):
            if not request.get('smoke'):raise ValueError('Profiling is diagnostic-only')
            import cProfile
            profiler=cProfile.Profile()
            try:code=profiler.runcall(ev.worker,request_path,phase)
            finally:profiler.dump_stats(str(run_dir/'profile.pstats'))
        else:
            code = ev.worker(request_path,phase)
        result = ev.read(run_dir/'result.json')
        result.update(ev.controller_identity(request['controller_selector']))
        recordings=list((Path(request['engine_dir'])/'recordings').glob('*.npz'))
        if result.get('status')=='ok':
            if len(recordings)!=1: raise ValueError('Expected one recording')
            verify_recording(recordings[0],request)
        result.update(actual_conditions=actual,engine_profile=profile,phase=phase,smoke=bool(request.get('smoke')),score_configuration=asdict(PlaceScoreConfig()))
        if phase=='red':
            telemetry['inventory_matches_layout'] = sum(r['placed_count'] for r in telemetry['per_drone'])==result.get('placed_count')
        ev.atomic_json(run_dir/'result.json',result)
        return code
    finally:
        ev.atomic_json(run_dir/'telemetry.json',telemetry)


if __name__=='__main__':
    # Parent attaches the Windows job before any engine/OpenGL/controller import.
    if sys.stdin.readline().strip()!='GO':
        raise SystemExit('Parent did not authorize worker startup')
    if os.name!='nt':
        from .resources import linux_limit
        linux_limit(int(os.environ['G3_MEMORY_BYTES']))
    raise SystemExit(run(Path(sys.argv[1])))
