"""Explicit semifinal map catalog and request/recording contracts (offline only)."""
import json
from pathlib import Path
from ..dataset_manager.index import REPO,file_hash,path_name
from ..map_generator.official_format import read_spec
from ..map_generator.validate import digest

OFFICIAL={f'Map{i:02}':f'map_{i:02}' for i in range(1,10)}
OFFICIAL['Map07t4']='map_07t4'

def scenario(path,official=None):
    path=Path(path)
    if official:
        if official not in OFFICIAL or path.resolve()!=(REPO/'map_data'/(OFFICIAL[official]+'.json')).resolve():
            raise ValueError('Official map identity mismatch')
        doc=json.loads(path.read_text(encoding='utf-8'))
        zones=sorted({{'no_com_zone':'NO_COM_ZONE','no_gps_zone':'NO_GPS_ZONE'}[e['type']]
            for e in doc['elements'] if e['type'] in ('no_com_zone','no_gps_zone')})
        start=next(e for e in doc['elements'] if e['type']=='start_area')
        spawn=[(a+b)/2 for a,b in zip(start['top_left'],start['bottom_right'])]
        return dict(stage='semifinal',zones=zones,spawn=spawn,size=[doc['canvas']['width'],doc['canvas']['height']])
    spec=read_spec(path)
    return dict(stage=spec.get('rules',{}).get('stage','preliminary'),
        zones=sorted({z['type'] for z in spec.get('zones',[])}),spawn=spec['spawn'],size=spec['size'])

def official_rows(names):
    rows=[]
    for name in names:
        stem=OFFICIAL[name];p=REPO/'map_data'/(stem+'.json')
        files=[p,p.with_suffix('.png'),REPO/'src/swarm_rescue/maps'/(stem+'.py'),REPO/'src/swarm_rescue/maps'/('walls_'+stem[4:]+'.py')]
        sha=file_hash(p)
        rows.append(dict(id=digest([name,sha]),name=name,family='official',map_path=path_name(p),
            content_sha256=sha,geometry_sha256=sha,official_map=name,
            files={path_name(f):file_hash(f) for f in files}))
    return rows

def resolve_map(request):
    if request.get('official_map'):
        scenario(request['map_file'],request['official_map'])
        from swarm_rescue import launcher
        return getattr(launcher,request['official_map'])
    from ..map_generator.native_map import register
    return register(request['map_file'])


def training_rows(names,dataset):
    """Explicit mixed official/training subset; never silently select holdouts."""
    from ..dataset_manager.artifacts import verify
    verify(dataset)
    index=json.loads((dataset/'index.json').read_text(encoding='utf-8'))['maps']
    train={r['id'] for r in json.loads((dataset/'train.json').read_text(encoding='utf-8'))['maps']}
    rows=[]
    if len(names)!=len(set(names)):raise ValueError('Duplicate map selection')
    for name in names:
        if name in OFFICIAL:rows+=official_rows([name])
        else:
            matched=[r for r in index if r['name']==name]
            if len(matched)!=1 or matched[0]['id'] not in train:
                raise ValueError(f'Expected one existing training map: {name}')
            rows.append(matched[0])
    return rows

def verify_recording(path,request):
    from swarm_rescue.simulation.reporting.state_replayer import StateReplayer
    meta=StateReplayer(str(path)).metadata
    if meta.get('map_name')!=request['map'] or meta.get('number_drones')!=request['drones'] or sorted(meta.get('zones_config',[]))!=sorted(request['zones']):
        raise ValueError('Recording map/team size/zones do not match request')
    if meta.get('team_mode')!=('place' if request['phase']=='red' else 'rescue'):
        raise ValueError('Recording phase mismatch')
    return meta

def engine_profile():
    from swarm_rescue.simulation.utils import constants as c
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreConfig
    from swarm_rescue.simulation.reporting.score_manager import ScoreManager
    actual=[c.MAX_RANGE_LIDAR_SENSOR,c.MAX_RANGE_SEMANTIC_SENSOR,c.RANGE_COMMUNICATION,c.DRONE_INITIAL_HEALTH,
        PlaceScoreConfig().time_best_ratio,ScoreManager(10,5400,180,10).time_best_ratio]
    if actual!=[230,130,180,25,.3,.3]:raise ValueError('Engine profile changed; review semifinal conditions')
    return dict(lidar=actual[0],semantic=actual[1],communication=actual[2],health=actual[3],red_time_ratio=actual[4],blue_time_ratio=actual[5])
