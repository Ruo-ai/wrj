"""Official editor schema and map_*/walls_* source export, confined to solutions."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from swarm_rescue.tools.map_wall_editor import load_elements, serialize_elements, render_elements_to_image
from .validate import geometry_hash, validate
from .naming import record_directory


CODE_ROOT = Path(__file__).resolve().parent.parent/'generated_maps'
FORMAT = 'official-editor-native-v1'


def editor_document(spec):
    w,h = spec['size']
    elements = []
    def rectangle(kind,l,t,r,b,color='#000000'):
        elements.append(dict(type=kind,top_left=[l,t],bottom_right=[r,b],color=color))
    # Editor border matches the implicit ClosedPlayground border; not added twice.
    for rect in ((0,0,w-1,6),(0,h-6,w-1,h-1),(0,6,6,h-6),(w-6,6,w-1,h-6)):
        rectangle('rect_wall',*rect)
    for kind,l,t,r,b in spec['walls']:
        dx,dy = r-l,b-t
        # The official loader coerces coordinates to int. Preserve odd thickness
        # without half-pixel truncation by exporting its exact rectangle.
        if kind=='box' or min(dx,dy)%2:
            rectangle('rect_wall',l,t,r,b)
        else:
            points = [[l,(t+b)//2],[r,(t+b)//2]] if dx>dy else [[(l+r)//2,t],[(l+r)//2,b]]
            elements.append(dict(type='line_wall',points=points,thickness=min(dx,dy),color='#000000'))
    x,y = spec['spawn']
    rectangle('start_area',*spec.get('start_area',[x-40,y-40,x+40,y+40]),color='#1E90FF')
    x,y = spec['center']
    hx,hy = [v//2 for v in spec.get('center_size',[48,48])]
    rectangle('disposal_center',x-hx,y-hy,x+hx,y+hy,color='#228B22')
    for zone in spec.get('zones',[]):
        kind = {'NO_COM_ZONE':'no_com_zone','NO_GPS_ZONE':'no_gps_zone'}[zone['type']]
        rectangle(kind,*zone['rect'],color='#FFFF00' if kind=='no_com_zone' else '#808080')
    for index,element in enumerate(elements,1):
        element['id'] = index
    doc = dict(version=1,canvas=dict(width=w,height=h,background='white'),elements=elements)
    # Upstream 6557940 supports zones: validate the complete editor round trip.
    if serialize_elements(w,h,load_elements(doc,w,h))!=doc:
        raise ValueError('Official editor would alter exported coordinates')
    return doc


def render_preview(spec,path):
    import cv2
    from PIL import Image
    doc = editor_document(spec)
    w,h = spec['size']
    bgr = render_elements_to_image(w,h,load_elements(doc,w,h))
    image = Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    image.save(path)


def sources(spec):
    validate(spec)
    w,h = spec['size']
    def world(p):
        return (p[0]-w/2,h/2-p[1])
    walls = ['"""Generated offline, official NormalWall/NormalBox interfaces."""',
             'from swarm_rescue.simulation.elements.normal_wall import NormalWall, NormalBox','']
    internal = editor_document(spec)['elements'][4:]
    for method,kind in (('add_walls','line_wall'),('add_boxes','rect_wall')):
        walls.append(f'def {method}(playground):')
        selected = [e for e in internal if e['type']==kind]
        if not selected:
            walls.append('    pass')
        for e in selected:
            if kind=='line_wall':
                start,end = map(world,e['points'])
                constructor = f'NormalWall(pos_start={start!r}, pos_end={end!r}, wall_thickness={e["thickness"]})'
            else:
                l,t = e['top_left']
                r,b = e['bottom_right']
                constructor = f'NormalBox(up_left_point={world([l,t])!r}, width={r-l}, height={b-t})'
            walls.extend([f'    wall = {constructor}','    playground.add(wall, wall.wall_coordinates)'])
        walls.append('')
    semifinal = 'rules' in spec
    limit = spec.get('rules',{}).get('timestep_limit',5000)
    zone_records = [(z['type'],tuple(world([(z['rect'][0]+z['rect'][2])/2,(z['rect'][1]+z['rect'][3])/2])),
                     (z['rect'][2]-z['rect'][0],z['rect'][3]-z['rect'][1])) for z in spec.get('zones',[])]
    map_source = f'''"""Generated map following the official MapAbstract construction sequence."""
import math
import random
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.elements.sensor_disablers import NoComZone, NoGpsZone, ZoneType
from .walls_generated import add_walls, add_boxes


class MapGenerated(MapAbstract):
    def __init__(self, drone_type, zones_config=(), number_drones=None):
        count = 5 if number_drones is None else number_drones
        allowed = (5,10) if {semifinal!r} else (5,)
        if count not in allowed or (zones_config and not {semifinal!r}):
            raise ValueError('Unsupported drone count or zone configuration')
        if any(z not in (ZoneType.NO_COM_ZONE,ZoneType.NO_GPS_ZONE) for z in zones_config):
            raise ValueError('Unsupported disable zone')
        super().__init__(drone_type, zones_config, number_drones=count)
        self._max_timestep_limit = {limit}
        self._max_walltime_limit = 180
        self._size_area = {tuple(spec['size'])!r}
        self._start_area_drones = {world(spec['spawn'])!r}
        self._disposal_center = DisposalCenter(size={tuple(spec.get('center_size',[48,48]))!r})
        self._disposal_center_pos = ({world(spec['center'])!r}, 0)
        self._bombs_pos, self._bombs_path = [], []
        self._number_bombs, self._number_drones = 0, count
        self._bombs, self._drones, self._drones_pos = [], [], []
        nb_per_side = math.ceil(math.sqrt(self._number_drones))
        sx = self._start_area_drones[0] - (nb_per_side - 1) * 20
        sy = self._start_area_drones[1] - (nb_per_side - 1) * 20
        for i in range(self._number_drones):
            pose = ((sx + i % nb_per_side * 40, sy + i // nb_per_side * 40), random.uniform(-math.pi, math.pi))
            self._drones_pos.append(pose)
        self._playground = ClosedPlayground(size=self._size_area)
        self._playground.add(self._disposal_center, self._disposal_center_pos)
        self._generated_zones = []
        for kind,position,size in {zone_records!r}:
            if getattr(ZoneType,kind) in self._zones_config:
                zone = (NoComZone if kind=='NO_COM_ZONE' else NoGpsZone)(size=size)
                self._playground.add(zone,(position,0))
                self._generated_zones.append(zone)
        add_walls(self._playground)
        add_boxes(self._playground)
        self._explored_map.initialize_walls(self._playground)
        misc = MiscData(size_area=self._size_area, number_drones=count,
                        max_timestep_limit={limit}, max_walltime_limit=180)
        for i, pose in enumerate(self._drones_pos):
            drone = drone_type(identifier=i, misc_data=misc)
            self._drones.append(drone)
            self._playground.add(drone, pose)
'''
    return {'__init__.py':'"""Generated offline map package."""\n',
            'map_generated.py':map_source,'walls_generated.py':'\n'.join(walls)+'\n'}


def code_folder(spec):
    return CODE_ROOT/('g_'+geometry_hash(spec))


def check_sources(spec):
    folder = code_folder(spec)
    for name,expected in sources(spec).items():
        if (folder/name).read_text(encoding='utf-8')!=expected:
            raise ValueError(f'Generated source differs from trusted template: {folder/name}')
    return folder


def publish_sources(spec):
    folder = code_folder(spec)
    if folder.exists():
        return check_sources(spec)
    CODE_ROOT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stage-',dir=CODE_ROOT) as tmp:
        stage = Path(tmp)/'package'
        stage.mkdir()
        for name,source in sources(spec).items():
            (stage/name).write_text(source,encoding='utf-8')
        stage.rename(folder)
    return folder


def load_class(spec):
    folder = check_sources(spec)
    name = '_g1official_'+geometry_hash(spec)
    package_spec = importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[name] = package
    package_spec.loader.exec_module(package)
    module_spec = importlib.util.spec_from_file_location(name+'.map_generated',folder/'map_generated.py')
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module.MapGenerated


def read_spec(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if 'canvas' not in data:
        return data  # Retained v2 diagnostic samples remain loadable.
    spec = json.loads((record_directory(path)/'generation.json').read_text(encoding='utf-8'))
    if editor_document(spec)!=data:
        raise ValueError('Official editor JSON differs from validated generation geometry')
    return spec
