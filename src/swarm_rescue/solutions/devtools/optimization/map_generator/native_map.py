"""Offline registration and one-map native load probe, with no controller dependency."""
import json
import math
from pathlib import Path
import random
import sys
import tempfile

import cv2
import numpy as np

from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.elements.normal_wall import NormalBox, NormalWall
from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
from swarm_rescue.simulation.utils.misc_data import MiscData
from .structures import spawn_points
from .validate import CLEARANCE, geometry_hash, validate
from .official_format import read_spec, load_class


def register(path):
    from swarm_rescue import launcher
    spec = read_spec(path)
    validate(spec)
    key = 'Generated'+geometry_hash(spec)
    if hasattr(launcher,key):
        raise ValueError(f'Map name already registered: {key}')
    if 'canvas' in json.loads(Path(path).read_text(encoding='utf-8')):
        cls = load_class(spec)
        cls.__name__ = key
        setattr(launcher,key,cls)
        return cls

    class GeneratedMap(MapAbstract):
        def __init__(self, drone_type, zones_config=(), number_drones=None):
            if zones_config or number_drones not in (None,5):
                raise ValueError('G1 maps support five drones and no disable zones')
            super().__init__(drone_type,zones_config,number_drones=5)
            w,h = spec['size']
            self._size_area = (w,h)
            self._max_timestep_limit, self._max_walltime_limit = 5000,180
            self._number_bombs, self._number_drones = 0,5
            self._bombs_pos, self._bombs_path = [],[]
            self._playground = pg = ClosedPlayground(size=(w,h))

            def world(x,y):
                return (x-w/2,h/2-y)

            self._start_area_drones = world(*spec['spawn'])
            self._disposal_center = DisposalCenter(size=(48,48))
            self._disposal_center_pos = (world(*spec['center']),0)
            pg.add(self._disposal_center,self._disposal_center_pos)
            for kind,l,t,r,b in spec['walls']:
                dx,dy = r-l,b-t
                if kind=='box':
                    wall = NormalBox(up_left_point=world(l,t),width=dx,height=dy)
                elif dx>dy:
                    wall = NormalWall(world(l,(t+b)/2),world(r,(t+b)/2),wall_thickness=dy)
                else:
                    wall = NormalWall(world((l+r)/2,t),world((l+r)/2,b),wall_thickness=dx)
                pg.add(wall,wall.wall_coordinates)
            self._explored_map.initialize_walls(pg)
            misc = MiscData(size_area=(w,h),number_drones=5,max_timestep_limit=5000,max_walltime_limit=180)
            self._drones_pos = [(world(x,y),random.uniform(-math.pi,math.pi))
                                for x,y in spawn_points(spec['spawn'])]
            for i,pose in enumerate(self._drones_pos):
                drone = drone_type(identifier=i,misc_data=misc)
                self._drones.append(drone)
                pg.add(drone,pose)

    GeneratedMap.__name__ = key
    setattr(launcher,key,GeneratedMap)
    return GeneratedMap


def probe(path):
    spec = read_spec(path)
    cls = register(path)
    if 'rules' in spec:
        from swarm_rescue.simulation.utils import constants as c
        from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreConfig
        from swarm_rescue.simulation.reporting.score_manager import ScoreManager
        expected = (230,130,180,25,.3)
        actual = (c.MAX_RANGE_LIDAR_SENSOR,c.MAX_RANGE_SEMANTIC_SENSOR,
                  c.RANGE_COMMUNICATION,c.DRONE_INITIAL_HEALTH,PlaceScoreConfig().time_best_ratio)
        if actual!=expected:
            raise ValueError(f'Engine does not match semifinal profile: {actual}')
        if ScoreManager(10,5400,180,10).time_best_ratio!=.3:
            raise ValueError('Blue time score does not match semifinal profile')
    for count in ((5,10) if 'rules' in spec else (5,)):
        probe_scenario(path,cls,count)


def probe_scenario(path,cls,count):
    import yaml
    from swarm_rescue import launcher
    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
    from swarm_rescue.simulation.reporting.place_score_manager import walls_and_boundary_grid
    from swarm_rescue.simulation.reporting.team_mode import TeamMode
    from swarm_rescue.solutions.devtools.evaluate_red import configure_ray_backend
    configure_ray_backend('cpu')
    spec = read_spec(path)
    zones = sorted({z['type'] for z in spec.get('zones',[])})
    mode = 'place' if count==5 else 'rescue'
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp)/'plan.yml'
        config.write_text(yaml.safe_dump(dict(stat_saving_enabled=False,state_recording_enabled=False,
            video_capture_enabled=False,evaluation_plan=[dict(map_name=cls.__name__,team_mode=mode,
            nb_rounds=1,config_weight=1,number_drones=count,initial_bombs_per_drone=2,zones_config=zones)])),encoding='ascii')
        instance = launcher.Launcher(config_path=str(config))
        scenario, = instance.eval_plan.list_eval_config
        world = getattr(launcher,scenario.map_name)(DroneMotionless,scenario.zones_config,scenario.number_drones)
        try:
            instance._configure_map_for_team(world,scenario,TeamMode.from_string(mode))
            if len(world.drones)!=count or (count==5 and any(d.carried_bombs_count()!=2 for d in world.drones)):
                raise ValueError('Native inventory mismatch')
            if world._max_timestep_limit!=spec.get('rules',{}).get('timestep_limit',5000) or world._max_walltime_limit!=180:
                raise ValueError('Native time budget mismatch')
            if len(getattr(world,'_generated_zones',[]))!=len(spec.get('zones',[])):
                raise ValueError('Native zones missing')
            for zone in getattr(world,'_generated_zones',[]):
                drone = world.drones[0]
                zone.disable(drone.gps())
                zone.disable(drone.compass())
                zone.disable(drone.communicator)
                if zone.__class__.__name__=='NoGpsZone' and not (drone.gps_is_disabled() and drone.compass_is_disabled()):
                    raise ValueError('NoGpsZone must disable GPS and compass')
                if zone.__class__.__name__=='NoComZone' and not drone.communicator_is_disabled():
                    raise ValueError('NoComZone must disable communication')
            spec = read_spec(path)
            w,h = spec['size']
            designed_positions = [(x-w/2,h/2-y) for x,y in spawn_points(spec['spawn'],count)]
            if not np.allclose([p for p,_ in world._drones_pos],designed_positions,atol=1e-8):
                raise ValueError('Native formation differs from validated generation geometry')
            if any(not np.allclose(d.true_position(),p,atol=.5)
                   for d,(p,_) in zip(world.drones,world._drones_pos)):
                raise ValueError('Engine moved a drone away from its validated spawn')
            for drone,pose in zip(world.drones,world._drones_pos):
                # Bound every orientation of the actual convex hull. Reserve
                # sqrt(2) pixels for raster-to-continuous boundary error.
                radius = max(v.length + shape.radius
                             for shape in drone.base.pm_shapes for v in shape.get_vertices())
                if radius+math.sqrt(2)>CLEARANCE:
                    raise ValueError(f'Native hull radius {radius:.3f} exceeds clearance model')
                if any(not hit.shape.sensor and hit.shape not in drone.base.pm_shapes
                       for shape in drone.base.pm_shapes
                       for hit in world.playground.space.shape_query(shape)):
                    raise ValueError('Native drone spawn physically overlaps another object')
            expected,_ = validate(spec)
            actual,_ = walls_and_boundary_grid(world.playground)
            # Native polygons use inclusive rounded edges; allow one pixel at each edge.
            kernel = np.ones((3,3),np.uint8)
            if (np.any(actual > cv2.dilate(expected,kernel)) or
                    np.any(actual < cv2.erode(expected,kernel))):
                raise ValueError('Native wall raster differs from designed geometry')
        finally:
            world.playground.window.close()
    print(f'Native load OK: {cls.__name__}, {mode} {count} drones, zones={zones}',flush=True)


if __name__=='__main__':
    probe(Path(sys.argv[1]))
