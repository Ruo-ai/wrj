"""Generated map following the official MapAbstract construction sequence."""
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
        allowed = (5,10) if True else (5,)
        if count not in allowed or (zones_config and not True):
            raise ValueError('Unsupported drone count or zone configuration')
        if any(z not in (ZoneType.NO_COM_ZONE,ZoneType.NO_GPS_ZONE) for z in zones_config):
            raise ValueError('Unsupported disable zone')
        super().__init__(drone_type, zones_config, number_drones=count)
        self._max_timestep_limit = 5400
        self._max_walltime_limit = 180
        self._size_area = (1700, 1100)
        self._start_area_drones = (38.0, -278.0)
        self._disposal_center = DisposalCenter(size=(96, 120))
        self._disposal_center_pos = ((506.0, -434.0), 0)
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
        for kind,position,size in [('NO_COM_ZONE', (-348.0, 179.5), (732, 511)), ('NO_GPS_ZONE', (192.0, 199.5), (808, 601))]:
            if getattr(ZoneType,kind) in self._zones_config:
                zone = (NoComZone if kind=='NO_COM_ZONE' else NoGpsZone)(size=size)
                self._playground.add(zone,(position,0))
                self._generated_zones.append(zone)
        add_walls(self._playground)
        add_boxes(self._playground)
        self._explored_map.initialize_walls(self._playground)
        misc = MiscData(size_area=self._size_area, number_drones=count,
                        max_timestep_limit=5400, max_walltime_limit=180)
        for i, pose in enumerate(self._drones_pos):
            drone = drone_type(identifier=i, misc_data=misc)
            self._drones.append(drone)
            self._playground.add(drone, pose)
