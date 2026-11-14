"""R8 independent ablations on retained RN7-B. No official entry changes."""
from swarm_rescue.solutions.my_drone_red_rn7 import MyDroneRedRN7B
from swarm_rescue.solutions.red_pixel_faces import PixelFaceMapper,PixelSnapshots
from swarm_rescue.solutions.red_footprint8 import FootprintMotion


class MyDroneRedR8(MyDroneRedRN7B):
    PIXEL_SUBMISSION=False
    FOOTPRINT_PASSAGE=False
    PAIR_FILL=True
    SAFETY_MARGIN=3.
    REFRESH_BUDGET=True
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.PIXEL_SUBMISSION:
            mapper=PixelFaceMapper(self.size_area);mapper.PAIR_FILL=self.PAIR_FILL
            self._submission_mapper=mapper
            self._radio_maps=PixelSnapshots(self._map.shape,self._n,mapper)
        if self.FOOTPRINT_PASSAGE:
            motion=FootprintMotion(int(self.identifier or 0));motion.SAFETY_MARGIN=self.SAFETY_MARGIN
            motion.REFRESH_BUDGET=self.REFRESH_BUDGET;self._rn7=motion


class MyDroneRedR8Mapping(MyDroneRedR8):PIXEL_SUBMISSION=True
class MyDroneRedR8Narrow(MyDroneRedR8):FOOTPRINT_PASSAGE=True
class MyDroneRedR8Combined(MyDroneRedR8):PIXEL_SUBMISSION=True;FOOTPRINT_PASSAGE=True
class MyDroneRedR8Surface(MyDroneRedR8Mapping):PAIR_FILL=False
class MyDroneRedR8OldBudget(MyDroneRedR8Narrow):REFRESH_BUDGET=False
