"""Generated offline, official NormalWall/NormalBox interfaces."""
from swarm_rescue.simulation.elements.normal_wall import NormalWall, NormalBox

def add_walls(playground):
    wall = NormalWall(pos_start=(-15.0, 394.0), pos_end=(-15.0, 24.0), wall_thickness=160)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-15.0, -14.0), pos_end=(-15.0, -394.0), wall_thickness=160)
    playground.add(wall, wall.wall_coordinates)

def add_boxes(playground):
    pass

