"""Generated offline, official NormalWall/NormalBox interfaces."""
from swarm_rescue.simulation.elements.normal_wall import NormalWall, NormalBox

def add_walls(playground):
    wall = NormalWall(pos_start=(-474.0, 17.0), pos_end=(-366.0, 17.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-296.0, 17.0), pos_end=(-148.0, 17.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-139.0, 394.0), pos_end=(-139.0, 256.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-139.0, 186.0), pos_end=(-139.0, 17.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-139.0, 17.0), pos_end=(-139.0, -138.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-139.0, -208.0), pos_end=(-139.0, -394.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-130.0, 17.0), pos_end=(168.0, 17.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(177.0, 394.0), pos_end=(177.0, 225.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(177.0, 155.0), pos_end=(177.0, 17.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(177.0, 17.0), pos_end=(177.0, -394.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(186.0, 17.0), pos_end=(276.0, 17.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(346.0, 17.0), pos_end=(474.0, 17.0), wall_thickness=18)
    playground.add(wall, wall.wall_coordinates)

def add_boxes(playground):
    wall = NormalBox(up_left_point=(-337.0, -166.0), width=60, height=44)
    playground.add(wall, wall.wall_coordinates)

