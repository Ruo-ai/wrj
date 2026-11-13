"""Generated offline, official NormalWall/NormalBox interfaces."""
from swarm_rescue.simulation.elements.normal_wall import NormalWall, NormalBox

def add_walls(playground):
    wall = NormalWall(pos_start=(-474.0, 15.0), pos_end=(-353.0, 15.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-277.0, 15.0), pos_end=(-168.0, 15.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-160.0, 394.0), pos_end=(-160.0, 228.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-160.0, 152.0), pos_end=(-160.0, 15.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-160.0, 15.0), pos_end=(-160.0, -162.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-160.0, -238.0), pos_end=(-160.0, -394.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-152.0, 15.0), pos_end=(167.0, 15.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(175.0, 394.0), pos_end=(175.0, 228.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(175.0, 152.0), pos_end=(175.0, 15.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(175.0, 15.0), pos_end=(175.0, -394.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(183.0, 15.0), pos_end=(310.0, 15.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(386.0, 15.0), pos_end=(474.0, 15.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)

def add_boxes(playground):
    wall = NormalBox(up_left_point=(-23.0, -167.0), width=60, height=44)
    playground.add(wall, wall.wall_coordinates)

