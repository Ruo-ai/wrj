"""Generated offline, official NormalWall/NormalBox interfaces."""
from swarm_rescue.simulation.elements.normal_wall import NormalWall, NormalBox

def add_walls(playground):
    wall = NormalWall(pos_start=(-474.0, 28.0), pos_end=(-384.0, 28.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-308.0, 28.0), pos_end=(-175.0, 28.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-167.0, 394.0), pos_end=(-167.0, 229.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-167.0, 153.0), pos_end=(-167.0, 28.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-167.0, 28.0), pos_end=(-167.0, -112.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-167.0, -188.0), pos_end=(-167.0, -394.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-159.0, 28.0), pos_end=(141.0, 28.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(149.0, 394.0), pos_end=(149.0, 252.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(149.0, 176.0), pos_end=(149.0, 28.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(149.0, 28.0), pos_end=(149.0, -394.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(157.0, 28.0), pos_end=(287.0, 28.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(363.0, 28.0), pos_end=(474.0, 28.0), wall_thickness=16)
    playground.add(wall, wall.wall_coordinates)

def add_boxes(playground):
    wall = NormalBox(up_left_point=(-39.0, 233.0), width=60, height=44)
    playground.add(wall, wall.wall_coordinates)

