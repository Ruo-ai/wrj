"""Generated offline, official NormalWall/NormalBox interfaces."""
from swarm_rescue.simulation.elements.normal_wall import NormalWall, NormalBox

def add_walls(playground):
    wall = NormalWall(pos_start=(-474.0, 148.0), pos_end=(-426.0, 148.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-474.0, -161.0), pos_end=(-430.0, -161.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-330.0, -161.0), pos_end=(-193.0, -161.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-326.0, 148.0), pos_end=(-193.0, 148.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-190.0, 394.0), pos_end=(-190.0, 346.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-190.0, 246.0), pos_end=(-190.0, -394.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(-187.0, -219.0), pos_end=(35.0, -219.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(135.0, -219.0), pos_end=(276.0, -219.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(146.0, 394.0), pos_end=(146.0, 288.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(146.0, 188.0), pos_end=(146.0, 76.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(146.0, -24.0), pos_end=(146.0, -216.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(149.0, 96.0), pos_end=(222.0, 96.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(322.0, 96.0), pos_end=(474.0, 96.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)
    wall = NormalWall(pos_start=(376.0, -219.0), pos_end=(474.0, -219.0), wall_thickness=6)
    playground.add(wall, wall.wall_coordinates)

def add_boxes(playground):
    pass

