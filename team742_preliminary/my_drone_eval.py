"""Team 742 red-stage submission: RL17-R."""
from swarm_rescue.simulation.reporting.team_mode import TeamMode
from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17


def drone_class_for_mode(mode: str):
    if TeamMode.from_string(mode) == TeamMode.PLACE:
        return MyDroneRedRepair17
    raise ValueError("This submission is for the red PLACE stage only.")
