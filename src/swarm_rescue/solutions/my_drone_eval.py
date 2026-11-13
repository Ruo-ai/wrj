"""
启动器使用的提交入口助手。
"""

from swarm_rescue.simulation.reporting.team_mode import TeamMode
from swarm_rescue.solutions.my_drone_red_advanced2 import MyDroneRedAdvanced2
from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample


def drone_class_for_mode(mode: str):
    """
    返回给定团队模式的具体无人机类。
    """
    team_mode = TeamMode.from_string(mode)
    if team_mode == TeamMode.PLACE:
        return MyDroneRedAdvanced2
    return MyDroneRescueExample
