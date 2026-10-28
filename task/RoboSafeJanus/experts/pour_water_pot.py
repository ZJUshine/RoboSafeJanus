"""pour_water_pot 的脚本专家:用杯子把水倒进锅(RoboTwin_objaverse_pot 00010)(流程见 ``cup_pour``)。"""

from task.RoboDojo.experts import cup_pour


class _Target(cup_pour.ContainerTarget):
    LABEL = "pot"


def play_once(env):
    cup_pour.play_once(env, _Target)
