"""pour_water_pot 的脚本专家:用杯子把水倒进锅(RoboTwin_objaverse_pot 00010)(流程见 ``pour_water_X``)。"""

from task.RoboDojo.experts import pour_water_X


class _Target(pour_water_X.ContainerTarget):
    LABEL = "pot"


def play_once(env):
    pour_water_X.play_once(env, _Target)
