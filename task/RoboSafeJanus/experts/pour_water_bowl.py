"""pour_water_bowl 的脚本专家:用杯子把水倒进碗(bowl 00011,口径 18 cm)(流程见 ``pour_water_X``)。"""

from task.RoboDojo.experts import pour_water_X


class _Target(pour_water_X.ContainerTarget):
    LABEL = "bowl"


def play_once(env):
    pour_water_X.play_once(env, _Target)
