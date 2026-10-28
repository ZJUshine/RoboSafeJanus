"""pour_water_bowl 的脚本专家:用杯子把水倒进碗(bowl 00011,口径 18 cm)(流程见 ``cup_pour``)。"""

from task.RoboDojo.experts import cup_pour


class _Target(cup_pour.ContainerTarget):
    LABEL = "bowl"


def play_once(env):
    cup_pour.play_once(env, _Target)
