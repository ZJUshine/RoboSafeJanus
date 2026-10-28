"""pour_water_plastic_box 的脚本专家:用杯子把水倒进开口朝上的塑料收纳盒(RoboTwin_062_plasticbox 00004,流程见 ``cup_pour``)。"""

from task.RoboDojo.experts import cup_pour


class _Target(cup_pour.ContainerTarget):
    LABEL = "plastic_box"


def play_once(env):
    cup_pour.play_once(env, _Target)
