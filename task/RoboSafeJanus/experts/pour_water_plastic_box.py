"""pour_water_plastic_box 的脚本专家:用杯子把水倒进开口朝上的塑料收纳盒(RoboTwin_062_plasticbox 00004,流程见 ``pour_water_X``)。"""

from task.RoboDojo.experts import pour_water_X


class _Target(pour_water_X.ContainerTarget):
    LABEL = "plastic_box"


def play_once(env):
    pour_water_X.play_once(env, _Target)
