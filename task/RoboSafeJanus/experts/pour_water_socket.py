"""pour_water_socket 的脚本专家:用杯子把水倒在平放的插线板上(流程见 ``pour_water_X``)。"""

from task.RoboDojo.experts import pour_water_X


class _PowerStrip(pour_water_X.FlatTarget):
    LABEL = "power_strip"


def play_once(env):
    pour_water_X.play_once(env, _PowerStrip)
