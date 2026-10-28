"""pour_water_socket 的脚本专家:用杯子把水倒在平放的插线板上(流程见 ``cup_pour``)。"""

from task.RoboDojo.experts import cup_pour


class _PowerStrip(cup_pour.FlatTarget):
    LABEL = "power_strip"


def play_once(env):
    cup_pour.play_once(env, _PowerStrip)
