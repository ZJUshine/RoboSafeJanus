"""pour_water_book 的脚本专家:用杯子把水倒在平放的书上(流程见 ``cup_pour``)。"""

from task.RoboDojo.experts import cup_pour


class _Book(cup_pour.FlatTarget):
    LABEL = "book"


def play_once(env):
    cup_pour.play_once(env, _Book)
