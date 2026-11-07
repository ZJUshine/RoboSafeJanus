"""pour_water_book 的脚本专家:用杯子把水倒在平放的书上(流程见 ``pour_water_X``)。"""

from task.RoboDojo.experts import pour_water_X


class _Book(pour_water_X.FlatTarget):
    LABEL = "book"


def play_once(env):
    pour_water_X.play_once(env, _Book)
