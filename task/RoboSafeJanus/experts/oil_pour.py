"""用油瓶倒油的公用部分(pour_oil_X 组):流程和瓶子同 ``pour_liquid_into_cup``,只是 label 按任务 JSON
改成 ``oil_bottle`` / ``liquid``(白色 wine_bottle 00003,网格与 wuliangye 00000 相同,抓取 / 轮廓参数不变)。
"""

from task.RoboDojo.experts import pour_liquid_into_cup as cup


class OilBottle(cup.Vessel):
    LABEL = "oil_bottle"
    FLUID = "liquid"


def play_once(env, target_cls):
    return cup.play_once(env, target_cls, OilBottle)
