"""pour_oil_X 组共用的脚本专家:流程和瓶子同 ``pour_common``,只是 label 按任务 JSON
改成 ``oil_bottle`` / ``liquid``(白色 wine_bottle 00003,网格与 wuliangye 00000 相同,抓取 / 轮廓参数不变)。
"""

from task.RoboDojo.experts import pour_common as pour


class OilBottle(pour.Vessel):
    LABEL = "oil_bottle"
    FLUID = "liquid"
    # 夹爪对称,同一个抓取绕接近轴转 180° 的第二版会让 joint6 差半圈;官方限位 ±120° 下先选转得少的
    # (瓶子的抓取缓存 symmetric=True,两版都会出现在候选里)。
    WRIST_PICK = 4


def play_once(env, target_cls):
    return pour.play_once(env, target_cls, OilBottle)
