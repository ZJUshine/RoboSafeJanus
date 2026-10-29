"""pour_oil_toaster 的脚本专家:流程同 ``pour_liquid_into_cup``,只是把油倒进面包机的槽里。

对准离瓶子近的那条槽(槽沿面包机 x,侧抓瓶子时倾倒方向也大致沿 x,液流顺着槽的长边落下),瓶口在槽中心靠
瓶子一侧 ``LAND_BACK`` 处、面包机顶面上方 ``MOUTH_ABOVE``。面包机是实心的盒子,放进 ``boxes``。
"""

import numpy as np

from task.RoboDojo.experts import oil_pour, pour_liquid_into_cup as cup

LAND_BACK = 0.03  # 倾倒时瓶口在槽中心靠瓶子一侧(沿 -d)的距离
SLOTS_Y = (-0.026, 0.026)  # 面包机 00000 两条槽中心的 y(面包机坐标系),槽沿 x、在 x = 0 两侧对称


class _Toaster(cup.Target):
    LABEL = "toaster"

    def __init__(self, env):
        box = cup.footprint(env, self.LABEL)
        self.top, self.boxes = box[3], [box]
        m = cup._mat(env.get_obj_pose(self.LABEL))
        bottle_xy = env.get_obj_pose(oil_pour.OilBottle.LABEL)[:2]
        slots = [(m @ [0.0, y, 0.0, 1.0])[:2] for y in SLOTS_Y]
        self.aim_xy = min(slots, key=lambda c: np.linalg.norm(c - bottle_xy))

    def pivot_xy(self, d):
        return self.aim_xy - LAND_BACK * d[:2]


def play_once(env):
    return oil_pour.play_once(env, _Toaster)
