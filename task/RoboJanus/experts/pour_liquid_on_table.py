"""pour_liquid_on_table 的脚本专家:流程同 ``pour_liquid_into_cup``,只是瓶口不在杯口上方,而在杯子
靠瓶子一侧的桌面上方,液体全部倒在杯子旁边的桌上,一点不进杯子。

瓶口 xy 在杯子水平轮廓(包围盒角点,含把手)外 ``TABLE_GAP`` 处(沿 -d),高度取瓶子 / 夹爪不碰桌面的最低值
(``MOUTH_ABOVE = 0``,由间隙抬高)。杯子放进 ``boxes``:瓶子 / 夹爪不从杯子上方压下去。
"""

import numpy as np

from task.RoboDojo.experts import pour_liquid_into_cup as cup

TABLE_GAP = 0.03  # 倾倒时瓶口在杯子水平轮廓外(沿 -d)的水平距离


class _BesideCup(cup.Target):
    MOUTH_ABOVE = 0.0

    def __init__(self, env):
        super().__init__(env)
        c, axes, half, _ = box = cup.footprint(env, self.LABEL)
        self.corners = np.array([c + axes @ (s * half) for s in [(1, 1), (1, -1), (-1, 1), (-1, -1)]])
        bottle = cup._mat(env.get_obj_pose("bottle"))
        self.top = float((bottle @ [0.0, 0.0, cup._bbox(env, "bottle")[0][2], 1.0])[2])  # 桌面
        self.boxes = [box]

    def pivot_xy(self, d):
        ext = float(np.max((self.corners - self.aim_xy) @ -d[:2]))
        return self.aim_xy - (ext + TABLE_GAP) * d[:2]


def play_once(env):
    cup.play_once(env, _BesideCup)
