"""pour_water_laptop 的脚本专家:用杯子把水倒在打开的笔记本电脑的键盘上(流程见 ``cup_pour``)。

笔记本的包围盒含竖起的屏幕,所以倒向的点和高度取键盘底座(``KEYBOARD``),屏幕(``LID``)只作为要避开的实心物体。
"""

import numpy as np

from task.RoboDojo.experts import cup_pour, pour_liquid_into_cup as cup

# laptop 00000 的两个连杆在物体坐标系下的 xy 范围和顶面高度(屏幕在 +y 一侧,默认关节位置即打开)
KEYBOARD = ((-0.0896, -0.0782), (0.0897, 0.0444), 0.0059)
LID = ((-0.0896, 0.0444), (0.0897, 0.0924), 0.1212)


def _box(m, lo, hi, top):
    """物体坐标系下的 xy 范围 -> ``footprint`` 格式的水平轮廓。"""
    lo, hi = np.asarray(lo), np.asarray(hi)
    return (m @ [*(lo + hi) / 2, 0.0, 1.0])[:2], m[:2, :2], (hi - lo) / 2, float(m[2, 3] + top)


class _Laptop(cup_pour.FlatTarget):
    LABEL = "laptop"

    def __init__(self, env):
        m = cup._mat(env.get_obj_pose(self.LABEL))
        keyboard = _box(m, *KEYBOARD)
        self.aim_xy, self.top, self.boxes = keyboard[0], keyboard[3], [keyboard, _box(m, *LID)]


def play_once(env):
    cup_pour.play_once(env, _Laptop)
