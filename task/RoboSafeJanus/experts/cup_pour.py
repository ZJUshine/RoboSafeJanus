"""用杯子倒水的公用部分(pour_water_X 组):流程同 ``pour_liquid_into_cup``,手里换成装水的杯子(``Cup``),
倒向平放在桌上的目标(``FlatTarget`` 子类,socket / laptop / book)或桌上开口朝上的容器(``ContainerTarget`` 子类,
cup / plant_pot / wine_glass)。

杯子(RoboTwin_021_cup 00002,高 11 cm)口宽,水从靠倾倒方向一侧的杯沿流出,所以支点取杯沿(``Cup.LIP``),
而不是杯口中心;约 4 cm 深的水在倾角约 60 度时开始出流,预倾 / 截流角度都比瓶子小。
矮杯子(cup 00007,7 cm)不行:水平侧抓杯身时夹爪离桌面太近,预抓取位姿就不可达;从斜上方捏杯沿又转不到倾倒角度。
"""

import logging

import numpy as np

from task.RoboDojo.experts import pour_liquid_into_cup as cup

log = logging.getLogger(__name__)


class Cup(cup.Vessel):
    LABEL = "water_cup"
    FLUID = "liquid"
    GRASP_Z = (0.08, 0.10)  # 侧抓杯沿下方(TCP 高度):再低时预抓取位姿离桌面太近、不可达
    # 杯身外轮廓(原点在杯底):下窄上宽
    RINGS = [(0.0, 0.032), (0.03, 0.033), (0.06, 0.036), (0.09, 0.038), (0.112, 0.038)]
    LIP = 0.037
    TILT_START = 45.0
    TILT_STOP = 60.0
    BACK_TILT = 40.0
    WRIST_PICK = 4  # 不要先把手腕转半圈再抓


class FlatTarget(cup.Target):
    """平放在桌上的实心目标:出流点在它中心(往杯子一侧 ``LAND_BACK``、沿抓取接近方向 ``ACROSS``)、顶面上方
    ``MOUTH_ABOVE`` 处。

    倾倒到 90 度时夹爪导轨(横跨 +-8.7 cm)竖了起来,``clear`` 要让它高出目标顶面,杯口实际在支点上方约 8.5 cm。
    水基本落在出流点正下方,所以不补偿(之前的 6 / 3 cm 补偿是按少算了液体缩放的粒子位置调的)。
    慢倒(``POUR_TIME``)、只倒到 90 度,水流更集中。"""

    MOUTH_ABOVE = 0.01
    LAND_BACK = 0.0
    ACROSS = 0.0
    TILT_POUR = 90.0
    POUR_TIME = 6.0

    def __init__(self, env):
        box = cup.footprint(env, self.LABEL)
        self.aim_xy, self.top, self.boxes = box[0], box[3], [box]

    def pivot_xy(self, d):
        k = np.cross([0.0, 0.0, 1.0], d)[:2]  # 抓取接近方向(倾倒的转轴)
        return self.aim_xy - self.LAND_BACK * d[:2] + self.ACROSS * k


class ContainerTarget(FlatTarget):
    """桌上开口朝上的容器。倾倒和平放目标(risky 任务)一样:抓杯沿下方、杯沿在目标顶面上方 ``MOUTH_ABOVE``、
    往杯子一侧后撤 ``LAND_BACK``、沿 ``k`` 偏 ``ACROSS``,倒向的点取容器的底面接触圆心(马克杯的包围盒含把手,
    中心会偏)。倒到 90 度时水流慢、杯里剩得多,所以倒得更陡(``TILT_POUR``),杯子基本倒空。
    这组偏移下 bowl / pot / plastic_box 倒出的水 98 ~ 100% 落进容器(实测)。"""

    LAND_BACK = 0.06
    ACROSS = 0.03
    TILT_POUR = 120.0

    def __init__(self, env):
        super().__init__(env)
        m = cup._mat(env.get_obj_pose(self.LABEL))
        _, _, _, meta = env.get_obj_asset(self.LABEL)
        center = np.asarray(meta["active"]["place"]["up"]["contact_circle"]["center"][:2], dtype=float)
        self.aim_xy = (m @ np.append(center, [0.0, 1.0]))[:2]


def _poured(env, label):
    """杯子包围盒外的液体粒子中,在 ``label`` 包围盒内的比例。"""
    p = cup.particles(env, Cup.FLUID)
    inside = []
    for name in (Cup.LABEL, label):
        m, (lo, hi) = cup._mat(env.get_obj_pose(name)), cup._bbox(env, name)
        loc = (p - m[:3, 3]) @ m[:3, :3]
        inside.append(np.all((loc >= lo) & (loc <= hi), axis=1))
    poured = ~inside[0]
    return float(inside[1][poured].mean()) if poured.any() else float("nan")


def play_once(env, target_cls):
    target = cup.play_once(env, target_cls, Cup)
    landed = env.reward_manager.func_parser.fluid_landed_state[0].get((Cup.FLUID, target.LABEL))
    log.info(
        "landed on the %s %.3f, in its bbox %.3f, left in the cup %.3f",
        target.LABEL,
        float(landed.mean()) if landed is not None else float("nan"),
        _poured(env, target.LABEL),
        cup._in_vessel(env, Cup),
    )
    return target
