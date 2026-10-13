"""pour_liquid_on_phone 的脚本专家:与 ``pour_liquid_into_cup`` 的专家相同(侧抓瓶颈、搬运预倾、
以瓶口为支点闭环倾倒、截流后把瓶子立起触发判定),只是倒向平放在桌上、屏幕朝上的手机。

和杯子相比只换了目标几何(``_PhonePour``):
- 手机是实心的平板,没有杯口:瓶子 / 夹爪在手机水平轮廓(外扩 ``CUP_MARGIN``)上方时要高出手机顶面
  ``CLEAR_RIM``,其余地方高出桌面 ``CLEAR_TABLE``。
- 出流时液体近乎水平射出,瓶口停在手机中心靠瓶子一侧 ``LAND_BACK`` 处,让液流落在屏幕中部。
- 液体落点与任务判定 ``is_A_fluid_on_B`` 用同一块区域(``Func_Parser.fluid_on_top_mask``):手机顶面上方
  ``z_above`` 到下方 ``z_below`` 之间。判定是累计的:倒的过程中落到过手机上的粒子都算,即使之后流到桌上。
"""

import logging

import numpy as np

from task.RoboDojo.experts import pour_liquid_into_cup as cup

log = logging.getLogger(__name__)

LAND_BACK = 0.03  # 倾倒时瓶口在手机中心靠瓶子一侧(沿 -d)的距离:液流出口近乎水平,要留出射程


class _PhonePour(cup._Pour):
    TARGET = "phone"

    def _init_target(self):
        """手机顶面高度 ``top``、中心 ``aim_xy``、水平轮廓 ``corners``;``axes`` / ``half`` 是轮廓的两条
        水平轴(手机坐标系中不朝上的两根轴在 xy 平面的投影)和半边长。"""
        env = self.env
        m = cup._mat(env.get_obj_pose("phone"))
        lo, hi = cup._bbox(env, "phone")
        self.phone_box = (lo, hi)
        self.region = self._check_args() or {}  # 落点区域和比例阈值,同任务判定
        self.landed = None  # 专家自己累计的落到过手机上的粒子(任务检查通过后就不再更新,这里一直更新)
        up = int(np.argmax(np.abs(m[2, :3])))
        side = [i for i in range(3) if i != up]
        verts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = verts @ m[:3, :3].T + m[:3, 3]
        self.top = float(world[:, 2].max())
        self.aim_xy = (m @ np.append((lo + hi) / 2, 1.0))[:2]
        self.axes = m[:2, side]
        self.half = (hi - lo)[side] / 2
        self.corners = np.array(
            [self.aim_xy + self.axes @ (s * self.half) for s in [(1, 1), (1, -1), (-1, 1), (-1, -1)]]
        )

    def _pivot_xy(self):
        return self.aim_xy - LAND_BACK * self.d[:2]

    def mouth_floor(self, m_xy, tilt):
        """瓶口在 ``m_xy`` 上方、倾角 ``tilt`` 时,瓶子和夹爪都不碰手机、桌面所需的最低瓶口高度。"""
        rel = (self.points - self.mouth) @ self.R(tilt).T
        xy = m_xy + rel[:, :2]
        loc = (xy - self.aim_xy) @ self.axes
        over = np.all(np.abs(loc) <= self.half + cup.CUP_MARGIN, axis=1)
        floor = np.where(over, self.top + cup.CLEAR_RIM, self.table_z + cup.CLEAR_TABLE)
        return float(np.max(floor - rel[:, 2]))

    def _check_args(self):
        """任务里 ``is_A_fluid_on_B`` 检查的参数;已通过(从检查列表移除)时返回 None。"""
        rm = self.env.reward_manager
        for group in rm.check_list[0]:
            for c in group:
                if isinstance(c, tuple) and c[0] == "is_A_fluid_on_B":
                    return c[1]
        return None

    def in_target(self, p):
        """粒子 ``p`` 是否在手机顶面上(同任务判定的区域),并累计到 ``landed``。"""
        pose, a = self.env.get_obj_pose("phone"), self.region
        on = self.env.reward_manager.func_parser.fluid_on_top_mask(
            p,
            pose[:3],
            pose[3:],
            np.stack(self.phone_box),
            xy_buffer=a.get("xy_buffer", 0.0),
            z_above=a.get("z_above", 0.01),
            z_below=a.get("z_below", 0.005),
        )
        self.landed = on if self.landed is None or len(self.landed) != len(on) else self.landed | on
        return on

    def judge(self):
        """按任务判定估算:累计落到手机上的比例(``landed`` 为专家一直累计的值)、瓶内残留比例、洒在桌上的
        液体沿倾倒方向相对手机中心的平均位置(正 = 射过了头)。两项检查都已通过时返回 None。"""
        env, rm = self.env, self.env.reward_manager
        if not rm.check_list[0] and not rm.trigger_check_list[0]:
            return None
        p = cup._particles(env)
        in_bottle = cup._inside(env, p, "bottle")
        out = ~(in_bottle | self.in_target(p)) & (p[:, 2] < self.table_z + 0.012)
        return {
            "landed": round(float(self.landed.mean()), 4),
            "need": self.region.get("percentage_threshold"),
            "landed_passed": self._check_args() is None,
            "residual": round(float(in_bottle.mean()), 4),
            "spill_along_d": round(float(((p[out, :2] - self.aim_xy) @ self.d[:2]).mean()), 3) if out.any() else None,
        }

    def log_target(self):
        log.info(
            "phone top %.3f (%.3f above the table), half size %s, pour toward %s; mouth %.3f above the phone at "
            "%.0f deg / %.3f at 110 deg",
            self.top,
            self.top - self.table_z,
            np.round(self.half, 3).tolist(),
            np.round(self.d[:2], 2).tolist(),
            self.cur[0][2] - self.top,
            cup.TILT_START,
            self.pour_state(110.0)[0][2] - self.top,
        )


def play_once(env):
    pour = cup.play_once(env, pour_cls=_PhonePour)
    pour.fluid()
    log.info(
        "landed on the phone in total: %.4f (need %s)", pour.landed.mean(), pour.region.get("percentage_threshold")
    )
