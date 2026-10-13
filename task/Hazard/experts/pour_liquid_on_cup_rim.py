"""pour_liquid_on_cup_rim 的脚本专家:与 ``pour_liquid_into_cup`` 的专家相同(侧抓瓶颈、搬运预倾、
以瓶口为支点闭环倾倒、截流后把瓶子立起触发判定),只是瓶口不在杯口上方,而在杯子靠瓶子一侧的杯壁外面、
略高于杯沿,液流打在杯沿 / 杯壁外侧,大部分顺着杯子外面流到桌上。

和杯子相比只换了倾倒点(``_RimPour``):
- 瓶口 xy 在杯身外壁外 ``RIM_GAP`` 处(沿 -d),高度不低于杯沿以上 ``MOUTH_ABOVE``;
  瓶子 / 夹爪与杯子、桌面的间隙仍由杯子的 ``mouth_floor`` 保证。
- 液体落点与任务判定 ``is_A_fluid_on_B`` 用同一块区域(``Func_Parser.fluid_on_top_mask``):杯子水平轮廓外扩
  ``xy_buffer``、杯沿以上 ``z_above`` 到以下 ``z_below`` 之间。判定是累计的,碰到过杯子上部的粒子都算;
  立起时再要求倒进杯里的比例低于阈值(``is_A_fluid_not_in_B``),区别于正常倒进杯子。
"""

import logging

import numpy as np

from task.RoboDojo.experts import pour_liquid_into_cup as cup

log = logging.getLogger(__name__)

RIM_GAP = 0.006  # 倾倒时瓶口在杯身外壁外(沿 -d)的水平距离:液流出口近乎水平,射程约等于它时打在杯沿上
MOUTH_ABOVE = 0.01  # 倾倒时瓶口高出杯沿的最小高度


class _RimPour(cup._Pour):
    def _init_target(self):
        super()._init_target()
        self.cup_box = cup._bbox(self.env, "cup")
        self.region = self._check_args() or {}  # 碰到杯子上部的区域和比例阈值,同任务判定
        self.touched = None  # 专家自己累计的碰到过杯子上部的粒子(任务检查通过后就不再更新,这里一直更新)

    def _pivot_xy(self):
        return self.aim_xy - (self.body_r + RIM_GAP) * self.d[:2]

    def mouth_floor(self, m_xy, tilt):
        return max(super().mouth_floor(m_xy, tilt), self.top + MOUTH_ABOVE)

    def _check_args(self):
        """任务里 ``is_A_fluid_on_B`` 检查的参数;已通过(从检查列表移除)时返回 None。"""
        for group in self.env.reward_manager.check_list[0]:
            for c in group:
                if isinstance(c, tuple) and c[0] == "is_A_fluid_on_B":
                    return c[1]
        return None

    def in_target(self, p):
        """粒子 ``p`` 是否在杯子里(同杯子任务);同时把碰到杯子上部(同任务判定的区域)的粒子累计到 ``touched``。"""
        pose, a = self.env.get_obj_pose("cup"), self.region
        on = self.env.reward_manager.func_parser.fluid_on_top_mask(
            p,
            pose[:3],
            pose[3:],
            np.stack(self.cup_box),
            xy_buffer=a.get("xy_buffer", 0.0),
            z_above=a.get("z_above", 0.01),
            z_below=a.get("z_below", 0.005),
        )
        on &= ~cup._inside(self.env, p, "bottle")
        self.touched = on if self.touched is None or len(self.touched) != len(on) else self.touched | on
        return super().in_target(p)

    def judge(self):
        """按任务判定估算:累计碰到杯子上部的比例(``touched`` 为专家一直累计的值)、杯内比例、瓶内残留比例、
        洒在桌上的液体沿倾倒方向相对杯子中心的平均位置(正 = 射过了头)。两项检查都已通过时返回 None。"""
        env, rm = self.env, self.env.reward_manager
        if not rm.check_list[0] and not rm.trigger_check_list[0]:
            return None
        p = cup._particles(env)
        in_bottle = cup._inside(env, p, "bottle")
        in_cup = self.in_target(p) & ~in_bottle
        out = ~(in_bottle | in_cup) & (p[:, 2] < self.table_z + 0.012)
        return {
            "touched": round(float(self.touched.mean()), 4),
            "need": self.region.get("percentage_threshold"),
            "touched_passed": self._check_args() is None,
            "in_cup": round(float(in_cup.mean()), 4),
            "residual": round(float(in_bottle.mean()), 4),
            "spill_along_d": round(float(((p[out, :2] - self.aim_xy) @ self.d[:2]).mean()), 3) if out.any() else None,
        }

    def log_target(self):
        cup_cat, cup_idx, _, _ = self.env.get_obj_asset("cup")
        log.info(
            "cup %s/%d: body r=%.3f, pour toward %s; mouth %.3f outside the wall, %.3f above the rim at %.0f deg / "
            "%.3f at 110 deg",
            cup_cat,
            cup_idx,
            self.body_r,
            np.round(self.d[:2], 2).tolist(),
            float(np.linalg.norm(self.pivot_xy - self.aim_xy)) - self.body_r,
            self.cur[0][2] - self.top,
            cup.TILT_START,
            self.pour_state(110.0)[0][2] - self.top,
        )


def play_once(env):
    pour = cup.play_once(env, pour_cls=_RimPour)
    f = pour.fluid()
    log.info(
        "touched the cup rim in total: %.4f (need %s); in the cup %.4f",
        pour.touched.mean(),
        pour.region.get("percentage_threshold"),
        f["cup"] / f["total"],
    )
