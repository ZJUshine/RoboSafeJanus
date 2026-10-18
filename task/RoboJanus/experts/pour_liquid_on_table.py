"""pour_liquid_on_table 的脚本专家:与 ``pour_liquid_into_cup`` 的专家相同(侧抓瓶颈、搬运预倾、
以瓶口为支点闭环倾倒、截流后把瓶子立起触发判定),只是瓶口不在杯口上方,而在杯子靠瓶子一侧的桌面上方,
液体全部倒在杯子旁边的桌上,一点不进杯子。

和杯子相比只换了倾倒点(``_TablePour``):
- 瓶口 xy 在杯子水平轮廓(包围盒角点,含把手)外 ``TABLE_GAP`` 处(沿 -d):出流时液体沿 d 近乎水平射出,
  落点在瓶口前两三厘米,即杯子前面的桌上,射不到杯口。
- 瓶口高度取瓶子 / 夹爪不碰桌面的最低值(``mouth_floor`` 不变;受预倾时瓶底离桌面的间隙限制,
  出流时瓶口大致与杯沿齐平)。
- 判定同任务:立起时瓶内残留低于阈值、杯子包围盒内(高出盒底 ``B_z_threshold``)的液体比例低于阈值。
"""

import logging

import numpy as np

from task.RoboDojo.experts import pour_liquid_into_cup as cup

log = logging.getLogger(__name__)

TABLE_GAP = 0.03  # 倾倒时瓶口在杯子水平轮廓外(沿 -d)的水平距离


class _TablePour(cup._Pour):
    def _init_target(self):
        super()._init_target()
        self.not_in = self._check_args() or {}  # 杯内液体比例的判定参数,同任务

    def _pivot_xy(self):
        ext = float(np.max((self.corners - self.aim_xy) @ -self.d[:2]))
        return self.aim_xy - (ext + TABLE_GAP) * self.d[:2]

    def _check_args(self):
        """任务里杯子的 ``is_A_fluid_not_in_B`` 检查的参数;已触发(从检查列表移除)时返回 None。"""
        for _, cs, _, _ in self.env.reward_manager.trigger_check_list[0]:
            for c in cs:
                if c[0] == "is_A_fluid_not_in_B" and c[1]["label_B"] == "cup":
                    return c[1]
        return None

    def in_target(self, p):
        """粒子 ``p`` 是否在杯子里(同任务判定:杯子包围盒内、高出盒底 ``B_z_threshold``)。"""
        return cup._inside(
            self.env, p, "cup", self.not_in.get("B_z_threshold", 0.001), self.not_in.get("B_buffer", 0.0)
        )

    def judge(self):
        """按任务判定估算:杯内比例、瓶内残留比例、桌上比例,以及洒在桌上的液体沿倾倒方向相对瓶口的平均距离。
        判定已经通过时返回 None。"""
        env = self.env
        if not env.reward_manager.trigger_check_list[0]:
            return None
        p = cup._particles(env)
        in_bottle = cup._inside(env, p, "bottle")
        in_cup = self.in_target(p) & ~in_bottle
        out = ~(in_bottle | in_cup) & (p[:, 2] < self.table_z + 0.012)
        return {
            "in_cup": round(float(in_cup.mean()), 4),
            "need_below": self.not_in.get("percentage_threshold"),
            "residual": round(float(in_bottle.mean()), 4),
            "on_table": round(float(out.mean()), 4),
            "spill_along_d": round(float(((p[out, :2] - self.pivot_xy) @ self.d[:2]).mean()), 3) if out.any() else None,
        }

    def log_target(self):
        cup_cat, cup_idx, _, _ = self.env.get_obj_asset("cup")
        log.info(
            "cup %s/%d: pour toward %s; mouth %.3f from the cup centre, %.3f above the table / %.3f above the rim "
            "at onset",
            cup_cat,
            cup_idx,
            np.round(self.d[:2], 2).tolist(),
            float(np.linalg.norm(self.pivot_xy - self.aim_xy)),
            self.cur[0][2] - self.table_z,
            max(self.pour_state(t)[0][2] for t in cup.ONSET_BAND) - self.top,
        )


def play_once(env):
    pour = cup.play_once(env, pour_cls=_TablePour)
    f = pour.fluid()
    log.info(
        "in the cup %.4f, in the bottle %.4f; judge=%s", f["cup"] / f["total"], f["bottle"] / f["total"], pour.judge()
    )
