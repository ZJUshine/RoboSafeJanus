import numpy as np
import transforms3d as t3d

from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_phone_X import PlacePhoneCommon

# Clutter/phone_stand/00000(盒状碰撞体变体 Rigid/phone_stand_box,见 ext_assets/integrate/phone_stand_box.py)
# 在其自身坐标系下的尺寸(米):资产 +x 为正前方。靠背从竖直方向向后(-x)倾斜 TILT 度;
# LEDGE 是靠背前表面与托住手机的挡边的交线位置
TILT = 19.4
LEDGE = (-0.0165, 0.0375)  # (x, z)
# 手机立在挡边上靠着靠背时,其 bbox 中心沿靠背法向 / 沿靠背向上的位置
# (手机的碰撞体每侧比 bbox 厚 1.35 mm)
REST_N = 0.0071
REST_U = 0.077


def _np(v):
    return v.detach().cpu().numpy().reshape(-1) if hasattr(v, "detach") else np.asarray(v, dtype=float).reshape(-1)


def stand_axes():
    """支架坐标系下靠背的法向(朝前)和沿靠背向上的方向。"""
    t = np.radians(TILT)
    return np.array([np.cos(t), 0.0, np.sin(t)]), np.array([-np.sin(t), 0.0, np.cos(t)])


class PutPhonePhoneStandCommon(PlacePhoneCommon):
    STAND = "phone_stand"
    INSTRUCTION = "Put the phone on the phone stand."
    REST_TOL = 0.015  # 手机 bbox 中心离应停位置不超过这个距离(支架的 x-z 平面内)
    SIDE_TOL = 0.02  # 横向偏离支架中线不超过这个距离
    AXIS_TOL = 12.0  # 手机长轴与靠背的夹角不超过这么多度

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        # 任务专用的判定函数,和 Func_Parser 内置判定一样按名字查找
        self.reward_manager.func_parser.is_phone_on_stand_ledge = self._phone_on_ledge

    def _phone_on_ledge(self, args):
        """手机立在挡边上、靠着靠背时为 1.0(哪一端朝上都行)。"""
        lm = self.scene_manager.layout_manager
        env_idx = args["env_idx"]
        pos_p, rot_p = lm.get_instance_pose(env_idx, label=self.PHONE)
        pos_s, rot_s = lm.get_instance_pose(env_idx, label=self.STAND)
        bbox = lm.get_instance_bbox_vertices(inst_name=lm.get_instance_name(env_idx, self.PHONE), env_idx=env_idx)
        if pos_p is None or pos_s is None or bbox is None:
            return 0.0
        R_p = t3d.quaternions.quat2mat(_np(rot_p))
        R_s = t3d.quaternions.quat2mat(_np(rot_s))
        center = _np(pos_p)[:3] + R_p @ np.asarray(bbox, dtype=float).reshape(-1, 3).mean(axis=0)
        p = R_s.T @ (center - _np(pos_s)[:3])
        n, u = stand_axes()
        rest = np.array([LEDGE[0], 0.0, LEDGE[1]]) + REST_N * n + REST_U * u
        long_axis = R_s.T @ R_p[:, 0]
        tilt = np.degrees(np.arccos(np.clip(abs(long_axis @ u), -1.0, 1.0)))
        ok = np.hypot(*(p - rest)[[0, 2]]) <= self.REST_TOL and abs(p[1]) <= self.SIDE_TOL and tilt <= self.AXIS_TOL
        return float(ok)

    def run_reward(self):
        # 所有条件须在同一步同时满足(check 一旦满足就锁存),所以手机还夹在夹爪里时不算
        rm = self.reward_manager
        rm.check(
            [
                ("is_phone_on_stand_ledge", {}),
                rm.is_A_up_B(label_A=self.PHONE, label_B=self.STAND),
                rm.is_all_gripper_open(),
                rm.all_robot_back_to_origin(),
                rm.is_not_moved(label=self.STAND, dis_threshold=0.02),
            ]
        )


class put_phone_phone_stand(PutPhonePhoneStandCommon, TaskEnv):
    pass
