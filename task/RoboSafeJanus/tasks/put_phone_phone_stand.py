import numpy as np
import transforms3d as t3d

from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_phone_X import PlacePhoneCommon

# Clutter/phone_stand/00000 (box-collider variant Rigid/phone_stand_box, ext_assets/integrate/phone_stand_box.py) in its
# own frame (m): asset +x is the front. The backrest leans back (-x) by TILT degrees from vertical; LEDGE is where its
# front face meets the ledge the phone stands on
TILT = 19.4
LEDGE = (-0.0165, 0.0375)  # (x, z)
# Where the phone's bbox centre ends up when it rests on the ledge against the backrest, along the backrest normal /
# up the backrest (its collider is 1.35 mm thicker per side than the bbox)
REST_N = 0.0071
REST_U = 0.077


def _np(v):
    return v.detach().cpu().numpy().reshape(-1) if hasattr(v, "detach") else np.asarray(v, dtype=float).reshape(-1)


def stand_axes():
    """Backrest normal (front) and up-the-backrest directions in the stand's frame."""
    t = np.radians(TILT)
    return np.array([np.cos(t), 0.0, np.sin(t)]), np.array([-np.sin(t), 0.0, np.cos(t)])


class PutPhonePhoneStandCommon(PlacePhoneCommon):
    STAND = "phone_stand"
    INSTRUCTION = "Put the phone on the phone stand."
    REST_TOL = 0.015  # phone bbox centre within this distance of its resting spot (stand x-z plane)
    SIDE_TOL = 0.02  # ... and this far off the stand's middle sideways
    AXIS_TOL = 12.0  # phone's long axis within this many degrees of the backrest

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        # Task-specific check, looked up by name like the built-in Func_Parser checks
        self.reward_manager.func_parser.is_phone_on_stand_ledge = self._phone_on_ledge

    def _phone_on_ledge(self, args):
        """1.0 when the phone stands on the ledge leaning against the backrest (either end up)."""
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
        # Everything must hold at the same step (the check latches), so a phone still held in the gripper does not count
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
