import numpy as np
import transforms3d as t3d

from env.reward_manager.reward_manager import RewardManager


def _np(v):
    return v.detach().cpu().numpy().reshape(-1) if hasattr(v, "detach") else np.asarray(v, dtype=float).reshape(-1)


class DropTableCommon:
    """drop_X_table 组:把物体举到桌面上方足够高处,松手让它落回桌面,然后机械臂回位。
    各 drop_<物体>_table 任务只设置 ``OBJ`` 和 ``INSTRUCTION``。"""

    OBJ = None
    INSTRUCTION = None
    LIFT = 0.15  # 物体中心须比起始位置至少高出这么多
    REST_TOL = 0.03  # 落定:bbox 中心离桌面的高度不超过 bbox 最长边的一半再加这个余量

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        # 任务专用的判定函数,和 Func_Parser 内置判定一样按名字查找
        self.reward_manager.func_parser.is_resting_on_table = self._resting_on_table
        self.step_lim = 400

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def _resting_on_table(self, args):
        """物体以任意朝向躺在桌面上(没被举着、也没掉到地上)时为 1.0。"""
        lm = self.scene_manager.layout_manager
        env_idx = args["env_idx"]
        pos, rot = lm.get_instance_pose(env_idx, label=args["label"])
        bbox = lm.get_instance_bbox_vertices(inst_name=lm.get_instance_name(env_idx, args["label"]), env_idx=env_idx)
        if pos is None or bbox is None:
            return 0.0
        bbox = np.asarray(bbox, dtype=float).reshape(-1, 3)
        center = _np(pos)[:3] + t3d.quaternions.quat2mat(_np(rot)) @ bbox.mean(axis=0)
        table_z = float(lm.table_info[env_idx].get("height", 0.765))
        return float(table_z <= center[2] <= table_z + np.ptp(bbox, axis=0).max() / 2 + self.REST_TOL)

    def run_reward(self):
        # 按顺序的两个阶段:先把物体举起;之后物体重新落在桌面上,且夹爪张开、机械臂回位
        # (三者在同一步满足,所以物体还夹在夹爪里时不算)
        rm = self.reward_manager
        rm.check([rm.is_lift(label=self.OBJ, z_threshold=self.LIFT)])
        rm.check(
            [
                ("is_resting_on_table", {"label": self.OBJ}),
                rm.is_all_gripper_open(),
                rm.all_robot_back_to_origin(),
            ]
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
