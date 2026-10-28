import numpy as np
import transforms3d as t3d

from env.reward_manager.reward_manager import RewardManager

# GenieSim_shelf/00001 顶层在置物架自身坐标系下的尺寸(米,未缩放):前沿(资产 +x)朝向机器人。
# 缩放变体(GenieSim_shelf_s080)把这些值乘以 ``rack_scale``
TOP_Z = 0.122
FRONT_X = 0.127
HALF_Y = 0.176
HEIGHT = 0.404


def rack_scale(layout_manager, env_idx, label):
    """置物架的等比缩放系数:bbox 高度除以未缩放的 ``HEIGHT``(布局元数据里没有 scaled_variant.py 写入的
    ``ext_link.scale``)。"""
    inst = layout_manager.get_instance_name(env_idx, label)
    bbox = np.asarray(layout_manager.get_instance_bbox_vertices(inst_name=inst, env_idx=env_idx), dtype=float)
    return round(float(np.ptp(bbox.reshape(-1, 3)[:, 2])) / HEIGHT, 3)


def _np(v):
    return v.detach().cpu().numpy().reshape(-1) if hasattr(v, "detach") else np.asarray(v, dtype=float).reshape(-1)


class PlaceCeramicBowlStorageRackCommon:
    """place_ceramic_bowl_storage_rack_X 组:把碗放到置物架顶层。各任务只设置 ``REGION``(碗最终要落在顶层的
    哪个区域:"center" 中央,或 "edge" 边缘,即部分悬出前沿)和 ``INSTRUCTION``。"""

    BOWL = "ceramic_bowl"
    RACK = "storage_rack"
    REGION = None
    INSTRUCTION = None
    CENTER_TOL = 0.04  # center:碗中心离顶层中心不超过这个距离
    EDGE_OVERHANG = (0.015, 0.06)  # edge:碗沿悬出前沿的距离范围
    BOTTOM_TOL = (-0.01, 0.02)  # 碗底相对顶层表面的高度范围
    UPRIGHT_THRESHOLD = 20

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        # 任务专用的判定函数,和 Func_Parser 内置判定一样按名字查找
        self.reward_manager.func_parser.is_bowl_in_rack_top_region = self._bowl_in_top_region
        self.step_lim = 400

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def _bowl_in_top_region(self, args):
        """碗停在置物架顶层的 ``args["region"]`` 区域内时为 1.0。"""
        lm = self.scene_manager.layout_manager
        env_idx = args["env_idx"]
        pos_b, _ = lm.get_instance_pose(env_idx, label=self.BOWL)
        pos_r, rot_r = lm.get_instance_pose(env_idx, label=self.RACK)
        bbox = lm.get_instance_bbox_vertices(inst_name=lm.get_instance_name(env_idx, self.BOWL), env_idx=env_idx)
        if pos_b is None or pos_r is None or bbox is None:
            return 0.0
        bbox = np.asarray(bbox, dtype=float).reshape(-1, 3)
        p = t3d.quaternions.quat2mat(_np(rot_r)).T @ (_np(pos_b)[:3] - _np(pos_r)[:3])
        radius = float(np.ptp(bbox[:, 0]) + np.ptp(bbox[:, 1])) / 4
        s = rack_scale(lm, env_idx, self.RACK)
        front = FRONT_X * s
        bottom = p[2] + bbox[:, 2].min() - TOP_Z * s
        if not (self.BOTTOM_TOL[0] <= bottom <= self.BOTTOM_TOL[1] and abs(p[1]) <= HALF_Y * s and p[0] <= front):
            return 0.0
        if args["region"] == "center":
            return float(np.hypot(p[0], p[1]) <= self.CENTER_TOL)
        overhang = p[0] + radius - front
        return float(self.EDGE_OVERHANG[0] <= overhang <= self.EDGE_OVERHANG[1])

    def run_reward(self):
        # 所有条件须在同一步同时满足(check 一旦满足就锁存),所以碗还夹在夹爪里时不算
        rm = self.reward_manager
        rm.check(
            [
                ("is_bowl_in_rack_top_region", {"region": self.REGION}),
                rm.is_axis_up(label=self.BOWL, axis=[0, 0, 1], threshold=self.UPRIGHT_THRESHOLD),
                rm.is_all_gripper_open(),
                rm.all_robot_back_to_origin(),
                rm.is_not_moved(label=self.RACK, dis_threshold=0.02),
            ]
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
