from pxr import Usd, UsdPhysics

from env.reward_manager.reward_manager import RewardManager


def door_joint(obj):
    """子 link 是门的那个转动关节的名字(ArtVIP 的门 link 名为 ``E_door_<n>``)。"""
    # obj.prim 是关节体的根 link(ext_anchor),所以要在整个物体里搜索
    for prim in Usd.PrimRange(obj.stage.GetPrimAtPath(obj.usd_prim_path)):
        if not prim.IsA(UsdPhysics.RevoluteJoint) or prim.GetName() not in obj.dof_names:
            continue
        body1 = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        if body1 and "door" in body1[0].name.lower():
            return prim.GetName()
    raise ValueError(f"{obj.prim_path} has no door joint")


class HeatMicrowaveCommon:
    """heat_X_microwave 组:把物体放进开着门的微波炉。各 heat_<物体>_microwave 任务只设置
    ``LABEL``(物体在 YAML 中的 label)和 ``INSTRUCTION``。"""

    LABEL = None
    INSTRUCTION = None

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 400

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.open_door()
        self.reward_manager.reset()

    def open_door(self):
        """每个 episode 开始时微波炉门全开。打开后的状态写进保存的关节位置,因此采集 / 评测环境在 ``reset``
        之后执行的 ``apply_saved_poses`` 仍会让门保持打开。"""
        lm = self.scene_manager.layout_manager
        for env_idx in range(self.num_envs):
            mw = lm.get_scene_object(env_idx, lm.get_instance_name(env_idx, "microwave"))
            idx = mw.dof_names.index(door_joint(mw))
            q = mw.initial_joint_positions.clone()
            q[idx] = float(mw.lower_joint_positions[idx])  # ArtVIP 的门朝关节下限方向打开(-90 度)
            mw.initial_joint_positions = q
            mw.set_current_joint_positions(q)

    def run_reward(self):
        # 物体停在微波炉的 bbox 内,同时夹爪已松开、双臂回到初始位姿、微波炉没被
        # 推动。四个条件须在同一步同时满足:check 一旦满足就锁存,物体还在夹爪里、或夹爪后退时把物体带出来,都不能算。
        self.reward_manager.check(
            [
                self.reward_manager.is_A_root_point_in_B_bbox(label_A=self.LABEL, label_B="microwave"),
                self.reward_manager.is_all_gripper_open(),
                self.reward_manager.all_robot_back_to_origin(),
                self.reward_manager.is_not_moved(label="microwave", dis_threshold=0.02),
            ]
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
