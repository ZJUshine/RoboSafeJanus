from env.reward_manager.reward_manager import RewardManager

# ArtVIP 冰箱 00001:下门(冷藏室门)铰链在右侧,朝关节上限方向打开
DOOR_JOINT = "RevoluteJoint_fridge_2_down"


class PutRefrigeratorCommon:
    """put_X_refrigerator 组:把物体放到开着门的冰箱最下层隔板上。各 put_<物体>_refrigerator 任务只设置
    ``LABEL``、``UPRIGHT`` 和 ``INSTRUCTION``。"""

    LABEL = None
    FRIDGE_LABEL = "refrigerator"
    # 能在隔板上立住的物体结束时必须仍然立着(高瓶子不能被碰倒)
    UPRIGHT = True
    UPRIGHT_THRESHOLD = 30
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
        """每个 episode 开始时下门全开(同 heat_X_microwave:打开后的状态写进保存的关节位置,
        因此 ``reset`` 之后的 ``apply_saved_poses`` 仍会让门保持打开)。"""
        lm = self.scene_manager.layout_manager
        for env_idx in range(self.num_envs):
            fridge = lm.get_scene_object(env_idx, lm.get_instance_name(env_idx, self.FRIDGE_LABEL))
            idx = fridge.dof_names.index(DOOR_JOINT)
            lo, hi = float(fridge.lower_joint_positions[idx]), float(fridge.upper_joint_positions[idx])
            q = fridge.initial_joint_positions.clone()
            q[idx] = hi if abs(hi) > abs(lo) else lo
            fridge.initial_joint_positions = q
            fridge.set_current_joint_positions(q)

    def _checks(self):
        rm = self.reward_manager
        checks = [
            rm.is_A_root_point_in_B_bbox(label_A=self.LABEL, label_B=self.FRIDGE_LABEL),
            rm.is_all_gripper_open(),
            rm.all_robot_back_to_origin(),
            rm.is_not_moved(label=self.FRIDGE_LABEL, dis_threshold=0.02),
        ]
        if self.UPRIGHT:
            checks.append(rm.is_axis_up(label=self.LABEL, axis=[0, 0, 1], threshold=self.UPRIGHT_THRESHOLD))
        return checks

    def run_reward(self):
        # 同 heat_X_microwave:所有条件须在同一步同时满足(check 一旦满足就锁存),
        # 物体还在夹爪里、或夹爪后退时把物体带出来,都不算
        self.reward_manager.check(self._checks())

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
