from env.reward_manager.reward_manager import RewardManager

# SimReady 台式微波炉(D00516)的门关节;门 link 是 part_21,关节 0 度(下限)就是门开 90 度
DOOR_JOINT = "piv_door_hinge"


class HeatMicrowaveCommon:
    """heat_X_microwave 组:把物体放进开着门的微波炉。各 heat_<物体>_microwave 任务只设置
    ``LABEL``(物体在 YAML 中的 label)和 ``INSTRUCTION``。

    ``INSTRUCTION`` 可以是一条字符串,也可以是多条同义指令的列表(与设计文件里的 ``S1.instruction`` 一致):
    ``gen_instruction`` 返回全部候选,采集(``save_episode`` 按 seed)和评测(``DescriptionManager``)各自随机抽一条。"""

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
            idx = mw.dof_names.index(DOOR_JOINT)
            q = mw.initial_joint_positions.clone()
            q[idx] = float(mw.lower_joint_positions[idx])
            mw.initial_joint_positions = q
            mw.set_current_joint_positions(q)

    def run_reward(self):
        # 物体停在微波炉的 bbox 内,同时夹爪已松开、双臂回到初始位姿、微波炉没被推动。
        # 四个条件须在同一步同时满足:check 一旦满足就锁存,物体还在夹爪里、或夹爪后退时把物体带出来,都不能算。
        self.reward_manager.check(
            [
                self.reward_manager.is_A_root_point_in_B_bbox(label_A=self.LABEL, label_B="microwave"),
                self.reward_manager.is_all_gripper_open(),
                self.reward_manager.all_robot_back_to_origin(),
                self.reward_manager.is_not_moved(label="microwave", dis_threshold=0.02),
            ]
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION] if isinstance(self.INSTRUCTION, str) else list(self.INSTRUCTION)
