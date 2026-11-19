from env.reward_manager.reward_manager import RewardManager

# SimReady 台式微波炉(D00516)的门关节;门 link 是 part_21,关节 0 度(下限)就是门开 90 度
DOOR_JOINT = "piv_door_hinge"

# 微波炉炉腔在微波炉局部坐标系下的范围(米)。资产自身的 oriented_bbox 把开着的门也算进去
# (0.543 x 0.763 x 0.299 m),物体摆在微波炉前面的桌面上也落在里面,不能用来判"放进炉腔"。
# 这里的数值与 experts/heat_X_microwave.py 的 MICROWAVES 一致(probe_cavity 量的炉腔内壁)。
CAVITY_BOX_MIN = (-0.224, -0.164, 0.012)
CAVITY_BOX_MAX = (0.119, 0.168, 0.267)
# 这几个物体夹住时的夹爪开度约 0.58--0.66(归一化,1 = 全开),原来的 0.6 阈值在夹着物体时也成立。
GRIPPER_OPEN_THRESHOLD = 0.9


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
        # 物体的整个包围盒都落在炉腔(微波炉局部坐标的盒子)内,同时夹爪已松开、双臂回到初始位姿、微波炉没被推动。
        # 用整个包围盒而不是原点:原点在炉腔盒内时物体仍可能一半卡在炉口。
        # 四个条件须在同一步同时满足:check 一旦满足就锁存,物体还在夹爪里、或夹爪后退时把物体带出来,都不能算。
        self.reward_manager.check(
            [
                self.reward_manager.is_A_bbox_in_B_local_box(
                    label_A=self.LABEL,
                    label_B="microwave",
                    box_min=CAVITY_BOX_MIN,
                    box_max=CAVITY_BOX_MAX,
                ),
                self.reward_manager.is_all_gripper_open(open_threshold=GRIPPER_OPEN_THRESHOLD),
                self.reward_manager.all_robot_back_to_origin(),
                self.reward_manager.is_not_moved(label="microwave", dis_threshold=0.02),
            ]
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION] if isinstance(self.INSTRUCTION, str) else list(self.INSTRUCTION)
