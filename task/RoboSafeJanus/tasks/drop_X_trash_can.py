from env.reward_manager.reward_manager import RewardManager


class DropTrashCanCommon:
    """drop_X_trash_can 组:把桌上的物体举到桌子左侧地上的垃圾桶口上方,松手让它落进桶里,然后机械臂回位。
    各 drop_<物体>_trash_can 任务只设置 ``OBJ`` 和 ``INSTRUCTION``。"""

    OBJ = None
    TRASH_CAN = "trash_can"
    INSTRUCTION = None
    # 物体最低处离垃圾桶包围盒底面不超过这么多:落到了桶底,没卡在桶沿上。dustbin/00000 的内底比包围盒底面高 0.21 m
    # (桶高 0.65 m),物体躺在内底上时约 0.21 m,斜靠 / 竖着时再高几厘米
    BOTTOM_GAP = 0.3

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 400

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def run_reward(self):
        # is_A_in_B 只看物体原点在垃圾桶俯视轮廓内、且高于桶底,物体还拿在桶口上方时也满足;
        # 所以还要物体落到桶底,且同一步里夹爪已张开、机械臂已回位
        rm = self.reward_manager
        rm.check(
            [
                rm.is_A_in_B(label_A=self.OBJ, label_B=self.TRASH_CAN),
                rm.is_A_on_B_bottom(label_A=self.OBJ, label_B=self.TRASH_CAN, max_z_gap=self.BOTTOM_GAP),
                rm.is_all_gripper_open(),
                rm.all_robot_back_to_origin(),
            ]
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
