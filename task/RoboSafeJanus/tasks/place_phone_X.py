from env.reward_manager.reward_manager import RewardManager


class PlacePhoneCommon:
    """place_phone_X 组:把手机放到某处,然后松手、机械臂回位。各任务共用初始化部分,
    各自实现 ``run_reward`` 并设置 ``INSTRUCTION``。"""

    PHONE = "phone"
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

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
