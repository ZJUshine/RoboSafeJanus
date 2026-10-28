from env.reward_manager.reward_manager import RewardManager


class PlacePhoneCommon:
    """place_phone_X group: put the phone somewhere, then let go and return. The tasks share the setup; each one
    writes its own ``run_reward`` and sets ``INSTRUCTION``."""

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
