from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from utils.transformer import *


class PourLiquidOnCupRimCommon:
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
        # Evaluated at every reward step and accumulated: enough of the liquid has hit the upper part of the cup
        # (rim and outer wall, footprint expanded by xy_buffer, from z_below under to z_above over the rim).
        self.reward_manager.check(
            [
                self.reward_manager.is_A_fluid_on_B(
                    "wine",
                    "cup",
                    percentage_threshold=0.5,
                    label_C="bottle",
                    xy_buffer=0.01,
                    z_above=0.01,
                    z_below=0.03,
                )
            ]
        )
        # The episode ends when the bottle is set upright again: most of the liquid is poured out of the bottle
        # and most of it ran down the outside of the cup instead of into it.
        self.reward_manager.trigger_check(
            [self.reward_manager.is_axis_up(label="bottle", axis=[0, 0, 1], threshold=30)],
            [
                self.reward_manager.is_A_fluid_not_in_B("wine", "bottle", percentage_threshold=0.15, B_buffer=0.0),
                self.reward_manager.is_A_fluid_not_in_B(
                    "wine", "cup", percentage_threshold=0.5, B_buffer=0.0, B_z_threshold=0.001
                ),
            ],
            trigger_mode="rising_edge",
        )

    def gen_instruction(self, env_idx):
        templates = ["Pour the liquid from the bottle onto the outer rim of the cup."]
        return templates


class pour_liquid_on_cup_rim(PourLiquidOnCupRimCommon, TaskEnv):
    pass
