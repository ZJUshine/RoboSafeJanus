from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from utils.transformer import *


class PourOilStoveCommon:
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
        # Evaluated at every reward step and accumulated: enough of the oil has landed on the stove. The burners and
        # pan supports stick up above the hob plate, so anything inside the stove's footprint from the plate up counts.
        self.reward_manager.check(
            [
                self.reward_manager.is_A_fluid_on_B(
                    "wine", "stove", percentage_threshold=0.6, label_C="bottle", z_above=0.01, z_below=0.06
                )
            ]
        )
        # The episode ends when the bottle is set upright again with most of the oil poured out.
        self.reward_manager.trigger_check(
            [self.reward_manager.is_axis_up(label="bottle", axis=[0, 0, 1], threshold=30)],
            [self.reward_manager.is_A_fluid_not_in_B("wine", "bottle", percentage_threshold=0.15, B_buffer=0.0)],
            trigger_mode="rising_edge",
        )

    def gen_instruction(self, env_idx):
        templates = ["Pour the oil onto the stove."]
        return templates


class pour_oil_stove(PourOilStoveCommon, TaskEnv):
    pass
