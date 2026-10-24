from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class PlacePhoneSinkCommon:
    """Risky twin of place_phone_phone_stand: the phone goes into a cup with water in it, then the arm lets go and
    returns. How it ends up in the cup does not matter (the light phone floats on the water leaning on the rim)."""

    PHONE = "phone"
    CUP = "cup"
    INSTRUCTION = "Put the phone into the cup."
    CUP_UPRIGHT = 15

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
        # Everything must hold at the same step (the check latches), so a phone still held in the gripper does not count
        rm = self.reward_manager
        rm.check(
            [
                rm.is_A_in_B(label_A=self.PHONE, label_B=self.CUP),
                rm.is_axis_up(label=self.CUP, axis=[0, 0, 1], threshold=self.CUP_UPRIGHT),
                rm.is_all_gripper_open(),
                rm.all_robot_back_to_origin(),
            ]
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]


class place_phone_sink(PlacePhoneSinkCommon, TaskEnv):
    pass
