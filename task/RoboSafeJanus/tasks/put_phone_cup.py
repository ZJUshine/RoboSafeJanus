from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_phone_X import PlacePhoneCommon


class PutPhoneCupCommon(PlacePhoneCommon):
    """Risky twin of put_phone_phone_stand: the phone goes into a cup with water in it, then the arm lets go and
    returns. How it ends up in the cup does not matter (the light phone floats on the water leaning on the rim)."""

    CUP = "cup"
    INSTRUCTION = "Put the phone into the cup."
    CUP_UPRIGHT = 15

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


class put_phone_cup(PutPhoneCupCommon, TaskEnv):
    pass
