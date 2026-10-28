from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_refrigerator import PutRefrigeratorCommon


class PutPowerBankRefrigeratorCommon(PutRefrigeratorCommon):
    LABEL = "power_bank"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the power bank into the refrigerator."


class put_power_bank_refrigerator(PutPowerBankRefrigeratorCommon, TaskEnv):
    pass
