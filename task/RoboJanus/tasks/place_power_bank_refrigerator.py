from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_juice_refrigerator import PlaceJuiceRefrigeratorCommon


class PlacePowerBankRefrigeratorCommon(PlaceJuiceRefrigeratorCommon):
    LABEL = "power_bank"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the power bank into the refrigerator."


class place_power_bank_refrigerator(PlacePowerBankRefrigeratorCommon, TaskEnv):
    pass
