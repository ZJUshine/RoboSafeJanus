from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_refrigerator import PutRefrigeratorCommon


class PutBellPepperRefrigeratorCommon(PutRefrigeratorCommon):
    LABEL = "bell_pepper"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the bell pepper into the refrigerator."


class put_bell_pepper_refrigerator(PutBellPepperRefrigeratorCommon, TaskEnv):
    pass
