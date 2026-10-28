from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_refrigerator import PutRefrigeratorCommon


class PutAppleRefrigeratorCommon(PutRefrigeratorCommon):
    LABEL = "apple"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the apple into the refrigerator."


class put_apple_refrigerator(PutAppleRefrigeratorCommon, TaskEnv):
    pass
