from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_refrigerator import PutRefrigeratorCommon


class PutLighterRefrigeratorCommon(PutRefrigeratorCommon):
    LABEL = "lighter"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the lighter into the refrigerator."


class put_lighter_refrigerator(PutLighterRefrigeratorCommon, TaskEnv):
    pass
