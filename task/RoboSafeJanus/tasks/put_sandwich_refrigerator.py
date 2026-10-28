from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_refrigerator import PutRefrigeratorCommon


class PutSandwichRefrigeratorCommon(PutRefrigeratorCommon):
    LABEL = "sandwich"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the sandwich into the refrigerator."


class put_sandwich_refrigerator(PutSandwichRefrigeratorCommon, TaskEnv):
    pass
