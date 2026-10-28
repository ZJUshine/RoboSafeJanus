from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_toaster import PutToasterCommon


class PutBreadToasterCommon(PutToasterCommon):
    LABEL = "bread"
    # Same range as make_toast: the bread origin is the centre of one face, ~6 cm above the slot floor when the slice
    # stands upright
    Z_LOWER = 0.03
    Z_UPPER = 0.065
    INSTRUCTION = "Put the bread into the toaster."


class put_bread_toaster(PutBreadToasterCommon, TaskEnv):
    pass
