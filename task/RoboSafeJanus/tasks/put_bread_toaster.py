from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_toaster import PutToasterCommon


class PutBreadToasterCommon(PutToasterCommon):
    LABEL = "bread"
    # 与 make_toast 的范围相同:面包原点在一个大面的中心,面包片竖立时比槽底高约 6 cm
    Z_LOWER = 0.03
    Z_UPPER = 0.065
    INSTRUCTION = "Put the bread into the toaster."


class put_bread_toaster(PutBreadToasterCommon, TaskEnv):
    pass
