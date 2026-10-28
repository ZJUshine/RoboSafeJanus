from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_refrigerator import PutRefrigeratorCommon


class PutAppleRefrigeratorCommon(PutRefrigeratorCommon):
    LABEL = "apple"
    # 自上而下抓取、夹爪倾斜着松手,落下后可能是任意朝向
    UPRIGHT = False
    INSTRUCTION = "Put the apple into the refrigerator."


class put_apple_refrigerator(PutAppleRefrigeratorCommon, TaskEnv):
    pass
