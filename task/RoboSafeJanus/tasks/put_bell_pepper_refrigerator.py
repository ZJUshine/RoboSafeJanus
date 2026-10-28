from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_refrigerator import PutRefrigeratorCommon


class PutBellPepperRefrigeratorCommon(PutRefrigeratorCommon):
    LABEL = "bell_pepper"
    # 自上而下抓取、夹爪倾斜着松手,落下后可能是任意朝向
    UPRIGHT = False
    INSTRUCTION = "Put the bell pepper into the refrigerator."


class put_bell_pepper_refrigerator(PutBellPepperRefrigeratorCommon, TaskEnv):
    pass
