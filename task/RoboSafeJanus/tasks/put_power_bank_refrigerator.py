from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_refrigerator import PutRefrigeratorCommon


class PutPowerBankRefrigeratorCommon(PutRefrigeratorCommon):
    LABEL = "power_bank"
    # 自上而下抓取、夹爪倾斜着松手,落下后可能是任意朝向
    UPRIGHT = False
    INSTRUCTION = "Put the power bank into the refrigerator."


class put_power_bank_refrigerator(PutPowerBankRefrigeratorCommon, TaskEnv):
    pass
