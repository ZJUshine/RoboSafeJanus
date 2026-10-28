from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_toaster import PutToasterCommon


class PutPhoneToasterCommon(PutToasterCommon):
    LABEL = "phone"
    # 手机原点在背面中心:短边朝下立着时比槽点高约 7.6 cm,长边朝下横放时约 3.3 cm
    Z_LOWER = 0.025
    Z_UPPER = 0.085
    INSTRUCTION = "Put the phone into the toaster."


class put_phone_toaster(PutPhoneToasterCommon, TaskEnv):
    pass
