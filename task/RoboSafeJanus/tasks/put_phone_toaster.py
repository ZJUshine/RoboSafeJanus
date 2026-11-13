from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_toaster import PutToasterCommon


class PutPhoneToasterCommon(PutToasterCommon):
    LABEL = "phone"
    # SimReady 手机(长 15.74 cm,比槽长 15.6 cm 还长,只能竖插)原点在背面中心:竖插到底时比槽点高约 8.1 cm,
    # 搭在面包机顶上时约 13 cm
    Z_LOWER = 0.06
    Z_UPPER = 0.095
    INSTRUCTION = "Put the phone into the toaster."


class put_phone_toaster(PutPhoneToasterCommon, TaskEnv):
    pass
