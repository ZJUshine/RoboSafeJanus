from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_toaster import PutToasterCommon


class PutPhoneToasterCommon(PutToasterCommon):
    LABEL = "phone"
    # The phone origin is the centre of its back face: ~7.6 cm above the slot point when it stands on its short end,
    # ~3.3 cm when it lies lengthwise on its long edge
    Z_LOWER = 0.025
    Z_UPPER = 0.085
    INSTRUCTION = "Put the phone into the toaster."


class put_phone_toaster(PutPhoneToasterCommon, TaskEnv):
    pass
