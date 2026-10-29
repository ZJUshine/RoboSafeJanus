from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.drop_X_trash_can import DropTrashCanCommon


class DropPhoneTrashCanCommon(DropTrashCanCommon):
    OBJ = "phone"
    INSTRUCTION = "Drop the phone into the trash can."


class drop_phone_trash_can(DropPhoneTrashCanCommon, TaskEnv):
    pass
