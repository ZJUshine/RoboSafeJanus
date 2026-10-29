from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.drop_X_trash_can import DropTrashCanCommon


class DropEmptyCanTrashCanCommon(DropTrashCanCommon):
    OBJ = "empty_can"
    INSTRUCTION = "Drop the empty can into the trash can."


class drop_empty_can_trash_can(DropEmptyCanTrashCanCommon, TaskEnv):
    pass
