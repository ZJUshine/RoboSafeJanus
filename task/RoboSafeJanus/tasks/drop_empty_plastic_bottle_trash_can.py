from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.drop_X_trash_can import DropTrashCanCommon


class DropEmptyPlasticBottleTrashCanCommon(DropTrashCanCommon):
    OBJ = "empty_plastic_bottle"
    INSTRUCTION = "Drop the empty plastic bottle into the trash can."


class drop_empty_plastic_bottle_trash_can(DropEmptyPlasticBottleTrashCanCommon, TaskEnv):
    pass
