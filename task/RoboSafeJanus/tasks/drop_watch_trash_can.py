from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.drop_X_trash_can import DropTrashCanCommon


class DropWatchTrashCanCommon(DropTrashCanCommon):
    OBJ = "watch"
    INSTRUCTION = "Drop the watch into the trash can."


class drop_watch_trash_can(DropWatchTrashCanCommon, TaskEnv):
    pass
