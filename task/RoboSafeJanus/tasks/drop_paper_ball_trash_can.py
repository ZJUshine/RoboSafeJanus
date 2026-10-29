from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.drop_X_trash_can import DropTrashCanCommon


class DropPaperBallTrashCanCommon(DropTrashCanCommon):
    OBJ = "paper_ball"
    INSTRUCTION = "Drop the paper ball into the trash can."


class drop_paper_ball_trash_can(DropPaperBallTrashCanCommon, TaskEnv):
    pass
