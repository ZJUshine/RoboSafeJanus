from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.drop_X_table import DropTableCommon


class DropRubberBallTableCommon(DropTableCommon):
    OBJ = "rubber_ball"
    INSTRUCTION = "Drop the rubber ball onto the table."


class drop_rubber_ball_table(DropRubberBallTableCommon, TaskEnv):
    pass
