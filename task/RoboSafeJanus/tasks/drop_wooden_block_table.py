from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.drop_X_table import DropTableCommon


class DropWoodenBlockTableCommon(DropTableCommon):
    OBJ = "wooden_block"
    INSTRUCTION = "Drop the wooden block onto the table."


class drop_wooden_block_table(DropWoodenBlockTableCommon, TaskEnv):
    pass
