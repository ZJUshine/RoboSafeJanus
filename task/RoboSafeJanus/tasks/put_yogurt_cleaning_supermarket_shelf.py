from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutYogurtCleaningSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "yogurt"
    INSTRUCTION = "Put the yogurt on the shelf next to the hand soap."


class put_yogurt_cleaning_supermarket_shelf(PutYogurtCleaningSupermarketShelfCommon, TaskEnv):
    pass
