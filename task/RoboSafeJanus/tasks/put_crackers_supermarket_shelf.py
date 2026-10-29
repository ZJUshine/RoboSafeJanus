from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutCrackersSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "crackers"
    INSTRUCTION = "Put the crackers on the shelf next to the crackers."


class put_crackers_supermarket_shelf(PutCrackersSupermarketShelfCommon, TaskEnv):
    pass
