from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutColaCleaningSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "cola"
    INSTRUCTION = "Put the cola on the shelf next to the shampoo."


class put_cola_cleaning_supermarket_shelf(PutColaCleaningSupermarketShelfCommon, TaskEnv):
    pass
