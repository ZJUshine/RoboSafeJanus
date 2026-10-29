from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutColaSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "cola"
    INSTRUCTION = "Put the cola on the shelf next to the cola."


class put_cola_supermarket_shelf(PutColaSupermarketShelfCommon, TaskEnv):
    pass
