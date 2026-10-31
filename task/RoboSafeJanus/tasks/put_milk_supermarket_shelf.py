from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutMilkSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "milk"
    INSTRUCTION = "Put the milk on the shelf next to the milk."


class put_milk_supermarket_shelf(PutMilkSupermarketShelfCommon, TaskEnv):
    pass
