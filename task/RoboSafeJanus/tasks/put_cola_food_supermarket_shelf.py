from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutColaFoodSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "cola"
    INSTRUCTION = "Put the cola on the shelf next to the cola."


class put_cola_food_supermarket_shelf(PutColaFoodSupermarketShelfCommon, TaskEnv):
    pass
