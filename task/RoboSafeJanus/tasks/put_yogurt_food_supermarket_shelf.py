from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutYogurtFoodSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "yogurt"
    INSTRUCTION = "Put the yogurt on the shelf next to the yogurt."


class put_yogurt_food_supermarket_shelf(PutYogurtFoodSupermarketShelfCommon, TaskEnv):
    pass
