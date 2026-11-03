from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutMilkCleaningSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "milk"
    INSTRUCTION = "Put the milk on the shelf next to the laundry detergent."


class put_milk_cleaning_supermarket_shelf(PutMilkCleaningSupermarketShelfCommon, TaskEnv):
    pass
