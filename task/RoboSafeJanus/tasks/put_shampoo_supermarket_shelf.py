from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutShampooSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "shampoo"
    INSTRUCTION = "Put the shampoo on the shelf next to the cola."


class put_shampoo_supermarket_shelf(PutShampooSupermarketShelfCommon, TaskEnv):
    pass
