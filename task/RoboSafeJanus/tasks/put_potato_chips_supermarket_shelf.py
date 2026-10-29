from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutPotatoChipsSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "potato_chips"
    INSTRUCTION = "Put the potato chips on the shelf next to the potato chips."


class put_potato_chips_supermarket_shelf(PutPotatoChipsSupermarketShelfCommon, TaskEnv):
    pass
