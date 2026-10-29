from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutDisinfectantSpraySupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "disinfectant_spray"
    INSTRUCTION = "Put the disinfectant spray on the shelf next to the cola."


class put_disinfectant_spray_supermarket_shelf(PutDisinfectantSpraySupermarketShelfCommon, TaskEnv):
    pass
