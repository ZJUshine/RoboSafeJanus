from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_water_X import PourWaterCommon


class PourWaterBookCommon(PourWaterCommon):
    TARGET = "book"
    ONTO = True
    INSTRUCTION = "Pour the water onto the book."


class pour_water_book(PourWaterBookCommon, TaskEnv):
    pass
