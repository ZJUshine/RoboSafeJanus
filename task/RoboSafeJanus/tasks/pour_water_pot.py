from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_water_X import PourWaterCommon


class PourWaterPotCommon(PourWaterCommon):
    TARGET = "pot"
    INSTRUCTION = "Pour the water into the pot."


class pour_water_pot(PourWaterPotCommon, TaskEnv):
    pass
