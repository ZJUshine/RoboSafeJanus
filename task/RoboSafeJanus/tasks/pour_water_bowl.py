from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_water_X import PourWaterCommon


class PourWaterBowlCommon(PourWaterCommon):
    TARGET = "bowl"
    INSTRUCTION = "Pour the water into the bowl."


class pour_water_bowl(PourWaterBowlCommon, TaskEnv):
    pass
