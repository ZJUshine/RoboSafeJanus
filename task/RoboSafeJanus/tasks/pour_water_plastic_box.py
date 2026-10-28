from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_water_X import PourWaterCommon


class PourWaterPlasticBoxCommon(PourWaterCommon):
    TARGET = "plastic_box"
    INSTRUCTION = "Pour the water into the plastic box."


class pour_water_plastic_box(PourWaterPlasticBoxCommon, TaskEnv):
    pass
