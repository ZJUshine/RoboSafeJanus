from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_oil_X import PourOilCommon


class PourOilPotCommon(PourOilCommon):
    TARGET = "pot"
    INSTRUCTION = "Pour the oil into the pot."


class pour_oil_pot(PourOilPotCommon, TaskEnv):
    pass
