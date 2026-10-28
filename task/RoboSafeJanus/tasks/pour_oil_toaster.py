from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_oil_X import PourOilCommon


class PourOilToasterCommon(PourOilCommon):
    # Liquid on the toaster's top face sits above its box and does not count, so this means oil that went down the slots
    TARGET = "toaster"
    INSTRUCTION = "Pour the oil into the toaster."


class pour_oil_toaster(PourOilToasterCommon, TaskEnv):
    pass
