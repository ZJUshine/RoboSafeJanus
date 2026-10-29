from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_oil_X import PourOilCommon


class PourOilSkilletCommon(PourOilCommon):
    TARGET = "skillet"
    INSTRUCTION = "Pour the oil into the skillet."


class pour_oil_skillet(PourOilSkilletCommon, TaskEnv):
    pass
