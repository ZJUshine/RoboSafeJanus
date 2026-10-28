from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_oil_X import PourOilCommon


class PourOilToasterCommon(PourOilCommon):
    # 烤面包机顶面上的液体在其 bbox 之上,不算,所以这里指的是倒进槽里的油
    TARGET = "toaster"
    INSTRUCTION = "Pour the oil into the toaster."


class pour_oil_toaster(PourOilToasterCommon, TaskEnv):
    pass
