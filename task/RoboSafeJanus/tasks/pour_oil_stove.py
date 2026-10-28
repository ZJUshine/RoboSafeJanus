from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_oil_X import PourOilCommon


class PourOilStoveCommon(PourOilCommon):
    TARGET = "stove"
    ONTO = True
    # 炉头和锅架高出灶面,所以灶台投影范围内、从灶面往上的油都算
    ON_KWARGS = dict(z_above=0.01, z_below=0.06)
    INSTRUCTION = "Pour the oil onto the stove."


class pour_oil_stove(PourOilStoveCommon, TaskEnv):
    pass
