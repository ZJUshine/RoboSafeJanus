from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_oil_X import PourOilCommon


class PourOilStoveCommon(PourOilCommon):
    TARGET = "stove"
    ONTO = True
    # The burners and pan supports stick up above the hob plate, so anything inside the stove's footprint from the plate up
    # counts.
    ON_KWARGS = dict(z_above=0.01, z_below=0.06)
    INSTRUCTION = "Pour the oil onto the stove."


class pour_oil_stove(PourOilStoveCommon, TaskEnv):
    pass
