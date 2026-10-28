from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatFrozenHamburgerMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "frozen_hamburger"
    INSTRUCTION = "Heat the frozen hamburger in the microwave."


class heat_frozen_hamburger_microwave(HeatFrozenHamburgerMicrowaveCommon, TaskEnv):
    pass
