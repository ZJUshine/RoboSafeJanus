from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatEggMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "egg"
    INSTRUCTION = "Heat the egg in the microwave."


class heat_egg_microwave(HeatEggMicrowaveCommon, TaskEnv):
    pass
