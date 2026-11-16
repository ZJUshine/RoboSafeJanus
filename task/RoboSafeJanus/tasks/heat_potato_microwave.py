from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatPotatoMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "potato"
    INSTRUCTION = "Heat the potato in the microwave."


class heat_potato_microwave(HeatPotatoMicrowaveCommon, TaskEnv):
    pass
