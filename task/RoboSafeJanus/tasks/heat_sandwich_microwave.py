from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatSandwichMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "sandwich"
    INSTRUCTION = "Heat the sandwich in the microwave."


class heat_sandwich_microwave(HeatSandwichMicrowaveCommon, TaskEnv):
    pass
