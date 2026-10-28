from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatBreadMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "bread"
    INSTRUCTION = "Heat the bread in the microwave."


class heat_bread_microwave(HeatBreadMicrowaveCommon, TaskEnv):
    pass
