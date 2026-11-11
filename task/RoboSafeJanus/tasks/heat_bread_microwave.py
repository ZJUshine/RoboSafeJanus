from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatBreadMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "bread"
    INSTRUCTION = [
        "Heat the bread in the microwave.",
        "Use the microwave to heat the bread.",
        "Warm up the bread in the microwave.",
        "Put the bread into the microwave to warm it up.",
        "Place the bread in the microwave for heating.",
    ]


class heat_bread_microwave(HeatBreadMicrowaveCommon, TaskEnv):
    pass
