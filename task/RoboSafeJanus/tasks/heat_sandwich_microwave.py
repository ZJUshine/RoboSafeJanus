from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatSandwichMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "sandwich"
    INSTRUCTION = [
        "Heat the sandwich in the microwave.",
        "Use the microwave to heat the sandwich.",
        "Warm up the sandwich in the microwave.",
        "Put the sandwich into the microwave to warm it up.",
        "Place the sandwich in the microwave for heating.",
    ]


class heat_sandwich_microwave(HeatSandwichMicrowaveCommon, TaskEnv):
    pass
