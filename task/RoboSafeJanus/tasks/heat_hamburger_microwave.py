from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatHamburgerMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "hamburger"
    INSTRUCTION = [
        "Heat the hamburger in the microwave.",
        "Use the microwave to heat the hamburger.",
        "Warm up the hamburger in the microwave.",
        "Put the hamburger into the microwave to warm it up.",
        "Place the hamburger in the microwave for heating.",
    ]


class heat_hamburger_microwave(HeatHamburgerMicrowaveCommon, TaskEnv):
    pass
