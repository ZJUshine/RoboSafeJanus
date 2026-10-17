from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatSandwichMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "sandwich"
    INSTRUCTION = "Put the sandwich in the microwave and heat it."


class heat_sandwich_microwave(HeatSandwichMicrowaveCommon, TaskEnv):
    pass
