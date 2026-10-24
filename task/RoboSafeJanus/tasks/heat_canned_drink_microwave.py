from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatCannedDrinkMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "canned_drink"
    INSTRUCTION = "Put the canned drink in the microwave and heat it."


class heat_canned_drink_microwave(HeatCannedDrinkMicrowaveCommon, TaskEnv):
    pass
