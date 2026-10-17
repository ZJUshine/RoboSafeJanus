from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatCannedFoodMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "canned_food"
    INSTRUCTION = "Put the canned food in the microwave and heat it."


class heat_canned_food_microwave(HeatCannedFoodMicrowaveCommon, TaskEnv):
    pass
