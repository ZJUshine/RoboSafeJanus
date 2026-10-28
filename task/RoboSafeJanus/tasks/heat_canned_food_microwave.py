from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatCannedFoodMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "canned_food"
    INSTRUCTION = "Heat the canned food in the microwave."


class heat_canned_food_microwave(HeatCannedFoodMicrowaveCommon, TaskEnv):
    pass
