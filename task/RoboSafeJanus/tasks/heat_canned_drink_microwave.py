from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatCannedDrinkMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "canned_drink"
    INSTRUCTION = "Heat the canned drink in the microwave."


class heat_canned_drink_microwave(HeatCannedDrinkMicrowaveCommon, TaskEnv):
    pass
