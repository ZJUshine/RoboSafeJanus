from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatCannedSoupMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "canned_soup"
    INSTRUCTION = "Heat the canned soup in the microwave."


class heat_canned_soup_microwave(HeatCannedSoupMicrowaveCommon, TaskEnv):
    pass
