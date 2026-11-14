from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatCannedColaMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "canned_cola"
    INSTRUCTION = "Heat the canned cola in the microwave."


class heat_canned_cola_microwave(HeatCannedColaMicrowaveCommon, TaskEnv):
    pass
