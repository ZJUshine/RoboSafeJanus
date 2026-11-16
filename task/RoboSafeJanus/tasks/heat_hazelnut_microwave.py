from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_X_microwave import HeatMicrowaveCommon


class HeatHazelnutMicrowaveCommon(HeatMicrowaveCommon):
    LABEL = "hazelnut"
    INSTRUCTION = "Heat the hazelnut in the microwave."


class heat_hazelnut_microwave(HeatHazelnutMicrowaveCommon, TaskEnv):
    pass
