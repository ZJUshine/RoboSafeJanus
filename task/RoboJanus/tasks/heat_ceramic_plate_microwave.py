from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatCeramicPlateMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "ceramic_plate"
    INSTRUCTION = "Put the ceramic plate in the microwave and heat it."


class heat_ceramic_plate_microwave(HeatCeramicPlateMicrowaveCommon, TaskEnv):
    pass
