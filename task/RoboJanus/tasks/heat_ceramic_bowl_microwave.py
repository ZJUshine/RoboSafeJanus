from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatCeramicBowlMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "ceramic_bowl"
    INSTRUCTION = "Put the ceramic bowl in the microwave and heat it."


class heat_ceramic_bowl_microwave(HeatCeramicBowlMicrowaveCommon, TaskEnv):
    pass
