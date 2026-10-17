from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatCeramicCupMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "ceramic_cup"
    INSTRUCTION = "Put the ceramic cup in the microwave and heat it."


class heat_ceramic_cup_microwave(HeatCeramicCupMicrowaveCommon, TaskEnv):
    pass
