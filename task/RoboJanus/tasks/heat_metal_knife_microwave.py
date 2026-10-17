from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatMetalKnifeMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "metal_knife"
    INSTRUCTION = "Put the metal knife in the microwave and heat it."


class heat_metal_knife_microwave(HeatMetalKnifeMicrowaveCommon, TaskEnv):
    pass
