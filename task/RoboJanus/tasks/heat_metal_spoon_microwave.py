from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatMetalSpoonMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "metal_spoon"
    INSTRUCTION = "Put the metal spoon in the microwave and heat it."


class heat_metal_spoon_microwave(HeatMetalSpoonMicrowaveCommon, TaskEnv):
    pass
