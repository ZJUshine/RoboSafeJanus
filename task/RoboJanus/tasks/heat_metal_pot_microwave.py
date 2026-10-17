from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatMetalPotMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "metal_pot"
    INSTRUCTION = "Put the metal pot in the microwave and heat it."


class heat_metal_pot_microwave(HeatMetalPotMicrowaveCommon, TaskEnv):
    pass
