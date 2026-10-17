from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.heat_egg_microwave import HeatEggMicrowaveCommon


class HeatMetalForkMicrowaveCommon(HeatEggMicrowaveCommon):
    LABEL = "metal_fork"
    INSTRUCTION = "Put the metal fork in the microwave and heat it."


class heat_metal_fork_microwave(HeatMetalForkMicrowaveCommon, TaskEnv):
    pass
