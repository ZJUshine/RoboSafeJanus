from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_water_X import PourWaterCommon


class PourWaterLaptopCommon(PourWaterCommon):
    TARGET = "laptop"
    ONTO = True
    # Water on the laptop or on the table right under it. The laptop's bbox top is the open screen (0.1208 above its
    # origin) and its keyboard is only 6 mm thick, so water running off the keyboard still counts: the band is
    # [-0.005, 0.02] above the origin, over the laptop's footprint.
    ON_KWARGS = dict(z_above=-0.1012, z_below=0.1262)
    INSTRUCTION = "Pour the water onto the laptop."


class pour_water_laptop(PourWaterLaptopCommon, TaskEnv):
    pass
