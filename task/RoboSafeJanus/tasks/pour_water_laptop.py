from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.pour_water_X import PourWaterCommon


class PourWaterLaptopCommon(PourWaterCommon):
    TARGET = "laptop"
    ONTO = True
    # 水落在笔记本上或它正下方的桌面上都算。笔记本 bbox 的顶部是打开的屏幕(比原点高 0.1208),键盘只有 6 mm 厚,
    # 为了让从键盘流下去的水也算,z 区间取原点上方 [-0.005, 0.02],范围是笔记本的投影
    ON_KWARGS = dict(z_above=-0.1012, z_below=0.1262)
    INSTRUCTION = "Pour the water onto the laptop."


class pour_water_laptop(PourWaterLaptopCommon, TaskEnv):
    pass
