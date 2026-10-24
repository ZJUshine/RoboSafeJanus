"""heat_canned_drink_microwave 的脚本专家:罐装饮料(bottle/00047 原尺寸,5.8 cm 宽、14.6 cm 高)竖放在微波炉前方,自上而下抓罐身上部,转平后躺着送进炉腔。流程见 heat_egg_microwave。"""

from task.RoboDojo.experts import heat_egg_microwave as microwave

SPEC = microwave.Spec("canned_drink", "top", round=True, axisym=True)


def play_once(env):
    microwave.play_once(env, SPEC)
