"""heat_bread_microwave 的脚本专家:面包卷(RoboTwin_075_bread/00004,9.5 x 5.0 x 4.0 cm)长边朝里放在微波炉前方,自上而下横跨 5.0 cm 宽的一边抓,转平后长边竖着送进炉腔。流程见 heat_egg_microwave。"""

from task.RoboDojo.experts import heat_egg_microwave as microwave

SPEC = microwave.Spec("bread", "top", center_slide=True)


def play_once(env):
    microwave.play_once(env, SPEC)
