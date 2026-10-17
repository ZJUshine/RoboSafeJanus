"""heat_sandwich_microwave 的脚本专家:三明治(Clutter/sandwich 0.65 倍)长边朝里放在微波炉前方,自上而下横跨 7.1 cm 宽的一边抓,转平后长边竖着送进炉腔。流程见 heat_egg_microwave。"""

from task.RoboDojo.experts import heat_egg_microwave as microwave

SPEC = microwave.Spec("sandwich", "top")


def play_once(env):
    microwave.play_once(env, SPEC)
