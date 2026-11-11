"""heat_hamburger_microwave 的脚本专家:汉堡(SimReady hamburger 0.85 倍,6.8 x 6.8 x 4.7 cm)放在微波炉前方,
自上而下抓,转平后送进炉腔。流程见 ``heat_X_microwave``。"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("hamburger")


def play_once(env):
    microwave.play_once(env, SPEC)
