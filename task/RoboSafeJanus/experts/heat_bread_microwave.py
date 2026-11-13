"""heat_bread_microwave 的脚本专家:``Rigid/SimReady_bread_s070``(SimReady 酸面包 x0.70,10.8 x 6.9 x 5.7 cm)。长边朝里放在微波炉前方,自上而下横跨 6.9 cm 短边抓,转平后长轴竖着送进炉腔(炉腔高 25.5 cm)。流程见 ``heat_X_microwave``。"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("bread")


def play_once(env):
    microwave.play_once(env, SPEC)
