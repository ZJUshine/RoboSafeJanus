"""heat_frozen_hamburger_microwave 的脚本专家:冷冻汉堡盒(GenieSim_frozen_food 0.7 倍,11.9 x 4.9 x 6.9 cm)长边朝里放在微波炉前方,自上而下横跨 4.9 cm 宽的一边抓,转平后长边竖着送进炉腔。合拢时盒子在指间挪 1.3 ~ 1.4 cm,同三明治放宽 reuse_tol。流程见 heat_egg_microwave。"""

from task.RoboDojo.experts import heat_egg_microwave as microwave

SPEC = microwave.Spec("frozen_hamburger", "top", center_slide=True, reuse_tol=0.02)


def play_once(env):
    microwave.play_once(env, SPEC)
