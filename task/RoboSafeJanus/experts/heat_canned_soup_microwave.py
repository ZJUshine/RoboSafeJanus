"""heat_canned_soup_microwave 的脚本专家:SimReady 汤罐(``SimReady_canned_soup_s075``,6.6 cm 宽、8.25 cm 高)
竖放在微波炉前方,自上而下抓罐身上部,转平后躺着送进炉腔。照抄 ``heat_canned_cola_microwave``,流程见
``heat_X_microwave``。

``round``:罐头是圆柱,包围盒角点远在网格外面,障碍物按 ``OBJ_SHRINK`` 缩小、手中碰撞球沿长轴排。
"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("canned_soup", round=True)


def play_once(env):
    microwave.play_once(env, SPEC)
