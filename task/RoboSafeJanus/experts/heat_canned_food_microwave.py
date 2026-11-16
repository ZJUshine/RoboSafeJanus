"""heat_canned_food_microwave 的脚本专家:SimReady 扁罐头(``SimReady_canned_food_s080``,6.8 x 6.8 x 3.2 cm)
平放在微波炉前方,自上而下抓,转平后送进炉腔。流程见 ``heat_X_microwave``。

``round``:罐头是圆柱,包围盒角点远在网格外面,障碍物按 ``OBJ_SHRINK`` 缩小、手中碰撞球沿长轴排。
"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("canned_food", round=True)


def play_once(env):
    microwave.play_once(env, SPEC)
