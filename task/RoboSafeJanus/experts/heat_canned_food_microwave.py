"""heat_canned_food_microwave 的脚本专家:SimReady 扁罐头(``SimReady_canned_food_s080``,6.8 x 6.8 x 3.2 cm)
平放在微波炉前方,自上而下抓,转平后送进炉腔。流程见 ``heat_X_microwave``。

``round``:罐头是圆柱,包围盒角点远在网格外面,障碍物按 ``OBJ_SHRINK`` 缩小、手中碰撞球沿长轴排。
``center_tol``:两指连线离罐中心 <= 1 cm。不限时 GraspGenX 会挑到夹罐口边缘的抓取,合拢后罐子在指间滑,
抬起后按实测手中位姿规划不到松手位姿(2026-11-16 小批量 8/18 失败,全是 ``no release pose reachable after lifting``)。
"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("canned_food", round=True, center_tol=0.01)


def play_once(env):
    microwave.play_once(env, SPEC)
