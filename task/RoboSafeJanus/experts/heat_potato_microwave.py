"""heat_potato_microwave 的脚本专家:土豆(SimReady potato 0.95 倍,6.83 x 9.41 x 6.80 cm)平放在微波炉前方,
长轴朝里;GraspGenX 横跨 6.8 cm 短径自上而下抓,转腕后长轴朝里送进炉腔。

``round``:土豆是圆润的椭球,包围盒角点远在网格外面,障碍物按 ``OBJ_SHRINK`` 缩小、手中碰撞球沿长轴排;
和鸡蛋一样,夹偏了合拢时会被挤出指间。流程见 ``heat_X_microwave``。
"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("potato", round=True)


def play_once(env):
    microwave.play_once(env, SPEC)
