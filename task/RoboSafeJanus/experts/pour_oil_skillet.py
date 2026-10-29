"""pour_oil_skillet 的脚本专家:流程同 ``pour_oil_pot``,只是倒进平底锅里。

平底锅很浅(锅沿 3.3 cm)、锅口很宽:瓶口在锅口中心靠瓶子一侧、锅沿上方 ``MOUTH_ABOVE`` 处。锅的包围盒含两个
把手(总长 0.266 m),``Target`` 的锅口半径取包围盒较窄方向的一半(乘 ``OPENING_SCALE``)。锅放进 ``boxes``:
倾倒时夹爪的导轨一端会比瓶口低,放进 ``boxes`` 后瓶口按夹爪 / 瓶子高出锅沿的需要抬高。
"""

from task.RoboDojo.experts import oil_pour, pour_liquid_into_cup as cup


class _Skillet(cup.Target):
    LABEL = "skillet"

    def __init__(self, env):
        super().__init__(env)
        self.boxes = [cup.footprint(env, self.LABEL)]


def play_once(env):
    return oil_pour.play_once(env, _Skillet)
