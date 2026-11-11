"""heat_sandwich_microwave 的脚本专家:三明治(SimReady sandwich 0.60 倍,7.4 x 6.9 x 4.4 cm)长边朝里放在
微波炉前方,自上而下抓,转平后送进炉腔。流程见 ``heat_X_microwave``。

关键是抓住后先在离炉口几十厘米的空中把腕部转成伸入姿态(``heat_X_microwave`` 的 ``prepare``):物体是摩擦夹持,腕部一转
它就往下荡十几厘米,和伸入叠在一起时会边进边沉、把微波炉顶走。
"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("sandwich")


def play_once(env):
    microwave.play_once(env, SPEC)
