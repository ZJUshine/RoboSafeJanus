"""heat_canned_food_microwave 的脚本专家:扁罐头(Clutter/can 0.75 倍,7.4 cm 宽、3.1 cm 高)自上而下抓,转平后送进炉腔。流程见 ``heat_X_microwave``。"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("canned_food", round=True)


def play_once(env):
    microwave.play_once(env, SPEC)
