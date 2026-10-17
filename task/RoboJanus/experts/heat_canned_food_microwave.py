"""heat_canned_food_microwave 的脚本专家:扁罐头(Clutter/can 0.75 倍)自上而下抓,转平后送进炉腔;GraspGenX 对这么扁的罐头只给夹上下底面的侧抓。流程见 heat_egg_microwave。"""

from task.RoboDojo.experts import heat_egg_microwave as microwave

SPEC = microwave.Spec("canned_food", "top", round=True)


def play_once(env):
    microwave.play_once(env, SPEC)
