"""place_sandwich_refrigerator 的脚本专家:同 ``place_juice_refrigerator``,三明治矮,自上而下抓(``grasp="top"``);
同 heat_sandwich_microwave 合到底(只合到比物体窄一点时,抬起就从指间滑掉)。"""

from task.RoboDojo.experts import place_juice_refrigerator as fridge

SPEC = fridge.Spec("sandwich", grasp="top", full_close=True)


def play_once(env):
    fridge.play_once(env, SPEC)
