"""place_lighter_refrigerator 的脚本专家:同 ``place_juice_refrigerator``,打火机小,自上而下抓(``grasp="top"``),合到底(只合到比物体窄一点时抬起就滑掉)。"""

from task.RoboDojo.experts import place_juice_refrigerator as fridge

SPEC = fridge.Spec("lighter", grasp="top", full_close=True)


def play_once(env):
    fridge.play_once(env, SPEC)
