"""put_lighter_refrigerator 的脚本专家:流程见 ``put_X_refrigerator``,打火机小,自上而下抓(``grasp="top"``),合到底(只合到比物体窄一点时抬起就滑掉)。"""

from task.RoboDojo.experts import put_X_refrigerator as fridge

SPEC = fridge.Spec("lighter", grasp="top", full_close=True)


def play_once(env):
    fridge.play_once(env, SPEC)
