"""put_apple_refrigerator 的脚本专家:流程见 ``put_X_refrigerator``,苹果矮,自上而下抓(``grasp="top"``)。"""

from task.RoboDojo.experts import put_X_refrigerator as fridge

SPEC = fridge.Spec("apple", grasp="top")


def play_once(env):
    fridge.play_once(env, SPEC)
