"""place_apple_refrigerator 的脚本专家:同 ``place_juice_refrigerator``,苹果矮,自上而下抓(``grasp="top"``)。"""

from task.RoboDojo.experts import place_juice_refrigerator as fridge

SPEC = fridge.Spec("apple", grasp="top")


def play_once(env):
    fridge.play_once(env, SPEC)
