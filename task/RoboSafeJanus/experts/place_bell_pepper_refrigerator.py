"""place_bell_pepper_refrigerator 的脚本专家:同 ``place_juice_refrigerator``,彩椒矮,自上而下抓(``grasp="top"``)。"""

from task.RoboDojo.experts import place_juice_refrigerator as fridge

SPEC = fridge.Spec("bell_pepper", grasp="top", squeeze=0.012, carry_slow=3.0)  # 圆滑的彩椒搬得快时在指间转、滑掉


def play_once(env):
    fridge.play_once(env, SPEC)
