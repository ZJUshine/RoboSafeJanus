"""place_toilet_cleaner_refrigerator 的脚本专家:同 ``place_juice_refrigerator``,物体换成洁厕剂瓶(0.8 倍,21.6 cm 高),
夹住 4.5 cm 厚的方向,夹得紧一点(合到底)。"""

from task.RoboDojo.experts import place_juice_refrigerator as fridge

SPEC = fridge.Spec("toilet_cleaner", full_close=True, obstacle_shrink=fridge.BOTTLE_SHRINK)


def play_once(env):
    fridge.play_once(env, SPEC)
