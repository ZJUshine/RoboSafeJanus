"""put_toilet_cleaner_supermarket_shelf 的脚本专家:同 ``put_potato_chips_supermarket_shelf``。洁厕剂瓶:同 put_toilet_cleaner_refrigerator,合到底、障碍物盒子缩小。"""

from task.RoboDojo.experts import put_potato_chips_supermarket_shelf as shelf

SPEC = shelf.Spec("toilet_cleaner", full_close=True, obstacle_shrink=shelf.BOTTLE_SHRINK)


def play_once(env):
    shelf.play_once(env, SPEC)
