"""put_hand_soap_supermarket_shelf 的脚本专家:同 ``put_potato_chips_supermarket_shelf``。洗手液瓶高 23.8 cm、重,合到底;障碍物盒子缩小。"""

from task.RoboDojo.experts import put_potato_chips_supermarket_shelf as shelf

SPEC = shelf.Spec("hand_soap", full_close=True, obstacle_shrink=shelf.BOTTLE_SHRINK)


def play_once(env):
    shelf.play_once(env, SPEC)
