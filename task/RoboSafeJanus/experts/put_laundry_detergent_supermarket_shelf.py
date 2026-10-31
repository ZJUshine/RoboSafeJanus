"""put_laundry_detergent_supermarket_shelf 的脚本专家:同 ``put_potato_chips_supermarket_shelf``。洗衣液袋(0.85 倍,18.5 cm 高)
重,合到底;袋子上窄下宽,障碍物盒子缩小。"""

from task.RoboDojo.experts import put_potato_chips_supermarket_shelf as shelf

SPEC = shelf.Spec("laundry_detergent", full_close=True, obstacle_shrink=shelf.BOTTLE_SHRINK)


def play_once(env):
    shelf.play_once(env, SPEC)
