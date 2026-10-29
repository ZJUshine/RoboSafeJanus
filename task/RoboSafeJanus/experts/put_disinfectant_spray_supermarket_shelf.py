"""put_disinfectant_spray_supermarket_shelf 的脚本专家:同 ``put_potato_chips_supermarket_shelf``。喷雾瓶细、顶上有喷头,抓在瓶身上。"""

from task.RoboDojo.experts import put_potato_chips_supermarket_shelf as shelf

SPEC = shelf.Spec("disinfectant_spray", tcp_z=(0.06, 0.12), obstacle_shrink=shelf.BOTTLE_SHRINK)


def play_once(env):
    shelf.play_once(env, SPEC)
