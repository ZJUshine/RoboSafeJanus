"""put_shampoo_supermarket_shelf 的脚本专家:同 ``put_potato_chips_supermarket_shelf``。Rejoice 洗发水(0.85 倍)
顶上有按压泵,抓在瓶身上;圆角瓶身比包围盒窄,只合到比包围盒窄 6 mm 时夹不住(smoke seed 100000 ~ 100004),
合到底;障碍物盒子缩小。"""

from task.RoboDojo.experts import put_potato_chips_supermarket_shelf as shelf

SPEC = shelf.Spec("shampoo", tcp_z=(0.06, 0.12), full_close=True, obstacle_shrink=shelf.BOTTLE_SHRINK)


def play_once(env):
    shelf.play_once(env, SPEC)
