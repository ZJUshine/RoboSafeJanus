"""put_milk_supermarket_shelf 的脚本专家:同 ``put_potato_chips_supermarket_shelf``。牛奶盒高 12 cm,TCP 放低到盒身中部。"""

from task.RoboDojo.experts import put_potato_chips_supermarket_shelf as shelf

SPEC = shelf.Spec("milk", tcp_z=(0.05, 0.09))


def play_once(env):
    shelf.play_once(env, SPEC)
