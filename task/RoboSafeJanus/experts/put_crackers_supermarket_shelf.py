"""put_crackers_supermarket_shelf 的脚本专家:同 ``put_potato_chips_supermarket_shelf``。饼干盒高 13.9 cm,TCP 放低。"""

from task.RoboDojo.experts import put_potato_chips_supermarket_shelf as shelf

SPEC = shelf.Spec("crackers", tcp_z=(0.06, 0.11))


def play_once(env):
    shelf.play_once(env, SPEC)
