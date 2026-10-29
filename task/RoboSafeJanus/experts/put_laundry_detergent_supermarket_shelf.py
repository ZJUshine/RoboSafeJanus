"""put_laundry_detergent_supermarket_shelf 的脚本专家:同 ``put_potato_chips_supermarket_shelf``。洗衣粉盒重(长边 18.6 cm 朝货架里),合到底。"""

from task.RoboDojo.experts import put_potato_chips_supermarket_shelf as shelf

SPEC = shelf.Spec("laundry_detergent", full_close=True)


def play_once(env):
    shelf.play_once(env, SPEC)
