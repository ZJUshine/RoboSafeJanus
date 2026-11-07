"""put_milk_food_supermarket_shelf 的脚本专家:牛奶放到货架上食品陈列旁边。流程见 ``put_X_supermarket_shelf``。"""

from task.RoboDojo.experts import put_X_supermarket_shelf as shelf

SPEC = shelf.Spec("milk", tcp_z=(0.05, 0.09))


def play_once(env):
    shelf.play_once(env, SPEC)
