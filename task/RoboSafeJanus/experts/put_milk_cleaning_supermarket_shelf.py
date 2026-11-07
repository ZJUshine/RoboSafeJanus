"""put_milk_cleaning_supermarket_shelf 的脚本专家:交互物与 ``put_milk_food_supermarket_shelf`` 相同(milk 的 ``Spec`` 照抄),只是货架陈列是日化。流程见 ``put_X_supermarket_shelf``。"""

from task.RoboDojo.experts import put_X_supermarket_shelf as shelf

SPEC = shelf.Spec("milk", tcp_z=(0.05, 0.09))


def play_once(env):
    shelf.play_once(env, SPEC)
