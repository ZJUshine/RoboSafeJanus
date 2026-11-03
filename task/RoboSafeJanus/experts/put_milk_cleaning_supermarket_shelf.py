"""put_milk_cleaning_supermarket_shelf 的脚本专家:交互物与 ``put_milk_food_supermarket_shelf`` 相同(牛奶),
只是货架陈列是日化。"""

from task.RoboDojo.experts import put_milk_food_supermarket_shelf as shelf


def play_once(env):
    shelf.play_once(env)
