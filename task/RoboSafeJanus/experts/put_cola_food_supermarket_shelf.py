"""put_cola_food_supermarket_shelf 的脚本专家:流程见 ``put_X_supermarket_shelf``。可乐罐矮(11.5 cm),TCP 放低到罐身中部。"""

from task.RoboDojo.experts import put_X_supermarket_shelf as shelf

SPEC = shelf.Spec("cola", tcp_z=(0.05, 0.09))


def play_once(env):
    shelf.play_once(env, SPEC)
