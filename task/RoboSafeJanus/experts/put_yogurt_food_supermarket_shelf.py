"""put_yogurt_food_supermarket_shelf 的脚本专家:同 ``put_milk_food_supermarket_shelf``。发酵乳瓶高 14.4 cm、
宽 7.66 cm(略高于默认 7.5 cm 夹持上限,单独放宽到 7.8 cm),瓶身圆、比包围盒窄,合到底(smoke 里只合到 7.06 cm 时抓起就滑掉)。"""

from task.RoboDojo.experts import put_milk_food_supermarket_shelf as shelf

SPEC = shelf.Spec("yogurt", finger_max=0.078, full_close=True)


def play_once(env):
    shelf.play_once(env, SPEC)
