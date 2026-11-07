"""put_toilet_cleaner_refrigerator 的脚本专家:侧抓洁厕液瓶身,竖着放上冰箱隔板,合到底;瓶子当障碍物时长宽方向缩小(``BOTTLE_SHRINK``)。流程见 ``put_X_refrigerator``。"""

from task.RoboDojo.experts import put_X_refrigerator as fridge

SPEC = fridge.Spec("toilet_cleaner", full_close=True, obstacle_shrink=fridge.BOTTLE_SHRINK)


def play_once(env):
    fridge.play_once(env, SPEC)
