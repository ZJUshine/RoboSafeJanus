"""drop_rubber_ball_table 的脚本专家:把橡胶球举高后松手,落回桌面。流程见 ``drop_X_table``。"""

from task.RoboDojo.experts import drop_X_table as drop

SPEC = drop.Spec("rubber_ball", round=True)


def play_once(env):
    drop.play_once(env, SPEC)
