"""drop_empty_plastic_bottle_trash_can 的脚本专家:同 ``drop_paper_ball_trash_can``。"""

from task.RoboDojo.experts import drop_paper_ball_trash_can as drop

SPEC = drop.Spec("empty_plastic_bottle", round=True)


def play_once(env):
    drop.play_once(env, SPEC)
