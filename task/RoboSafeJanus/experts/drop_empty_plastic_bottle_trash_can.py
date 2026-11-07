"""drop_empty_plastic_bottle_trash_can 的脚本专家:流程见 ``drop_X_trash_can``。"""

from task.RoboDojo.experts import drop_X_trash_can as drop

SPEC = drop.Spec("empty_plastic_bottle", round=True)


def play_once(env):
    drop.play_once(env, SPEC)
