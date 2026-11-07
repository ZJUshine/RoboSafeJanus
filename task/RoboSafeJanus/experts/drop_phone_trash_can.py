"""drop_phone_trash_can 的脚本专家:流程见 ``drop_X_trash_can``。"""

from task.RoboDojo.experts import drop_X_trash_can as drop

SPEC = drop.Spec("phone")


def play_once(env):
    drop.play_once(env, SPEC)
