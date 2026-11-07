"""put_bread_toaster 的脚本专家:吐司片竖插进面包架。流程见 ``put_X_toaster``。"""

from task.RoboDojo.experts import put_X_toaster as toast

SPEC = toast.Spec("bread")


def play_once(env):
    toast.play_once(env, SPEC)
