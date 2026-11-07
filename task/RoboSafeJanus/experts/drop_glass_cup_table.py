"""drop_glass_cup_table 的脚本专家:流程见 ``drop_X_table``,抓取不限部位。"""

from task.RoboDojo.experts import drop_X_table as drop

SPEC = drop.Spec("glass_cup")


def play_once(env):
    drop.play_once(env, SPEC)
