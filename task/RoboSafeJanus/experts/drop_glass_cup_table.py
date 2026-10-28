"""drop_glass_cup_table 的脚本专家:同 ``drop_rubber_ball_table``,抓取不限部位。"""

from task.RoboDojo.experts import drop_rubber_ball_table as drop

SPEC = drop.Spec("glass_cup")


def play_once(env):
    drop.play_once(env, SPEC)
