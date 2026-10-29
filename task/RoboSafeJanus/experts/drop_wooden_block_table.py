"""drop_wooden_block_table 的脚本专家:同 ``drop_rubber_ball_table``,木块是 6 cm 的立方体,障碍物用缩小的包围盒。"""

from task.RoboDojo.experts import drop_rubber_ball_table as drop

SPEC = drop.Spec("wooden_block")


def play_once(env):
    drop.play_once(env, SPEC)
