"""drop_wine_bottle_table 的脚本专家:同 ``drop_rubber_ball_table``,夹住瓶颈(物体坐标系 z 0.075 ~ 0.145 m,宽约 3 cm)。
障碍物按瓶身 / 瓶肩 / 瓶颈分段(截面量自资产网格)。"""

from task.RoboDojo.experts import drop_rubber_ball_table as drop

SPEC = drop.Spec(
    "wine_bottle",
    obj_z=(0.075, 0.145),
    prefer_pitch=20.0,
    profile=((-0.06, 0.05, 0.083, 0.083), (0.05, 0.068, 0.062, 0.062), (0.068, 0.159, 0.032, 0.032)),
)


def play_once(env):
    drop.play_once(env, SPEC)
