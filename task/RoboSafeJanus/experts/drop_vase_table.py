"""drop_vase_table 的脚本专家:同 ``drop_rubber_ball_table``,夹住瓶颈(物体坐标系 z 0.025 ~ 0.055 m,截面 5.8 ~ 6.5 x 4.4 ~ 4.9 cm)。
障碍物按底部 / 瓶身 / 瓶肩 / 瓶颈 / 瓶口分段(截面量自资产网格,椭圆:x 宽、y 窄)。"""

from task.RoboDojo.experts import drop_rubber_ball_table as drop

SPEC = drop.Spec(
    "vase",
    obj_z=(0.025, 0.055),
    profile=(
        (-0.074, -0.052, 0.091, 0.073),
        (-0.052, 0.0, 0.101, 0.078),
        (0.0, 0.022, 0.087, 0.062),
        (0.022, 0.059, 0.068, 0.051),
        (0.059, 0.074, 0.081, 0.063),
    ),
)


def play_once(env):
    drop.play_once(env, SPEC)
