"""drop_empty_can_trash_can 的脚本专家:同 ``drop_paper_ball_trash_can``。压扁的可乐罐斜躺在资产坐标系里(轴线绕 z 转 45 度),
包围盒 11.0 x 10.9 cm,罐身只有 11.0 x 7.0 x 6.0 cm(0.8 倍变体;网格在 xy 平面的最小外接矩形),障碍物 / 手中碰撞球按罐身的盒子取。"""

from task.RoboDojo.experts import drop_paper_ball_trash_can as drop

SPEC = drop.Spec("empty_can", box=((-0.0012, -0.002, 0.0), (0.1101, 0.0696, 0.0598), 45.0))


def play_once(env):
    drop.play_once(env, SPEC)
