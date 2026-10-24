"""place_ceramic_bowl_storage_rack_edge 的脚本专家:同 ``place_ceramic_bowl_storage_rack_center``,碗放在顶层隔板前沿,
碗沿伸出前沿外 2.5 ~ 3.5 cm(碗中心仍在隔板上,碗底圈整个在隔板上,放得稳)。"""

from task.RoboDojo.experts import place_ceramic_bowl_storage_rack_center as rack

SPEC = rack.Spec(targets=((None, 0.0), (None, 0.03), (None, -0.03)), overhang=(0.03, 0.025, 0.035))


def play_once(env):
    rack.play_once(env, SPEC)
