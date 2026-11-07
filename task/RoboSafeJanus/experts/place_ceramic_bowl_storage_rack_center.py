"""place_ceramic_bowl_storage_rack_center 的脚本专家:碗放在置物架顶层隔板中间。流程见 ``place_ceramic_bowl_storage_rack_X``。"""

from task.RoboDojo.experts import place_ceramic_bowl_storage_rack_X as rack

SPEC = rack.Spec()


def play_once(env):
    rack.play_once(env, SPEC)
