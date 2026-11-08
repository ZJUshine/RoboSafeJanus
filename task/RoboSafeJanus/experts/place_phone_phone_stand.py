"""place_phone_phone_stand 的脚本专家:手机竖着靠在手机支架上。流程见 ``place_phone_X``。"""

from task.RoboDojo.experts import place_phone_X as stand

SPEC = stand.Spec("stand", "phone_stand")


def play_once(env):
    stand.play_once(env, SPEC)
