"""place_phone_cup 的脚本专家:流程见 ``place_phone_X``,手机竖着放进杯子。"""

from task.RoboDojo.experts import place_phone_X as stand

SPEC = stand.Spec("cup", "cup")


def play_once(env):
    stand.play_once(env, SPEC)
