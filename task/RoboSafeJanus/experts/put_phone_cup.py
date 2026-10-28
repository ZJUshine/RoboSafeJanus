"""put_phone_cup 的脚本专家:同 ``put_phone_phone_stand``,手机竖着放进杯子。"""

from task.RoboDojo.experts import put_phone_phone_stand as stand

CUP = stand.Spec("cup", "cup")


def play_once(env):
    stand.play_once(env, CUP)
