"""put_power_bank_refrigerator 的脚本专家:同 ``put_toilet_cleaner_refrigerator``,充电宝矮,自上而下抓(``grasp="top"``)。"""

from task.RoboDojo.experts import put_toilet_cleaner_refrigerator as fridge

SPEC = fridge.Spec("power_bank", grasp="top")


def play_once(env):
    fridge.play_once(env, SPEC)
