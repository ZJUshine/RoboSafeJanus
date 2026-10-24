"""pour_oil_toaster 的脚本专家:与 ``pour_liquid_toaster`` 相同(只是液体换成黄色的油)。"""

from task.RoboDojo.experts import pour_liquid_toaster as toaster


def play_once(env):
    toaster.play_once(env)
