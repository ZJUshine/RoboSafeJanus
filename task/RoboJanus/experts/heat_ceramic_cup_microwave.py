"""heat_ceramic_cup_microwave 的脚本专家:杯子(直径 7.2 cm,高 7.2 cm,无把手);竖直夹住近侧杯壁(倾斜着夹竖直的杯壁只有指尖一角碰到,一抬就滑);流程见 heat_metal_knife_microwave。"""

from task.RoboDojo.experts import heat_metal_knife_microwave as microwave

SPEC = microwave.Spec("ceramic_cup", "rim", tilts=(0.0,), carry_tilts=(45.0, 30.0), depth=0.04, rim_in=0.003)


def play_once(env):
    microwave.play_once(env, SPEC)
