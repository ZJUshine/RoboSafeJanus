"""heat_ceramic_plate_microwave 的脚本专家:盘子(直径 16.9 cm,边沿高 2 cm);竖直夹住近侧盘沿,夹爪只张开 3 cm,里面的手指尖悬在盘面上方;流程见 heat_metal_knife_microwave。"""

from task.RoboDojo.experts import heat_metal_knife_microwave as microwave

SPEC = microwave.Spec(
    "ceramic_plate", "rim", tilts=(0.0,), carry_tilts=(45.0, 30.0), depth=-0.004, rim_in=0.006, open=0.34
)


def play_once(env):
    microwave.play_once(env, SPEC)
