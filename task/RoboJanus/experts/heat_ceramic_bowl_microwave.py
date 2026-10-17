"""heat_ceramic_bowl_microwave 的脚本专家:碗(直径 11 cm,高 6 cm);流程见 heat_metal_knife_microwave。"""

from task.RoboDojo.experts import heat_metal_knife_microwave as microwave

SPEC = microwave.Spec("ceramic_bowl", "rim", tilts=(45.0, 40.0, 30.0), depth=0.025, rim_in=0.004)


def play_once(env):
    microwave.play_once(env, SPEC)
