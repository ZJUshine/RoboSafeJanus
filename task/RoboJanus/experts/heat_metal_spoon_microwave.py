"""heat_metal_spoon_microwave 的脚本专家:勺子(18.5 cm),平放;竖直抓在体积质心(z +0.015)往勺柄一侧 3 cm 处;流程见 heat_metal_knife_microwave。"""

from task.RoboDojo.experts import heat_metal_knife_microwave as microwave

SPEC = microwave.Spec(
    "metal_spoon",
    "handle",
    handles=((0.0, 0.0, -0.015),),
    tip_clear=0.001,
    forward=(0.0, 0.0, 1.0),
    tilts=(0.0,),
    carry_tilts=(45.0, 30.0),
    width=0.009,
)


def play_once(env):
    microwave.play_once(env, SPEC)
