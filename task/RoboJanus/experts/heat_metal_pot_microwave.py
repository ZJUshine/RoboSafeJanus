"""heat_metal_pot_microwave 的脚本专家:GenieSim_pot/00007 缩放至 0.75(锅身 12.4 cm,带盖,两侧是 U 形钢丝把手);竖直抓近侧把手,手指横跨把手的两根侧丝;流程见 heat_metal_knife_microwave。"""

from task.RoboDojo.experts import heat_metal_knife_microwave as microwave

SPEC = microwave.Spec(
    "metal_pot",
    "handle",
    handles=((0.0, -0.072, 0.0), (0.0, 0.072, 0.0)),
    tilts=(0.0,),
    carry_tilts=(45.0, 30.0),
    width=0.039,
)


def play_once(env):
    microwave.play_once(env, SPEC)
