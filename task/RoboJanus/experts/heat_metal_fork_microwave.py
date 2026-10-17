"""heat_metal_fork_microwave 的脚本专家:叉子(18.3 cm),平放;叉柄中段离桌面约 5 mm(叉身是弯的),竖直抓在体积质心(z -0.005)往叉柄一侧 1.5 cm 处;流程见 heat_metal_knife_microwave。"""

from task.RoboDojo.experts import heat_metal_knife_microwave as microwave

SPEC = microwave.Spec(
    "metal_fork",
    "handle",
    handles=((0.0, 0.0, -0.02),),
    tip_clear=0.001,
    forward=(0.0, 0.0, 1.0),
    tilts=(0.0,),
    carry_tilts=(45.0, 30.0),
    width=0.009,
)


def play_once(env):
    microwave.play_once(env, SPEC)
