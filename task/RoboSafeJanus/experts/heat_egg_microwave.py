"""heat_egg_microwave 的脚本专家:鸡蛋侧躺在微波炉前方,GraspGenX 的抓取、手指横跨鸡蛋短径,伸入时鸡蛋长轴朝里(松手后只能向两侧滚,被炉壁挡住)。流程见 ``heat_X_microwave``。"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("egg", round=True, approach_z=(-1.0, -0.25), prefer_pitch=40.0)


def play_once(env):
    microwave.play_once(env, SPEC)
