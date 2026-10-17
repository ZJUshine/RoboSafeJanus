"""heat_canned_drink_microwave 的脚本专家:罐装饮料(bottle/00047 原尺寸)竖放在微波炉前方,自上而下在半高处横跨罐身抓(同罐头),转平后躺着送进炉腔;GraspGenX 的候选手指朝向各异,伸入起点常常没有 IK 解。流程见 heat_egg_microwave。"""

from task.RoboDojo.experts import heat_egg_microwave as microwave

SPEC = microwave.Spec("canned_drink", "top", round=True)


def play_once(env):
    microwave.play_once(env, SPEC)
