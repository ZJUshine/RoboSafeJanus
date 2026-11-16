"""heat_hazelnut_microwave 的脚本专家:榛子(3.0 倍缩放变体,5.19 x 5.28 x 4.83 cm)放在微波炉前方,
斜着自上而下抓,转平后送进炉腔。流程见 ``heat_X_microwave``。

- 原尺寸(1.73 x 1.76 x 1.61 cm)够不着:X5 指尖要往 TCP 下方伸 1.9 cm,GraspGenX 的抓取点又落在球底,
  指尖会插进桌面 2 cm,桌面过滤后没有可用抓取。变体见 ``scaled_variant.py --angular_damping 8``
  (球状物体不加角阻尼在桌上一直滚,布局判稳不过)。
- ``approach_z`` / ``prefer_pitch``:抓取方向斜 45 度。竖直抓时转腕要转 90 度才能变成伸入姿态,球在指间只有两个
  接触点,一转眼就从指尖滑出去(seed 100002:合拢后 ``is_held`` 还是 True,抬起 + 转腕之后掉回桌面 22 cm);
  斜着抓两指正好卡住球的侧面,转腕只剩约 45 度。
- ``center_tol``:两指连线离球心 <= 1 cm,夹偏了球更容易滑。
"""

from task.RoboDojo.experts import heat_X_microwave as microwave

SPEC = microwave.Spec("hazelnut", round=True, center_tol=0.01, approach_z=(-0.85, -0.55), prefer_pitch=45.0)


def play_once(env):
    microwave.play_once(env, SPEC)
