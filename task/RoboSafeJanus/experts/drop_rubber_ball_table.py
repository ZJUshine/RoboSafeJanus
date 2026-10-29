"""drop_rubber_ball_table / drop_vase_table / drop_glass_cup_table 共用的脚本专家(右臂)。本模块是 drop_rubber_ball_table 的专家,
其余任务的专家只是用自己的 ``Spec`` 调用 ``play_once``。

三段运动,每段一次 cuRobo 规划,只在夹爪开合时停(``heat_egg_microwave._pick_place_direct``):初始 -> 抓取 -> 合拢 ->
松手位姿 -> 松开(物体落回桌面)-> 回初始。物体放进 cuRobo 的碰撞世界,手指和手中物体用 tool 碰撞球补上。

1. 抓取:GraspGenX 的抓取,接近方向 z 分量在 ``APPROACH_Z`` 内,偏好从机器人基座朝物体、向下倾 ``Spec.prefer_pitch`` 度
   (同 heat_egg_microwave;偏好自上而下时,瓶颈、杯沿这种高处的抓取末端要竖直朝下、离基座又近,大多规划不到,
   data/debug/drop/r1)。``Spec.obj_z`` 给定时只抓物体坐标系里 TCP 高度在这个范围内的部位(花瓶、玻璃瓶的瓶颈)。
   指尖离桌面不足 ``heat_egg_microwave.TOP_TIP_CLEAR`` 时往回提。去抓取时物体当障碍物:默认是缩小的包围盒;瓶颈比
   包围盒窄得多的物体用 ``Spec.profile`` 按高度分段的方盒子,否则张开的手指夹瓶颈时就在包围盒里,cuRobo 判碰撞
   (data/debug/drop/r3、r5)。
2. 松手位姿:抓取位姿竖直往上平移(姿态不变),让手中物体最低处离桌面 ``DROP_HEIGHTS``,依次再试水平往机器人方向
   挪 ``SHIFTS``(够不着时)。
3. 松开,物体落回桌面,回初始位姿。
"""

from dataclasses import dataclass

import numpy as np

from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts import heat_egg_microwave as mw

ARM = "right_arm"


@dataclass(frozen=True)
class Spec:
    label: str
    obj_z: tuple = None  # 抓取中心(TCP)在物体坐标系下的高度范围;None 时不限
    round: bool = False  # 圆的物体:手中碰撞球沿长轴排 3 个、包围盒缩小 OBJ_SHRINK 当障碍物(同 heat_egg_microwave)
    prefer_pitch: float = 40.0  # 偏好的接近方向向下倾的角度(度)
    # 障碍物形状:[(z_lo, z_hi, x 宽, y 宽), ...],物体坐标系下、以包围盒的 xy 中心为中心的盒子;None 时用缩小的包围盒
    profile: tuple = None


BALL = Spec("rubber_ball", round=True)

APPROACH_Z = (-1.0, 0.1)
MAX_GRASPS = 40
DROP_HEIGHTS = (0.25, 0.22, 0.19)  # 松手时手中物体最低处离桌面
SHIFTS = (0.0, 0.04, 0.08)  # 松手位姿水平往机器人基座方向挪这么多
CARRY_SLOW = 1.5


def _grasps(env, spec):
    base = np.asarray(env.robot(ARM).entity_origin_pose[:3], dtype=float)
    outward = mw._horizontal(mw._center(env, spec.label) - base)
    p = np.radians(spec.prefer_pitch)
    prefer = np.cos(p) * outward - np.sin(p) * np.array([0.0, 0.0, 1.0])
    candidates = env.grasp_candidates(
        spec.label, ARM, prefer=prefer, approach_z=APPROACH_Z, obj_z=spec.obj_z, max_candidates=1000
    )
    candidates = [(mw._lift_tips(env, g), s) for g, s in candidates]
    if not candidates:
        raise PlanFailError(f"{ARM}: no grasp candidate for {spec.label}")
    return candidates[:MAX_GRASPS]


def _obstacles(env, spec, shrink):
    if spec.profile is None:
        return {spec.label: mw._obj_box(env, spec.label, shrink)}
    lo, hi = mw._bbox(env, spec.label)
    cx, cy = (lo[:2] + hi[:2]) / 2
    obj = mw._mat(env.get_obj_pose(spec.label))
    return {
        f"{spec.label}_{i}": mw._box((cx - wx / 2, cy - wy / 2, z0), (cx + wx / 2, cy + wy / 2, z1), obj)
        for i, (z0, z1, wx, wy) in enumerate(spec.profile)
    }


def _releases(env, spec, ee, ee_T_obj):
    lo, hi = mw._bbox(env, spec.label)
    pts = mw._held_points(spec, lo, hi)
    obj = ee @ ee_T_obj
    bottom = float(((np.c_[pts, np.ones(len(pts))] @ obj.T)[:, 2]).min())
    base = np.asarray(env.robot(ARM).entity_origin_pose[:3], dtype=float)
    toward = mw._horizontal(base - ee[:3, 3])
    out = []
    for h in DROP_HEIGHTS:
        for d in SHIFTS:
            T = ee.copy()
            T[2, 3] += mw._table_z(env) + h - bottom
            T[:3, 3] += d * toward
            out.append(mw._pose(T))
    return out


def _play(env, spec):
    label = spec.label
    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    root = env.robot(ARM).entity_origin_pose
    lo, hi = mw._bbox(env, label)
    pts_obj = mw._held_points(spec, lo, hi)
    shrink = mw.OBJ_SHRINK if spec.round else mw.BOX_SHRINK

    def set_world(with_obj, tool):
        planner.set_obstacles(_obstacles(env, spec, shrink) if with_obj else {}, root)
        planner.set_tool_spheres(tool)

    def held_tool(ee_T_obj):
        held_ee = (np.c_[pts_obj, np.ones(len(pts_obj))] @ ee_T_obj.T)[:, :3]
        width = float(np.ptp(held_ee[:, 1]))  # 物体沿手指方向(ee y)的宽度
        obj = mw._held_spheres(ee_T_obj, lo, hi) if spec.round else mw._bbox_spheres(ee_T_obj, lo, hi)
        return mw._finger_spheres(min(width, mw.GRIPPER_OPEN)) + obj

    mw._pick_place_direct(
        env,
        label,
        set_world=set_world,
        grasps=_grasps(env, spec),
        open_tool=mw._finger_spheres(mw.GRIPPER_OPEN),
        held_tool=held_tool,
        releases=lambda ee, ee_T_obj: _releases(env, spec, ee, ee_T_obj),
        close=lambda g: env.close_gripper(ARM),
        carry_slow=CARRY_SLOW,
    )


def play_once(env, spec=BALL):
    mw._run_direct(env, lambda: _play(env, spec))
