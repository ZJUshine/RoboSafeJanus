"""drop_X_trash_can 组共用的脚本专家(左臂)。本模块是 drop_paper_ball_trash_can 的专家,其余任务的专家只是用自己的
``Spec`` 调用 ``play_once``。

垃圾桶在桌子左侧的地上(中心约 (-0.63, -0.10)),只有左臂够得到。流程同 ``drop_rubber_ball_table``:三段运动,每段一次
cuRobo 规划(``heat_egg_microwave._pick_place_direct``):初始 -> 抓取 -> 合拢 -> 桶口上方的松手位姿 -> 松开(物体落进桶里)
-> 回初始。垃圾桶整个包围盒当障碍物,手中物体最低处要高出桶口 ``DROP_HEIGHTS``。

``heat_egg_microwave`` 的流程函数都用模块级的 ``ARM``(右臂),这里在运行期间临时换成左臂。
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import transforms3d as t3d

from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts import heat_egg_microwave as mw

ARM = "left_arm"
TRASH_CAN = "trash_can"


@dataclass(frozen=True)
class Spec:
    label: str
    round: bool = False  # 圆的物体:手中碰撞球沿长轴排 3 个、包围盒缩小 OBJ_SHRINK 当障碍物(同 heat_egg_microwave)
    prefer_pitch: float = 60.0  # 偏好的接近方向向下倾的角度(度)
    # 物体的盒子(障碍物、手中碰撞球、松手高度都用它):((cx, cy, cz), (dx, dy, dz), 绕物体 z 轴转的角度),物体坐标系;
    # None 时用包围盒。斜着躺在资产坐标系里的物体(压扁的易拉罐)包围盒比物体大得多,夹住它的手指都在盒子里
    box: tuple = None


PAPER_BALL = Spec("paper_ball", round=True)

APPROACH_Z = (-1.0, 0.1)
MAX_GRASPS = 40
DROP_HEIGHTS = (0.08, 0.12, 0.05)  # 松手时手中物体最低处高出桶口
# 松手点:垃圾桶中心水平往左臂基座方向挪这么多(桶口内沿离中心约 0.21 m,中心离基座约 0.48 m)
SHIFTS = (0.06, 0.0, 0.12)
CAN_INFLATE = 0.01  # 垃圾桶包围盒交给 cuRobo 前各向外扩这么多
CARRY_SLOW = 1.5


@contextmanager
def _left_arm():
    old = mw.ARM
    mw.ARM = ARM
    try:
        yield
    finally:
        mw.ARM = old


def _can(env):
    """垃圾桶的位姿矩阵、包围盒(物体坐标系)和桶口高度。"""
    can = mw._mat(env.get_obj_pose(TRASH_CAN))
    lo, hi = mw._bbox(env, TRASH_CAN)
    top = float(((np.c_[mw.GRID * (hi - lo) + lo, np.ones(len(mw.GRID))] @ can.T)[:, 2]).max())
    return can, lo, hi, top


def _obj_box_frame(env, spec):
    """物体盒子:(物体坐标系 -> 盒子坐标系的变换, 盒子坐标系下的 lo, hi)。"""
    if spec.box is None:
        lo, hi = mw._bbox(env, spec.label)
        return np.eye(4), lo, hi
    center, dims, yaw = spec.box
    T = np.eye(4)
    T[:3, :3] = t3d.axangles.axangle2mat([0.0, 0.0, 1.0], np.radians(yaw))
    T[:3, 3] = center
    h = np.asarray(dims, dtype=float) / 2
    return T, -h, h


def _grasps(env, spec):
    base = np.asarray(env.robot(ARM).entity_origin_pose[:3], dtype=float)
    outward = mw._horizontal(mw._center(env, spec.label) - base)
    p = np.radians(spec.prefer_pitch)
    prefer = np.cos(p) * outward - np.sin(p) * np.array([0.0, 0.0, 1.0])
    candidates = env.grasp_candidates(spec.label, ARM, prefer=prefer, approach_z=APPROACH_Z, max_candidates=1000)
    candidates = [(mw._lift_tips(env, g), s) for g, s in candidates]
    if not candidates:
        raise PlanFailError(f"{ARM}: no grasp candidate for {spec.label}")
    return candidates[:MAX_GRASPS]


def _releases(env, spec, ee, ee_T_obj):
    """抓取位姿平移(姿态不变)到手中物体包围盒中心在松手点正上方、最低处高出桶口 ``DROP_HEIGHTS``。"""
    T_box, lo, hi = _obj_box_frame(env, spec)
    pts = mw._held_points(spec, lo, hi)
    box = ee @ ee_T_obj @ T_box
    world = (np.c_[pts, np.ones(len(pts))] @ box.T)[:, :3]
    center = (box @ np.append((lo + hi) / 2, 1.0))[:3]
    bottom = float(world[:, 2].min())
    can, _, _, top = _can(env)
    base = np.asarray(env.robot(ARM).entity_origin_pose[:3], dtype=float)
    toward = mw._horizontal(base - can[:3, 3])
    out = []
    for h in DROP_HEIGHTS:
        for d in SHIFTS:
            T = ee.copy()
            target = can[:3, 3] + d * toward
            T[:2, 3] += target[:2] - center[:2]
            T[2, 3] += top + h - bottom
            out.append(mw._pose(T))
    return out


def _play(env, spec):
    label = spec.label
    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    root = env.robot(ARM).entity_origin_pose
    T_box, lo, hi = _obj_box_frame(env, spec)
    pts_box = mw._held_points(spec, lo, hi)
    shrink = mw.OBJ_SHRINK if spec.round else mw.BOX_SHRINK
    can, can_lo, can_hi, _ = _can(env)
    can_box = mw._box(can_lo - CAN_INFLATE, can_hi + CAN_INFLATE, can)

    def set_world(with_obj, tool):
        obstacles = {TRASH_CAN: can_box}
        if with_obj:
            c, h = (lo + hi) / 2, (hi - lo) / 2 * shrink
            obstacles[label] = mw._box(c - h, c + h, mw._mat(env.get_obj_pose(label)) @ T_box)
        planner.set_obstacles(obstacles, root)
        planner.set_tool_spheres(tool)

    def held_tool(ee_T_obj):
        ee_T_box = ee_T_obj @ T_box
        held_ee = (np.c_[pts_box, np.ones(len(pts_box))] @ ee_T_box.T)[:, :3]
        width = float(np.ptp(held_ee[:, 1]))  # 物体沿手指方向(ee y)的宽度
        obj = mw._held_spheres(ee_T_box, lo, hi) if spec.round else mw._bbox_spheres(ee_T_box, lo, hi)
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


def play_once(env, spec=PAPER_BALL):
    with _left_arm():
        mw._run_direct(env, lambda: _play(env, spec))
