"""put_X_supermarket_shelf 组共用的脚本专家。本模块也定义 put_potato_chips_supermarket_shelf 的专家,其余任务的专家只是
用自己的 ``Spec`` 调用 ``play_once``。

超市货架(``Geometry/GenieSim_supermarket_shelf_box/00006``)立在机器人正前方的桌面上,开口朝 -y。最下层一半是食品区、
一半是日化区,两个区每局随机左右;商品立在货架前方桌面中间,要放到食品区里靠货架中心的空位。用食品区那一侧的手臂
(``heat_egg_microwave`` 的流程函数都用模块级的 ``ARM``,运行期间临时换掉,同 ``drop_paper_ball_trash_can``)。

流程同 ``put_toilet_cleaner_refrigerator``:三段运动,每段一次 cuRobo 规划(``heat_egg_microwave._pick_place_direct``):
初始 -> 抓取 -> 合拢 -> 层板上的松手位姿 -> 松开 -> 沿接近方向退出 -> 回初始。货架碰撞盒子(与资产的碰撞体相同)、
四件陈列品和手上的商品放进 cuRobo 的碰撞世界,手指和手中物体用 tool 碰撞球补上。

1. 抓取(``Spec.grasp``,候选都来自 GraspGenX):
   ``"side"``(竖着放进去的商品):接近方向接近水平(z 分量在 ``APPROACH_Z`` 内)、手指水平、接近方向的水平投影与货架朝里
   方向(+y)夹角不超过 ``Spec.max_yaw``,TCP 离桌面在 ``Spec.tcp_z`` 之间、离物体顶面不少于 ``Spec.top_margin``,
   夹持宽度不超过 ``FINGER_MAX``。
   ``"top"``(矮的商品):自上而下(接近方向 z 分量在 ``TOP_APPROACH_Z`` 内,偏好朝货架里向下倾 45 度)。
   合拢:``Spec.full_close`` 时慢慢合到底,否则慢慢合到比物体窄 ``Spec.squeeze``。
2. 松手位姿:``side`` 时手中物体先扶正(物体 z 轴转到竖直),再绕竖直轴转到"接近方向的水平投影朝货架里"(再加
   ``YAW_EXTRA``);``top`` 时夹爪转到朝货架里、向下倾 ``INSERT_PITCHES``。物体包围盒中心落在层板上 ``TARGETS``
   (货架坐标系:x 是进深,y 朝食品区那一侧取正)处,离层板前沿 / 背板 / 侧板 / 陈列品不少于 ``CLEAR``,物体底面离层板
   ``RELEASE_GAPS``、顶面离上一层层板底面不少于 ``CLEAR``。整条轨迹按 ``CARRY_SLOW`` 放慢。
3. 松开后先沿接近方向退出 ``RETREAT`` 再回初始位姿(两次规划):直接回程会把竖着的商品蹭倒。
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import transforms3d as t3d

from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts import heat_egg_microwave as mw
from task.RoboDojo.experts.put_bread_toaster import _grip_val

SHELF = "shelf"
ZONE = "food_display_1"  # 用它判断食品区在货架中心的哪一侧
DISPLAYS = ("food_display_1", "food_display_2", "cleaning_display_1", "cleaning_display_2")


@dataclass(frozen=True)
class Spec:
    label: str
    grasp: str = "side"  # "side" | "top",见模块说明第 1 条
    tcp_z: tuple = (0.08, 0.16)  # side:TCP 离桌面
    squeeze: float = 0.006  # 夹爪合到比物体(沿手指方向)窄这么多
    top_margin: float = 0.025  # side:TCP 离物体顶面至少这么远
    max_yaw: float = 50.0  # side:接近方向的水平投影与货架朝里方向的最大夹角
    full_close: bool = False  # 慢慢合到底(用满夹紧力)
    # 物体当障碍物时长宽方向的缩放:圆角、细颈的瓶子包围盒比瓶身大,抓取 / 松手位姿上 cuRobo 偏保守的腕部碰撞球就伸进了
    # 包围盒,判碰撞,要缩小(同 put_toilet_cleaner_refrigerator)
    obstacle_shrink: tuple = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Shelf:
    """货架碰撞体(货架坐标系,米;开口朝 +x,宽度沿 y),与 ``ext_assets/integrate/supermarket_shelf_box.py`` 的盒子相同。"""

    boxes: tuple
    board_x: tuple  # 最下层层板的进深范围(前沿在 +x)
    side: float  # 侧板内侧 |y|
    floor: float  # 最下层层板上表面
    ceiling: float  # 上一层层板底面


SHELVES = {
    "GenieSim_supermarket_shelf_box/00006": Shelf(
        boxes=(
            ((-0.183, -0.525, -0.944), (0.170, 0.525, -0.840)),
            ((-0.183, -0.525, -0.944), (-0.140, 0.525, 0.944)),
            ((-0.183, -0.525, -0.944), (0.170, -0.500, 0.944)),
            ((-0.183, 0.500, -0.944), (0.170, 0.525, 0.944)),
            ((-0.145, -0.500, -0.555), (0.166, 0.500, -0.519)),
            ((-0.145, -0.500, -0.255), (0.166, 0.500, -0.226)),
            ((-0.145, -0.500, 0.059), (0.166, 0.500, 0.088)),
            ((-0.145, -0.500, 0.366), (0.166, 0.500, 0.403)),
            ((-0.145, -0.500, 0.908), (0.166, 0.500, 0.944)),
        ),
        board_x=(-0.145, 0.166),
        side=0.500,
        floor=-0.840,
        ceiling=-0.555,
    ),
}

# 抓取
APPROACH_Z = (-0.35, 0.35)
FINGER_Z_MAX = 0.35
TOP_APPROACH_Z = (-1.0, -0.6)
MAX_GRASPS = 40
GENTLE_STEPS = 150
FINGER_SPACING = 0.008  # 手指碰撞球铺满的间距(同 put_toilet_cleaner_refrigerator)
BOTTLE_SHRINK = (0.8, 0.8, 1.0)
DISPLAY_SHRINK = 0.95  # 陈列品当障碍物时按这个比例缩小(都竖着立在层板上,手中物体从旁边擦过)
SHELF_INFLATE = 0.005  # 货架盒子交给 cuRobo 前各向外扩这么多
# 放入
YAW_EXTRA = (0.0, 15.0, -15.0)
# 物体包围盒中心在货架坐标系的 (x, |y|):进深中间偏里;离货架中心 0.12 ~ 0.20 m(陈列品内侧在 ~0.23 m)
TARGETS = ((-0.01, 0.15), (-0.01, 0.18), (-0.01, 0.12), (0.03, 0.15), (-0.05, 0.15))
CLEAR = 0.015  # 物体包围盒离层板前沿 / 背板 / 侧板 / 陈列品 / 上一层层板底面的最小距离
INSERT_PITCHES = (45.0, 30.0, 15.0, 0.0)
RELEASE_GAPS = {"side": (0.008,), "top": (0.008, 0.02)}
CARRY_SLOW = 2.0
RETREAT = 0.10


@contextmanager
def _arm(arm):
    old = mw.ARM
    mw.ARM = arm
    try:
        yield
    finally:
        mw.ARM = old


def _shelf(env):
    category, index, _, _ = env.get_obj_asset(SHELF)
    key = f"{category}/{index:05d}"
    if key not in SHELVES:
        raise PlanFailError(f"no board geometry for {key}; measure it and add it to SHELVES")
    return SHELVES[key]


def _zone_sign(env, shelf):
    """食品区在货架坐标系 y 的哪一侧(+1 / -1)。"""
    y = (np.linalg.inv(shelf) @ np.append(mw._center(env, ZONE), 1.0))[1]
    return 1.0 if y > 0 else -1.0


def _heading(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _rz(deg):
    m = np.eye(4)
    m[:3, :3] = t3d.axangles.axangle2mat([0.0, 0.0, 1.0], np.radians(deg))
    return m


def _rot_to(a, b):
    """把单位向量 ``a`` 转到 ``b`` 的最小旋转(4x4)。"""
    m = np.eye(4)
    axis = np.cross(a, b)
    if np.linalg.norm(axis) > 1e-6:
        m[:3, :3] = t3d.axangles.axangle2mat(axis, np.arccos(np.clip(a @ b, -1.0, 1.0)))
    return m


def _down_inward(inward, pitch):
    p = np.radians(pitch)
    return np.cos(p) * inward + np.sin(p) * np.array([0.0, 0.0, -1.0])


def _grasps(env, spec, inward):
    """GraspGenX 抓取候选(见模块说明第 1 条),按优先级。"""
    arm, label = mw.ARM, spec.label
    face = mw._bbox_points(env, label)[mw.FACE_CENTERS]
    if spec.grasp == "top":
        candidates = env.grasp_candidates(
            label, arm, prefer=_down_inward(inward, 45.0), approach_z=TOP_APPROACH_Z, max_candidates=1000
        )
        keep = [
            (mw._lift_tips(env, g), sc) for g, sc in candidates if np.ptp(face @ mw._mat(g)[:3, 1]) <= mw.FINGER_MAX
        ]
    else:
        top = mw._bbox_points(env, label)[:, 2].max()
        table_z = mw._table_z(env)
        bias = env.robot(arm).gripper_bias
        candidates = env.grasp_candidates(
            label, arm, prefer=inward, approach_z=APPROACH_Z, finger_z_max=FINGER_Z_MAX, max_candidates=1000
        )
        keep = []
        for g, score in candidates:
            gm = mw._mat(g)
            a = gm[:3, 0]
            tcp = gm[:3, 3] + bias * a
            yaw = abs((_heading(a) - _heading(inward) + 180.0) % 360.0 - 180.0)
            if (
                yaw <= spec.max_yaw
                and spec.tcp_z[0] <= tcp[2] - table_z <= spec.tcp_z[1]
                and tcp[2] <= top - spec.top_margin
                and np.ptp(face @ gm[:3, 1]) <= mw.FINGER_MAX
            ):
                keep.append((g, score))
    if not keep:
        raise PlanFailError(f"{arm}: no {spec.grasp} grasp for {label} ({len(candidates)} candidates)")
    return keep[:MAX_GRASPS]


def _close(env, spec, grasp):
    """慢慢合拢(同 put_toilet_cleaner_refrigerator)。"""
    fingers = mw._mat(grasp)[:3, 1]
    width = float(np.ptp(mw._bbox_points(env, spec.label)[mw.FACE_CENTERS] @ fingers))
    if spec.full_close or width > mw.FINGER_MAX:
        env.close_gripper(mw.ARM, 0.0, GENTLE_STEPS)
    else:
        env.close_gripper(mw.ARM, _grip_val(env, width - spec.squeeze), GENTLE_STEPS)


def _display_rects(env, shelf):
    """陈列品包围盒在货架坐标系的 xy 范围 ``[(lo, hi), ...]``。"""
    inv = np.linalg.inv(shelf)
    out = []
    for label in DISPLAYS:
        pts = (np.c_[mw._bbox_points(env, label), np.ones(27)] @ inv.T)[:, :2]
        out.append((pts.min(axis=0), pts.max(axis=0)))
    return out


def _release_poses(env, spec, geom, shelf, sign, inward, ee, ee_T_obj, local):
    """层板上的松手位姿候选 ``[ee 位姿, ...]``,见模块说明第 2 条;``local`` 是物体包围盒采样点(物体坐标系)。"""
    obj = ee @ ee_T_obj
    obj_pts = (np.c_[local, np.ones(len(local))] @ obj.T)[:, :3]
    center = obj_pts.mean(axis=0)
    floor_z = (shelf @ [0.0, 0.0, geom.floor, 1.0])[2]
    ceil_z = (shelf @ [0.0, 0.0, geom.ceiling, 1.0])[2]
    if spec.grasp == "top":
        rots = [
            _rot_to(ee[:3, 0], _rz(extra)[:3, :3] @ _down_inward(inward, pitch))
            for pitch in INSERT_PITCHES
            for extra in YAW_EXTRA
        ]
    else:
        base_delta = (_heading(inward) - _heading(mw._horizontal(ee[:3, 0])) + 180.0) % 360.0 - 180.0
        level = _rot_to(obj[:3, 2], np.array([0.0, 0.0, 1.0]))
        rots = [_rz(base_delta + extra) @ r for r in (level, np.eye(4)) for extra in YAW_EXTRA]
    inv = np.linalg.inv(shelf)
    displays = _display_rects(env, shelf)
    bx0, bx1 = geom.board_x
    out = []
    for rot in rots:
        for tx, ty in TARGETS:
            goal = (shelf @ [tx, sign * ty, 0.0, 1.0])[:3]
            for gap in RELEASE_GAPS[spec.grasp]:
                m = rot.copy()
                m[:2, 3] = goal[:2] - (rot[:3, :3] @ center)[:2]
                m[2, 3] = floor_z + gap - (obj_pts @ rot[:3, :3].T)[:, 2].min()
                moved = (np.c_[obj_pts, np.ones(len(obj_pts))] @ m.T)[:, :3]
                if moved[:, 2].max() > ceil_z - CLEAR:
                    continue
                p = (np.c_[moved, np.ones(len(moved))] @ inv.T)[:, :2]
                lo, hi = p.min(axis=0), p.max(axis=0)
                if (
                    lo[0] < bx0 + CLEAR
                    or hi[0] > bx1 - CLEAR
                    or lo[1] < -geom.side + CLEAR
                    or hi[1] > geom.side - CLEAR
                ):
                    continue
                if any(np.all(lo < dhi + CLEAR) and np.all(hi > dlo - CLEAR) for dlo, dhi in displays):
                    continue
                out.append(mw._pose(m @ ee))
    if not out:
        raise PlanFailError(f"{spec.label} does not fit next to the food displays in any orientation")
    return out


def _play(env, spec):
    label = spec.label
    geom = _shelf(env)
    shelf = mw._mat(env.get_obj_pose(SHELF))
    inward = mw._horizontal(-shelf[:3, 0])
    sign = _zone_sign(env, shelf)
    planner = env.robot_manager.planner[env.robot(mw.ARM).robot_name]
    root = env.robot(mw.ARM).entity_origin_pose
    pad = np.full(3, SHELF_INFLATE)
    fixed = {
        f"shelf_{i}": mw._box(np.subtract(lo, pad), np.add(hi, pad), shelf) for i, (lo, hi) in enumerate(geom.boxes)
    }
    fixed.update({d: mw._obj_box(env, d, DISPLAY_SHRINK) for d in DISPLAYS})
    lo, hi = mw._bbox(env, label)
    local = mw.GRID * (hi - lo) + lo

    def set_world(with_obj, tool):
        obstacles = dict(fixed)
        if with_obj:
            obstacles[label] = mw._obj_box(env, label, 1.0 if with_obj == "return" else spec.obstacle_shrink)
        planner.set_obstacles(obstacles, root)
        planner.set_tool_spheres(tool)

    def held_tool(ee_T_obj):
        held_ee = (np.c_[local, np.ones(len(local))] @ ee_T_obj.T)[:, :3]
        width = min(float(np.ptp(held_ee[:, 1])), mw.GRIPPER_OPEN)
        return mw._finger_spheres(width, FINGER_SPACING) + mw._bbox_spheres(ee_T_obj, lo, hi)

    mw._pick_place_direct(
        env,
        label,
        set_world=set_world,
        grasps=_grasps(env, spec, inward),
        open_tool=mw._finger_spheres(mw.GRIPPER_OPEN, FINGER_SPACING),
        held_tool=held_tool,
        releases=lambda ee, ee_T_obj: _release_poses(env, spec, geom, shelf, sign, inward, ee, ee_T_obj, local),
        close=lambda g: _close(env, spec, g),
        carry_slow=CARRY_SLOW,
        retreat=RETREAT if spec.grasp == "side" else 0.0,
    )


def _zone_arm(env):
    """食品区那一侧的手臂:食品区在世界 -x 一侧(机器人左边)用左臂。"""
    shelf = mw._mat(env.get_obj_pose(SHELF))
    return "left_arm" if mw._center(env, ZONE)[0] < shelf[0, 3] else "right_arm"


SPEC = Spec("potato_chips")


def play_once(env, spec=None):
    with _arm(_zone_arm(env)):
        mw._run_direct(env, lambda: _play(env, SPEC if spec is None else spec))
