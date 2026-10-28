"""place_ceramic_bowl_storage_rack_center / _edge 共用的脚本专家(右臂)。本模块是 center 的专家,edge 的专家只是用自己的
``Spec`` 调用 ``play_once``。

置物架(GenieSim_shelf/00001)转 -90 度放在右臂基座正前方,资产 +x(开放的长边)朝向机器人;碗放在它左边的桌面上,要放到顶层
隔板(桌面以上 0.324 m,上方开放)上:center 放在隔板中央,edge 放在前沿、碗沿伸出前沿外 ``Spec.overhang``。

三段运动,每段一次 cuRobo 规划,只在夹爪开合时停(``heat_egg_microwave._pick_place_direct``):初始 -> 抓取 -> 合拢 ->
隔板上的松手位姿 -> 松开 -> 沿接近方向退出 ``RETREAT`` -> 回初始。置物架的 5 个碰撞盒子(两端实心侧墙、三层隔板)和顶层
后沿上方的横杆、碗壁(``_bowl_boxes``:一圈斜放的薄盒子 + 碗底)放进 cuRobo 的碰撞世界,手指和手中的碗用 tool 碰撞球
补上。夹爪只张开到 ``OPEN_GAP``(去抓取、松手时都是):碗沿是斜着抓的,手指方向倾斜约 35 度,全开时里面那根手指伸到碗底
以下,下去时把碗推开(data/debug/bowl_center/r1、r2)。

1. 抓取:GraspGenX 的碗沿抓取(一指在碗里、一指在碗外),接近方向向下倾 ``APPROACH_Z``。碗旋转对称,每个候选绕碗的轴
   转到接近方向的水平投影朝置物架里(``inward``),只留抓在靠机器人一侧碗沿上的(夹爪在碗外);按接近方向与
   "朝里向下倾 ``PREFER_PITCH``" 的夹角排序。GraspGenX 的 TCP 在碗壁里侧 0.5 ~ 1.5 cm(沿手指方向),两指同时合拢时
   先碰到碗壁的那根把碗推开,所以每个候选沿手指方向平移到碗壁截面中线正好在两指中间(``_center_on_wall``),只留碗壁
   交点高度在 ``WALL_Z`` 内的。慢慢合到底(碗沿薄,包围盒量不出夹持宽度)。
2. 松手位姿:手中的碗只绕竖直轴转(夹爪的倾角不变,碗保持水平;先试把碗扶正),碗中心落在 ``Spec.targets``
   (置物架坐标系的 x, y),碗底离隔板 ``RELEASE_GAPS``。
3. 松开后先沿接近方向退出(夹爪斜着往回、往上抽出碗沿;顶层在右臂够得到的上限附近,直线退不出时改为普通规划,
   再不行就直接回程,碗壁在碰撞世界里),夹爪全开,再回初始位姿(任务判定要求此时碗在顶层的目标区域里、
   置物架没被推动)。
"""

from dataclasses import dataclass

import numpy as np
import transforms3d as t3d

from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts import heat_egg_microwave as mw
from task.RoboDojo.experts.put_bread_toaster import _grip_val
from task.RoboDojo.experts.put_toilet_cleaner_refrigerator import _heading, _rot_to, _rz
from task.RoboDojo.tasks.place_ceramic_bowl_storage_rack_X import FRONT_X, TOP_Z, rack_scale

ARM = "right_arm"
BOWL = "ceramic_bowl"
RACK = "storage_rack"

# 置物架碰撞体(GenieSim_shelf/00001 资产坐标系,米;缩放变体乘 rack_scale):USD 碰撞网格的 5 个盒子 + 顶层后沿上方的横杆
# (只有视觉网格)
RACK_BOXES = (
    ((-0.129, -0.184, -0.2024), (0.129, -0.176, 0.203)),
    ((-0.129, 0.176, -0.2024), (0.129, 0.184, 0.203)),
    ((-0.128, -0.177, -0.178), (0.126, 0.176, -0.174)),
    ((-0.128, -0.177, -0.030), (0.126, 0.176, -0.026)),
    ((-0.128, -0.177, 0.117), (0.126, 0.176, 0.122)),
    ((-0.127, -0.180, 0.196), (-0.123, 0.180, 0.200)),
)


@dataclass(frozen=True)
class Spec:
    # 碗中心在置物架坐标系的 (x, y) 候选,按优先级;None 时按 overhang 放在前沿
    targets: tuple = ((0.0, 0.0), (0.015, 0.0), (-0.015, 0.0), (0.0, 0.02), (0.0, -0.02))
    overhang: tuple = ()  # edge:碗沿伸出前沿外的距离候选,碗中心 x = FRONT_X - 半径 + overhang


CENTER = Spec()

APPROACH_Z = (-0.8, -0.45)  # 接近方向单位向量 z 分量:斜着向下抓碗沿;更陡的在顶层松手时够不到(腕部太高)
PREFER_PITCH = 30.0
NEAR_RIM = 0.02  # TCP 至少比碗中心更靠机器人这么多(抓在近侧碗沿上,夹爪在碗外)
MAX_GRASPS = 40
GENTLE_STEPS = 150
FINGER_SPACING = 0.008
RIM_GAP = 0.012  # 合拢后两指间距(碗沿厚度)取这么多算手指碰撞球
OPEN_GAP = 0.035  # 去抓取 / 松手时两指间距
# bowl/00010 碗壁截面(碗坐标系 r, z,/tmp/hz/bowl/wall_profile.py):壁厚约 5 mm,中线从碗底边 (0.034, -0.016) 到碗沿
# (0.0587, 0.025);碗底内表面 z -0.0175、外底 z -0.025
WALL = ((0.034, -0.016), (0.0587, 0.025))
WALL_T = 0.008  # 碗壁盒子厚度(实际约 5 mm)
WALL_Z = (-0.002, 0.018)  # 抓取时碗壁交点的高度范围(碗坐标系):太低里面的手指碰碗底,太高抓不牢
FLOOR = (-0.025, -0.0165, 0.052)  # 碗底盒子 z 范围、边长(两个转 45 度的方盒子拼成八边形)
RING_N = 16
YAW_EXTRA = (0.0, 15.0, -15.0, 30.0, -30.0)
RELEASE_GAPS = (0.008, 0.015)
RETREAT = 0.05


def _rack(env):
    rack = mw._mat(env.get_obj_pose(RACK))
    # 资产 +x 是开放的长边,必须朝向机器人(-y);YAML 里的 rotate_deg 改了时这里先报错
    if rack[:3, 0] @ [0.0, -1.0, 0.0] < 0.9:
        raise PlanFailError(f"{RACK} front (asset +x) does not face the robot: {rack[:3, 0].round(3).tolist()}")
    return rack


def _bowl_boxes(env):
    """碗壁(``RING_N`` 个沿碗壁斜放的薄盒子)和碗底(两个方盒子) -> {名字: (dims, 世界位姿)}。"""
    obj = mw._mat(env.get_obj_pose(BOWL))
    (r0, z0), (r1, z1) = WALL
    length = float(np.hypot(r1 - r0, z1 - z0))
    width = 2 * r1 * np.tan(np.pi / RING_N) * 1.1
    out = {}
    for i in range(RING_N):
        th = 2 * np.pi * i / RING_N
        u, t = np.array([np.cos(th), np.sin(th), 0.0]), np.array([-np.sin(th), np.cos(th), 0.0])
        s_dir = ((r1 - r0) * u + np.array([0.0, 0.0, z1 - z0])) / length
        f = np.eye(4)
        f[:3, :3] = np.stack([s_dir, t, np.cross(s_dir, t)], axis=1)
        f[:3, 3] = (r0 + r1) / 2 * u + [0.0, 0.0, (z0 + z1) / 2]
        h = np.array([length, width, WALL_T]) / 2
        out[f"bowl_wall_{i}"] = mw._box(-h, h, obj @ f)
    za, zb, side = FLOOR
    for i, deg in enumerate((0.0, 45.0)):
        out[f"bowl_floor_{i}"] = mw._box((-side / 2, -side / 2, za), (side / 2, side / 2, zb), obj @ _rz(deg))
    return out


def _center_on_wall(m, obj, bias):
    """ee 位姿 ``m`` 沿手指方向平移,让碗壁截面中线落在 TCP 上 -> (新位姿, 交点在碗坐标系的高度);手指方向上找不到
    交点时返回 None。"""
    inv = np.linalg.inv(obj)
    tcp = (inv @ np.append(m[:3, 3] + bias * m[:3, 0], 1.0))[:3]
    f = inv[:3, :3] @ m[:3, 1]
    (r0, z0), (r1, z1) = WALL

    def side(lam):  # 碗壁中线的有向距离(碗外为正)
        p = tcp + lam * f
        return (np.hypot(p[0], p[1]) - r0) * (z1 - z0) - (p[2] - z0) * (r1 - r0)

    lams = np.linspace(-0.03, 0.03, 61)
    vals = np.array([side(x) for x in lams])
    idx = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(idx) == 0:
        return None
    i = idx[np.argmin(np.abs(lams[idx]))]
    lam = lams[i] - vals[i] * (lams[i + 1] - lams[i]) / (vals[i + 1] - vals[i])
    out = m.copy()
    out[:3, 3] += lam * m[:3, 1]
    return out, float((tcp + lam * f)[2])


def _grasps(env, inward):
    """GraspGenX 碗沿抓取候选(见模块说明第 1 条),按优先级。"""
    p = np.radians(PREFER_PITCH)
    prefer = np.cos(p) * inward - np.sin(p) * np.array([0.0, 0.0, 1.0])
    candidates = env.grasp_candidates(BOWL, ARM, prefer=prefer, approach_z=APPROACH_Z, max_candidates=1000)
    obj = mw._mat(env.get_obj_pose(BOWL))
    axis, center = obj[:3, 2], mw._center(env, BOWL)
    bias = env.robot(ARM).gripper_bias
    out = []
    for g, _ in candidates:
        m = mw._mat(g)
        delta = np.radians(_heading(inward) - _heading(mw._horizontal(m[:3, 0])))
        R = np.eye(4)
        R[:3, :3] = t3d.axangles.axangle2mat(axis, delta)
        T = np.eye(4)
        T[:3, 3] = center - R[:3, :3] @ center
        hit = _center_on_wall(T @ R @ m, obj, bias)
        if hit is None or not WALL_Z[0] <= hit[1] <= WALL_Z[1]:
            continue
        m = hit[0]
        tcp = m[:3, 3] + bias * m[:3, 0]
        if (tcp - center) @ inward <= -NEAR_RIM:
            out.append((float(np.arccos(np.clip(m[:3, 0] @ prefer, -1.0, 1.0))), mw._pose(m)))
    if not out:
        raise PlanFailError(f"{ARM}: no near-rim grasp for {BOWL} ({len(candidates)} candidates)")
    out.sort(key=lambda t: t[0])
    return [(g, -cost) for cost, g in out[:MAX_GRASPS]]


def _targets(spec, lo, hi, scale):
    if not spec.overhang:
        return spec.targets
    radius = float(hi[0] - lo[0]) / 2
    return [(FRONT_X * scale - radius + o, y) for o in spec.overhang for _, y in spec.targets]


def _release_poses(spec, rack, scale, inward, ee, ee_T_obj, local, lo, hi):
    """顶层隔板上的松手位姿候选 ``[ee 位姿, ...]``,见模块说明第 2 条;``local`` 是碗包围盒采样点(碗坐标系)。"""
    obj = ee @ ee_T_obj
    obj_pts = (np.c_[local, np.ones(len(local))] @ obj.T)[:, :3]
    center = obj_pts.mean(axis=0)
    top_z = (rack @ [0.0, 0.0, TOP_Z * scale, 1.0])[2]
    base_delta = (_heading(inward) - _heading(mw._horizontal(ee[:3, 0])) + 180.0) % 360.0 - 180.0
    level = _rot_to(obj[:3, 2], np.array([0.0, 0.0, 1.0]))
    out = []
    for r in (level, np.eye(4)):
        for extra in YAW_EXTRA:
            rot = _rz(base_delta + extra) @ r
            for tx, ty in _targets(spec, lo, hi, scale):
                goal = (rack @ [tx, ty, 0.0, 1.0])[:3]
                for gap in RELEASE_GAPS:
                    m = rot.copy()
                    m[:2, 3] = goal[:2] - (rot[:3, :3] @ center)[:2]
                    m[2, 3] = top_z + gap - (obj_pts @ rot[:3, :3].T)[:, 2].min()
                    out.append(mw._pose(m @ ee))
    return out


def _play(env, spec):
    rack = _rack(env)
    inward = mw._horizontal(-rack[:3, 0])
    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    root = env.robot(ARM).entity_origin_pose
    scale = rack_scale(env.scene_manager.layout_manager, 0, RACK)
    fixed = {
        f"rack_{i}": mw._box(np.multiply(lo, scale), np.multiply(hi, scale), rack)
        for i, (lo, hi) in enumerate(RACK_BOXES)
    }
    lo, hi = mw._bbox(env, BOWL)
    local = mw.GRID * (hi - lo) + lo

    def set_world(with_obj, tool):
        # 去抓取、松手后退出 / 回程时碗在桌上 / 隔板上,碗壁当障碍物;搬运时碗在手里(tool 碰撞球)
        planner.set_obstacles({**fixed, **_bowl_boxes(env)} if with_obj else dict(fixed), root)
        planner.set_tool_spheres(tool)

    mw._pick_place_direct(
        env,
        BOWL,
        set_world=set_world,
        grasps=_grasps(env, inward),
        open_tool=mw._finger_spheres(OPEN_GAP, FINGER_SPACING),
        held_tool=lambda ee_T_obj: mw._finger_spheres(RIM_GAP, FINGER_SPACING) + mw._bbox_spheres(ee_T_obj, lo, hi),
        releases=lambda ee, ee_T_obj: _release_poses(spec, rack, scale, inward, ee, ee_T_obj, local, lo, hi),
        close=lambda g: env.close_gripper(ARM, 0.0, GENTLE_STEPS),
        retreat=RETREAT,
        grasp_open=_grip_val(env, OPEN_GAP),
        release_open=_grip_val(env, OPEN_GAP),
        retreat_fallback=True,
    )


def play_once(env, spec=CENTER):
    mw._run_direct(env, lambda: _play(env, spec))
