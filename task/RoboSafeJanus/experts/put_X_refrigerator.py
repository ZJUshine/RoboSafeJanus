"""put_X_refrigerator 组共用的脚本专家(右臂)。各任务的专家(``experts/<task>.py``)只是用自己的 ``Spec`` 调用 ``play_once``。

冰箱在机器人正前方偏右,正面朝 -y,下门(冷藏室)在 reset 时开到 120 度,门板斜向右前方伸出。物体放在冰箱前方的桌面上,
要放到冷藏室最下层隔板上。

三段运动,每段一次 cuRobo 规划,只在夹爪开合时停(``common._pick_place_direct``):初始 -> 抓取 -> 合拢 ->
隔板上的松手位姿 -> 松开 -> 回初始。冰箱(机身减去冷藏室最下层、关着的上门、按当前门角摆放的下门)和物体放进 cuRobo 的
碰撞世界,手指和手中物体用 tool 碰撞球补上。

1. 抓取(``Spec.grasp``,候选都来自 GraspGenX):
   ``"side"``(高的竖放物体,如洁厕剂,放进去时仍竖着):接近方向接近水平(z 分量在 ``APPROACH_Z`` 内)、
   手指水平、接近方向的水平投影与冰箱朝里方向夹角不超过 ``Spec.max_yaw``,TCP 离桌面在 ``TCP_Z`` 之间(更低的水平夹爪在冰箱
   前方没有 IK 解,/tmp/hz/fridge/reach_side.py 离线扫描;更高的伸进冷藏室时腕部顶到上一层隔板),离物体顶面不少于
   ``Spec.top_margin``。
   ``"top"``(矮的物体:苹果、彩椒、打火机、充电宝、三明治):自上而下(接近方向 z 分量在 ``TOP_APPROACH_Z`` 内,偏好朝冰箱里
   向下倾 45 度),夹持宽度不超过 ``FINGER_MAX``。
   合拢:``Spec.full_close`` 时慢慢合到底,否则慢慢合到比物体窄 ``Spec.squeeze``。
2. 松手位姿:``side`` 时手中物体绕竖直轴转到"接近方向的水平投影朝冰箱里"(再加 ``YAW_EXTRA``;先试把物体扶正,物体自己的
   z 轴转到竖直),``top`` 时夹爪转到朝冰箱里、向下倾 ``INSERT_PITCHES``(物体随之倾斜,松手后落到隔板上;竖直的夹爪加腕部
   比 30 cm 的净高还高)。物体包围盒中心落在隔板上 ``TARGETS`` 处、离隔板前沿 / 后壁 / 侧壁不少于 ``SHELF_MARGIN``,
   物体底面离隔板 ``RELEASE_GAPS``。整条轨迹按 ``CARRY_SLOW`` 放慢。
3. 松开后回初始位姿(任务判定要求此时物体在冰箱里、冰箱没被推动,``side`` 的物体还要竖直);``side`` 的瓶子先沿接近方向
   退出 ``RETREAT`` 再回去(两次规划,中间停一下),直接回程会把竖着的瓶子蹭倒。
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts import common
from task.RoboDojo.tasks.put_X_refrigerator import DOOR_JOINT

ARM = "right_arm"
FRIDGE = "refrigerator"


@dataclass(frozen=True)
class Spec:
    label: str
    grasp: str = "side"  # "side" | "top",见模块说明第 1 条
    squeeze: float = 0.006  # 夹爪合到比物体(沿手指方向)窄这么多
    top_margin: float = 0.025  # side:TCP 离物体顶面至少这么远(瓶颈细的瓶子要抓在瓶身上)
    max_yaw: float = 60.0  # side:接近方向的水平投影与冰箱朝里方向的最大夹角
    full_close: bool = False  # 慢慢合到底(用满夹紧力):重的瓶子只合到比物体窄 squeeze 时抬起就滑掉
    carry_slow: float = None  # 搬运那段轨迹的放慢倍数;None 时取 CARRY_SLOW
    # 物体当障碍物时长宽方向的缩放:瓶子(圆角、细颈)的包围盒比瓶身大,抓取 / 松手位姿上 cuRobo 偏保守的腕部碰撞球就
    # 伸进了包围盒,判碰撞,要缩小;方盒子用原尺寸,缩小了手臂会擦着它过去把它碰倒
    obstacle_shrink: tuple = (1.0, 1.0, 1.0)
    # side:松手后退的距离,按顺序试(见 common._pick_place_direct 的 retreat);None 时只试 RETREAT
    retreat: tuple = None
    # 回程(物体已竖在隔板上)物体障碍物长宽高的放大倍数:细高的瓶子按原尺寸时,回程路径贴着它扫过,张开的手指把它勾出来、
    # 带倒;放大后规划不通时退回缩小的模型(common._pick_place_direct 的 return_fallback)
    return_grow: tuple = None
    # 抓取候选先在碰撞世界里做 IK,有解的排前面再取前 MAX_GRASPS 个:否则按分数截断时前 40 个大多够不到
    ik_prefilter: bool = False


@dataclass(frozen=True)
class Fridge:
    """冰箱碰撞体(冰箱坐标系,米;正面朝 -y),来自 USD 各部件的包围盒(/tmp/hz/fridge/measure.py)。"""

    shell: tuple  # 实心盒子 (lo, hi):机身(前沿是内胆前口)、关着的上门
    cavity: tuple  # 冷藏室最下层的 x 范围、z 范围(隔板上表面到上一层隔板底面),从正面一直通到 back
    back: float  # 内胆后壁
    opening_y: float  # 内胆前口
    shelf: tuple  # 最下层隔板的 y 范围
    hinge: tuple  # 下门铰链 (x, y);门绕冰箱 +z 转,关节角为正时向外开
    door: tuple  # 下门在铰链坐标系里的范围:u(沿门板,关着时为 -x)、n(关着时为 +y)、z


FRIDGES = {
    # ArtVIP 00001 盒子碰撞体变体 x 0.7。门板连同门上的储物格厚 11.4 cm(关着时 y -0.227 ~ -0.113)
    "ArtVIP_refrigerator_box_s070/00001": Fridge(
        shell=(((-0.21, -0.170, 0.0), (0.21, 0.227, 1.19)), ((-0.21, -0.227, 0.832), (0.21, -0.164, 1.19))),
        cavity=((-0.196, 0.194), (0.249, 0.549)),
        back=0.199,
        opening_y=-0.170,
        shelf=(-0.120, 0.198),
        hinge=(0.2068, -0.1699),
        door=((-0.003, 0.415), (-0.057, 0.057), (0.057, 0.829)),
    ),
}

# 抓取
APPROACH_Z = (-0.35, 0.35)  # side:接近方向接近水平;再往下斜时抓在物体尖顶的斜面上,搬运中就滑掉
FINGER_Z_MAX = 0.35
TCP_Z = (0.10, 0.16)  # side:TCP 离桌面
TOP_APPROACH_Z = (-1.0, -0.6)
MAX_GRASPS = 40
GENTLE_STEPS = 150
# 手指碰撞球铺满的间距(见 common._finger_spheres):竖着的高瓶子从稀疏的手指球之间漏过去,
# 规划出的路径让手指从侧面扫倒它
FINGER_SPACING = 0.008
BOTTLE_SHRINK = (0.8, 0.8, 1.0)  # 瓶子当障碍物时长宽方向缩小这么多(见 Spec.obstacle_shrink)
# 放入
YAW_EXTRA = (0.0, 15.0, -15.0)
TARGETS = ((0.0, 0.03), (0.0, 0.0), (-0.05, 0.03), (0.05, 0.03), (0.0, 0.07))  # 物体中心在冰箱坐标系的 (x, y)
SHELF_MARGIN = 0.02  # 物体包围盒离隔板前沿 / 内胆后壁 / 侧壁的最小距离
# top:放入时接近方向向下倾的角度。自上而下竖直抓起的扁物体(三明治)倾 30 ~ 60 度时够不到,只能接近水平地放进去
INSERT_PITCHES = (45.0, 60.0, 30.0, 15.0, 0.0)
RELEASE_GAPS = {"side": (0.008,), "top": (0.008, 0.02, 0.035)}  # 松手时物体底面离隔板;手指碰隔板时从高一点松手
CARRY_SLOW = 2.0  # 手里拿着物体的那段整条轨迹按这个倍数放慢(瓶子重、高)
RETREAT = 0.10  # side:松手后先沿接近方向退出这么远再回初始位姿(见 common._pick_place_direct)


def _fridge(env):
    category, index, _, _ = env.get_obj_asset(FRIDGE)
    key = f"{category}/{index:05d}"
    if key not in FRIDGES:
        raise PlanFailError(f"no shelf geometry for {key}; measure it and add it to FRIDGES")
    return FRIDGES[key]


def _door_angle(env):
    lm = env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, FRIDGE))
    q = obj.get_current_joint_positions()
    q = q.detach().cpu().numpy().reshape(-1) if hasattr(q, "detach") else np.asarray(q).reshape(-1)
    return float(q[obj.dof_names.index(DOOR_JOINT)])


def _fridge_boxes(geom):
    """冰箱碰撞体(冰箱坐标系)的轴对齐盒子:机身减去冷藏室最下层(左右壁、隔板以下、上一层隔板以上、后壁)、关着的上门。"""
    (x0, y0, z0), (x1, y1, z1) = geom.shell[0]
    (cx0, cx1), (cz0, cz1) = geom.cavity
    return [
        ((x0, y0, z0), (cx0, y1, z1)),
        ((cx1, y0, z0), (x1, y1, z1)),
        ((cx0, y0, z0), (cx1, y1, cz0)),
        ((cx0, y0, cz1), (cx1, y1, z1)),
        ((cx0, geom.back, cz0), (cx1, y1, cz1)),
        *geom.shell[1:],
    ]


def _door_box(geom, door_angle, fridge):
    """开着的下门(铰链坐标系里的盒子绕冰箱 z 转 ``door_angle``)-> (dims, 世界位姿)。"""
    c, s = np.cos(door_angle), np.sin(door_angle)
    u, n = np.array([-c, -s, 0.0]), np.array([-s, c, 0.0])
    (u0, u1), (n0, n1), (z0, z1) = geom.door
    m = np.eye(4)
    m[:3, :3] = np.stack([u, n, np.cross(u, n)], axis=1)  # 第三列是 -z:盒子对称,方向无所谓
    m[:3, 3] = [*geom.hinge, 0.0] + (u0 + u1) / 2 * u + (n0 + n1) / 2 * n + [0.0, 0.0, (z0 + z1) / 2]
    w = fridge @ m
    return [u1 - u0, n1 - n0, z1 - z0], common._pose(w)


def _grasps(env, spec, inward):
    """GraspGenX 抓取候选(见模块说明第 1 条),按优先级。"""
    label = spec.label
    face = common._bbox_points(env, label)[common.FACE_CENTERS]
    if spec.grasp == "top":
        candidates = env.grasp_candidates(
            label, ARM, prefer=common._down_inward(inward, 45.0), approach_z=TOP_APPROACH_Z, max_candidates=1000
        )
        keep = [(g, sc) for g, sc in candidates if np.ptp(face @ common._mat(g)[:3, 1]) <= common.FINGER_MAX]
    else:
        top = common._bbox_points(env, label)[:, 2].max()
        table_z = common._table_z(env)
        bias = env.robot(ARM).gripper_bias
        candidates = env.grasp_candidates(
            label, ARM, prefer=inward, approach_z=APPROACH_Z, finger_z_max=FINGER_Z_MAX, max_candidates=1000
        )
        keep = []
        for g, score in candidates:
            gm = common._mat(g)
            a = gm[:3, 0]
            tcp = gm[:3, 3] + bias * a
            yaw = abs((common._heading(a) - common._heading(inward) + 180.0) % 360.0 - 180.0)
            # 夹持宽度不超过 FINGER_MAX:沿物体宽的一边夹时,张开的手指贴着物体过去,把它推倒
            if (
                yaw <= spec.max_yaw
                and TCP_Z[0] <= tcp[2] - table_z <= TCP_Z[1]
                and tcp[2] <= top - spec.top_margin
                and np.ptp(face @ gm[:3, 1]) <= common.FINGER_MAX
            ):
                keep.append((g, score))
    if not keep:
        raise PlanFailError(f"{ARM}: no {spec.grasp} grasp for {label} ({len(candidates)} candidates)")
    return keep if spec.ik_prefilter else keep[:MAX_GRASPS]


def _ik_sorted(env, planner, root, grasps, set_grasp_world):
    """``Spec.ik_prefilter``:在去抓取时的碰撞世界里逐个做 IK,有解的候选按原顺序排前面(没解的不丢,排在后面)。"""
    set_grasp_world()
    q0 = np.asarray(env.get_arm_joints(ARM), dtype=float)
    ok = [planner.solve_ik_to_joint(q0, list(g), deepcopy(root))["status"] == "Success" for g, _ in grasps]
    return [c for c, k in zip(grasps, ok) if k] + [c for c, k in zip(grasps, ok) if not k]


def _close(env, spec, grasp):
    """慢慢合拢(同 put_X_toaster):X5 手指驱动很硬,一下合到底会把物体挤歪、挤穿。包围盒沿手指方向比
    ``FINGER_MAX`` 宽时(GraspGenX 夹的是细的地方)包围盒量不出夹持宽度,慢慢合到底。"""
    fingers = common._mat(grasp)[:3, 1]
    width = float(np.ptp(common._bbox_points(env, spec.label)[common.FACE_CENTERS] @ fingers))
    if spec.full_close or width > common.FINGER_MAX:
        env.close_gripper(ARM, 0.0, GENTLE_STEPS)
    else:
        env.close_gripper(ARM, common._grip_val(env, width - spec.squeeze), GENTLE_STEPS)


def _release_poses(spec, geom, fridge, inward, ee, ee_T_obj, local):
    """隔板上的松手位姿候选 ``[ee 位姿, ...]``,见模块说明第 2 条;``local`` 是物体包围盒采样点(物体坐标系)。"""
    obj = ee @ ee_T_obj
    obj_pts = (np.c_[local, np.ones(len(local))] @ obj.T)[:, :3]
    center = obj_pts.mean(axis=0)
    shelf_z = (fridge @ [0.0, 0.0, geom.cavity[1][0], 1.0])[2]
    if spec.grasp == "top":
        rots = [
            common._rot_to(ee[:3, 0], common._rz(extra)[:3, :3] @ common._down_inward(inward, pitch))
            for pitch in INSERT_PITCHES
            for extra in YAW_EXTRA
        ]
    else:
        base_delta = (common._heading(inward) - common._heading(common._horizontal(ee[:3, 0])) + 180.0) % 360.0 - 180.0
        # 先试把手中物体扶正(物体自己的 z 轴转到竖直):细瓶颈处抓起的瓶子在指间歪 10 ~ 20 度,歪着放下松手就倒
        level = common._rot_to(obj[:3, 2], np.array([0.0, 0.0, 1.0]))
        rots = [common._rz(base_delta + extra) @ r for r in (level, np.eye(4)) for extra in YAW_EXTRA]
    fridge_inv = np.linalg.inv(fridge)
    (sx0, sx1), (sy0, sy1) = geom.cavity[0], geom.shelf
    out = []
    for rot in rots:
        for tx, ty in TARGETS:
            goal = (fridge @ [tx, ty, 0.0, 1.0])[:3]
            for gap in RELEASE_GAPS[spec.grasp]:
                m = rot.copy()
                m[:2, 3] = goal[:2] - (rot[:3, :3] @ center)[:2]
                m[2, 3] = shelf_z + gap - (obj_pts @ rot[:3, :3].T)[:, 2].min()
                moved = (np.c_[obj_pts, np.ones(len(obj_pts))] @ m.T)[:, :3]
                pts_f = (np.c_[moved, np.ones(len(moved))] @ fridge_inv.T)[:, :3]
                if (
                    pts_f[:, 0].min() < sx0 + SHELF_MARGIN
                    or pts_f[:, 0].max() > sx1 - SHELF_MARGIN
                    or pts_f[:, 1].min() < sy0 + SHELF_MARGIN
                    or pts_f[:, 1].max() > sy1 - SHELF_MARGIN
                ):
                    continue
                out.append(common._pose(m @ ee))
    if not out:
        raise PlanFailError(f"{spec.label} does not fit on the shelf in any orientation")
    return out


def _play(env, spec):
    label = spec.label
    geom = _fridge(env)
    fridge = common._mat(env.get_obj_pose(FRIDGE))
    inward = common._horizontal(fridge[:3, 1])
    door = _door_angle(env)
    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    root = env.robot(ARM).entity_origin_pose
    fixed = {f"fridge_{i}": common._box(lo, hi, fridge) for i, (lo, hi) in enumerate(_fridge_boxes(geom))}
    fixed["fridge_door"] = _door_box(geom, door, fridge)
    lo, hi = common._bbox(env, label)
    local = common.GRID * (hi - lo) + lo
    open_tool = common._finger_spheres(common.GRIPPER_OPEN, FINGER_SPACING)

    def set_world(with_obj, tool):
        obstacles = dict(fixed)
        if with_obj:
            # 去抓取时、以及松手后沿接近方向退出时("return_tight"),按 Spec.obstacle_shrink 缩小。回程一定要
            # 带着物体:不带时回程路径贴着竖放在隔板上的瓶子扫过去,一蹭就倒
            if with_obj == "return":
                scale = spec.return_grow or 1.0
            else:
                scale = spec.obstacle_shrink
            obstacles[label] = common._obj_box(env, label, scale)
        planner.set_obstacles(obstacles, root)
        planner.set_tool_spheres(tool)

    def held_tool(ee_T_obj):
        held_ee = (np.c_[local, np.ones(len(local))] @ ee_T_obj.T)[:, :3]
        width = min(float(np.ptp(held_ee[:, 1])), common.GRIPPER_OPEN)
        return common._finger_spheres(width, FINGER_SPACING) + common._bbox_spheres(ee_T_obj, lo, hi)

    grasps = _grasps(env, spec, inward)
    if spec.ik_prefilter:
        grasps = _ik_sorted(env, planner, root, grasps, lambda: set_world("grasp", open_tool))
    common._pick_place_direct(
        env,
        label,
        set_world=set_world,
        grasps=grasps[:MAX_GRASPS],
        open_tool=open_tool,
        held_tool=held_tool,
        releases=lambda ee, ee_T_obj: _release_poses(spec, geom, fridge, inward, ee, ee_T_obj, local),
        close=lambda g: _close(env, spec, g),
        carry_slow=spec.carry_slow or CARRY_SLOW,
        retreat=(spec.retreat or RETREAT) if spec.grasp == "side" else 0.0,
        return_fallback=spec.return_grow is not None,
    )


def play_once(env, spec):
    common._run_direct(env, lambda: _play(env, spec))
