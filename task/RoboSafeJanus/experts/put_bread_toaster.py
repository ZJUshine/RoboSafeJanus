"""place_<物体>_toaster 系列任务共用的脚本专家(右臂)。本模块是 put_bread_toaster 的专家,
put_phone_toaster 的专家只是用自己的 ``Spec`` 调用 ``play_once``。

物体竖着插在面包架(``bread_shelf``)的一个槽里,薄的方向沿架子 x;要竖着放进面包机(``toaster``)的一个槽里。

三段运动,每段一次 cuRobo 规划,只在夹爪开合时停(``heat_egg_microwave._pick_place_direct``):
面包机(外壳减两个槽)、面包架(实心,只留物体所在的槽)、物体
放进 cuRobo 的碰撞世界(``CuroboPlanner.set_obstacles``),手指和手中物体用 ``CuroboPlanner.set_tool_spheres`` 补上。

1. 初始 -> 抓取:水平地从机器人一侧抓物体顶边中部。GraspGenX 候选里只留接近方向接近水平(z 分量在 ``APPROACH_Z`` 内)、
   与 ``PREFER`` 夹角不超过 ``MAX_YAW``、手指横跨物体厚度方向(余弦 > ``FINGER_ACROSS``)、TCP 在顶边以下 ``TOP_DEPTH``、
   沿接近方向在宽度的 ``WIDTH_FRAC`` 之间的,再沿手指方向平移到物体厚度中面上;相机朝上的排前面。
   取第一个第 1 段(夹爪张开,物体是障碍物)和预测的第 2 段都能规划的候选。合拢夹爪只合到比物体薄 ``Spec.squeeze``。
2. 抓取 -> 槽里的松手位姿:物体竖直、厚度方向沿槽宽、包围盒中心对准槽中心,手指最低处离面包机顶面 ``FINGER_CLEAR``
   (槽比物体深,手指不进槽);两个槽 x 两个朝向 x 三个高度,取第一个能规划的。物体按实测的 ee->物体 变换挂到夹爪上
   (薄片上按 ``HELD_SPACING`` 排一层碰撞球,半径 ``Spec.held_r``;物体在面包架槽里离铁丝只有 2 mm,起点按需缩小球半径),整条轨迹按 ``CARRY_SLOW`` 放慢。
3. 松开,物体落到槽底;松手位姿 -> 初始关节一次规划(物体作为障碍物;起点就碰到它时去掉再试,再不通规划到初始末端位姿)。
"""

from dataclasses import dataclass

import numpy as np

from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts import heat_egg_microwave as mw

ARM = "right_arm"


@dataclass(frozen=True)
class Spec:
    label: str
    squeeze: float = 0.004  # 夹爪合到比物体包围盒厚度窄这么多(手指驱动刚度 2300 N/m,每指约 4.6 N)
    held_r: float = None  # 手中物体碰撞球半径;None 时取 HELD_R_FRAC x 包围盒半厚度(碰撞体比外观厚的物体要给真实半厚度)


BREAD = Spec("bread")


@dataclass(frozen=True)
class Toaster:
    """面包机碰撞体(面包机坐标系,米),来自碰撞网格的竖直射线(/tmp/hz/toaster/heightmap.py)。"""

    shell: tuple  # 外壳 (lo, hi)
    slots: tuple  # 每个槽的 (x 范围, y 范围);槽从 floor 一直通到顶面以上
    floor: float  # 槽底


TOASTERS = {
    # 拉杆、旋钮都在外壳盒子之内;两个槽长 15.6、宽 2.6、深 12.9 cm。toaster_boxslot 是同一个面包机,
    # 机身碰撞体换成了盒子(ext_assets/integrate/toaster_boxslot.py),尺寸相同
    "toaster/00000": Toaster(
        shell=((-0.1115, -0.0783, -0.0817), (0.1152, 0.0772, 0.0802)),
        slots=(((-0.078, 0.078), (-0.039, -0.013)), ((-0.078, 0.078), (0.013, 0.039))),
        floor=-0.049,
    ),
}
TOASTERS["toaster_boxslot/00000"] = TOASTERS["toaster/00000"]

# 抓取
APPROACH_Z = (-0.4, 0.4)
PREFER = (0.0, 1.0, 0.0)  # 偏好的接近方向:从机器人一侧向前
MAX_YAW = 25.0  # 接近方向水平投影与 PREFER 的最大夹角:斜着(30 度以上)夹时抬起中物体绕指尖转、滑掉
FINGER_ACROSS = 0.95  # 手指与物体厚度方向夹角余弦的下限:斜着夹时一合拢就把吐司片拧歪
TOP_DEPTH = (0.015, 0.05)  # TCP 在物体顶边以下的深度
WIDTH_FRAC = (0.35, 0.65)  # TCP 沿接近方向在物体宽度上的位置(0 = 靠机器人的边)
MAX_GRASPS = 40
GENTLE_STEPS = 150
# 放入
FINGER_CLEAR = (
    0.01,
    0.025,
    0.045,
)  # 松手时手指最低处离面包机顶面:低的放不下(夹爪 / 相机比槽宽,碰到顶面)就从高一点松手
CARRY_CLEAR = 0.02  # 手指离得太低时物体底面至少高出面包机顶面这么多再松手
CARRY_SLOW = 2.0  # 手里拿着物体的那段整条轨迹按这个倍数放慢(时间上统一拉长,不分段)
HELD_R_FRAC = 0.8  # 手中物体碰撞球半径 = 这个比例 x 物体半厚度(Spec.held_r 未给时)
# 手中物体碰撞球在厚度中面上的间距:15 cm 长的手机按 2.4 cm 排时斜着进槽,球之间的空当擦到槽口,手机在指间被拧歪 20 多度
HELD_SPACING = 0.012
OBJ_SHRINK_XY = 0.95  # 物体当障碍物时长宽方向缩小这么多(手指 / 掌心抓取时本来就贴着它)
# 面包架碰撞体(面包架坐标系)。仿真里用的是原资产 bread_shelf/00000,碰撞体是整个网格的 convexDecomposition:凸包把铁丝
# 之间的空隙填上了一部分(只建铁丝时 cuRobo 让手指钻进物体旁边的空隙,一到抓取位姿物理就炸)。所以
# 整个面包架(含两端把手)按实心盒子算到铁丝圈顶面,只把物体所在的槽(夹着它的两个铁丝圈之间)留空,物体从槽里竖直抽出
_LOOPS_X = ((-0.0566, -0.0466), (-0.0296, -0.0206), (-0.0046, 0.0044), (0.0204, 0.0294), (0.0464, 0.0565))
SHELF_LO = (-0.0792, -0.0417, -0.0381)
SHELF_TOP = 0.0381  # 铁丝圈顶梁上表面
SHELF_Y1 = 0.0419
SLOT_MARGIN = 0.002  # 留空的槽在物体两侧再让出这么多
# 手指碰撞球铺满的间距(见 heat_egg_microwave._finger_spheres):面包架铁丝只有 4 ~ 9 mm 粗
FINGER_SPACING = 0.008  # 球半径 6 mm:间距 <= 6 x sqrt(2) mm 时球面之间没有空隙
HELD_START_CLEAR = 0.001  # 刚抓起时手中物体碰撞球离铁丝至少这么远(物体斜靠在铁丝上时按需缩小球半径)
HELD_R_MIN = 0.0015
# 手指闭合在物体上时的采样点(ee_link 坐标系):x 沿接近方向,y 在物体两侧 GRIP_Y 之间,z 半厚度同 mw.FINGER_PROFILE
GRIP_Y = (0.002, 0.012)


def _finger_pts(width):
    """夹住厚 ``width`` 的物体时,两根手指的采样点。"""
    ys = [s * (width / 2 + d) for s in (-1, 1) for d in GRIP_Y]
    return np.array([[x, y, z] for x, h in mw.FINGER_PROFILE for y in ys for z in (-h, h)])


def _grip_val(env, gap):
    """两指间距 ``gap``(米)对应的归一化夹爪开度。每根手指的关节值是间距的一半,开度 v 对应关节值
    ``v * (scale[1] - scale[0]) + scale[0]``(X5:scale = [-0.01, 0.044],v = 0 时把手指往 -1 cm 压,即用满力夹紧)。"""
    lo, hi = env.robot(ARM).gripper_scale
    return float(np.clip((gap / 2 - lo) / (hi - lo), 0.0, 1.0))


def _toaster(env):
    category, index, _, _ = env.get_obj_asset("toaster")
    key = f"{category}/{index:05d}"
    if key not in TOASTERS:
        raise PlanFailError(f"no slot geometry for {key}; measure it and add it to TOASTERS")
    return TOASTERS[key]


def _upright_axes(m):
    """物体坐标系的 x / y 轴里朝上(最接近世界 +z)的那根的下标和符号(物体 z 是厚度方向,竖插时水平)。"""
    up = int(np.argmax(np.abs(m[2, :2])))
    return up, float(np.sign(m[2, up]))


def _grasp_candidates(env, spec):
    """顶边中部的水平侧抓候选(见模块说明第 1 条),已平移到物体厚度中面上,相机朝上的排前面。"""
    label = spec.label
    m = mw._mat(env.get_obj_pose(label))
    thick = m[:3, 2]
    pts = mw._bbox_points(env, label)
    top = pts[:, 2].max()
    mid = mw._center(env, label)  # 包围盒中心:在厚度中面上
    bias = env.robot(ARM).gripper_bias
    candidates = env.grasp_candidates(label, ARM, prefer=PREFER, approach_z=APPROACH_Z, max_candidates=1000)
    keep = []
    for g, score in candidates:
        gm = mw._mat(g)
        a = gm[:3, 0]
        tcp = gm[:3, 3] + bias * a
        along = pts @ a
        frac = (tcp @ a - along.min()) / np.ptp(along)
        if (
            a[:2] @ PREFER[:2] / np.linalg.norm(a[:2]) > np.cos(np.radians(MAX_YAW))
            and abs(gm[:3, 1] @ thick) > FINGER_ACROSS
            and TOP_DEPTH[0] <= top - tcp[2] <= TOP_DEPTH[1]
            and WIDTH_FRAC[0] <= frac <= WIDTH_FRAC[1]
        ):
            # 沿手指方向平移,让两指的中面落在物体厚度中面上:GraspGenX 的候选偏 2 ~ 5 mm,一合拢就把吐司片上端
            # 推偏;它下半截卡在只比它宽 5 mm 的槽里,被推偏后斜卡在铁丝之间,往上提时从指间滑掉
            finger = gm[:3, 1]
            g = g.copy()
            g[:3] += ((mid - tcp) @ finger) * finger
            keep.append((g, score))
    # 相机朝上(ee z 朝上)的排前面:相机支架装在腕部 ee -z 一侧,朝下时竖直放进槽的最后几厘米相机碰到面包机顶面
    keep.sort(key=lambda c: mw._mat(c[0])[2, 2] < 0)
    if not keep:
        raise PlanFailError(f"{ARM}: no top-edge side grasp for {label} ({len(candidates)} side grasps)")
    return keep[:MAX_GRASPS]


def _toaster_boxes(geom):
    """面包机碰撞体(面包机坐标系):外壳减去两个槽 = 槽两侧和中间的长条、槽两端、槽底以下。"""
    (x0, y0, z0), (x1, y1, z1) = geom.shell
    (sx0, sx1), _ = geom.slots[0]
    ys = sorted(geom.slots, key=lambda s: s[1][0])
    edges = [y0] + [v for _, (a, b) in ys for v in (a, b)] + [y1]
    boxes = [((x0, edges[i], z0), (x1, edges[i + 1], z1)) for i in range(0, len(edges), 2)]
    for _, (a, b) in ys:
        boxes += [((x0, a, z0), (sx0, b, z1)), ((sx1, a, z0), (x1, b, z1)), ((sx0, a, z0), (sx1, b, geom.floor))]
    return boxes


def _slab_spheres(ee_T_obj, lo, hi, r=None):
    """手中薄片物体(包围盒 lo / hi,物体坐标系,厚度沿物体 z)在厚度中面上按 ``HELD_SPACING`` 排一层半径 ``r`` 的碰撞球,
    ee_link 坐标系。"""
    if r is None:
        r = HELD_R_FRAC * float(hi[2] - lo[2]) / 2

    def axis(a, b):
        return np.linspace(a + r, b - r, max(2, int(np.ceil((b - a - 2 * r) / HELD_SPACING)) + 1))

    xs, ys = axis(lo[0], hi[0]), axis(lo[1], hi[1])
    z = (lo[2] + hi[2]) / 2
    return [[*(ee_T_obj @ [x, y, z, 1.0])[:3], r] for x in xs for y in ys]


def _box_dist(pts, lo, hi):
    """点到轴对齐盒子(同一坐标系)的有符号距离,盒内为负。"""
    lo, hi = np.asarray(lo), np.asarray(hi)
    d_out = np.linalg.norm(np.maximum(np.maximum(lo - pts, pts - hi), 0.0), axis=1)
    d_in = np.minimum(np.min(pts - lo, axis=1), np.min(hi - pts, axis=1))
    return np.where(d_out > 0, d_out, -d_in)


def _shelf_boxes(env, label):
    """面包架碰撞盒(面包架坐标系):物体所在的槽两侧各一块实心盒子(见 SHELF_LO 处的说明);留空的范围是夹着物体的
    两个铁丝圈之间,再并上物体自己在铁丝圈顶面以下占的范围(两侧各让 ``SLOT_MARGIN``)。"""
    shelf = mw._mat(env.get_obj_pose("bread_shelf"))
    mid_x = float((np.linalg.inv(shelf) @ np.append(mw._center(env, label), 1.0))[0])
    inner = [x1 for _, x1 in _LOOPS_X if x1 <= mid_x], [x0 for x0, _ in _LOOPS_X if x0 >= mid_x]
    if not inner[0] or not inner[1]:
        raise PlanFailError(f"{label} is not between two shelf wire loops (shelf-frame x {mid_x:.3f})")
    gap_lo, gap_hi = max(inner[0]), min(inner[1])
    # 再加上物体自己(铁丝圈顶面以下的部分)占的范围:斜靠着的物体(15 cm 高的手机)下半截伸进了旁边的空隙
    pts = mw._bbox_points(env, label)
    pts = (np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(shelf).T)[:, :3]
    pts = pts[pts[:, 2] <= SHELF_TOP]
    if len(pts):
        gap_lo = min(gap_lo, float(pts[:, 0].min()) - SLOT_MARGIN)
        gap_hi = max(gap_hi, float(pts[:, 0].max()) + SLOT_MARGIN)
    x0, y0, z0 = SHELF_LO
    return [((x0, y0, z0), (gap_lo, SHELF_Y1, SHELF_TOP)), ((gap_hi, y0, z0), (-x0, SHELF_Y1, SHELF_TOP))]


def _fit_held(held, ee_world, shelf, boxes):
    """手中物体还在面包架槽里(离铁丝只有 2 mm,常斜靠着一根):按需统一缩小碰撞球,让起点离铁丝至少
    ``HELD_START_CLEAR``,否则 cuRobo 判起点碰撞。返回缩小后的球和缩小量。"""
    held = np.asarray(held, dtype=float)
    pts = (np.c_[held[:, :3], np.ones(len(held))] @ (np.linalg.inv(shelf) @ ee_world).T)[:, :3]
    gap = min(float((_box_dist(pts, lo, hi) - held[:, 3]).min()) for lo, hi in boxes)
    shrink = max(0.0, HELD_START_CLEAR - gap)
    held[:, 3] = np.maximum(held[:, 3] - shrink, HELD_R_MIN)
    return held.tolist(), shrink


def _obj_box(env, label):
    lo, hi = mw._bbox(env, label)
    c, h = (lo + hi) / 2, (hi - lo) / 2 * np.array([OBJ_SHRINK_XY, OBJ_SHRINK_XY, 1.0])
    return mw._box(c - h, c + h, mw._mat(env.get_obj_pose(label)))


def _target_obj(m_obj, slot_axes, flip):
    """手中物体放进槽时的姿态:竖直、厚度轴(物体 z)沿槽宽 ``slot_axes[1]``(取与当前相近的符号,``flip`` 时取反)。"""
    up_idx, up_sign = _upright_axes(m_obj)
    width = slot_axes[1] * (1.0 if m_obj[:3, 2] @ slot_axes[1] >= 0 else -1.0)
    if flip:
        width = -width
    R = np.zeros((3, 3))
    R[:, 2] = width
    R[:, up_idx] = up_sign * np.array([0.0, 0.0, 1.0])
    other = 1 - up_idx
    # 右手系:x = y x z 或 y = z x x
    R[:, other] = np.cross(R[:, 1], R[:, 2]) if other == 0 else np.cross(R[:, 2], R[:, 0])
    return R


def _release_poses(env, geom, toaster, toaster_top, obj, ee_T_obj, lo, hi, finger_pts):
    """槽里的松手位姿候选(ee_link 位姿,依次槽 x 朝向 x 高度):物体竖直、厚度沿槽宽、包围盒中心对准槽中心,手指最低处离面包机顶面
    ``FINGER_CLEAR``(物体底面至少高出顶面 ``CARRY_CLEAR``,物体不够长时就从这个高度松手)。"""
    local = mw.GRID * (hi - lo) + lo
    out = []
    for (x0, x1), (y0, y1) in geom.slots:
        slot_c = toaster @ [(x0 + x1) / 2, (y0 + y1) / 2, 0.0, 1.0]
        for flip in (False, True):
            R_obj = _target_obj(obj, (toaster[:3, 0], toaster[:3, 1]), flip)
            T = np.eye(4)
            T[:3, :3] = R_obj
            T[:2, 3] = slot_c[:2] - (R_obj @ ((lo + hi) / 2))[:2]
            T[2, 3] = toaster_top + CARRY_CLEAR - (local @ R_obj.T)[:, 2].min()
            ee_rel = T @ np.linalg.inv(ee_T_obj)
            tip_z = float((finger_pts @ ee_rel[:3, :3].T)[:, 2].min() + ee_rel[2, 3])
            for finger_clear in FINGER_CLEAR:
                m = ee_rel.copy()
                m[2, 3] -= max(0.0, tip_z - (toaster_top + finger_clear))
                out.append(mw._pose(m))
    return out


def _play(env, spec):
    label = spec.label
    geom = _toaster(env)
    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    root = env.robot(ARM).entity_origin_pose
    toaster = mw._mat(env.get_obj_pose("toaster"))
    toaster_top = (toaster @ [0.0, 0.0, geom.shell[1][2], 1.0])[2]
    shelf = mw._mat(env.get_obj_pose("bread_shelf"))
    fixed = {f"toaster_{i}": mw._box(lo, hi, toaster) for i, (lo, hi) in enumerate(_toaster_boxes(geom))}
    shelf_boxes = []

    def refresh_shelf():
        """面包架留空的槽按物体当前的位姿算:合拢夹爪时手指会把斜靠着的物体拧一下,下半截偏到旁边的空隙里。"""
        shelf_boxes[:] = _shelf_boxes(env, label)
        fixed.update({f"shelf_{i}": mw._box(lo, hi, shelf) for i, (lo, hi) in enumerate(shelf_boxes)})

    refresh_shelf()
    lo, hi = mw._bbox(env, label)
    width = float(hi[2] - lo[2])
    closed = mw._finger_spheres(width - spec.squeeze, FINGER_SPACING)

    def set_world(with_obj, tool):
        obstacles = dict(fixed)
        if with_obj:
            obstacles[label] = _obj_box(env, label)
        planner.set_obstacles(obstacles, root)
        planner.set_tool_spheres(tool)

    def held_tool(ee_T_obj):
        # 第 2 段的起点物体还在面包架的槽里(这时它的实测位姿就是起点位姿)
        ee_start = mw._mat(env.get_obj_pose(label)) @ np.linalg.inv(ee_T_obj)
        held, _ = _fit_held(_slab_spheres(ee_T_obj, lo, hi, spec.held_r), ee_start, shelf, shelf_boxes)
        return closed + held

    def releases(ee, ee_T_obj):
        return _release_poses(env, geom, toaster, toaster_top, ee @ ee_T_obj, ee_T_obj, lo, hi, _finger_pts(width))

    mw._pick_place_direct(
        env,
        label,
        set_world=set_world,
        grasps=_grasp_candidates(env, spec),
        open_tool=mw._finger_spheres(mw.GRIPPER_OPEN, FINGER_SPACING),
        held_tool=held_tool,
        releases=releases,
        # 只合到比物体薄 spec.squeeze、并且慢慢合:X5 的手指驱动很硬,一下合到底时两指压进 1.2 cm 的吐司片 7 mm
        # (PhysX 接触被挤穿),抬起时吐司片从指间滑掉
        close=lambda g: env.close_gripper(ARM, _grip_val(env, width - spec.squeeze), GENTLE_STEPS),
        carry_slow=CARRY_SLOW,
        after_close=refresh_shelf,
    )


def play_once(env, spec=BREAD):
    mw._run_direct(env, lambda: _play(env, spec))
