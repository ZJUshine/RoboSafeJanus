"""put_phone_phone_stand / put_phone_cup 共用的脚本专家(右臂)。本模块是 put_phone_phone_stand 的专家,
put_phone_cup 的专家只是用自己的 ``Spec`` 调用 ``play_once``。

手机(RoboTwin_077_phone_hull/00001,碰撞体 15.1 x 6.4 x 1.4 cm)屏幕朝上平放在桌上,长边大致沿世界 y。

三段运动,每段一次 cuRobo 规划,只在夹爪开合时停(``heat_egg_microwave._pick_place_direct``):初始 -> 抓取 -> 合拢 ->
松手位姿 -> 松开 -> 沿接近方向退出 ``RETREAT`` -> 回初始。支架的碰撞盒子(同 phone_stand_box 变体)/ 杯壁(一圈斜放的
薄盒子 + 杯底)放进 cuRobo 的碰撞世界,手指和手中的手机用 tool 碰撞球补上。

1. 抓取:GraspGenX 的自上而下抓取(缓存里补了一批,见 data/grasp_pose/.../RoboTwin_077_phone_hull/00001.npz 的 meta),
   手指横跨手机宽度,TCP 在手机一端(手机坐标系 |x| 在 ``END_X`` 内):放好后夹住的这一端朝上,手指在支架靠背 / 杯沿
   以上。沿手指方向平移到手机宽度中面上,再沿接近方向移到指尖离桌面 ``TIP_CLEAR``(只用指尖夹住
   1.4 cm 厚的手机);靠机器人一端的排前面。慢慢合到比包围盒宽度窄 ``SQUEEZE``。
2. 松手位姿:手机竖起来,夹住的一端朝上。
   - 支架(``STAND``):屏幕朝前(手机 z 沿靠背法向 n)、长边沿靠背向上(u),包围盒中心在靠在靠背上、底边落在托架上的
     位置(``REST_N`` / ``REST_U``,任务里也按它判定)再往前 ``GAPS_N``、往上 ``GAPS_U``;夹爪接近方向 = -n,水平偏下。
   - 杯子(``CUP``):长边竖直,包围盒中心对准杯子轴线,底边离杯底 ``CUP_GAPS``;接近方向水平,朝向依次取 ``HEADINGS``。
3. 松开(手机落到托架上靠住靠背 / 落进杯底),沿接近方向退出,回初始位姿。
"""

from dataclasses import dataclass

import numpy as np

from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts import heat_egg_microwave as mw, put_bread_toaster as toast
from task.RoboDojo.tasks.put_phone_phone_stand import LEDGE, REST_N, REST_U, stand_axes

ARM = "right_arm"
PHONE = "phone"


@dataclass(frozen=True)
class Spec:
    target: str  # "stand" | "cup"
    label: str


STAND = Spec("stand", "phone_stand")

# 抓取
APPROACH_Z = (-1.0, -0.85)
FINGER_ACROSS = 0.9  # 手指方向与手机宽度方向(手机 y)夹角余弦的下限
END_X = (0.035, 0.062)  # TCP 在手机坐标系的 |x|:离中心至少这么远,手指才在靠背顶(手机底边以上 9.2 cm)以上
# 指尖最低处离桌面:手机碰撞体顶面离桌面 1.28 cm,heat_egg_microwave 的 TOP_TIP_CLEAR(8 mm)时手指只有最后 5 mm 夹在
# 手机边上,一转腕手机就绕指尖转出去(data/debug/place_phone/r1);cuRobo 的桌面比真实桌面低 2.5 cm,不会判碰撞
TIP_CLEAR = 0.003
MAX_GRASPS = 40
GENTLE_STEPS = 150
# 夹爪合到比手机包围盒宽度(6.15 cm)窄这么多:指尖处手指在关节值 0.0308(两指间距 6.16 cm)就碰到手机,合到包围盒宽度
# 时夹紧力为零,一抬手机就从指尖滑出去(data/debug/place_phone/r1、r2);多合 1 ~ 2 cm 时都夹得住(d_sq0.01、d_sq0.02)
SQUEEZE = 0.012
FINGER_SPACING = 0.008
HELD_R = 0.0072  # 手中手机碰撞球半径:碰撞体半厚度(同 put_phone_toaster)
CARRY_SLOW = 2.0
RETREAT = 0.06
COLLIDER_PAD = 0.00135  # 手机碰撞体每侧比包围盒多出这么多
# 支架:松手时手机比靠着靠背、落在托架上的位置再往前 / 往上这么多
GAPS_N = (0.002, 0.004)
GAPS_U = (0.006, 0.012, 0.018)
# 支架碰撞盒子(资产坐标系;靠背 / 托架 / 钩子在靠背坐标系 (n, y, u) 里),同 ext_assets/integrate/phone_stand_box.py
STAND_FIXED = (
    ((-0.0556, -0.033, 0.0004), (0.0555, 0.033, 0.0096)),
    ((-0.0555, -0.011, 0.0096), (-0.043, 0.011, 0.105)),
)
STAND_PLATE = (
    ((-0.004, -0.033, -0.003), (0.0, 0.033, 0.0915)),
    ((-0.004, -0.033, -0.005), (0.020, 0.033, 0.0)),
    ((0.017, -0.033, -0.005), (0.020, -0.017, 0.0055)),
    ((0.017, 0.017, -0.005), (0.020, 0.033, 0.0055)),
)


@dataclass(frozen=True)
class Cup:
    """杯子(杯子坐标系,原点在杯底中心,米):杯壁内表面从 (r, z) ``wall[0]`` 到 ``wall[1]``,杯底内表面高 ``floor``。"""

    wall: tuple
    floor: float


CUPS = {
    # cup/00007 放大 1.4 倍(scaled_variant.py);截面量自原资产的碰撞网格(/tmp/hz/phone/section.py)
    "cup_s140/00007": Cup(wall=((0.0356, 0.003), (0.0480, 0.0994)), floor=0.003),
}
CUP_WALL_T = 0.004
CUP_RING_N = 16
CUP_GAPS = (0.015, 0.03, 0.045)
HEADINGS = (90.0, 120.0, 60.0, 150.0, 30.0)  # 接近方向的水平朝向(度,世界坐标;90 = 从机器人一侧向前)


def _phone_box(env):
    return toast._obj_box(env, PHONE)


def _set_tips(env, g):
    """自上而下的抓取 ``g`` 沿接近方向移动(只改位置),让指尖最低处离桌面 ``TIP_CLEAR``。"""
    m = mw._mat(g)
    a = m[:3, 0]
    tip_z = m[2, 3] + (env.robot(ARM).gripper_bias + mw.TIP_BEYOND_TCP) * a[2]
    g = np.array(g, dtype=float)
    g[:3] += (tip_z - mw._table_z(env) - TIP_CLEAR) / -a[2] * a
    return g


def _grasps(env):
    """自上而下、手指横跨宽度、TCP 在手机一端的 GraspGenX 抓取(见模块说明第 1 条),按优先级。"""
    obj = mw._mat(env.get_obj_pose(PHONE))
    inv = np.linalg.inv(obj)
    mid = mw._center(env, PHONE)
    bias = env.robot(ARM).gripper_bias
    candidates = env.grasp_candidates(PHONE, ARM, prefer=(0.0, 0.0, -1.0), approach_z=APPROACH_Z, max_candidates=1000)
    keep = []
    for g, score in candidates:
        m = mw._mat(g)
        tcp = m[:3, 3] + bias * m[:3, 0]
        if (
            abs(m[:3, 1] @ obj[:3, 1]) < FINGER_ACROSS
            or not END_X[0] <= abs((inv @ np.append(tcp, 1.0))[0]) <= END_X[1]
        ):
            continue
        finger = m[:3, 1]
        g = np.array(g, dtype=float)
        g[:3] += ((mid - tcp) @ finger) * finger
        keep.append(((tcp - mid)[1] > 0, _set_tips(env, g), score))
    if not keep:
        raise PlanFailError(f"{ARM}: no top-down end grasp for {PHONE} ({len(candidates)} top-down candidates)")
    keep.sort(key=lambda k: k[0])  # 靠机器人一端(y 小)的排前面,其余保持 grasp_candidates 的顺序
    return [(g, s) for _, g, s in keep[:MAX_GRASPS]]


def _up_sign(ee_T_obj, bias):
    """夹住的是手机的 +x 端(1)还是 -x 端(-1):这一端放好后朝上。"""
    tcp = np.linalg.inv(ee_T_obj) @ [bias, 0.0, 0.0, 1.0]
    return 1.0 if tcp[0] >= 0 else -1.0


def _frame(x, z):
    R = np.eye(4)
    R[:3, 0], R[:3, 2] = x, z
    R[:3, 1] = np.cross(z, x)
    return R


def _stand_boxes(stand):
    n, u = stand_axes()
    plate = np.eye(4)
    plate[:3, 0], plate[:3, 2] = n, u
    plate[:3, 3] = [LEDGE[0], 0.0, LEDGE[1]]
    boxes = {f"stand_{i}": mw._box(lo, hi, stand) for i, (lo, hi) in enumerate(STAND_FIXED)}
    boxes.update({f"stand_plate_{i}": mw._box(lo, hi, stand @ plate) for i, (lo, hi) in enumerate(STAND_PLATE)})
    return boxes


def _stand_releases(stand, ee_T_obj, sign, c_local):
    n, u = stand_axes()
    n_w, u_w = stand[:3, :3] @ n, stand[:3, :3] @ u
    ledge = (stand @ [LEDGE[0], 0.0, LEDGE[1], 1.0])[:3]
    T = _frame(sign * u_w, n_w)
    out = []
    for du in GAPS_U:
        for dn in GAPS_N:
            T[:3, 3] = ledge + (REST_N + dn) * n_w + (REST_U + du) * u_w - T[:3, :3] @ c_local
            out.append(mw._pose(T @ np.linalg.inv(ee_T_obj)))
    return out


def _cup(env, label):
    category, index, _, _ = env.get_obj_asset(label)
    key = f"{category}/{index:05d}"
    if key not in CUPS:
        raise PlanFailError(f"no wall geometry for cup {key}; measure it and add it to CUPS")
    return CUPS[key]


def _cup_boxes(cup, geom):
    """杯壁(``CUP_RING_N`` 个沿内表面斜放的薄盒子,向外厚 ``CUP_WALL_T``)和杯底 -> {名字: (dims, 世界位姿)}。"""
    (r0, z0), (r1, z1) = geom.wall
    length = float(np.hypot(r1 - r0, z1 - z0))
    width = 2 * r1 * np.tan(np.pi / CUP_RING_N) * 1.1
    out = {}
    for i in range(CUP_RING_N):
        th = 2 * np.pi * i / CUP_RING_N
        radial, t = np.array([np.cos(th), np.sin(th), 0.0]), np.array([-np.sin(th), np.cos(th), 0.0])
        s_dir = ((r1 - r0) * radial + np.array([0.0, 0.0, z1 - z0])) / length
        f = np.eye(4)
        f[:3, :3] = np.stack([s_dir, -t, np.cross(s_dir, -t)], axis=1)  # z = 杯壁法向朝外
        f[:3, 3] = (r0 + r1) / 2 * radial + [0.0, 0.0, (z0 + z1) / 2]
        h = np.array([length / 2, width / 2, CUP_WALL_T / 2])
        out[f"cup_wall_{i}"] = mw._box(-h + [0.0, 0.0, CUP_WALL_T / 2], h + [0.0, 0.0, CUP_WALL_T / 2], cup @ f)
    out["cup_floor"] = mw._box((-r0, -r0, 0.0), (r0, r0, geom.floor), cup)
    return out


def _cup_releases(cup, geom, ee_T_obj, sign, c_local, half_len):
    axis = (cup @ [0.0, 0.0, geom.floor, 1.0])[:3]
    out = []
    for gap in CUP_GAPS:
        for h in HEADINGS:
            a = np.array([np.cos(np.radians(h)), np.sin(np.radians(h)), 0.0])
            T = _frame(np.array([0.0, 0.0, sign]), -a)  # 屏幕朝着夹爪,夹住的一端朝上
            T[:3, 3] = axis + [0.0, 0.0, gap + half_len] - T[:3, :3] @ c_local
            out.append(mw._pose(T @ np.linalg.inv(ee_T_obj)))
    return out


def _play(env, spec):
    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    root = env.robot(ARM).entity_origin_pose
    bias = env.robot(ARM).gripper_bias
    target = mw._mat(env.get_obj_pose(spec.label))
    lo, hi = mw._bbox(env, PHONE)
    c_local = (lo + hi) / 2
    width = float(hi[1] - lo[1])
    if spec.target == "stand":
        if target[:3, 0] @ [0.0, -1.0, 0.0] < 0.9:
            # 支架正面(资产 +x)必须朝向机器人;YAML 里的 rotate_deg 改了时这里先报错
            raise PlanFailError(f"{spec.label} front (asset +x) does not face the robot: {target[:3, 0].round(3)}")
        fixed = _stand_boxes(target)

        def releases(ee, ee_T_obj):
            return _stand_releases(target, ee_T_obj, _up_sign(ee_T_obj, bias), c_local)

    else:
        geom = _cup(env, spec.label)
        fixed = _cup_boxes(target, geom)
        half_len = float(hi[0] - lo[0]) / 2 + COLLIDER_PAD

        def releases(ee, ee_T_obj):
            return _cup_releases(target, geom, ee_T_obj, _up_sign(ee_T_obj, bias), c_local, half_len)

    def set_world(with_obj, tool):
        planner.set_obstacles({**fixed, PHONE: _phone_box(env)} if with_obj else dict(fixed), root)
        planner.set_tool_spheres(tool)

    closed = mw._finger_spheres(width, FINGER_SPACING)
    mw._pick_place_direct(
        env,
        PHONE,
        set_world=set_world,
        grasps=_grasps(env),
        open_tool=mw._finger_spheres(mw.GRIPPER_OPEN, FINGER_SPACING),
        held_tool=lambda ee_T_obj: closed + toast._slab_spheres(ee_T_obj, lo, hi, HELD_R),
        releases=releases,
        close=lambda g: env.close_gripper(ARM, toast._grip_val(env, width - SQUEEZE), GENTLE_STEPS),
        carry_slow=CARRY_SLOW,
        retreat=RETREAT,
        retreat_fallback=True,
    )


def play_once(env, spec=STAND):
    mw._run_direct(env, lambda: _play(env, spec))
