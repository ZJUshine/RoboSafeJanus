"""heat_<餐具>_microwave 系列(刀、叉、勺、锅、碗、杯、盘)共用的脚本专家(右臂;物体在微波炉前方,门开在左侧)。
本模块是 heat_metal_knife_microwave 的专家,其余任务只是用自己的 ``Spec`` 调用 ``play_once``。

与 heat_egg_microwave 的"自上而下抓再转平、手伸进炉腔"不同,这里抓物体靠近机器人的一端,物体在前、夹爪在后
被送进炉腔,手留在炉口外:

1. 抓取:接近方向自上而下、朝"抓取点 -> 物体中心"方向倾 ``Spec.tilts`` 度。送入时夹爪要倾 30 ~ 45 度:夹爪竖直时
   右臂 link3 碰到炉口上沿时 TCP 还在炉口外 9 ~ 10 cm,倾 30 度能到炉口平面,倾 45 度能伸进 3 ~ 4 cm
   (/tmp/hz/tw/push_sweep.py 离线扫描)。倾斜的夹爪只有指尖的一个角最低(TCP 下方约 1.6 cm),贴着桌面夹薄物体时
   只有这个角碰到物体,一合就把刀、叉挤飞;这类物体竖直抓(``tilts=(0,)``),抬起后再朝炉腔倾 ``Spec.carry_tilts``
   (物体前端随之抬起,不转平)。
   - ``"handle"``(刀、叉、勺、锅):TCP 在 ``Spec.handles`` 里离机器人最近的那个点(物体坐标系),手指水平、横跨把手;
     薄的物体指尖贴着桌面(``Spec.tip_clear``)。手指横跨时物体绕手指连线的转动只靠指垫的扭转摩擦挡住,
     所以长餐具抓在柄上靠近质心处,不抓柄端。把手截面是圆鼓的,一下合到底会把它像西瓜子一样挤出去,所以给了
     ``Spec.width`` 的物体慢慢合到比它窄 ``SQUEEZE``。
   - ``"rim"``(碗、杯、盘):TCP 在物体边沿离机器人最近处(包围盒半径减 ``Spec.rim_in``,边沿顶面下 ``Spec.depth``),
     手指沿"朝炉腔里"方向夹住边沿,里面的手指在物体内侧。
2. 竖直抬起,绕竖直轴转到抓取点 -> 物体中心的方向正对炉腔(并倾到 ``carry_tilts``),停在炉口外,再沿炉腔方向直线送进去:物体最低点离转盘
   ``GAPS``,物体中心横向对准转盘中心,TCP 停在炉口平面里外 ``DEPTHS`` 处(取第一个可行的,越深越好),要求物体中心
   伸进炉口至少 ``MIN_IN``、最里面离后壁至少 ``BACK_CLEAR``。
3. 松开夹爪,沿接近方向退一点 / 竖直抬起,再水平退出,右臂回到初始位姿。失败时也松开夹爪、回到初始位姿再抛出异常。

每条候选路径都先规划,再用 heat_egg_microwave 的 ``_Clearance``(cuRobo 碰撞球 + 手指 + 手中物体采样点,对微波炉
外壳减去炉腔)检查。
"""

from dataclasses import dataclass
import logging

import numpy as np
import transforms3d as t3d

from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts.heat_egg_microwave import (
    ARM,
    FINGER_PTS,
    GRIPPER_OPEN,
    GraspLostError,
    _bbox_points,
    _center,
    _choose_clear,
    _Clearance,
    _horizontal,
    _inward,
    _mat,
    _table_z,
    _tcp_dist,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Spec:
    """一个物体的抓取方式,见模块说明第 1 条。"""

    label: str
    grip: str  # "handle" | "rim"
    handles: tuple = ()  # handle:候选抓取点(物体坐标系),取离机器人最近的
    forward: tuple = None  # handle:物体坐标系里从把手指向另一头的轴;为空时用"抓取点 -> 包围盒中心"
    tilts: tuple = (45.0, 30.0)  # 接近方向朝物体中心倾斜的角度(度),按优先级
    tip_clear: float = 0.002  # 指尖离桌面的最小高度
    depth: float = 0.015  # rim:TCP 在边沿顶面下这么深
    rim_in: float = 0.004  # rim:TCP 在包围盒边缘往里这么多(边沿壁厚的一半)
    carry_tilts: tuple = None  # 送入时夹爪的倾角(度),按优先级;为空时保持抓取时的倾角
    open: float = 1.0  # 抓取前夹爪张开的归一化开度(盘子:里面的手指不压到盘面)
    width: float = None  # 手指之间的物体宽度(米);给了就慢慢合到比它窄 SQUEEZE,不合到底


# 抓在质心(网格体积质心 z -0.047,刀柄比刀刃重)往刀柄一侧 1.3 cm 处:物体中心(包围盒中心)离 TCP 6 cm,送得进炉腔
KNIFE = Spec(
    "metal_knife",
    "handle",
    handles=((0.001, 0.0, -0.06),),
    forward=(0.0, 0.0, 1.0),
    tilts=(0.0,),
    carry_tilts=(45.0, 30.0),
    width=0.016,
)

# 抓取
PRE_GRASP = 0.08
GRASP_SETTLE = 30
SQUEEZE = 0.006  # width:合到比物体窄这么多
GENTLE_STEPS = 150
LIFT = (0.10, 0.08, 0.06)
SLIP_TOL = 0.02  # TCP 到物体中心的距离比刚抓好时变大超过它算抓丢
# 送入
GAPS = (0.012, 0.02, 0.03)  # 物体最低的采样点离转盘的高度
DEPTHS = (0.05, 0.04, 0.03, 0.02, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04)  # TCP 伸过炉口平面的深度(负数:在炉口外)
MIN_IN = 0.03  # 物体中心至少伸进炉口这么深(碗的底座此时整个落在炉腔里)
BACK_CLEAR = 0.015  # 物体最里面离炉腔后壁的最小距离
START_OUT = 0.03  # 直线送入的起点:物体最里面在炉口外这么远
INSERT_SLOW = 1.5
# 退出
BACK_OFF = (0.05, 0.03, 0.0)  # 松手后先沿接近方向后退的距离
LIFTS_OUT = (0.03, 0.015, 0.0)  # 之后竖直抬起的高度
RETREAT = 0.15


def play_once(env, spec=KNIFE):
    try:
        _play(env, spec)
    except Exception:
        # 同 heat_egg_microwave:手里拿着东西、停在别处结束的 episode 之后,同一进程的下一次 reset 会让机器人位姿变成 NaN
        try:
            env.open_gripper(ARM)
            env.back_to_origin(ARM)
        except Exception as e:
            log.warning("cleanup after a failed attempt failed: %s: %s", type(e).__name__, e)
        raise


def _rot(toward, tilt, finger):
    """ee 姿态:x(接近方向)= 竖直向下、朝 ``toward``(水平)倾 ``tilt`` 度;y(手指)取 ``finger`` 垂直于 x 的分量。"""
    t = np.radians(tilt)
    x = np.sin(t) * toward + np.cos(t) * np.array([0.0, 0.0, -1.0])
    y = finger - (finger @ x) * x
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _pose(tcp, rot, bias):
    return np.concatenate([tcp - bias * rot[:, 0], t3d.quaternions.mat2quat(rot)])


def _grasp_candidates(env, spec, inward):
    """``(ee 位姿, 抓取点 -> 物体中心的水平方向)`` 的候选,按优先级排序。"""
    bias = env.robot(ARM).gripper_bias
    up = np.array([0.0, 0.0, 1.0])
    table = _table_z(env)
    c = _center(env, spec.label)
    top = _bbox_points(env, spec.label)[:, 2].max()
    if spec.grip == "handle":
        m = _mat(env.get_obj_pose(spec.label))
        pts = [(m @ np.append(p, 1.0))[:3] for p in spec.handles]
        p = min(pts, key=lambda q: (q - c) @ inward)  # 离机器人最近(最靠炉腔外)的把手
        toward = _horizontal(m[:3, :3] @ spec.forward if spec.forward is not None else c - p)
        finger = np.cross(up, toward)
    else:
        radius = float(np.ptp(_bbox_points(env, spec.label)[:, :2] @ inward[:2])) / 2
        toward = inward
        p = c - (radius - spec.rim_in) * inward
        p[2] = top - spec.depth
        finger = inward
    out = []
    for tilt in spec.tilts:
        for f in (finger, -finger):
            rot = _rot(toward, tilt, f)
            # 手指最低点(倾斜时是指尖的一个角,在 TCP 下方约 1.6 cm)离桌面不低于 tip_clear
            low = float(((FINGER_PTS - [bias, 0.0, 0.0]) @ rot.T)[:, 2].min())
            tcp = p.copy()
            tcp[2] = max(tcp[2], table + spec.tip_clear - low)
            out.append((_pose(tcp, rot, bias), toward))
    return out


def _grasp(env, spec, inward, clearance):
    candidates = _grasp_candidates(env, spec, inward)
    chains = []
    for g, _ in candidates:
        approach = _mat(g)[:3, 0]
        pre = g.copy()
        pre[:3] -= PRE_GRASP * approach
        chains.append([pre, *env._straight_poses(pre, g[:3] - pre[:3], 0.02)])
    i, plans = _choose_clear(env, clearance, chains, "grasp", table="body")
    env.open_gripper(ARM, spec.open)
    for res in plans:
        env.execute_plan(ARM, res)
    if spec.width is None:
        env.close_gripper(ARM)
    else:
        env.close_gripper(ARM, max(0.0, (spec.width - SQUEEZE) / GRIPPER_OPEN), GENTLE_STEPS)
    env.delay(GRASP_SETTLE)
    return candidates[i]


def _play(env, spec):
    label = spec.label
    mw, k, geom, inward = _inward(env)
    clearance = _Clearance(env, mw, geom)
    grasp, toward = _grasp(env, spec, inward, clearance)
    held0 = _tcp_dist(env, label)
    tilt = np.degrees(np.arccos(np.clip(-_mat(grasp)[2, 0], -1, 1)))
    log.info(
        "%s %s grasp at %s, tilt %.0f deg; tcp-%s %.3f after closing",
        label,
        spec.grip,
        np.round(grasp[:3], 3).tolist(),
        tilt,
        label,
        held0,
    )

    def check_held(stage):
        d = _tcp_dist(env, label)
        if d > held0 + SLIP_TOL:
            raise GraspLostError(f"{label} slipped while {stage} (tcp-{label} {held0:.3f} -> {d:.3f})")

    env.move_up(ARM, LIFT)
    check_held("lifting")

    # 手中物体的采样点(ee 坐标系),按抬起后实测的 ee -> 物体变换算
    ee = _mat(env.get_arm_pose(ARM))
    ee_inv = np.linalg.inv(ee)
    pts = _bbox_points(env, label)
    held = (np.c_[pts, np.ones(len(pts))] @ ee_inv.T)[:, :3]
    center_ee = (ee_inv @ np.append(_center(env, label), 1.0))[:3]
    bias = env.robot(ARM).gripper_bias
    # 绕竖直轴转 delta,让抓取点 -> 物体中心的方向正对炉腔
    delta = np.arctan2(np.cross(toward, inward)[2], toward @ inward)
    yawed = t3d.axangles.axangle2mat([0.0, 0.0, 1.0], delta) @ ee[:3, :3]
    rots = [yawed]
    if spec.carry_tilts is not None:
        # 再绕水平的"炉腔横向"轴转,让接近方向从竖直向下朝炉腔里倾 carry_tilt 度(物体前端随之抬起)
        axis = np.cross([0.0, 0.0, -1.0], inward)
        rots = [
            t3d.axangles.axangle2mat(axis, np.radians(a) - np.arccos(np.clip(-yawed[2, 0], -1, 1))) @ yawed
            for a in spec.carry_tilts
        ]
    side = np.cross([0.0, 0.0, 1.0], inward)
    plate = (mw @ [*geom.plate_xy, geom.plate_z, 1.0])[:3]
    opening = (mw @ [0.0, geom.opening_y, 0.0, 1.0])[:3]
    cavity_depth = float(geom.cavity_back - geom.opening_y)

    candidates, chains = [], []
    for rot in rots:
        rel = held @ rot.T  # 物体采样点相对 ee_link 原点,世界方向
        rel_c = rot @ center_ee
        tcp_off = bias * rot[:, 0]  # TCP 相对 ee_link 原点
        carry = round(float(np.degrees(np.arccos(np.clip(-rot[2, 0], -1, 1)))))
        for gap in GAPS:
            for d in DEPTHS:
                # ee_link 原点 o:TCP 伸过炉口平面 d;物体中心横向对准转盘中心;物体最低点离转盘 gap
                o = opening + (d - tcp_off @ inward) * inward
                o += ((plate - o - rel_c) @ side) * side
                o[2] = plate[2] + gap - rel[:, 2].min()
                inner = float(((o + rel - opening) @ inward).max())
                center_in = float((o + rel_c - opening) @ inward)
                if inner > cavity_depth - BACK_CLEAR or center_in < MIN_IN:
                    continue
                run = inner + START_OUT
                far = np.concatenate([o - run * inward, t3d.quaternions.mat2quat(rot)])
                line = env._straight_poses(far, run * inward, 0.02)
                candidates.append((carry, gap, d, round(center_in, 3)))
                chains.append([far, *line])
    if not chains:
        raise PlanFailError(f"{label}: no insertion depth puts its center {MIN_IN} m into the cavity")
    i, plans = _choose_clear(env, clearance, chains, "insertion", held)
    log.info("%s insert (tilt, gap above plate, tcp depth past opening, center depth) %s", label, candidates[i])
    env.execute_plan(ARM, plans[0])
    check_held("turning toward the microwave")
    for res in plans[1:]:
        env.execute_plan(ARM, res, INSERT_SLOW)
    check_held("inserting")

    env.open_gripper(ARM)
    env.delay(30)
    # 退出:沿接近方向后退 + 竖直抬起 + 水平后退,整段不碰微波炉
    here = env.get_arm_pose(ARM)
    approach = _mat(here)[:3, 0]
    chains, opts = [], []
    for back in BACK_OFF:
        for lift in LIFTS_OUT:
            a = here.copy()
            a[:3] -= back * approach
            b = a.copy()
            b[2] += lift
            chain = ([a] if back > 0 else []) + ([b] if lift > 0 else [])
            chains.append(chain + env._straight_poses(b, -RETREAT * inward, 0.03))
            opts.append((back, lift))
    i, plans = _choose_clear(env, clearance, chains, "retreat")
    log.info("retreat (back off, lift) %s", opts[i])
    for res in plans:
        env.execute_plan(ARM, res)
    env.back_to_origin(ARM)
    end = np.linalg.solve(_mat(env.get_obj_pose("microwave")), np.append(_center(env, label), 1.0))[:3] / k
    log.info(
        "%s center at the end, in unscaled microwave coordinates: %s; microwave moved %.3f m",
        label,
        np.round(end, 3).tolist(),
        float(np.linalg.norm(env.get_obj_pose("microwave")[:3] - mw[:3, 3])),
    )
