"""pour_liquid_into_cup 的脚本专家(单臂:瓶子所在一侧的手臂)。

1. 抓取:GraspGenX 候选里只留水平侧抓瓶颈的(接近方向、手指都水平),偏好从机器人一侧向前(+y)接近;
   并且只选从抓取位姿出发,抬起、搬运、整段倾倒都有连续 IK 解的候选(腕部 joint6 要有足够的转动余量)。
2. 竖直抬起,沿笛卡尔路径从瓶子所在的一侧搬到杯旁,同时预倾到 ``TILT_START``(此时还不会出流)。
3. 以瓶口为支点倾倒:先按实测抓取把位姿平滑修正到开倒状态(搬运中瓶子会在手里滑转)。瓶子绕水平的抓取
   接近轴转动,瓶口 xy 固定在杯口中心靠瓶子一侧(出流时液体近乎水平射出,要留出射程),高度取满足间隙的
   最低值(随倾角连续变化)。闭环读液体粒子:接近出流角时放慢,出流后按流量加减倾角,倾转速度始终连续;
   瓶内剩余 <= ``RESIDUAL_DONE`` 或超时后停稳结束。
4. 回到 ``TILT_STOP`` 截流,等空中的液滴落完;再抬高、后撤,把瓶子立起停在杯子旁边(瓶子一侧)——
   立起(倾角 < 30 度)时任务 ``is_axis_up`` 的上升沿触发液体判定,episode 到此结束(评测也在这时结束),
   瓶子留在手里,不放回、不回初始位姿。

抓取之后的目标都写成瓶子的状态 (瓶口位置, 倾角),再用 ee->瓶子 变换换算成末端位姿;倾倒时每一步都重新测量
这个变换,抓取略有滑动也能倒准。路点之间用自己的关节插值执行(时间可控),IK 解要求与上一个路点连续。
cuRobo 的碰撞世界只有桌子,瓶子 / 夹爪与杯子、桌面的间隙由 ``_Pour.mouth_floor`` 用采样点保证。
"""

from collections import deque
import logging

import numpy as np
import transforms3d as t3d

from src.data_collect.collect_env import PlanFailError

log = logging.getLogger(__name__)

# 抓取
NECK_Z = (0.085, 0.12)  # 夹爪中心(TCP)在瓶子坐标系下的高度范围(瓶颈约 0.068 ~ 0.158)
APPROACH_Z = (-0.25, 0.25)  # 接近方向单位向量的 z 分量:水平侧抓
FINGER_Z_MAX = 0.25  # 手指方向 |z| 上限:手指水平,绕接近轴转腕即倾倒
PREFER = (0.0, 1.0, 0.0)  # 偏好的接近方向:从机器人一侧向前
MIN_SCORE = 0.9  # GraspGenX 判别器分数下限:低分抓取闭合时会把瓶子推偏几厘米、转动近 10 度
DEPTH_MAX = 0.02  # 瓶颈轴线在夹爪中心之后(靠掌心)的最大距离:抓得太深时倾倒中瓶子会沿瓶颈滑出几厘米
MAX_GRASP_TRIES = 10  # 最多对几个可到达的候选做整段倾倒的可达性检查
ONSET_BAND = (100.0, 105.0, 110.0, 115.0)  # 出流前后的倾角:按这些角度下瓶口能压到多低给抓取排序
PRE_GRASP = 0.08  # 预抓取位姿沿接近方向后退的距离
APPROACH_TIME = 0.5  # 预抓取 -> 抓取的直线接近时间(秒)
GRASP_SETTLE = 30  # 闭合夹爪后等待的物理步数
LIFT = 0.06  # 抓起后竖直抬升的高度
LIFT_TIME = 0.4
HELD_TOL = 0.025  # 夹爪中心在瓶子坐标系下偏离抓取时位置超过它就算抓丢
IN_HAND_TURN = 12.0  # 倾倒中瓶子相对夹爪转动超过它(度,相对开倒时)说明在手指间打滑,提前放弃
CARRY_SETTLE = 0.2  # 搬运结束后停住等瓶子不晃的时间(秒)
# 间隙(cuRobo 不知道杯子)
CLEAR_OPEN = 0.002  # 在杯口内侧上方的点距杯沿高度的最小值
CLEAR_RIM = 0.008  # 在杯沿 / 杯壁 / 把手上方的点(杯子水平轮廓外扩 CUP_MARGIN)距杯沿高度的最小值
CUP_MARGIN = 0.01
HANDLE_HALF_WIDTH = 0.008  # 马克杯把手的半宽(网格量得 5 ~ 7 mm)
OPEN_MARGIN = 0.004  # 杯口内侧区域 = 杯口半径减去它
CLEAR_TABLE = 0.015  # 距桌面的最小高度
# 倾倒(角度单位:度)
OPENING_SCALE = 0.8  # 杯口内半径估计 = 它 x 杯身外半径(包围盒水平尺寸较小者的一半,马克杯的把手只在一个方向)
MOUTH_BACK = 0.5  # 瓶口在杯口中心靠瓶子一侧 MOUTH_BACK x 杯口半径处:出流时液体近乎水平射出,要留出射程
TILT_START = 75.0  # 搬运结束时的预倾角,此时还不会出流
TILT_ONSET = 95.0  # 超过它之后放慢,满瓶约 100 ~ 105 度出流
TILT_MAX = 150.0
TILT_STOP = 75.0  # 倒完回到这个角度截流
TILT_STEP = 2.5  # 闭环倾倒每次调整的角度
CHECK_STEP = 7.5  # 选抓取时检查倾倒弧线可达性的角度间隔
RATE_FAST = 45.0  # 远离出流角时的倾转速度(度/秒)
RATE_ONSET = 10.0  # 接近出流角时的倾转速度:出流瞬间会涌出一大股
RATE_SLOW = 15.0  # 出流后调整倾角的速度
ONSET = 0.003  # 流出的粒子比例超过它算已经出流
FLOW_MIN = 0.15  # 流量(每秒流出的粒子占总数的比例)低于它就加大倾角
FLOW_MAX = 0.35  # 高于它就减小倾角
FLOW_WINDOW = 0.2  # 估计流量的时间窗(秒)
HOLD = 0.08  # 流量合适时每次保持的时间(秒)
RESIDUAL_DONE = 0.04  # 瓶内剩余比例低于它就结束倾倒(判定要求 <= 0.15)
POUR_TIMEOUT = 7.0  # 闭环倾倒最长时间(秒)
REALIGN_TIME = 0.4  # 开倒前按实测抓取修正位姿的平滑运动时间(秒)
STOP_TIME = 0.1  # 倾转中要保持 / 反向 / 结束时,匀减速停下的时间(秒)
FLOOR_SLOPE = 0.001  # 倾倒弧线上瓶口高度随倾角变化的最大斜率(米/度),抹平间隙要求的突变
FLOOR_WINDOW = 20.0  # 抹平时前后看的倾角范围(度)
FLOOR_SAMPLE = 0.5  # 抹平时的倾角采样间隔(度)
DRIP_AIR = 0.005  # 空中粒子比例低于它算液滴落完
DRIP_TIMEOUT = 1.0  # 截流后等液滴落完的最长时间(秒)
# 立起
BACK_UP = 0.04  # 截流后瓶口抬高的距离
BACK_OFF = 0.03  # 截流后瓶口沿倾倒方向反向后撤的距离
BACK_TILT = 40.0  # 后撤到这里时的倾角(仍 > 30 度,判定在立起时才触发)
END_HEIGHT = 0.05  # 立起后瓶底离桌面的高度
END_CLEAR = 0.02  # 立起后瓶子包围盒与杯子的水平间隙:判定时包围盒里杯中的液体会被算成瓶内残留
# 笛卡尔路径
MOVE_SPEED = 0.3  # 平均平移速度(米/秒)
TURN_SPEED = 90.0  # 平均倾转速度(度/秒)
STOP_SPEED = 120.0  # 截流时的平均倾转速度
MIN_MOVE_TIME = 0.3
WAYPOINT_STEP = (0.03, 7.5)  # 路点间距(米, 度)
MAX_JOINT_JUMP = 0.6  # 相邻路点 IK 解的最大关节变化(弧度),超过视为换了 IK 分支 / 腕部绕了一圈
# wuliangye 外形的采样圆环 (瓶子坐标系下的高度, 半径),来自网格
BOTTLE_RINGS = [
    (-0.06, 0.042),
    (-0.02, 0.042),
    (0.02, 0.042),
    (0.045, 0.040),
    (0.058, 0.030),
    (0.068, 0.018),
    (0.09, 0.017),
    (0.115, 0.017),
    (0.14, 0.017),
    (0.158, 0.017),
]
# 夹爪在 ee_link(link6)坐标系下的包围盒(x = 接近方向,y = 手指张合方向),来自 X5 夹爪网格:
# 电机座、横跨 +-8.7 cm 的导轨、夹住瓶颈时的两根手指(越往指尖越窄)
GRIPPER_BOXES = [
    ((0.0, -0.03, -0.03), (0.06, 0.03, 0.03)),
    ((0.06, -0.087, -0.03), (0.08, 0.087, 0.03)),
    ((0.08, -0.03, -0.03), (0.097, 0.03, 0.03)),
    ((0.097, -0.03, -0.018), (0.127, 0.03, 0.018)),
    ((0.127, -0.03, -0.01), (0.167, 0.03, 0.01)),
]


class GraspLostError(Exception):
    pass


def _mat(pose):
    m = np.eye(4)
    m[:3, :3] = t3d.quaternions.quat2mat(pose[3:7])
    m[:3, 3] = pose[:3]
    return m


def _pose(m):
    return np.concatenate([m[:3, 3], t3d.quaternions.mat2quat(m[:3, :3])])


def _rot(axis, deg):
    return t3d.axangles.axangle2mat(axis, np.radians(deg))


def _bbox(env, label):
    """``label`` 的包围盒在物体坐标系下的 (min, max);与任务的液体判定用同一个包围盒。"""
    lm = env.scene_manager.layout_manager
    v = np.asarray(lm.get_instance_bbox_vertices(lm.get_instance_name(0, label), 0), dtype=float).reshape(-1, 3)
    return v.min(axis=0), v.max(axis=0)


def _box_points(lo, hi, spacing=0.012):
    axes = [np.linspace(a, b, max(int(np.ceil((b - a) / spacing)) + 1, 2)) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _bottle_points():
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    return np.concatenate(
        [np.stack([r * np.cos(ang), r * np.sin(ang), np.full_like(ang, z)], 1) for z, r in BOTTLE_RINGS]
    )


def _tilt_of(m):
    """瓶子轴线与竖直方向的夹角(度)。"""
    return float(np.degrees(np.arccos(np.clip(m[2, 2], -1.0, 1.0))))


def _ik_near(env, arm, pose, q_prev):
    """``pose`` 的 IK 解,要求与 ``q_prev`` 连续:从 ``q_prev`` 出发规划,取路径终点。"""
    q = np.asarray(env.plan_to_pose(arm, pose, start_joints=q_prev)["position"][-1], dtype=float)
    jump = np.abs(q - q_prev).max()
    if jump > MAX_JOINT_JUMP:
        raise PlanFailError(f"{arm}: IK for {np.round(pose, 3).tolist()} jumps {jump:.2f} rad from the previous joints")
    return q


def _ik_chain(env, arm, poses, q):
    out = []
    for pose in poses:
        q = _ik_near(env, arm, pose, q)
        out.append(q)
    return out


def _command(env, arm, pos, vel):
    key = env.robot_manager.process_name(arm)
    env.execute([{key: {"position": p.tolist(), "velocity": v.tolist()}} for p, v in zip(pos, vel)])


def _run(env, arm, qs, duration):
    """沿关节折线 ``qs``(``qs[0]`` 为起点)在 ``duration`` 秒内运动,按关节空间弧长分配时间,
    起止速度为零(smoothstep)。"""
    qs = np.asarray(qs, dtype=float)
    s_way = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(qs, axis=0), axis=1) + 1e-9)])
    u = np.linspace(0.0, 1.0, max(int(round(duration / env.dt)), 1) + 1)[1:]
    u = u * u * (3 - 2 * u)
    pos = np.stack([np.interp(u * s_way[-1], s_way, qs[:, j]) for j in range(qs.shape[1])], axis=1)
    vel = np.diff(np.vstack([qs[:1], pos]), axis=0) / env.dt
    vel[-1] = 0.0
    _command(env, arm, pos, vel)


def _ramp(env, arm, q0, q1, v0, v1, dist):
    """倾倒的一小步 ``q0`` -> ``q1``(倾角走 ``dist`` 度),倾转速度从 ``v0`` 匀变到 ``v1``(度/秒,都 >= 0、
    不同时为零)。上一步的末速度就是这一步的 ``v0``,相邻小步之间速度连续,不会因速度阶跃抖动。返回用时(秒)。"""
    dur = 2.0 * dist / (v0 + v1)
    n = max(int(round(dur / env.dt)), 1)
    tau = np.arange(1, n + 1) / n
    a = 2.0 * v0 / (v0 + v1)  # 起点处的 ds / dtau;s(tau) 是二次函数,即匀加 / 减速
    s = a * tau + (1.0 - a) * tau**2
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    pos = q0 + s[:, None] * (q1 - q0)
    vel = np.diff(np.vstack([q0, pos]), axis=0) / env.dt
    if v1 == 0:
        vel[-1] = 0.0
    _command(env, arm, pos, vel)
    return n * env.dt


def _hold(env, arm, q, duration):
    """保持关节角 ``q``;显式给零速度(空控制会沿用上一条指令的速度)。"""
    n = max(int(round(duration / env.dt)), 1)
    _command(env, arm, np.repeat(np.asarray(q, dtype=float)[None], n, 0), np.zeros((n, len(q))))


def _particles(env):
    lm = env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "wine"))
    p = np.asarray(obj.get_particle_positions()[0], dtype=float).reshape(-1, 3)
    R = t3d.quaternions.quat2mat(np.asarray(obj.init_ori, dtype=float).reshape(-1)[:4])
    return p @ R.T + np.asarray(obj.init_pos, dtype=float).reshape(-1)[:3]


def _inside(env, p, label, z_min=0.0, buffer=0.0):
    """粒子 ``p`` 是否在 ``label`` 的包围盒(外扩 ``buffer``)内、且高出盒底 ``z_min``(同任务判定)。"""
    m, (lo, hi) = _mat(env.get_obj_pose(label)), _bbox(env, label)
    lo, hi = lo - buffer, hi + buffer
    loc = (p - m[:3, 3]) @ m[:3, :3]
    return np.all((loc >= lo) & (loc <= hi), axis=1) & (loc[:, 2] - lo[2] >= z_min)


def _fluid(pour):
    """液体粒子分布(与任务判定同一套区域):瓶内 / 目标处(杯内)/ 空中 / 洒在桌上 的粒子数。"""
    p = _particles(pour.env)
    in_bottle = _inside(pour.env, p, "bottle")
    in_target = pour.in_target(p) & ~in_bottle
    out = ~(in_bottle | in_target)
    air = out & (p[:, 2] > pour.table_z + 0.012)
    return {
        "total": len(p),
        "bottle": int(in_bottle.sum()),
        pour.TARGET: int(in_target.sum()),
        "air": int(air.sum()),
        "spill": int((out & ~air).sum()),
    }


def _judge(env):
    """按任务液体判定(参数取自尚未触发的那条检查)估算:倒出部分在杯内的比例、瓶内残留比例、
    被当作零散粒子忽略的比例,以及现在立起瓶子能否通过。判定已经通过时返回 None。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    rm = env.reward_manager
    checks = [c for _, cs, _, _ in rm.trigger_check_list[0] for c in cs if c[0] == "is_A_fluid_in_B"]
    if not checks:
        return None
    a = checks[0][1]
    p = _particles(env)
    in_c = _inside(env, p, a["label_C"], a["C_z_threshold"], a.get("C_buffer", a["B_buffer"]))
    in_b = _inside(env, p, a["label_B"], a["B_z_threshold"], a["B_buffer"]) & ~in_c
    out = np.where(~(in_b | in_c))[0]
    ignored = 0
    if a["ignore_scattered"] and len(out):
        pairs = cKDTree(p[out, :2]).query_pairs(a["scatter_connect_radius"], output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(out), len(out)))
        _, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        ignored = int((sizes[labels] < a["scatter_min_component_size"]).sum())
    eff = len(p) - ignored
    residual = in_c.sum() / eff
    inside = in_b.sum() / max(eff - in_c.sum(), 1)
    ok = ignored / len(p) <= a["max_ignore_ratio"] and residual <= a["C_residual_threshold"]
    ok = ok and inside >= a["percentage_threshold"]
    return {"in_cup": round(float(inside), 4), "residual": round(float(residual), 4), "ignored": ignored, "ok": ok}


class _Pour:
    """倾倒几何:瓶子状态 (瓶口位置 M, 倾角 tilt) <-> 瓶子位姿 <-> 末端位姿,以及间隙计算。

    ``tilt`` 时瓶子姿态 = 绕水平轴 ``k`` 转 ``tilt`` 度 x 正立姿态 ``R_up``;``k`` 是抓取接近方向的水平投影,
    转动方向使瓶口倒向目标(倾倒方向 ``d``)。

    倒向的目标(这里是杯子)由子类可覆盖的 ``TARGET`` / ``_init_target`` / ``_pivot_xy`` / ``mouth_floor`` /
    ``in_target`` / ``judge`` / ``log_target`` 描述,其他任务(例如倒在手机上)复用其余部分。
    """

    TARGET = "cup"

    def __init__(self, env, arm, table_z, bottle_m, b_T_ee):
        self.env, self.arm, self.table_z = env, arm, table_z
        _, _, _, meta = env.get_obj_asset("bottle")
        R_place = t3d.quaternions.quat2mat(meta["active"]["place"]["up"]["orientation"])
        tilted = bottle_m[:3, :3] @ R_place.T
        self.R_up = _rot([0, 0, 1], np.degrees(np.arctan2(tilted[1, 0], tilted[0, 0]))) @ R_place
        self.bottle0 = bottle_m
        self.bottle_box = _bbox(env, "bottle")
        self.mouth = np.array([0.0, 0.0, self.bottle_box[1][2]])
        self._init_target()

        approach = (bottle_m @ b_T_ee)[:3, 0]
        a_h = np.array([approach[0], approach[1], 0.0]) / np.linalg.norm(approach[:2])
        d = np.cross(a_h, [0.0, 0.0, 1.0])
        self.d = d if d[:2] @ (self.aim_xy - bottle_m[:2, 3]) > 0 else -d
        self.k = np.cross([0.0, 0.0, 1.0], self.d)  # 绕 k 转正角度时瓶口倒向 d
        self.pivot_xy = self._pivot_xy()
        self.set_grasp(b_T_ee)
        self.q, self.cur = None, None  # 最近一次指令的关节角 / 瓶子状态

    # ------------------------------------------------------------------ 目标(杯子)
    def _init_target(self):
        """目标的几何:``top``(顶面高度)、``aim_xy``(倒向的点)、``corners``(水平轮廓的角点),以及杯子专用的量。

        杯子的水平轮廓:杯身是以底面接触圆心为圆心(马克杯的包围盒含把手)、包围盒较窄方向一半为半径的圆,
        包围盒里超出杯身圆的部分是把手;杯口内半径按杯身外半径估计。"""
        env = self.env
        cup_m = _mat(env.get_obj_pose("cup"))
        _, _, _, cup_meta = env.get_obj_asset("cup")
        verts = np.asarray(cup_meta["geometry"]["aligned_bbox"]["vertices"], dtype=float)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        self.top = float(cup_m[2, 3] + hi[2])
        self.cup_xy, self.cup_R2, self.cup_lo2, self.cup_hi2 = cup_m[:2, 3], cup_m[:2, :2], lo[:2], hi[:2]
        corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        self.corners = corners @ cup_m[:2, :2].T + cup_m[:2, 3]
        self.body_c = np.asarray(cup_meta["active"]["place"]["up"]["contact_circle"]["center"][:2], dtype=float)
        self.body_r = float(np.min(hi[:2] - lo[:2])) / 2
        self.long_axis = int(np.argmax(hi[:2] - lo[:2]))  # 把手所在的方向
        self.aim_xy = (cup_m @ np.append(self.body_c, [0.0, 1.0]))[:2]
        self.open_r = OPENING_SCALE * self.body_r

    def _pivot_xy(self):
        """倾倒时瓶口的水平位置。"""
        return self.aim_xy - MOUTH_BACK * self.open_r * self.d[:2]

    def in_target(self, p):
        """粒子 ``p`` 是否已到目标处(同任务判定:杯子包围盒内、高出盒底 1 mm)。"""
        return _inside(self.env, p, "cup", 0.001)

    def fluid(self):
        return _fluid(self)

    def judge(self):
        return _judge(self.env)

    def log_target(self):
        cup_cat, cup_idx, _, _ = self.env.get_obj_asset("cup")
        log.info(
            "cup %s/%d: opening r=%.3f, mouth %.3f above the rim at %.0f deg / %.3f at 110 deg",
            cup_cat,
            cup_idx,
            self.open_r,
            self.cur[0][2] - self.top,
            TILT_START,
            self.pour_state(110.0)[0][2] - self.top,
        )

    # ------------------------------------------------------------------ 位姿
    def set_grasp(self, b_T_ee):
        self.b_T_ee = b_T_ee
        grip = np.concatenate([_box_points(lo, hi) for lo, hi in GRIPPER_BOXES])
        self.points = np.concatenate([_bottle_points(), grip @ b_T_ee[:3, :3].T + b_T_ee[:3, 3]])

    def regrasp(self):
        """重新测量 ee->瓶子 变换。"""
        env = self.env
        self.set_grasp(np.linalg.inv(_mat(env.get_obj_pose("bottle"))) @ _mat(env.get_arm_pose(self.arm)))

    def R(self, tilt):
        return _rot(self.k, tilt) @ self.R_up

    def bottle_m(self, m, tilt):
        b = np.eye(4)
        b[:3, :3] = self.R(tilt)
        b[:3, 3] = m - b[:3, :3] @ self.mouth
        return b

    def ee_pose(self, m, tilt):
        return _pose(self.bottle_m(m, tilt) @ self.b_T_ee)

    def measured(self):
        b = _mat(self.env.get_obj_pose("bottle"))
        return (b @ np.append(self.mouth, 1))[:3], _tilt_of(b)

    # ------------------------------------------------------------------ 间隙与路径
    def mouth_floor(self, m_xy, tilt):
        """瓶口在 ``m_xy`` 上方、倾角 ``tilt`` 时,瓶子和夹爪都不碰杯子、桌面所需的最低瓶口高度。"""
        rel = (self.points - self.mouth) @ self.R(tilt).T
        xy = m_xy + rel[:, :2]
        loc = (xy - self.cup_xy) @ self.cup_R2
        in_box = np.all((loc >= self.cup_lo2 - CUP_MARGIN) & (loc <= self.cup_hi2 + CUP_MARGIN), axis=1)
        body = loc - self.body_c
        handle = in_box & (np.abs(body[:, self.long_axis]) > self.body_r + CUP_MARGIN)
        handle &= np.abs(body[:, 1 - self.long_axis]) < HANDLE_HALF_WIDTH + CUP_MARGIN
        over_cup = handle | (np.linalg.norm(body, axis=1) < self.body_r + CUP_MARGIN)
        over_open = np.linalg.norm(xy - self.aim_xy, axis=1) < self.open_r - OPEN_MARGIN
        floor = np.where(over_cup, self.top + CLEAR_RIM, self.table_z + CLEAR_TABLE)
        floor = np.where(over_open, self.top + CLEAR_OPEN, floor)
        return float(np.max(floor - rel[:, 2]))

    def pour_state(self, tilt):
        """倾倒弧线上 ``tilt`` 处的状态:瓶口在 ``pivot_xy`` 上方,高度取满足间隙的最低值。

        采样点越过杯沿 / 杯口(或手机轮廓)边界时,最低高度会随倾角突变;这里取前后 ``FLOOR_WINDOW`` 度内
        ``mouth_floor(τ) - FLOOR_SLOPE * |τ - tilt|`` 的最大值,高度随倾角连续变化,瓶口在突变前就逐渐抬到位,
        不会在一小步里猛跳。结果不低于 ``tilt`` 处的最低高度,间隙只会更大。"""
        taus = np.arange(-FLOOR_WINDOW, FLOOR_WINDOW + 1e-9, FLOOR_SAMPLE) + tilt
        taus = np.concatenate([[tilt], np.clip(taus, TILT_START, TILT_MAX)])
        z = max(self.mouth_floor(self.pivot_xy, x) - FLOOR_SLOPE * abs(x - tilt) for x in taus)
        return np.array([*self.pivot_xy, z]), float(tilt)

    def pivot_path(self, t0, t1, step=5.0):
        n = max(int(np.ceil(abs(t1 - t0) / step)), 1)
        return [self.pour_state(t) for t in np.linspace(t0, t1, n + 1)[1:]]

    def path(self, start, goal):
        """``start`` -> ``goal`` 的路点:瓶口位置和倾角线性插值,每个路点的瓶口抬到间隙要求以上。"""
        (m0, t0), (m1, t1) = start, goal
        n = max(
            int(np.ceil(np.linalg.norm(m1 - m0) / WAYPOINT_STEP[0])), int(np.ceil(abs(t1 - t0) / WAYPOINT_STEP[1])), 1
        )
        out = []
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            m, t = m0 + s * (m1 - m0), t0 + s * (t1 - t0)
            m[2] = max(m[2], self.mouth_floor(m[:2], t))
            out.append((m, float(t)))
        return out

    def lifted_state(self):
        b = self.bottle0.copy()
        b[2, 3] += LIFT
        return (b @ np.append(self.mouth, 1))[:3], 0.0

    def end_state(self):
        """立起后的状态:瓶子正立在目标靠瓶子一侧(沿 -d),包围盒与目标轮廓(杯子含把手)水平相隔 ``END_CLEAR``,
        瓶底离桌面 ``END_HEIGHT``。"""
        lo, hi = self.bottle_box
        half_diag = float(np.hypot(*np.max(np.abs([lo[:2], hi[:2]]), axis=0)))
        ext = float(np.max((self.corners - self.aim_xy) @ -self.d[:2]))
        axis_xy = self.aim_xy - (ext + half_diag + END_CLEAR) * self.d[:2]
        origin = np.array([*axis_xy, self.table_z + END_HEIGHT - (self.R_up @ [0.0, 0.0, lo[2]])[2]])
        return origin + self.R_up @ self.mouth, 0.0

    # ------------------------------------------------------------------ 执行
    def check(self, q):
        """从抓取关节角 ``q`` 出发,抬起、搬运、整段倾倒是否都有连续的 IK 解(只规划,不执行)。"""
        ee = self.bottle0 @ self.b_T_ee
        lift = [ee.copy() for _ in range(2)]
        lift[0][2, 3] += LIFT / 2
        lift[1][2, 3] += LIFT
        q = _ik_chain(self.env, self.arm, [_pose(m) for m in lift], q)[-1]
        waypoints = self.path(self.lifted_state(), self.pour_state(TILT_START))
        waypoints += self.pivot_path(TILT_START, TILT_MAX, CHECK_STEP)
        _ik_chain(self.env, self.arm, [self.ee_pose(m, t) for m, t in waypoints], q)

    def go(self, waypoints, speed=TURN_SPEED):
        """依次经过 ``waypoints``(瓶子状态),一次平滑运动。"""
        qs = _ik_chain(self.env, self.arm, [self.ee_pose(m, t) for m, t in waypoints], self.q)
        states = [self.cur, *waypoints]
        dist = sum(np.linalg.norm(b[0] - a[0]) for a, b in zip(states, states[1:]))
        turn = sum(abs(b[1] - a[1]) for a, b in zip(states, states[1:]))
        _run(self.env, self.arm, [self.q, *qs], max(dist / MOVE_SPEED, turn / speed, MIN_MOVE_TIME))
        self.q, self.cur = qs[-1], waypoints[-1]


def _tcp_local(env, arm):
    """夹爪中心(TCP)在瓶子坐标系下的位置。"""
    ee = _mat(env.get_arm_pose(arm))
    tcp = ee[:3, 3] + env.robot(arm).gripper_bias * ee[:3, 0]
    return (np.linalg.inv(_mat(env.get_obj_pose("bottle"))) @ np.append(tcp, 1))[:3]


def _check_held(env, arm, tcp0, stage):
    """夹爪中心在瓶子坐标系下是否还在抓取时的位置附近(GraspGenX 的抓取深浅不一,不能和瓶子轴线比)。"""
    tcp = _tcp_local(env, arm)
    if np.linalg.norm(tcp - tcp0) > HELD_TOL:
        raise GraspLostError(
            f"bottle slipped in the gripper while {stage} (tcp in bottle frame {np.round(tcp, 3)}, "
            f"grasped at {np.round(tcp0, 3)})"
        )


def _grasp(env, arm, bottle0, table_z, pour_cls):
    """侧抓瓶颈:候选按出流前后瓶口能压到多低排序(夹爪离杯沿越远越低,液体越不容易射出杯外),
    同一档内保持 GraspGenX 的顺序;选第一个预抓取 / 直线接近 / 后续整段倾倒都可达的,执行抓取。"""
    cands = env.grasp_candidates(
        "bottle", arm, prefer=PREFER, approach_z=APPROACH_Z, finger_z_max=FINGER_Z_MAX, obj_z=NECK_Z
    )
    bias = env.robot(arm).gripper_bias

    def depth(pose):
        """抓取时瓶颈轴线在夹爪中心之后多远(沿水平接近方向,瓶子坐标系)。"""
        local = np.linalg.inv(bottle0) @ _mat(pose)
        a = local[:2, 0] / np.linalg.norm(local[:2, 0])
        return float((local[:2, 3] + bias * local[:2, 0]) @ a)

    cands = [c for c in cands if c[1] >= MIN_SCORE and depth(c[0]) <= DEPTH_MAX]
    if not cands:
        raise PlanFailError(f"{arm}: no GraspGenX neck grasp passes the filters")
    ranked = []
    for i, (pose, score) in enumerate(cands):
        pour = pour_cls(env, arm, table_z, bottle0, np.linalg.inv(bottle0) @ _mat(pose))
        h = max(pour.pour_state(t)[0][2] for t in ONSET_BAND) - pour.top
        ranked.append((round(h / 0.005), i, h, pose, score, pour))
    ranked.sort(key=lambda r: r[:2])
    q0, tried = env.get_arm_joints(arm), 0
    for _, i, h, pose, score, pour in ranked:
        g = _mat(pose)
        pre = g.copy()
        pre[:3, 3] -= PRE_GRASP * g[:3, 0]
        try:
            to_pre = env.plan_to_pose(arm, _pose(pre), start_joints=q0)
        except PlanFailError:
            continue
        tried += 1
        line = [pre.copy() for _ in range(4)]
        for j, m in enumerate(line):
            m[:3, 3] += (j + 1) / 4 * PRE_GRASP * g[:3, 0]
        try:
            q_app = _ik_chain(env, arm, [_pose(m) for m in line], np.asarray(to_pre["position"][-1], dtype=float))
            pour.check(q_app[-1])
        except PlanFailError as e:
            log.info("grasp %d (score %.3f, mouth %.3f) rejected: %s", i, score, h, e)
            if tried >= MAX_GRASP_TRIES:
                break
            continue
        tcp = (np.linalg.inv(bottle0) @ np.append(g[:3, 3] + bias * g[:3, 0], 1))[2]
        log.info(
            "grasp %d (score %.3f, depth %.3f, height %.3f, mouth %.3f above the rim at onset), approach %s, "
            "pour toward %s",
            i,
            score,
            depth(pose),
            tcp,
            h,
            np.round(g[:3, 0], 2),
            np.round(pour.d, 2),
        )
        env.open_gripper(arm)
        env.execute_plan(arm, to_pre)
        _run(env, arm, [env.get_arm_joints(arm), *q_app], APPROACH_TIME)
        env.settle(arm, q_app[-1])
        env.close_gripper(arm)
        env.delay(GRASP_SETTLE)
        return pour
    raise PlanFailError(f"{arm}: none of {tried} reachable neck grasps (of {len(cands)}) can reach the whole pour")


def _pour(env, pour):
    """闭环倾倒:出流前加大倾角(接近出流角时放慢);出流后流量小于 ``FLOW_MIN`` 加大、大于 ``FLOW_MAX``
    减小倾角,其余时候保持。返回结束时的倾角。

    倾转速度在相邻小步之间连续(``_ramp``):从静止起步、换速时匀加 / 减速;要保持、反向或结束时先在
    ``STOP_TIME`` 内匀减速停下(会顺着原方向多转不到 1 度),而不是突然停住。"""
    arm, tilt, cap, t, v = pour.arm, TILT_START, TILT_MAX, 0.0, 0.0  # v: 当前倾转速度(度/秒,带符号)
    # 搬运中瓶子会在手里滑转:按实测的 ee->瓶子 变换先平滑地回到开倒状态,
    # 否则这段修正会挤进第一小步(几十毫秒),手臂猛地一抖
    pour.regrasp()
    q = _ik_near(env, arm, pour.ee_pose(*pour.pour_state(TILT_START)), pour.q)
    _run(env, arm, [pour.q, q], REALIGN_TIME)
    env.settle(arm, q)
    pour.q, pour.cur = q, pour.pour_state(TILT_START)
    R_ref = pour.b_T_ee[:3, :3]  # 开倒时实测的瓶子->ee 姿态

    def move(new, v_end):
        """以末速度 ``v_end``(>= 0)转到倾角 ``new``,起步速度沿用当前的 ``v``(反向时从 0 起步)。"""
        nonlocal tilt, t, v
        pour.regrasp()
        q = _ik_near(env, arm, pour.ee_pose(*pour.pour_state(new)), pour.q)
        sign = float(np.sign(new - tilt))
        dur = _ramp(env, arm, pour.q, q, max(sign * v, 0.0), v_end, abs(new - tilt))
        pour.q, tilt, t, v = q, new, t + dur, sign * v_end

    def stop():
        """正在转动时,在 ``STOP_TIME`` 内匀减速停下(不越过倾角范围);转不动时原地停住。"""
        nonlocal t, v
        if v == 0:
            return
        room = cap - tilt if v > 0 else tilt - TILT_START
        d = min(abs(v) * STOP_TIME / 2, room)
        if d > 1e-3:
            try:
                move(tilt + np.sign(v) * d, 0.0)
                return
            except PlanFailError:
                pass
        _hold(env, arm, pour.q, HOLD)
        t, v = t + HOLD, 0.0

    f = pour.fluid()
    total, full, onset, hist, next_log = f["total"], f["bottle"], False, deque(), 0.0
    while True:
        f = pour.fluid()
        onset = onset or f["bottle"] < full - ONSET * total
        hist.append((t, f["bottle"]))
        while len(hist) > 2 and hist[1][0] <= t - FLOW_WINDOW:
            hist.popleft()
        flow = (hist[0][1] - f["bottle"]) / total / max(t - hist[0][0], 1e-6) if len(hist) > 1 else 0.0
        measured = pour.measured()[1]
        if t >= next_log:
            log.info("pour t=%.2fs tilt=%.1f (measured %.1f) flow=%.2f/s %s", t, tilt, measured, flow, f)
            next_log += 0.5
        # 只看瓶子相对夹爪的转动:手臂跟随指令的滞后不算
        R_now = (np.linalg.inv(_mat(env.get_obj_pose("bottle"))) @ _mat(env.get_arm_pose(arm)))[:3, :3]
        turn = np.degrees(np.arccos(np.clip((np.trace(R_ref.T @ R_now) - 1) / 2, -1.0, 1.0)))
        if turn > IN_HAND_TURN:
            raise GraspLostError(
                f"bottle turned {turn:.1f} deg in the gripper while pouring (tilt {measured:.1f}, commanded {tilt:.1f})"
            )
        if f["bottle"] <= RESIDUAL_DONE * total or t >= POUR_TIMEOUT:
            break
        if not onset:
            new, rate = tilt + TILT_STEP, RATE_FAST if tilt < TILT_ONSET else RATE_ONSET
        elif flow < FLOW_MIN:
            new, rate = tilt + TILT_STEP, RATE_SLOW
        elif flow > FLOW_MAX:
            new, rate = tilt - TILT_STEP, RATE_SLOW
        else:
            new = tilt
        new = float(np.clip(new, TILT_START, cap))
        if v != 0 and (new == tilt or np.sign(new - tilt) != np.sign(v)):
            stop()  # 要保持或反向:先停稳,下一轮再按流量决定
            continue
        if new == tilt:
            _hold(env, arm, pour.q, HOLD)
            t += HOLD
            continue
        try:
            move(new, rate)
        except PlanFailError as e:
            log.info("pour capped at %.1f deg: %s", tilt, e)
            cap = min(cap, max(tilt, new - TILT_STEP))
    stop()  # 停稳后再截流
    pour.cur = pour.pour_state(tilt)
    log.info("pour done after %.2fs at tilt %.1f: %s judge=%s", t, tilt, f, pour.judge())
    return tilt


def _wait_drips(env, pour):
    t = 0.0
    while t < DRIP_TIMEOUT:
        f = pour.fluid()
        if f["air"] <= DRIP_AIR * f["total"]:
            break
        _hold(env, pour.arm, pour.q, 0.04)
        t += 0.04
    log.info("drips settled after %.2fs: %s judge=%s", t, f, pour.judge())


def play_once(env, pour_cls=_Pour):
    bottle0 = _mat(env.get_obj_pose("bottle"))
    table_z = float((bottle0 @ np.array([0.0, 0.0, _bbox(env, "bottle")[0][2], 1.0]))[2])  # 瓶底即桌面
    arm = "left_arm" if bottle0[0, 3] < 0 else "right_arm"

    # 1. 侧抓瓶颈,竖直抬起
    pour = _grasp(env, arm, bottle0, table_z, pour_cls)
    planned = pour.b_T_ee
    tcp0 = _tcp_local(env, arm)
    ee = _mat(env.get_arm_pose(arm))
    lift = [ee.copy() for _ in range(2)]
    lift[0][2, 3] += LIFT / 2
    lift[1][2, 3] += LIFT
    q = env.get_arm_joints(arm)
    qs = _ik_chain(env, arm, [_pose(m) for m in lift], q)
    _run(env, arm, [q, *qs], LIFT_TIME)
    env.settle(arm, qs[-1])
    _check_held(env, arm, tcp0, "lifting")
    # 闭合夹爪时手指可能把瓶子推偏、转动:按实测的 ee->瓶子 变换重建倾倒几何(倾倒轴、间隙采样点)
    b_T_ee = np.linalg.inv(_mat(env.get_obj_pose("bottle"))) @ _mat(env.get_arm_pose(arm))
    turn = np.degrees(np.arccos(np.clip((np.trace(planned[:3, :3].T @ b_T_ee[:3, :3]) - 1) / 2, -1.0, 1.0)))
    pour = pour_cls(env, arm, table_z, bottle0, b_T_ee)
    log.info(
        "in hand: ee moved %.3f m / turned %.1f deg from the planned grasp; mouth %.3f above the rim at onset",
        np.linalg.norm(b_T_ee[:3, 3] - planned[:3, 3]),
        turn,
        max(pour.pour_state(t)[0][2] for t in ONSET_BAND) - pour.top,
    )
    pour.q, pour.cur = qs[-1], (pour.measured()[0], 0.0)

    # 2. 搬到目标旁并预倾,瓶口到倾倒位置上方
    pour.go(pour.path(pour.cur, pour.pour_state(TILT_START)))
    env.settle(arm, pour.q)
    _hold(env, arm, pour.q, CARRY_SETTLE)
    _check_held(env, arm, tcp0, "carrying")
    pour.log_target()

    # 3. 闭环倾倒;4. 截流、等液滴落完
    tilt = _pour(env, pour)
    pour.go(pour.pivot_path(tilt, TILT_STOP), speed=STOP_SPEED)
    _wait_drips(env, pour)
    _check_held(env, arm, tcp0, "pouring")

    # 抬高后撤,把瓶子立起停在杯子旁边:立起时触发任务的液体判定,episode 到此结束
    m_stop = pour.cur[0]
    back = (m_stop + [0.0, 0.0, BACK_UP] - BACK_OFF * pour.d, BACK_TILT)
    pour.go(pour.path(pour.cur, back) + pour.path(back, pour.end_state()))
    env.settle(arm, pour.q)
    judge = pour.judge()
    log.log(
        logging.INFO if judge is None else logging.WARNING,
        "upright at %.2fs (liquid check %s): %s",
        env._step_cnt * env.dt,
        "passed" if judge is None else f"NOT passed, judge={judge}",
        pour.fluid(),
    )
    return pour
