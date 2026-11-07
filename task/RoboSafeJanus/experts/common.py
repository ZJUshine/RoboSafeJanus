"""RoboSafeJanus 各组脚本专家共用的抓取 / 规划工具(各组的专家在 ``experts/<group>.py``,任务的专家
``experts/<task>.py`` 只给出自己的 ``Spec`` 并调用组模块的 ``play_once``)。

``_pick_place_direct``:三段运动,每段一次 cuRobo 规划,只在夹爪开合时停:初始 -> 抓取 -> 合拢 -> 松手位姿 -> 松开 ->
回初始。各组传入自己的碰撞世界(``CuroboPlanner.set_obstacles``)、抓取候选和松手位姿候选;手指和手中物体用
``CuroboPlanner.set_tool_spheres`` 补上(cuRobo 的碰撞球只到腕部)。

1. 取第一个第 1 段(初始 -> 抓取,夹爪张开、物体是障碍物)和预测的第 2 段都能规划的抓取候选;合拢夹爪。
2. 合拢后按实测的 ee->物体 变换把物体挂到夹爪上、从碰撞世界移除;抓取 -> 松手位姿一次规划,整条轨迹按 ``carry_slow``
   放慢。
3. 松开后物体放回碰撞世界;松手位姿 -> 初始关节一次规划,起点就碰到物体时去掉物体再试,关节空间规划不通时改为规划到
   初始末端位姿。

规划出的路径腕部向上臂折回(``_folds``,cuRobo 的自碰撞模型漏掉这种构型,仿真里物理直接变成 NaN)时重新规划。
规划不到、判定不过的尝试直接交给采集脚本丢弃,换下一个 seed。

流程函数都用模块级的 ``ARM``(默认右臂);用左臂的组(drop_X_trash_can、put_X_supermarket_shelf)在运行期间临时改
``common.ARM``。
"""

import logging

import numpy as np
import transforms3d as t3d

from src.data_collect.collect_env import PlanFailError

log = logging.getLogger(__name__)

ARM = "right_arm"

# top:指尖最低处(TCP 沿接近方向再往前 TIP_BEYOND_TCP)离桌面至少这么远,不够时把抓取沿接近方向往回提:
# GraspGenX 夹在矮物体(三明治)中心高度的抓取,指尖贴着桌面,cuRobo 判碰撞,一个都规划不到
TOP_TIP_CLEAR = 0.008
TIP_BEYOND_TCP = 0.019
FINGER_MAX = 0.075  # 夹持宽度上限(X5 夹爪最大开口 8.8 cm)
GRASP_SETTLE = 30
GRIPPER_OPEN = 0.088  # X5 夹爪全开时两指间距(归一化开度 1)
RELEASE_SETTLE = 40  # 松开夹爪后等这么多步再回程
# 手里拿着物体的那段整条轨迹按这个倍数放慢(时间上统一拉长,不分段):按 cuRobo 的速度跑时鸡蛋会被甩出去
CARRY_SLOW = 1.5
PRED_REUSE_TOL = 0.01  # 合拢后按实测手中位姿规划不到第 2 段时,物体在指间挪动不超过这么多就沿用预测的那条
# 碰撞模型
OBJ_SHRINK = 0.85  # 圆的物体:包围盒按这个比例缩小当障碍物
BOX_SHRINK = 0.95  # 其余物体(手指 / 掌心抓取时本来就贴着它)
FINGER_R = 0.006  # 手指碰撞球半径
HELD_R_FRAC = 0.9  # 手中物体的碰撞球半径 = 这个比例 x 包围盒最短(圆的物体:次短)边的一半
# X5 的 URDF 关节限位是 +-10 rad,cuRobo 会规划出腕部向上臂折回的路径(例如 joint2 2.1 rad 时 joint4 1.8 rad):
# 仿真里这里自碰,机械臂物理直接变成 NaN。上限取自已成功的鸡蛋 / 倒水数据(|joint4| <= 1.76,joint2 + joint4 <= 3.57)
J4_MAX = 1.8
J2_PLUS_J4_MAX = 3.65
FOLD_RETRIES = 3  # 一次规划的流程:规划出折回的路径时最多重新规划这么多次
# 手指(link7 / link8,不在 cuRobo 的碰撞球里)在 ee_link 坐标系下的截面:x 沿接近方向,z 方向的半厚度按 X5 手指网格
# 越往指尖越窄:(x, 半厚度),``_finger_spheres`` 按它摆碰撞球
FINGER_PROFILE = ((0.075, 0.03), (0.097, 0.03), (0.11, 0.018), (0.127, 0.018), (0.142, 0.01), (0.158, 0.01))


def _mat(pose):
    m = np.eye(4)
    m[:3, :3] = t3d.quaternions.quat2mat(pose[3:7])
    m[:3, 3] = pose[:3]
    return m


def _bbox(env, label):
    """``label`` 包围盒在物体坐标系下的 (min, max)。"""
    v = np.asarray(env.get_obj_asset(label)[3]["geometry"]["aligned_bbox"]["vertices"], dtype=float)
    return v.min(axis=0), v.max(axis=0)


GRID = np.array([[a, b, c] for a in (0, 0.5, 1) for b in (0, 0.5, 1) for c in (0, 0.5, 1)])
FACE_CENTERS = np.sum(GRID == 0.5, axis=1) == 2  # 6 个面中心:圆的物体(鸡蛋、罐头、汉堡)包围盒的角点远在物体外面


def _bbox_points(env, label):
    """``label`` 包围盒的角点、棱中点、面中心和中心(27 个点,顺序同 ``GRID``),世界坐标。"""
    lo, hi = _bbox(env, label)
    local = GRID * (hi - lo) + lo
    return (np.c_[local, np.ones(len(local))] @ _mat(env.get_obj_pose(label)).T)[:, :3]


def _center(env, label):
    """``label`` 包围盒中心的世界坐标(资产原点不一定在物体中心,鸡蛋的就偏了约 2 cm)。"""
    lo, hi = _bbox(env, label)
    return (_mat(env.get_obj_pose(label)) @ np.append((lo + hi) / 2, 1.0))[:3]


def _table_z(env):
    return float(env.scene_manager.layout_manager.table_info[0].get("height", 0.765))


def _folds(plans):
    """路径中是否有腕部向上臂折回的构型(``J4_MAX`` / ``J2_PLUS_J4_MAX``);有时返回描述。"""
    q = np.concatenate([np.asarray(res["position"]) for res in plans])
    j4, j24 = float(np.abs(q[:, 3]).max()), float((q[:, 1] + q[:, 3]).max())
    if j4 > J4_MAX or j24 > J2_PLUS_J4_MAX:
        return f"folds the wrist onto the arm (|joint4| {j4:.2f}, joint2 + joint4 {j24:.2f})"
    return None


def _horizontal(v):
    v = np.array([v[0], v[1], 0.0])
    return v / np.linalg.norm(v)


def _pose(m):
    return np.concatenate([m[:3, 3], t3d.quaternions.mat2quat(m[:3, :3])])


def _box(lo, hi, world_T_frame):
    """``frame`` 坐标系里的轴对齐盒子 -> (dims, 世界位姿)。"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.eye(4)
    c[:3, 3] = (lo + hi) / 2
    return (hi - lo).tolist(), _pose(world_T_frame @ c)


def _obj_box(env, label, shrink=OBJ_SHRINK):
    """物体包围盒(长宽高都按 ``shrink`` 缩小,标量或 3 个分量)当障碍物 -> (dims, 世界位姿)。"""
    lo, hi = _bbox(env, label)
    c, h = (lo + hi) / 2, (hi - lo) / 2 * np.asarray(shrink, dtype=float)
    return _box(c - h, c + h, _mat(env.get_obj_pose(label)))


def _bbox_spheres(ee_T_obj, lo, hi, r_frac=HELD_R_FRAC):
    """手中物体(包围盒 lo / hi,物体坐标系)填满包围盒的碰撞球,ee_link 坐标系:半径 = ``r_frac`` x 最短边的一半,
    沿每条比 3 个半径还长的边按半径的间距排(球面之间没有空隙)。"""
    ext = hi - lo
    r = r_frac * float(ext.min()) / 2
    axes = []
    for i in range(3):
        if ext[i] - 2 * r < r:
            axes.append([(lo[i] + hi[i]) / 2])
        else:
            n = int(np.ceil((ext[i] - 2 * r) / r)) + 1
            axes.append(np.linspace(lo[i] + r, hi[i] - r, n))
    return [[*(ee_T_obj @ [x, y, z, 1.0])[:3], r] for x in axes[0] for y in axes[1] for z in axes[2]]


def _finger_spheres(gap, spacing=None):
    """两根手指(间距 ``gap``)的碰撞球,ee_link 坐标系;x / 半厚度取 ``FINGER_PROFILE``。

    默认每个 FINGER_PROFILE 截面放 1 ~ 3 个球(球间有 1 ~ 2 cm 的空隙,对微波炉这种大块的墙够用)。``spacing`` 给定时
    沿手指长度和高度都按这个间距铺满(取不大于 FINGER_R 时球之间没有空隙):细铁丝(面包架)会从稀疏的球之间穿过去,
    cuRobo 规划出手指穿过铁丝的路径,合拢夹爪时物理炸掉。"""
    out = []
    if spacing is None:
        profile = [(x, h, 3 if h > 2 * FINGER_R else 1) for x, h in FINGER_PROFILE]
    else:
        xs = np.arange(FINGER_PROFILE[0][0], FINGER_PROFILE[-1][0] + 1e-9, spacing)
        hs = np.interp(xs, [x for x, _ in FINGER_PROFILE], [h for _, h in FINGER_PROFILE])
        profile = [(x, h, max(1, int(np.ceil((2 * h - 2 * FINGER_R) / spacing)) + 1)) for x, h in zip(xs, hs)]
    for x, h, nz in profile:
        for y in (-(gap / 2 + FINGER_R), gap / 2 + FINGER_R):
            for z in np.linspace(-h + FINGER_R, h - FINGER_R, nz) if nz > 1 else [0.0]:
                out.append([float(x), y, float(z), FINGER_R])
    return out


def _held_spheres(ee_T_obj, lo, hi):
    """手中的圆物体(包围盒 lo / hi,物体坐标系)沿长轴排 3 个球,半径 = ``HELD_R_FRAC`` x 次短边的一半,ee_link 坐标系。"""
    ext = hi - lo
    long_ax = int(np.argmax(ext))
    r = HELD_R_FRAC * float(np.sort(ext)[1]) / 2
    c = (lo + hi) / 2
    out = []
    for t in np.linspace(-1, 1, 3) * max(ext[long_ax] / 2 - r, 0.0):
        p = c.copy()
        p[long_ax] += t
        out.append([*(ee_T_obj @ np.append(p, 1.0))[:3], r])
    return out


def _plan_once(env, pose, start=None, tries=FOLD_RETRIES):
    """一次规划到 ``pose``;规划出腕部向上臂折回的路径(``_folds``)时重新规划,最多 ``tries`` 次,都折回时返回 None。
    cuRobo 的自碰撞模型漏掉这种构型,仿真里手腕撞上上臂,物理直接变成 NaN(面包架紧挨着右臂基座时很常见)。"""
    global LAST_PLAN_FAIL
    for _ in range(tries):
        try:
            res = env.plan_to_pose(ARM, pose, start_joints=start)
        except PlanFailError as e:
            LAST_PLAN_FAIL = str(e).split(":", 1)[-1].strip()[:50]
            break
        fold = _folds([res])
        if fold is None:
            return res
        LAST_PLAN_FAIL = fold
    return None


def _plan_linear(env, pose, start=None, axis="x"):
    """一次规划到 ``pose``,用 cuRobo 自己的直线运动约束(``ToolPoseCriteria.linear_motion``,同 cuRobo 的 plan_grasp):
    除了沿 ee ``axis`` 方向,中间路点的位置和姿态都贴着目标位姿。规划不到(或都折回)时返回 None。"""
    from curobo.types import ToolPoseCriteria

    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    frames = list(planner.motion_planner.tool_frames)
    linear = ToolPoseCriteria.linear_motion(axis=axis, non_terminal_scale=1.0, project_distance_to_goal=True)
    planner.motion_planner.update_tool_pose_criteria({f: linear for f in frames})
    try:
        res = None
        for _ in range(FOLD_RETRIES):
            try:
                r = env.plan_to_pose(ARM, pose, start_joints=start)
            except PlanFailError:
                break
            if _folds([r]) is None:
                res = r
                break
    finally:
        planner.motion_planner.update_tool_pose_criteria({f: ToolPoseCriteria() for f in frames})
    return res


LAST_PLAN_FAIL = ""  # _plan_once 最近一次失败的原因(写进失败信息里)


def _plan_joint_once(env, planner, goal, tries=FOLD_RETRIES):
    """同 ``_plan_once``,关节空间规划到 ``goal``。"""
    for _ in range(tries):
        res = planner.plan_joint(env.get_arm_joints(ARM), np.asarray(goal, float))
        if res["status"] != "Success":
            return None
        if _folds([res]) is None:
            return res
    return None


def _pick_place_direct(
    env,
    label,
    *,
    set_world,
    grasps,
    open_tool,
    held_tool,
    releases,
    close,
    carry_slow=CARRY_SLOW,
    after_close=None,
    retreat=0.0,
    reuse_tol=PRED_REUSE_TOL,
    grasp_open=1.0,
    release_open=1.0,
    retreat_fallback=False,
):
    """三段运动、每段一次 cuRobo 规划的抓取 + 放置(见模块说明):

    - ``set_world(with_obj, tool)``:设置 cuRobo 的碰撞世界(固定障碍物,``with_obj`` 为真时加上物体:第 1 段传 "grasp"、
      第 3 段先传 "return"、松手后退时传 "return_tight",物体模型可以按阶段取不同的大小)和 tool 碰撞球;
    - ``grasps``:抓取候选 ``[(ee 位姿, score), ...]``,按优先级;``open_tool``:夹爪张开时的手指碰撞球;
    - ``held_tool(ee_T_obj)``:夹住物体后的 tool 碰撞球(手指 + 物体);
    - ``releases(ee, ee_T_obj)``:松手位姿候选 ``[ee 位姿, ...]``(``ee`` 是抓取时的 ee 位姿矩阵);
    - ``close(grasp)``:合拢夹爪;``after_close()``:合拢后的钩子(例如按物体的新位姿重新设置障碍物);
    - ``retreat`` > 0:松开后先沿接近方向退出这么远(一次规划,不带物体),再带着原尺寸的物体回初始位姿。竖放在隔板上的
      瓶子要这样:松手位姿上 cuRobo 偏保守的腕部碰撞球已经伸进了瓶子的包围盒,直接回程只能把物体缩小 / 去掉,回程路径
      贴着瓶子扫过去,一蹭就倒。
    - ``grasp_open`` / ``release_open``:去抓取前 / 松手时夹爪张开到的归一化开度(默认全开)。碗沿这种斜着抓的薄壁,
      全开时里面那根手指伸到碗底以下,要只张开一部分。``release_open`` < 1 时退出后、回初始位姿前再全开(任务判定要求
      夹爪张开)。
    - ``retreat_fallback``:沿接近方向的直线退出规划不到时,改为不带直线约束规划到同一位姿,还不行就不退、直接回程
      (默认 False:直接报错)。
    """
    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    home_pose = env.get_arm_pose(ARM)
    obj0 = _mat(env.get_obj_pose(label))

    def plan_release(ee, ee_T_obj, start=None):
        set_world(False, held_tool(ee_T_obj))
        for rel in releases(ee, ee_T_obj):
            res = _plan_once(env, rel, start=start)
            if res is not None:
                return res
        return None

    # ---- 1. 初始 -> 抓取:取第一个第 1 段和预测的第 2 段都能规划的候选
    reasons, chosen = [], None
    for i, (g, _) in enumerate(grasps):
        set_world("grasp", open_tool)
        r1 = _plan_once(env, g)
        if r1 is None:
            reasons.append(f"{i}: grasp unreachable ({LAST_PLAN_FAIL})")
            continue
        gm = _mat(g)
        r2 = plan_release(gm, np.linalg.inv(gm) @ obj0, start=np.asarray(r1["position"][-1]))
        if r2 is None:
            reasons.append(f"{i}: no release pose reachable from this grasp ({LAST_PLAN_FAIL})")
            continue
        chosen = g, r1, r2
        break
    if chosen is None:
        raise PlanFailError(f"{ARM}: no grasp with a reachable release: " + "; ".join(reasons[:4] + reasons[-2:]))
    g, r1, pred_r2 = chosen
    pred_ee_T_obj = np.linalg.inv(_mat(g)) @ obj0
    env.open_gripper(ARM, grasp_open)
    env.execute_plan(ARM, r1)
    close(g)
    env.delay(GRASP_SETTLE)
    if after_close is not None:
        after_close()

    # ---- 2. 按实测的手中位姿:抓取 -> 松手位姿,一次规划
    ee = _mat(env.get_arm_pose(ARM))
    ee_T_obj = np.linalg.inv(ee) @ _mat(env.get_obj_pose(label))
    r2 = plan_release(ee, ee_T_obj)
    if r2 is None:
        # 合拢时物体在指间挪动了,按实测的手中位姿规划不到时,退回到选候选时预测的那条第 2 段(起点就是现在的关节角)
        r2 = pred_r2
        moved_by = float(np.linalg.norm(ee_T_obj[:3, 3] - pred_ee_T_obj[:3, 3]))
        if moved_by > reuse_tol:
            raise PlanFailError(
                f"{ARM}: no release pose reachable with the measured grasp ({label} moved {moved_by:.3f} m)"
            )
    env.execute_plan(ARM, r2, carry_slow)
    env.open_gripper(ARM, release_open)
    env.delay(RELEASE_SETTLE)
    tries = ("return", False)
    if retreat > 0:
        ee = _mat(env.get_arm_pose(ARM))
        back = ee.copy()
        back[:3, 3] -= retreat * ee[:3, 0]
        rb = None
        for with_obj in ("return_tight", False):  # 带着(缩小的)物体退,免得退的路上扫到它
            set_world(with_obj, open_tool)
            rb = _plan_linear(env, _pose(back))
            if rb is not None:
                break
        if rb is None and retreat_fallback:
            rb = _plan_once(env, _pose(back))
        if rb is not None:
            env.execute_plan(ARM, rb)
            tries = ("return",)
        elif not retreat_fallback:
            raise PlanFailError(f"{ARM}: cannot back {retreat:.2f} m out of the release pose")
    if release_open < 1.0:
        env.open_gripper(ARM)

    # ---- 3. 松手位姿 -> 初始关节,一次规划。先把物体当障碍物(免得退出时碰到它);物体松手后滚 / 倒到指间时起点就和它的
    # 盒子重叠,cuRobo 判起点碰撞,这时不带物体再规划。关节空间规划不通时改为规划到初始末端位姿
    r3 = None
    for with_obj in tries:
        set_world(with_obj, open_tool)
        r3 = _plan_joint_once(env, planner, env.init_joints[ARM]) or _plan_once(env, home_pose)
        if r3 is not None:
            break
    if r3 is None:
        raise PlanFailError(f"{ARM}: no way back to the start pose")
    env.execute_plan(ARM, r3)


def _run_direct(env, play):
    """运行 ``play()``;无论成败都清掉任务障碍物和 tool 碰撞球。失败时(先清掉,手指常贴着物体 / 障碍物,回初始位姿的规划会判
    起点碰撞)也像成功时那样松开夹爪、回到初始位姿:手里拿着东西、停在别处结束的 episode 之后,同一进程的下一次 reset 会让
    机器人位姿变成 NaN。"""
    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    root = env.robot(ARM).entity_origin_pose
    try:
        play()
    except Exception:
        planner.set_obstacles({}, root)
        planner.set_tool_spheres(None)
        try:
            env.open_gripper(ARM)
            env.back_to_origin(ARM)
        except Exception as e:
            log.warning("cleanup after a failed attempt failed: %s: %s", type(e).__name__, e)
        raise
    finally:
        planner.set_obstacles({}, root)
        planner.set_tool_spheres(None)


def _lift_tips(env, g):
    """自上而下的抓取 ``g``:指尖离桌面不足 ``TOP_TIP_CLEAR`` 时沿接近方向往回提(只改位置)。"""
    m = _mat(g)
    a = m[:3, 0]
    tip_z = m[2, 3] + (env.robot(ARM).gripper_bias + TIP_BEYOND_TCP) * a[2]
    deficit = _table_z(env) + TOP_TIP_CLEAR - tip_z
    if deficit <= 0 or a[2] > -0.1:
        return g
    g = np.array(g, dtype=float)
    g[:3] -= deficit / -a[2] * a
    return g


def _held_points(spec, lo, hi):
    """手中物体的采样点(物体坐标系):包围盒 27 点,圆的物体只用面中心和中心。"""
    pts = GRID * (hi - lo) + lo
    return pts[FACE_CENTERS | (np.sum(GRID == 0.5, axis=1) == 3)] if spec.round else pts


# ------------------------------------------------------------------ 夹爪开度、薄片物体、朝向

SLAB_R_FRAC = 0.8  # 手中物体碰撞球半径 = 这个比例 x 物体半厚度(Spec.held_r 未给时)
# 手中物体碰撞球在厚度中面上的间距:15 cm 长的手机按 2.4 cm 排时斜着进槽,球之间的空当擦到槽口,手机在指间被拧歪 20 多度
SLAB_SPACING = 0.012
OBJ_SHRINK_XY = 0.95  # 物体当障碍物时长宽方向缩小这么多(手指 / 掌心抓取时本来就贴着它)


def _grip_val(env, gap):
    """两指间距 ``gap``(米)对应的归一化夹爪开度。每根手指的关节值是间距的一半,开度 v 对应关节值
    ``v * (scale[1] - scale[0]) + scale[0]``(X5:scale = [-0.01, 0.044],v = 0 时把手指往 -1 cm 压,即用满力夹紧)。"""
    lo, hi = env.robot(ARM).gripper_scale
    return float(np.clip((gap / 2 - lo) / (hi - lo), 0.0, 1.0))


def _slab_spheres(ee_T_obj, lo, hi, r=None):
    """手中薄片物体(包围盒 lo / hi,物体坐标系,厚度沿物体 z)在厚度中面上按 ``SLAB_SPACING`` 排一层半径 ``r`` 的碰撞球,
    ee_link 坐标系。"""
    if r is None:
        r = SLAB_R_FRAC * float(hi[2] - lo[2]) / 2

    def axis(a, b):
        return np.linspace(a + r, b - r, max(2, int(np.ceil((b - a - 2 * r) / SLAB_SPACING)) + 1))

    xs, ys = axis(lo[0], hi[0]), axis(lo[1], hi[1])
    z = (lo[2] + hi[2]) / 2
    return [[*(ee_T_obj @ [x, y, z, 1.0])[:3], r] for x in xs for y in ys]


def _obj_box_xy(env, label):
    """物体包围盒长宽方向缩小 ``OBJ_SHRINK_XY``(高不变)当障碍物 -> (dims, 世界位姿)。"""
    lo, hi = _bbox(env, label)
    c, h = (lo + hi) / 2, (hi - lo) / 2 * np.array([OBJ_SHRINK_XY, OBJ_SHRINK_XY, 1.0])
    return _box(c - h, c + h, _mat(env.get_obj_pose(label)))


def _rz(deg):
    m = np.eye(4)
    m[:3, :3] = t3d.axangles.axangle2mat([0.0, 0.0, 1.0], np.radians(deg))
    return m


def _heading(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


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
