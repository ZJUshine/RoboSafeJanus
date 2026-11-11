"""heat_<物体>_microwave 系列任务共用的脚本专家(右臂;物体在微波炉右前方,门开在左侧)。
各任务的专家(``experts/<task>.py``)只是用自己的 ``Spec`` 调用 ``play_once``。

三步(``common._pick_place_direct``,只在夹爪开合时停):

1. 抓取:GraspGenX 候选,接近方向 z 分量在 ``Spec.approach_z`` 内(偏好朝炉腔里向下倾 ``Spec.prefer_pitch``)、手指水平且垂直于炉腔朝里的方向(``FINGER_ACROSS_MIN``)、夹持宽度不超过 ``FINGER_MAX``;取第一个能规划到、且能接着规划到松手位姿的。
2. 放置:先原地抬起 ``LIFT`` 并把腕部转成伸入姿态(接近方向水平朝炉腔里),再伸到转盘中心上方松手。物体是摩擦夹持,腕部
   一转它就往下荡;转腕和伸入叠在一起时物体边进边沉,把微波炉顶走,所以转腕单独做完。手中那段按 ``CARRY_SLOW`` 放慢。
3. 退出归位:松开后沿接近方向退出炉口(直接回程的起点在炉腔里,cuRobo 规划不出来),再回初始关节。

碰撞世界:微波炉(外壳减炉腔、控制面板、开着的门)和物体交给 cuRobo(``CuroboPlanner.set_obstacles``),手指和手中物体用
``CuroboPlanner.set_tool_spheres`` 补上(cuRobo 的碰撞球只到腕部)。规划不到的尝试交给采集脚本丢弃,换下一个 seed。
"""

from dataclasses import dataclass

import numpy as np
import transforms3d as t3d

from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts import common


@dataclass(frozen=True)
class Spec:
    label: str
    # 圆的物体(鸡蛋、罐头):包围盒角点远在物体外面,障碍物按 OBJ_SHRINK 缩小、手中碰撞球沿长轴排 3 个
    round: bool = False
    approach_z: tuple = (-1.0, -0.7)  # 接近方向单位向量 z 分量的范围,默认自上而下
    prefer_pitch: float = 90.0  # 偏好的接近方向:朝炉腔里向下倾这么多度(90 = 竖直向下)


@dataclass(frozen=True)
class Microwave:
    """微波炉碰撞体(微波炉坐标系,米;前面朝 -y,炉门在左侧)。"""

    shell: tuple  # 外壳
    panel: tuple  # 比炉口平面再往前凸出的部分(控制面板、旋钮),若干个盒子
    cavity: tuple  # 炉口 / 炉腔的 x 范围、z 范围
    cavity_back: float  # 炉腔后壁的 y;炉腔从炉口一直通到这里
    opening_y: float  # 炉口平面
    plate_z: float  # 转盘上表面
    plate_xy: tuple  # 转盘中心
    door: tuple  # 门全开时门 link 的包围盒(下角、上角);任务每个 episode 都把门开到同一个角度
    # 碰撞盒子各向外扩这么多:不够时腕部 / 相机蹭到炉口把微波炉推走,多了手指 + 物体的碰撞球过不去
    inflate: float = 0.006


MICROWAVES = {
    # SimReady 台式微波炉(D00516):body 0.51 x 0.37 x 0.30 m,门在左边缘、关节 0 度就是开 90 度。
    # 数字来自 src/data_collect/probe_articulation.py(各 part 的 bbox)和 probe_cavity.py(炉腔内壁射线)。
    "SimReady_microwave/00000": Microwave(
        shell=((-0.254, -0.164, 0.0), (0.254, 0.203, 0.299)),
        panel=(((0.150, -0.175, 0.030), (0.254, -0.164, 0.240)),),
        cavity=((-0.224, 0.119), (0.012, 0.267)),
        cavity_back=0.168,
        opening_y=-0.164,
        plate_z=0.076,
        plate_xy=(-0.050, 0.010),
        door=((-0.2889, -0.560, 0.015), (-0.2511, -0.160, 0.298)),
    ),
}

FINGER_ACROSS_MIN = 0.9  # 手指方向与"垂直于炉腔朝里"方向夹角余弦的下限:转成伸入姿态时只绕手指轴转
MAX_GRASPS = 200  # 上游按判别器分数排好,只试前面这些
LIFT = 0.06  # 抬起量:转腕时物体不蹭桌面
TCP_HEIGHTS = (0.06, 0.065, 0.055, 0.05, 0.07)  # 松手时 TCP 离转盘的高度(按顺序试)
BACK_CLEAR = 0.015  # 物体最里面的点离炉腔后壁至少这么远(长物体转平后从 TCP 往里伸出很多)
OBJ_CLEAR = 0.005  # 松手时物体离转盘 / 炉腔顶至少这么远
RETREAT = (0.12, 0.08, 0.05)  # 松手后沿接近方向退出的距离(按顺序试,长的受直线约束 / 关节限位退不出来时退短一点)


# ------------------------------------------------------------------ 碰撞世界


def _microwave(env):
    """微波炉位姿、几何(``MICROWAVES`` 按 ``<类目>/<编号>`` 查)、炉腔朝里的水平方向(微波炉坐标系 +y)。"""
    mw = common._mat(env.get_obj_pose("microwave"))
    category, index, _, _ = env.get_obj_asset("microwave")
    key = f"{category}/{index:05d}"
    if key not in MICROWAVES:
        raise PlanFailError(f"no cavity geometry for microwave {key}; measure it and add it to MICROWAVES")
    return mw, MICROWAVES[key], common._horizontal(mw[:3, 1])


def _microwave_obstacles(geom, mwm):
    """微波炉碰撞盒(外壳减炉腔:左壁、右侧电控区、腔底、腔顶、后壁;控制面板;开着的门),各向外扩 ``geom.inflate``。"""
    (x0, y0, z0), (x1, y1, z1) = geom.shell
    (cx0, cx1), (cz0, cz1) = geom.cavity
    back = geom.cavity_back
    boxes = [
        ((x0, y0, z0), (cx0, y1, z1)),
        ((cx1, y0, z0), (x1, y1, z1)),
        ((cx0, y0, z0), (cx1, y1, cz0)),
        ((cx0, y0, cz1), (cx1, y1, z1)),
        ((cx0, back, cz0), (cx1, y1, cz1)),
        *geom.panel,
        geom.door,
    ]
    return {
        f"mw_{i}": common._box(np.asarray(lo) - geom.inflate, np.asarray(hi) + geom.inflate, mwm)
        for i, (lo, hi) in enumerate(boxes)
    }


# ------------------------------------------------------------------ 位姿


def _insert_rot(inward, finger):
    """伸入姿态:x(接近方向)= ``inward``,y(手指)水平、与 ``finger`` 同向。"""
    y = np.cross([0.0, 0.0, 1.0], inward)
    if y @ finger < 0:
        y = -y
    return np.stack([inward, y, np.cross(inward, y)], axis=1)


def _grasps(env, spec, inward):
    """GraspGenX 抓取候选(见模块说明第 1 条),按优先级。"""
    candidates = env.grasp_candidates(
        spec.label,
        common.ARM,
        prefer=common._down_inward(inward, spec.prefer_pitch),
        approach_z=spec.approach_z,
        max_candidates=1000,
    )
    across = np.cross([0.0, 0.0, 1.0], inward)
    pts = common._bbox_points(env, spec.label)[common.FACE_CENTERS]
    out = [
        (g, sc)
        for g, sc in candidates
        if abs(common._mat(g)[:3, 1] @ across) > FINGER_ACROSS_MIN
        and np.ptp(pts @ common._mat(g)[:3, 1]) <= common.FINGER_MAX
    ]
    if not out:
        raise PlanFailError(f"{common.ARM}: no grasp candidate for {spec.label} ({len(candidates)} before filtering)")
    return out[:MAX_GRASPS]


def _release_poses(env, geom, mwm, inward, finger, held_ee, face_ee):
    """炉腔内的松手位姿候选(按 ``TCP_HEIGHTS``):TCP 在转盘中心上方,物体最里面的点(``held_ee``,ee 坐标系)离后壁
    ``BACK_CLEAR``;物体(包围盒面中心 ``face_ee``:转平后包围盒的角点比物体本身低得多)碰到转盘或炉腔顶的去掉。"""
    rot = _insert_rot(inward, finger)
    pts = held_ee @ rot.T  # 物体采样点相对 ee 原点的位移(世界坐标系)
    face_z = (face_ee @ rot.T)[:, 2]
    bias = env.robot(common.ARM).gripper_bias
    plate = (mwm @ [*geom.plate_xy, geom.plate_z, 1.0])[:3]
    opening = (mwm @ [0.0, geom.opening_y, 0.0, 1.0])[:3]
    ceiling = (mwm @ [0.0, 0.0, geom.cavity[1][1], 1.0])[2]
    depth = geom.cavity_back - geom.opening_y
    quat = t3d.quaternions.mat2quat(rot)
    out = []
    for h in TCP_HEIGHTS:
        end = plate + [0.0, 0.0, h] - bias * inward
        inner = float(((end + pts - opening) @ inward).max())
        end -= max(0.0, inner - (depth - BACK_CLEAR)) * inward
        z = end[2] + face_z
        if z.min() >= plate[2] + OBJ_CLEAR and z.max() <= ceiling - OBJ_CLEAR:
            out.append(np.concatenate([end, quat]))
    return out


# ------------------------------------------------------------------ 流程


def _play(env, spec):
    label = spec.label
    mwm, geom, inward = _microwave(env)
    planner = env.robot_manager.planner[env.robot(common.ARM).robot_name]
    root = env.robot(common.ARM).entity_origin_pose
    fixed = _microwave_obstacles(geom, mwm)
    lo, hi = common._bbox(env, label)
    pts_obj = common._held_points(spec, lo, hi)
    face_obj = (common.GRID * (hi - lo) + lo)[common.FACE_CENTERS]
    shrink = common.OBJ_SHRINK if spec.round else common.BOX_SHRINK

    def set_world(with_obj, tool):
        obstacles = dict(fixed)
        if with_obj:
            obstacles[label] = common._obj_box(env, label, shrink)
        planner.set_obstacles(obstacles, root)
        planner.set_tool_spheres(tool)

    def held_ee(ee_T_obj, pts=pts_obj):
        return (np.c_[pts, np.ones(len(pts))] @ ee_T_obj.T)[:, :3]

    def held_tool(ee_T_obj):
        width = float(np.ptp(held_ee(ee_T_obj)[:, 1]))  # 物体沿手指方向(ee y)的宽度
        obj = common._held_spheres(ee_T_obj, lo, hi) if spec.round else common._bbox_spheres(ee_T_obj, lo, hi)
        return common._finger_spheres(min(width, common.GRIPPER_OPEN)) + obj

    def prepare(ee):
        m = ee.copy()
        m[:3, :3] = _insert_rot(inward, ee[:3, 1])
        m[2, 3] += LIFT
        return [common._pose(m)]

    def releases(ee, ee_T_obj):
        return _release_poses(env, geom, mwm, inward, ee[:3, 1], held_ee(ee_T_obj), held_ee(ee_T_obj, face_obj))

    common._pick_place_direct(
        env,
        label,
        set_world=set_world,
        grasps=_grasps(env, spec, inward),
        open_tool=common._finger_spheres(common.GRIPPER_OPEN),
        held_tool=held_tool,
        releases=releases,
        prepare=prepare,
        retreat=RETREAT,
        close=lambda g: env.close_gripper(common.ARM),
    )


def play_once(env, spec):
    common._run_direct(env, lambda: _play(env, spec))
