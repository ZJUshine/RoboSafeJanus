"""heat_<物体>_microwave 系列任务共用的脚本专家(右臂;物体在微波炉右前方,门开在左侧)。
各任务的专家(``experts/<task>.py``)只是用自己的 ``Spec`` 调用 ``play_once``。

三段运动,每段一次 cuRobo 规划,只在夹爪开合时停:初始 -> 抓取 -> 合拢 -> 炉腔内松手位姿 -> 松开 -> 回初始
(``common._pick_place_direct``)。微波炉(外壳减炉腔、控制面板、按当前门角摆放的门)和物体放进
cuRobo 的碰撞世界(``CuroboPlanner.set_obstacles``),手指和手中物体用 ``CuroboPlanner.set_tool_spheres`` 补上
(cuRobo 的碰撞球只到腕部)。

1. 抓取(``Spec.grasp``,候选都来自 GraspGenX):
   - ``"graspgen"``(鸡蛋):接近方向 z 分量在 ``Spec.approach_z`` 内、手指水平(``FINGER_Z_MAX``),``Spec.across_long_axis``
     时手指横跨物体短径;偏好朝炉腔里、向下倾 ``Spec.prefer_pitch`` 的。侧躺的鸡蛋只会沿垂直于长轴的方向滚:伸入时手指
     垂直于炉腔朝里的方向,鸡蛋长轴就朝里,松手后只能向两侧滚,被炉壁挡住。
   - ``"top"``(罐头、罐装饮料、三明治):自上而下(接近方向 z 分量在 ``TOP_APPROACH_Z`` 内),手指与"垂直于炉腔朝里"的
     方向夹角余弦大于 ``FINGER_ACROSS_MIN``(伸入时末端只绕手指轴转平),夹持宽度不超过 ``FINGER_MAX``。
   取第一个第 1 段(初始 -> 抓取,夹爪张开、物体是障碍物)和预测的第 2 段都能规划的候选;合拢夹爪合到底。
2. 合拢后按实测的 ee->物体 变换把物体挂到夹爪上、从碰撞世界移除;抓取 -> 松手位姿一次规划(抬起、转向、伸入都在这一条
   轨迹里),整条轨迹按 ``CARRY_SLOW`` 放慢。松手位姿:接近方向水平(或下倾 ``INSERT_PITCHES``)朝炉腔里、手指水平,TCP 在
   转盘中心上方 ``TCP_HEIGHTS*``;长物体往外退,让它最里面的点离后壁 ``BACK_CLEAR``,物体底面离转盘不低于 ``OBJ_FLOOR``。
3. 松开后物体放回碰撞世界;松手位姿 -> 初始关节一次规划,起点就碰到物体时去掉物体再试,关节空间规划不通时改为规划到
   初始末端位姿。

规划出的路径腕部向上臂折回(``_folds``,cuRobo 的自碰撞模型漏掉这种构型,仿真里物理直接变成 NaN)时重新规划。
规划不到、判定不过的尝试直接交给采集脚本丢弃,换下一个 seed。
"""

from dataclasses import dataclass
import json
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import transforms3d as t3d

from env.global_configs import OBJECTS_PATH
from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.experts import common
from task.RoboDojo.tasks.heat_X_microwave import door_joint


@dataclass(frozen=True)
class Spec:
    """一个物体的抓取方式,见模块说明第 1 条。"""

    label: str
    grasp: str  # "graspgen" | "top"
    prefer_pitch: float = 40.0  # graspgen:偏好的接近方向向下倾的角度
    approach_z: tuple = (-1.0, -0.25)  # graspgen:接近方向单位向量 z 分量的范围
    across_long_axis: bool = False  # graspgen:手指横跨物体短径
    round: bool = False  # 圆的物体:沿长轴排 3 个碰撞球、包围盒缩小 OBJ_SHRINK 当障碍物(包围盒角点远在物体外面)
    # 绕物体 z 轴旋转对称(罐头、罐装饮料):GraspGenX 的抓取绕这条轴转一圈也都是这个物体的抓取,用来补齐手指朝向
    # (扁罐头 GraspGenX 只给两三个自上而下的抓取,朝向不对就没有可用的)
    axisym: bool = False
    # 去抓取时物体当障碍物的缩放(长宽高,物体坐标系);None 时圆的物体取 OBJ_SHRINK、其余取 BOX_SHRINK。软的扁物体
    # (三明治 4.7 cm 高)自上而下抓时,张开的手指在抓取位姿上就伸到了包围盒里,高度要压低
    grasp_shrink: tuple = None
    # top:把 GraspGenX 的抓取沿 ee z(垂直于接近方向和手指)平移到物体包围盒中心线上。长物体(三明治)转平后这条轴竖着,
    # 抓在偏一头 2 cm 的地方时物体下半截垂得太低,松手位姿就得抬到腕部顶住炉腔顶(同 put_X_toaster 把抓取沿手指方向
    # 挪到厚度中面)
    center_slide: bool = False
    # 合拢后物体在指间挪动不超过这么多时,沿用合拢前预测的第 2 段(见 _pick_place_direct 的 reuse_tol);None 时取
    # PRED_REUSE_TOL。软的三明治一夹就被挤偏 1 ~ 1.5 cm
    reuse_tol: float = None


# 抓取
FINGER_Z_MAX = 0.35  # graspgen:手指方向 |z| 上限:手指水平,夹爪竖直方向才能放进炉口
FINGER_ALONG_MAX = 0.3  # graspgen:手指方向与物体长轴(物体坐标系 z)夹角余弦的上限
FINGER_ACROSS_MIN = 0.9  # top:手指方向与"垂直于炉腔朝里"方向夹角余弦的下限
TOP_APPROACH_Z = (-1.0, -0.7)  # top:接近方向单位向量 z 分量的范围
# top:TCP 离物体包围盒中心(水平)不超过这么远:抓在长物体(三明治)一头时,转平后物体往下垂,炉腔里放不下
TOP_CENTER_TOL = 0.025
AXISYM_STEP = 10.0  # axisym:绕物体 z 轴每隔这么多度复制一份抓取
MAX_GRASPS = 40
MW_INFLATE = 0.003  # 微波炉碰撞盒子交给 cuRobo 前各向外扩这么多


@dataclass(frozen=True)
class Microwave:
    """微波炉原尺寸下的碰撞体(微波炉坐标系,米;前面朝 -y,炉门在左侧),来自 PhysX 射线探测
    (/tmp/hz/collision_probe.py、/tmp/hz/mw2/probe2.py)。"""

    shell: tuple  # 外壳
    panel: tuple  # 比炉口平面再往前凸出的部分(控制面板、旋钮),若干个盒子
    cavity: tuple  # 炉口 / 炉腔的 x 范围、z 范围
    cavity_back: float  # 炉腔后壁的 y;炉腔从炉口一直通到这里
    opening_y: float  # 炉口平面
    plate_z: float  # 转盘上表面
    plate_xy: tuple  # 转盘中心

    def scaled(self, k):
        """``scaled_variant.py`` 按 ``k`` 等比缩放后的资产的几何。"""
        return Microwave(*(np.asarray(v, dtype=float) * k for v in vars(self).values()))


# 按原始资产(<category>/<index>)查;scaled_variant.py 生成的缩放资产按 metadata 的 ext_link.scaled_from 查
MICROWAVES = {
    "ArtVIP_microwave/00000": Microwave(
        shell=((-0.284, -0.188, -0.012), (0.290, 0.203, 0.329)),
        panel=(((0.155, -0.229, -0.012), (0.290, -0.188, 0.329)),),  # 控制面板那一侧前面板往前 4 cm
        cavity=((-0.215, 0.145), (0.088, 0.269)),
        cavity_back=0.131,
        opening_y=-0.188,
        plate_z=0.083,
        plate_xy=(-0.044, -0.023),
    ),
    # 炉腔比 00000 高(转盘以上 24 cm)、深(转盘中心离炉口 22 cm);控制面板往前 3 cm,面板上的旋钮再往前 2 cm
    "ArtVIP_microwave/00002": Microwave(
        shell=((-0.278, -0.179, 0.0), (0.280, 0.210, 0.329)),
        panel=(((0.140, -0.212, 0.0), (0.280, -0.179, 0.329)), ((0.170, -0.232, 0.075), (0.235, -0.212, 0.135))),
        cavity=((-0.240, 0.125), (0.077, 0.310)),
        cavity_back=0.198,
        opening_y=-0.179,
        plate_z=0.072,
        plate_xy=(-0.068, 0.039),
    ),
}


# 伸入
# 0.75 倍 00000 的炉口只有约 13.6 cm 高,能过的只有接近水平、TCP 离转盘 4.5 ~ 5.3 cm 这一小段(/tmp/hz/insert_sweep.py)
TCP_HEIGHTS = (0.050, 0.047, 0.053, 0.045)  # 伸到底时 TCP 离转盘的高度
INSERT_PITCHES = (0.0, 5.0)  # 接近方向向下倾斜的角度(度)
# 炉腔在转盘以上不低于 TALL_CAVITY 时(原尺寸 ArtVIP 00000 18.6 cm、0.75 倍 00002 17.8 cm)夹爪可以放高一点:水平伸入
# 时 TCP 离转盘 5 ~ 7 cm、倾 5 度时 4 ~ 5 cm 都能过(/tmp/hz/full_size/reach_fine.py、/tmp/hz/mw2/reach_mw.py)
TALL_CAVITY = 0.16
TCP_HEIGHTS_TALL = (0.06, 0.065, 0.055, 0.05, 0.07)
# 长物体(罐装饮料)抓在顶部、转平后从 TCP 往里伸出十几厘米:TCP 停在转盘中心上方时会顶到炉腔后壁,
# 这时松手位姿往外退,让物体最里面的采样点离后壁 BACK_CLEAR
BACK_CLEAR = 0.015
# 直线伸入的起点:TCP 在炉口外这么远。从物体在微波炉前方抓起时(鸡蛋、罐头、汉堡)起点要在物体后方,否则夹爪边转向
# 边前移,指尖扫到控制面板;从微波炉右侧抓起时(饮料、三明治)远的起点离基座太近,没有 IK 解,只能用近的
OBJ_FLOOR = 0.005  # 松手时物体最低的包围盒面中心至少高出转盘这么多


def _insert_rot(inward, finger, pitch):
    """伸入时的 ee 姿态:x(接近方向)= ``inward`` 向下倾 ``pitch`` 度,y(手指)水平且与当前手指同向。"""
    p = np.radians(pitch)
    x = np.cos(p) * inward + np.sin(p) * np.array([0.0, 0.0, -1.0])
    y = np.cross([0.0, 0.0, 1.0], inward)
    if y @ finger < 0:
        y = -y
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _microwave(env):
    """微波炉资产的缩放比例(``scaled_variant.py`` 生成的资产是 ``metadata.json`` 的 ``ext_link.scale``,其余为 1)
    和缩放后的几何。要直接读文件:layout manager 载入的 metadata 只保留 physics / visual / geometry / active / passive。"""
    category, index, _, _ = env.get_obj_asset("microwave")
    with open(os.path.join(OBJECTS_PATH, "Articulation", category, f"{index:05d}", "metadata.json")) as f:
        link = json.load(f).get("ext_link", {})
    source = link.get("scaled_from", f"Articulation/{category}/{index:05d}").split("/", 1)[1]
    if source not in MICROWAVES:
        raise PlanFailError(f"no cavity geometry for microwave {source}; measure it and add it to MICROWAVES")
    k = float(link.get("scale", 1.0))
    return k, MICROWAVES[source].scaled(k)


def _inward(env):
    """微波炉位姿、缩放后的几何、炉腔朝里的水平方向(微波炉坐标系 +y)。"""
    mw = common._mat(env.get_obj_pose("microwave"))
    _, geom = _microwave(env)
    inward = mw[:3, 1].copy()
    inward[2] = 0.0
    return mw, geom, inward / np.linalg.norm(inward)


# ------------------------------------------------------------------ 一次规划的流程(微波炉 / 烤面包机 / 冰箱共用)


def _microwave_boxes(geom):
    """微波炉碰撞体(缩放后的微波炉坐标系):外壳减去炉腔(左壁、右侧电控区、腔底、腔顶、后壁)+ 控制面板 / 旋钮。"""
    (x0, y0, z0), (x1, y1, z1) = np.asarray(geom.shell)
    (cx0, cx1), (cz0, cz1) = np.asarray(geom.cavity)
    back = float(geom.cavity_back)
    boxes = [
        ((x0, y0, z0), (cx0, y1, z1)),
        ((cx1, y0, z0), (x1, y1, z1)),
        ((cx0, y0, z0), (cx1, y1, cz0)),
        ((cx0, y0, cz1), (cx1, y1, z1)),
        ((cx0, back, cz0), (cx1, y1, cz1)),
    ]
    boxes += [tuple(map(tuple, b)) for b in np.asarray(geom.panel)]
    return boxes


def _door_box(env):
    """开着的门:门 link 的外观包围盒(门 link 坐标系,来自 USD,静态)绕铰链转到当前关节角 -> (dims, 世界位姿)。"""
    lm = env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "microwave"))
    stage = obj.stage
    joint_prim = next(
        p for p in Usd.PrimRange(stage.GetPrimAtPath(obj.usd_prim_path)) if p.GetName() == door_joint(obj)
    )
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    if (joint.GetAxisAttr().Get() or "Z") != "Z":
        raise PlanFailError(f"door joint {joint_prim.GetName()} axis is not Z")
    body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    body1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
    root = stage.GetPrimAtPath(obj.usd_prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    rng = cache.ComputeRelativeBound(body1, body1).ComputeAlignedRange()
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())

    def rel(prim):  # prim 在微波炉根坐标系里的位姿(USD 里的装配关系,不随仿真变)
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        r = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array(m * r.GetInverse(), dtype=float).T

    def joint_frame(local_pos, local_rot):
        m = np.eye(4)
        q = local_rot if local_rot is not None else Gf.Quatf(1, 0, 0, 0)
        m[:3, :3] = t3d.quaternions.quat2mat([q.GetReal(), *q.GetImaginary()])
        m[:3, 3] = np.asarray(local_pos if local_pos is not None else (0, 0, 0), float)
        return m

    j0 = rel(body0) @ joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
    j1_in_body1 = joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
    q = obj.get_current_joint_positions()
    q = q.detach().cpu().numpy().reshape(-1) if hasattr(q, "detach") else np.asarray(q).reshape(-1)
    angle = float(q[obj.dof_names.index(door_joint(obj))])
    rz = np.eye(4)
    rz[:3, :3] = t3d.axangles.axangle2mat([0, 0, 1], angle)
    root_T_body1 = j0 @ rz @ np.linalg.inv(j1_in_body1)
    return common._box(lo, hi, common._mat(env.get_obj_pose("microwave")) @ root_T_body1)


# ------------------------------------------------------------------ 微波炉


def _grasps(env, spec, inward):
    """GraspGenX 抓取候选(见模块说明第 1 条),按优先级。"""
    up = np.array([0.0, 0.0, 1.0])
    if spec.grasp == "graspgen":
        p = np.radians(spec.prefer_pitch)
        prefer = np.cos(p) * inward - np.sin(p) * up
        long_axis = common._mat(env.get_obj_pose(spec.label))[:3, 2]
        candidates = env.grasp_candidates(
            spec.label,
            common.ARM,
            prefer=prefer,
            approach_z=spec.approach_z,
            finger_z_max=FINGER_Z_MAX,
            max_candidates=1000,
        )
        if spec.across_long_axis:
            candidates = [(g, sc) for g, sc in candidates if abs(common._mat(g)[:3, 1] @ long_axis) < FINGER_ALONG_MAX]
    elif spec.grasp == "top":
        candidates = env.grasp_candidates(
            spec.label,
            common.ARM,
            prefer=-up,
            approach_z=TOP_APPROACH_Z,
            finger_z_max=FINGER_Z_MAX,
            max_candidates=1000,
        )
        if spec.axisym:
            candidates = _rotate_about_axis(env, spec.label, candidates)
        across = np.cross(up, inward)
        pts = common._bbox_points(env, spec.label)[common.FACE_CENTERS]
        center = common._center(env, spec.label)
        bias = env.robot(common.ARM).gripper_bias

        def off_center(g):
            m = common._mat(g)
            return float(np.linalg.norm((m[:3, 3] + bias * m[:3, 0] - center)[:2]))

        if spec.center_slide:
            candidates = [(_slide_to_center(env, g, center), sc) for g, sc in candidates]
        candidates = [
            (common._lift_tips(env, g), sc)
            for g, sc in candidates
            if abs(common._mat(g)[:3, 1] @ across) > FINGER_ACROSS_MIN
            and np.ptp(pts @ common._mat(g)[:3, 1]) <= common.FINGER_MAX
            and off_center(g) <= TOP_CENTER_TOL
        ]
    else:
        raise ValueError(f"unknown grasp mode {spec.grasp}")
    if not candidates:
        raise PlanFailError(f"{common.ARM}: no {spec.grasp} grasp candidate for {spec.label}")
    return candidates[:MAX_GRASPS]


def _slide_to_center(env, g, center):
    """抓取 ``g`` 沿 ee z 轴平移,让 TCP 在这条轴上与 ``center`` 对齐。"""
    m = common._mat(g)
    z = m[:3, 2]
    tcp = m[:3, 3] + env.robot(common.ARM).gripper_bias * m[:3, 0]
    g = np.array(g, dtype=float)
    g[:3] += ((center - tcp) @ z) * z
    return g


def _rotate_about_axis(env, label, candidates):
    """``candidates``(世界坐标的 ee 位姿)绕物体包围盒中心的物体 z 轴每 ``AXISYM_STEP`` 度复制一份(旋转对称的物体)。"""
    obj = common._mat(env.get_obj_pose(label))
    axis = obj[:3, 2]
    c = common._center(env, label)
    out = []
    for deg in np.arange(0.0, 360.0, AXISYM_STEP):
        R = np.eye(4)
        R[:3, :3] = t3d.axangles.axangle2mat(axis, np.radians(deg))
        T = np.eye(4)
        T[:3, 3] = c
        Ti = np.eye(4)
        Ti[:3, 3] = -c
        M = T @ R @ Ti
        out += [(common._pose(M @ common._mat(g)), sc) for g, sc in candidates]
    out.sort(key=lambda gs: -gs[1])
    return out


def _release_poses(env, geom, mwm, inward, held_ee, finger_y, floor_ee=None):
    """炉腔内的松手位姿候选 ``[ee 位姿, ...]``(依次 TCP 高度 x 下倾角),见模块说明第 2 条;``held_ee`` 是手中物体的采样点
    (ee 坐标系),``floor_ee`` 是算物体底面 / 顶面高度用的采样点(默认同 ``held_ee``;方物体只用面中心:转平后包围盒的
    角点比物体本身低得多)。"""
    floor_ee = held_ee if floor_ee is None else floor_ee
    bias = env.robot(common.ARM).gripper_bias
    plate = (mwm @ [*geom.plate_xy, geom.plate_z, 1.0])[:3]
    opening = (mwm @ [0.0, geom.opening_y, 0.0, 1.0])[:3]
    heights = TCP_HEIGHTS_TALL if geom.cavity[1][1] - geom.plate_z >= TALL_CAVITY else TCP_HEIGHTS
    depth_max = float(geom.cavity_back - geom.opening_y)
    out = []
    for h in heights:
        for pitch in INSERT_PITCHES:
            rot = _insert_rot(inward, finger_y, pitch)
            end = plate + [0.0, 0.0, h] - bias * rot[:, 0]
            inner = float(((end + held_ee @ rot.T - opening) @ inward).max())
            end -= max(0.0, inner - (depth_max - BACK_CLEAR)) * inward
            bottom = end[2] + (floor_ee @ rot.T)[:, 2].min()
            if bottom < plate[2] + OBJ_FLOOR:
                # 物体在手里往下垂得太多(抓在三明治这种长物体的上沿,转平后下半截比 TCP 低好几厘米):把松手位姿整体
                # 往上提,只要提完物体顶面还在炉腔顶以下
                lift = plate[2] + OBJ_FLOOR - bottom
                top = end[2] + lift + (floor_ee @ rot.T)[:, 2].max()
                if top > (mwm @ [0.0, 0.0, geom.cavity[1][1], 1.0])[2] - OBJ_FLOOR:
                    continue
                end = end + [0.0, 0.0, lift]
            out.append(np.concatenate([end, t3d.quaternions.mat2quat(rot)]))
    return out


def _play(env, spec):
    label = spec.label
    mwm, geom, inward = _inward(env)
    planner = env.robot_manager.planner[env.robot(common.ARM).robot_name]
    root = env.robot(common.ARM).entity_origin_pose
    # 盒子各向外扩 MW_INFLATE 再交给 cuRobo:不扩时伸到炉口的腕部 / 相机蹭到炉口上沿,把微波炉推走 4 ~ 5 cm;
    # 扩 6 mm 时 0.75 倍微波炉的炉口窄了 1.2 cm,三明治的松手位姿连 IK 都没有
    fixed = {
        f"mw_{i}": common._box(np.asarray(lo) - MW_INFLATE, np.asarray(hi) + MW_INFLATE, mwm)
        for i, (lo, hi) in enumerate(_microwave_boxes(geom))
    }
    dims, pose = _door_box(env)
    fixed["mw_door"] = ([d + 2 * MW_INFLATE for d in dims], pose)
    lo, hi = common._bbox(env, label)
    pts_obj = common._held_points(spec, lo, hi)
    shrink = common.OBJ_SHRINK if spec.round else common.BOX_SHRINK

    def set_world(with_obj, tool):
        obstacles = dict(fixed)
        if with_obj:
            s = spec.grasp_shrink if (with_obj == "grasp" and spec.grasp_shrink is not None) else shrink
            obstacles[label] = common._obj_box(env, label, s)
        planner.set_obstacles(obstacles, root)
        planner.set_tool_spheres(tool)

    def held_tool(ee_T_obj):
        held_ee = (np.c_[pts_obj, np.ones(len(pts_obj))] @ ee_T_obj.T)[:, :3]
        width = float(np.ptp(held_ee[:, 1]))  # 物体沿手指方向(ee y)的宽度
        obj = common._held_spheres(ee_T_obj, lo, hi) if spec.round else common._bbox_spheres(ee_T_obj, lo, hi)
        return common._finger_spheres(min(width, common.GRIPPER_OPEN)) + obj

    def releases(ee, ee_T_obj):
        held_ee = (np.c_[pts_obj, np.ones(len(pts_obj))] @ ee_T_obj.T)[:, :3]
        floor_ee = held_ee if spec.round else held_ee[common.FACE_CENTERS]
        return _release_poses(env, geom, mwm, inward, held_ee, ee[:3, 1], floor_ee)

    common._pick_place_direct(
        env,
        label,
        set_world=set_world,
        grasps=_grasps(env, spec, inward),
        open_tool=common._finger_spheres(common.GRIPPER_OPEN),
        held_tool=held_tool,
        releases=releases,
        close=lambda g: env.close_gripper(common.ARM),
        reuse_tol=spec.reuse_tol or common.PRED_REUSE_TOL,
    )


def play_once(env, spec):
    common._run_direct(env, lambda: _play(env, spec))
