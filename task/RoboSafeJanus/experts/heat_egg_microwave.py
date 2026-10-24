"""heat_<物体>_microwave 系列任务共用的脚本专家(右臂;物体在微波炉右前方,门开在左侧)。本模块是
heat_egg_microwave 的专家,其余任务(罐头、罐装饮料、三明治)的专家只是用自己的 ``Spec`` 调用 ``play_once``。

三段运动,每段一次 cuRobo 规划,只在夹爪开合时停:初始 -> 抓取 -> 合拢 -> 炉腔内松手位姿 -> 松开 -> 回初始
(``_pick_place_direct``,烤面包机 / 冰箱的专家也用它)。微波炉(外壳减炉腔、控制面板、按当前门角摆放的门)和物体放进
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
import logging
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import transforms3d as t3d

from env.global_configs import OBJECTS_PATH
from src.data_collect.collect_env import PlanFailError
from task.RoboDojo.tasks.heat_egg_microwave import door_joint

log = logging.getLogger(__name__)

ARM = "right_arm"


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
    # 抓在偏一头 2 cm 的地方时物体下半截垂得太低,松手位姿就得抬到腕部顶住炉腔顶(同 place_bread_toaster 把抓取沿手指方向
    # 挪到厚度中面)
    center_slide: bool = False
    # 合拢后物体在指间挪动不超过这么多时,沿用合拢前预测的第 2 段(见 _pick_place_direct 的 reuse_tol);None 时取
    # PRED_REUSE_TOL。软的三明治一夹就被挤偏 1 ~ 1.5 cm
    reuse_tol: float = None


EGG = Spec("egg", "graspgen", prefer_pitch=40.0, across_long_axis=True, round=True)

# 抓取
FINGER_Z_MAX = 0.35  # graspgen:手指方向 |z| 上限:手指水平,夹爪竖直方向才能放进炉口
FINGER_ALONG_MAX = 0.3  # graspgen:手指方向与物体长轴(物体坐标系 z)夹角余弦的上限
FINGER_ACROSS_MIN = 0.9  # top:手指方向与"垂直于炉腔朝里"方向夹角余弦的下限
TOP_APPROACH_Z = (-1.0, -0.7)  # top:接近方向单位向量 z 分量的范围
# top:TCP 离物体包围盒中心(水平)不超过这么远:抓在长物体(三明治)一头时,转平后物体往下垂,炉腔里放不下
TOP_CENTER_TOL = 0.025
AXISYM_STEP = 10.0  # axisym:绕物体 z 轴每隔这么多度复制一份抓取
# top:指尖最低处(TCP 沿接近方向再往前 TIP_BEYOND_TCP)离桌面至少这么远,不够时把抓取沿接近方向往回提:
# GraspGenX 夹在矮物体(三明治)中心高度的抓取,指尖贴着桌面,cuRobo 判碰撞,一个都规划不到
TOP_TIP_CLEAR = 0.008
TIP_BEYOND_TCP = 0.019
MAX_GRASPS = 40
FINGER_MAX = 0.075  # 夹持宽度上限(X5 夹爪最大开口 8.8 cm)
GRASP_SETTLE = 30
GRIPPER_OPEN = 0.088  # X5 夹爪全开时两指间距(归一化开度 1)
RELEASE_SETTLE = 40  # 松开夹爪后等这么多步再回程
MW_INFLATE = 0.003  # 微波炉碰撞盒子交给 cuRobo 前各向外扩这么多
# 手里拿着物体的那段整条轨迹按这个倍数放慢(时间上统一拉长,不分段):按 cuRobo 的速度跑时鸡蛋会被甩出去
CARRY_SLOW = 1.5
PRED_REUSE_TOL = 0.01  # 合拢后按实测手中位姿规划不到第 2 段时,物体在指间挪动不超过这么多就沿用预测的那条
# 碰撞模型
OBJ_SHRINK = 0.85  # 圆的物体:包围盒按这个比例缩小当障碍物
BOX_SHRINK = 0.95  # 其余物体(手指 / 掌心抓取时本来就贴着它)
FINGER_R = 0.006  # 手指碰撞球半径
HELD_R_FRAC = 0.9  # 手中物体的碰撞球半径 = 这个比例 x 包围盒最短(圆的物体:次短)边的一半


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


# X5 的 URDF 关节限位是 +-10 rad,cuRobo 会规划出腕部向上臂折回的路径(例如 joint2 2.1 rad 时 joint4 1.8 rad):
# 仿真里这里自碰,机械臂物理直接变成 NaN。上限取自已成功的鸡蛋 / 倒水数据(|joint4| <= 1.76,joint2 + joint4 <= 3.57)
J4_MAX = 1.8
J2_PLUS_J4_MAX = 3.65
FOLD_RETRIES = 3  # 一次规划的流程:规划出折回的路径时最多重新规划这么多次
# 手指(link7 / link8,不在 cuRobo 的碰撞球里)在 ee_link 坐标系下的截面:x 沿接近方向,z 方向的半厚度按 X5 手指网格
# 越往指尖越窄:(x, 半厚度),``_finger_spheres`` 按它摆碰撞球
FINGER_PROFILE = ((0.075, 0.03), (0.097, 0.03), (0.11, 0.018), (0.127, 0.018), (0.142, 0.01), (0.158, 0.01))
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
    mw = _mat(env.get_obj_pose("microwave"))
    _, geom = _microwave(env)
    inward = mw[:3, 1].copy()
    inward[2] = 0.0
    return mw, geom, inward / np.linalg.norm(inward)


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


# ------------------------------------------------------------------ 一次规划的流程(微波炉 / 烤面包机 / 冰箱共用)


def _pose(m):
    return np.concatenate([m[:3, 3], t3d.quaternions.mat2quat(m[:3, :3])])


def _box(lo, hi, world_T_frame):
    """``frame`` 坐标系里的轴对齐盒子 -> (dims, 世界位姿)。"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.eye(4)
    c[:3, 3] = (lo + hi) / 2
    return (hi - lo).tolist(), _pose(world_T_frame @ c)


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
    return _box(lo, hi, _mat(env.get_obj_pose("microwave")) @ root_T_body1)


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
    """两根手指(间距 ``gap``)的碰撞球,ee_link 坐标系;x / 半厚度取 heat_egg_microwave.FINGER_PROFILE。

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


# ------------------------------------------------------------------ 微波炉


def _grasps(env, spec, inward):
    """GraspGenX 抓取候选(见模块说明第 1 条),按优先级。"""
    up = np.array([0.0, 0.0, 1.0])
    if spec.grasp == "graspgen":
        p = np.radians(spec.prefer_pitch)
        prefer = np.cos(p) * inward - np.sin(p) * up
        long_axis = _mat(env.get_obj_pose(spec.label))[:3, 2]
        candidates = env.grasp_candidates(
            spec.label, ARM, prefer=prefer, approach_z=spec.approach_z, finger_z_max=FINGER_Z_MAX, max_candidates=1000
        )
        if spec.across_long_axis:
            candidates = [(g, sc) for g, sc in candidates if abs(_mat(g)[:3, 1] @ long_axis) < FINGER_ALONG_MAX]
    elif spec.grasp == "top":
        candidates = env.grasp_candidates(
            spec.label, ARM, prefer=-up, approach_z=TOP_APPROACH_Z, finger_z_max=FINGER_Z_MAX, max_candidates=1000
        )
        if spec.axisym:
            candidates = _rotate_about_axis(env, spec.label, candidates)
        across = np.cross(up, inward)
        pts = _bbox_points(env, spec.label)[FACE_CENTERS]
        center = _center(env, spec.label)
        bias = env.robot(ARM).gripper_bias

        def off_center(g):
            m = _mat(g)
            return float(np.linalg.norm((m[:3, 3] + bias * m[:3, 0] - center)[:2]))

        if spec.center_slide:
            candidates = [(_slide_to_center(env, g, center), sc) for g, sc in candidates]
        candidates = [
            (_lift_tips(env, g), sc)
            for g, sc in candidates
            if abs(_mat(g)[:3, 1] @ across) > FINGER_ACROSS_MIN
            and np.ptp(pts @ _mat(g)[:3, 1]) <= FINGER_MAX
            and off_center(g) <= TOP_CENTER_TOL
        ]
    else:
        raise ValueError(f"unknown grasp mode {spec.grasp}")
    if not candidates:
        raise PlanFailError(f"{ARM}: no {spec.grasp} grasp candidate for {spec.label}")
    return candidates[:MAX_GRASPS]


def _slide_to_center(env, g, center):
    """抓取 ``g`` 沿 ee z 轴平移,让 TCP 在这条轴上与 ``center`` 对齐。"""
    m = _mat(g)
    z = m[:3, 2]
    tcp = m[:3, 3] + env.robot(ARM).gripper_bias * m[:3, 0]
    g = np.array(g, dtype=float)
    g[:3] += ((center - tcp) @ z) * z
    return g


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


def _rotate_about_axis(env, label, candidates):
    """``candidates``(世界坐标的 ee 位姿)绕物体包围盒中心的物体 z 轴每 ``AXISYM_STEP`` 度复制一份(旋转对称的物体)。"""
    obj = _mat(env.get_obj_pose(label))
    axis = obj[:3, 2]
    c = _center(env, label)
    out = []
    for deg in np.arange(0.0, 360.0, AXISYM_STEP):
        R = np.eye(4)
        R[:3, :3] = t3d.axangles.axangle2mat(axis, np.radians(deg))
        T = np.eye(4)
        T[:3, 3] = c
        Ti = np.eye(4)
        Ti[:3, 3] = -c
        M = T @ R @ Ti
        out += [(_pose(M @ _mat(g)), sc) for g, sc in candidates]
    out.sort(key=lambda gs: -gs[1])
    return out


def _held_points(spec, lo, hi):
    """手中物体的采样点(物体坐标系):包围盒 27 点,圆的物体只用面中心和中心。"""
    pts = GRID * (hi - lo) + lo
    return pts[FACE_CENTERS | (np.sum(GRID == 0.5, axis=1) == 3)] if spec.round else pts


def _release_poses(env, geom, mwm, inward, held_ee, finger_y, floor_ee=None):
    """炉腔内的松手位姿候选 ``[ee 位姿, ...]``(依次 TCP 高度 x 下倾角),见模块说明第 2 条;``held_ee`` 是手中物体的采样点
    (ee 坐标系),``floor_ee`` 是算物体底面 / 顶面高度用的采样点(默认同 ``held_ee``;方物体只用面中心:转平后包围盒的
    角点比物体本身低得多)。"""
    floor_ee = held_ee if floor_ee is None else floor_ee
    bias = env.robot(ARM).gripper_bias
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
    planner = env.robot_manager.planner[env.robot(ARM).robot_name]
    root = env.robot(ARM).entity_origin_pose
    # 盒子各向外扩 MW_INFLATE 再交给 cuRobo:不扩时伸到炉口的腕部 / 相机蹭到炉口上沿,把微波炉推走 4 ~ 5 cm;
    # 扩 6 mm 时 0.75 倍微波炉的炉口窄了 1.2 cm,三明治的松手位姿连 IK 都没有
    fixed = {
        f"mw_{i}": _box(np.asarray(lo) - MW_INFLATE, np.asarray(hi) + MW_INFLATE, mwm)
        for i, (lo, hi) in enumerate(_microwave_boxes(geom))
    }
    dims, pose = _door_box(env)
    fixed["mw_door"] = ([d + 2 * MW_INFLATE for d in dims], pose)
    lo, hi = _bbox(env, label)
    pts_obj = _held_points(spec, lo, hi)
    shrink = OBJ_SHRINK if spec.round else BOX_SHRINK

    def set_world(with_obj, tool):
        obstacles = dict(fixed)
        if with_obj:
            s = spec.grasp_shrink if (with_obj == "grasp" and spec.grasp_shrink is not None) else shrink
            obstacles[label] = _obj_box(env, label, s)
        planner.set_obstacles(obstacles, root)
        planner.set_tool_spheres(tool)

    def held_tool(ee_T_obj):
        held_ee = (np.c_[pts_obj, np.ones(len(pts_obj))] @ ee_T_obj.T)[:, :3]
        width = float(np.ptp(held_ee[:, 1]))  # 物体沿手指方向(ee y)的宽度
        obj = _held_spheres(ee_T_obj, lo, hi) if spec.round else _bbox_spheres(ee_T_obj, lo, hi)
        return _finger_spheres(min(width, GRIPPER_OPEN)) + obj

    def releases(ee, ee_T_obj):
        held_ee = (np.c_[pts_obj, np.ones(len(pts_obj))] @ ee_T_obj.T)[:, :3]
        floor_ee = held_ee if spec.round else held_ee[FACE_CENTERS]
        return _release_poses(env, geom, mwm, inward, held_ee, ee[:3, 1], floor_ee)

    _pick_place_direct(
        env,
        label,
        set_world=set_world,
        grasps=_grasps(env, spec, inward),
        open_tool=_finger_spheres(GRIPPER_OPEN),
        held_tool=held_tool,
        releases=releases,
        close=lambda g: env.close_gripper(ARM),
        reuse_tol=spec.reuse_tol or PRED_REUSE_TOL,
    )


def play_once(env, spec=EGG):
    _run_direct(env, lambda: _play(env, spec))
