"""heat_<物体>_microwave 系列任务共用的脚本专家(右臂;物体在微波炉右前方,门开在左侧)。本模块是
heat_egg_microwave 的专家,其余任务(罐头、罐装饮料、三明治)的专家只是用自己的 ``Spec`` 调用
``play_once``。

1. 抓取(``Spec.grasp``):
   - ``"graspgen"``(鸡蛋):GraspGenX 候选里只留接近方向在 ``Spec.approach_z`` 范围内、
     手指水平的,``Spec.across_long_axis`` 时
     还要求手指横跨物体短径(与物体坐标系 z 轴垂直),偏好与伸入方向(炉腔朝里、向下倾 ``Spec.prefer_pitch``)一致的。
     侧躺的鸡蛋只会沿垂直于长轴的方向滚:伸入时手指垂直于炉腔朝里的方向,鸡蛋长轴就朝里,松手后只能向两侧滚,
     被炉壁挡住;手指沿长轴夹时鸡蛋会朝炉门滚出去。
   - ``"top"``(罐头、汉堡):自上而下抓包围盒中心,手指水平且垂直于炉腔朝里的方向;指尖离桌面 ``TIP_CLEAR``。
   - ``"side"``(三明治):手指水平的侧抓,从机器人一侧接近包围盒中心,夹爪中心(TCP)
     离物体底面 ``Spec.grip_height``(手指上下各 3.1 cm,TCP 太低手指会碰桌面)。物体离右臂基座只有 15 ~ 25 cm,
     正对物体水平伸出去在这么近的地方没有 IK 解,所以候选是偏航 ``SIDE_YAWS`` x 向下倾 ``SIDE_PITCHES`` 的网格。
     ``Spec.box`` 的物体只沿它的两条水平轴接近(手指夹住一对侧面,夹持宽度不超过 ``FINGER_MAX``)。三明治只能
     这样抓:自上而下抓再转平会把 11.9 cm 的长边竖起来,炉腔放不下;GraspGenX 的侧抓夹在物体半高处,手指碰桌面。
     送进炉腔时末端转回接近水平,物体随之后仰抓取时的倾角,由手中物体的碰撞采样点检查。
2. 竖直抬起,把末端转成"接近水平(``INSERT_PITCHES``)、接近方向的水平投影朝炉腔里、手指水平"的姿态,先停在炉口外
   远处(物体后方,夹爪转向时指尖不会扫到控制面板),再沿炉腔方向直线伸进去:TCP 停在转盘中心上方
   ``TCP_HEIGHTS`` 处。按夹爪而不是物体定高度:间隙由夹爪 / 相机 / 腕部决定;物体在手里的位置按抬起后
   实测的 ee->物体 变换算,只要求它的包围盒底面离转盘不低于 ``OBJ_FLOOR``、顶面不碰炉腔顶。
3. 松开夹爪,在炉腔顶允许的范围内竖直抬起一点(``LIFTS_OUT``,手指离开鸡蛋),再水平退出炉腔,右臂回到初始位姿
   (任务判定要求此时鸡蛋仍在炉内、微波炉没被推动)。失败时也松开夹爪、回到初始位姿再抛出异常。

cuRobo 的碰撞世界只有桌子,不知道微波炉,所以伸入 / 退出的每条候选路径都先规划,再用 cuRobo 自己的机械臂碰撞球
(link1 ~ link6、相机、相机支架,由同一个 cuRobo 模型正运动学算出)加上手指、手中物体包围盒的采样点,逐个轨迹点
检查是否碰到微波炉(``_Clearance``:外壳盒子减去炉腔,尺寸取自碰撞体射线探测,随资产缩放),手指和物体还要高于桌面。只按夹爪本身算间隙是不够的:
link6 后面的腕部关节 link5 比 link6 的轴线高 8.5 cm,向下倾 20 度伸入时会插进炉口上沿的面板,把微波炉推走。
"""

from dataclasses import dataclass
import json
import logging
import os

import numpy as np
import torch
import transforms3d as t3d

from env.global_configs import OBJECTS_PATH
from src.data_collect.collect_env import SIM_JOINT_BOUNDS, PlanFailError

log = logging.getLogger(__name__)

ARM = "right_arm"


@dataclass(frozen=True)
class Spec:
    """一个物体的抓取方式,见模块说明第 1 条。"""

    label: str
    grasp: str  # "graspgen" | "top" | "side"
    prefer_pitch: float = 40.0  # graspgen:偏好的接近方向向下倾的角度
    approach_z: tuple = (-1.0, -0.25)  # graspgen:接近方向单位向量 z 分量的范围
    across_long_axis: bool = False  # graspgen:手指横跨物体短径
    grip_height: float = 0.038  # side:TCP 离物体底面的高度
    box: bool = False  # side:只沿物体的两条水平轴接近
    round: bool = False  # 圆的物体:碰撞 / 离转盘高度只用包围盒的面中心和中心(角点远在物体外面)
    from_above: bool = False  # 高的物体:先到物体上方再竖直下到预抓取位姿,免得手臂扫过物体
    fingers_across_inward: bool = False  # graspgen:只留手指与炉腔朝里方向垂直的抓取(同 top)
    gentle_close: bool = False  # 夹爪慢慢合到比物体略窄,不合到底
    top_depth: float = None  # top:TCP 在物体顶面下这么深(默认在物体中心高度)
    carry_to_front: bool = False  # 在微波炉右侧抓起的物体:抬起后先水平搬到炉口正前方,再转向伸入


EGG = Spec("egg", "graspgen", prefer_pitch=40.0, across_long_axis=True, round=True)

# 抓取
FINGER_Z_MAX = 0.35  # graspgen:手指方向 |z| 上限:手指水平,夹爪竖直方向才能放进炉口
FINGER_ALONG_MAX = 0.3  # graspgen:手指方向与物体长轴(物体坐标系 z)夹角余弦的上限
FINGER_ACROSS_MIN = 0.9  # graspgen(fingers_across_inward):手指方向与"垂直于炉腔朝里"方向夹角余弦的下限
MAX_GRASPS = 40
FINGER_MAX = 0.075  # 夹持宽度上限(X5 夹爪最大开口 8.8 cm)
TIP_CLEAR = 0.006  # top:指尖离桌面的高度
SIDE_YAWS = (0.0, 25.0, -25.0, 50.0, -50.0, 75.0, -75.0)  # side:接近方向相对"基座 -> 物体"方向的偏航
SIDE_PITCHES = (0.0, 20.0, 40.0)  # side:接近方向向下倾的角度
SIDE_MAX_YAW = 80.0  # side(box):接近方向与"基座 -> 物体"方向的最大夹角
TOP_YAWS = (0.0, 15.0, -15.0)  # top:手指方向相对"垂直于炉腔朝里"的偏航
PRE_GRASP = 0.08
ABOVE_TOP = 0.06  # 预抓取前先到的高点:手指尖高出物体顶面这么多
START_LIFT = 0.10  # 抓取前先从初始位姿竖直抬起的高度
GRASP_SETTLE = 30
GRIPPER_OPEN = 0.088  # X5 夹爪全开时两指间距(归一化开度 1)
GENTLE_SQUEEZE = 0.01  # gentle_close:合到比物体窄这么多
GENTLE_STEPS = 150
LIFT = (0.12, 0.10, 0.08)
SLIP_TOL = 0.02  # TCP 到物体中心的距离比刚抓好时变大超过它算抓丢


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
CLEAR = 0.006  # 机械臂碰撞球 / 手指与微波炉的最小间隙
# X5 的 URDF 关节限位是 +-10 rad,cuRobo 会规划出腕部向上臂折回的路径(例如 joint2 2.1 rad 时 joint4 1.8 rad):
# 仿真里这里自碰,机械臂物理直接变成 NaN。上限取自已成功的鸡蛋 / 倒水数据(|joint4| <= 1.76,joint2 + joint4 <= 3.57)
J4_MAX = 1.8
J2_PLUS_J4_MAX = 3.65
OBJ_CLEAR = 0.003  # 手中物体(包围盒采样点,本身已偏保守)与微波炉的最小间隙
TABLE_CLEAR = 0.003  # 手指 / 手中物体与桌面的最小间隙
# 手指(link7 / link8,不在 cuRobo 的碰撞球里)在 ee_link 坐标系下的采样点:x 沿接近方向,y 手指张合方向,
# z 方向的半厚度按 X5 手指网格越往指尖越窄(同 pour_liquid_into_cup.GRIPPER_BOXES):(x, 半厚度)
FINGER_PROFILE = ((0.075, 0.03), (0.097, 0.03), (0.11, 0.018), (0.127, 0.018), (0.142, 0.01), (0.158, 0.01))
FINGER_TIP = 0.127  # 指尖段(x >= 它)去抓取时允许碰到桌面:GraspGenX 的抓取常让指尖贴桌,实际抓得起来
FINGER_PTS = np.array([[x, y, z] for x, h in FINGER_PROFILE for y in (-0.06, -0.035, 0.035, 0.06) for z in (-h, h)])
# 伸入
# 0.75 倍微波炉的炉口只有约 13.6 cm 高:夹爪放低了手指碰炉口下沿,放高了 / 倾角大了相机、相机支架、link4 碰炉口
# 上沿,能过的只有接近水平、TCP 离转盘 4.5 ~ 5.3 cm 这一小段(/tmp/hz/insert_sweep.py 离线扫描)
TCP_HEIGHTS = (0.050, 0.047, 0.053, 0.045)  # 伸到底时 TCP 离转盘的高度
INSERT_PITCHES = (0.0, 5.0)  # 接近方向向下倾斜的角度(度)
# 炉腔在转盘以上不低于 TALL_CAVITY 时(原尺寸 ArtVIP 00000 18.6 cm、0.75 倍 00002 17.8 cm)夹爪可以放高一点:水平伸入
# 时 TCP 离转盘 5 ~ 7 cm、倾 5 度时 4 ~ 5 cm 都能过(/tmp/hz/full_size/reach_fine.py、/tmp/hz/mw2/reach_mw.py)
TALL_CAVITY = 0.16
TCP_HEIGHTS_TALL = (0.06, 0.065, 0.055, 0.05, 0.07)
# 长物体(罐装饮料)抓在顶部、转平后从 TCP 往里伸出十几厘米:TCP 停在转盘中心上方时会顶到炉腔后壁,
# 这时整条伸入路径往外退,让物体最里面的采样点离后壁 BACK_CLEAR
BACK_CLEAR = 0.015
# 直线伸入的起点:TCP 在炉口外这么远。从物体在微波炉前方抓起时(鸡蛋、罐头、汉堡)起点要在物体后方,否则夹爪边转向
# 边前移,指尖扫到控制面板;从微波炉右侧抓起时(饮料、三明治)远的起点离基座太近,没有 IK 解,只能用近的
STAGE_OUT = (0.27, 0.18, 0.12, 0.08)
OBJ_FLOOR = 0.005  # 松手时物体最低的包围盒面中心至少高出转盘这么多
INSERT_SLOW = 1.5
CARRY_FRONT = 0.22  # carry_to_front:搬到 TCP 在炉口外这么远处
# 伸入起点的 IK 种子:正手、腕部不翻转的构型(/tmp/hz/insert_sweep.py 离线解出的伸到底时的关节角),手指两个朝向各一个
INSERT_SEEDS = ((0.63, 2.1, 1.07, 1.1, 0.63, 0.0), (0.63, 2.1, 1.07, 1.1, 0.63, 3.14))
LIFTS_OUT = (0.02, 0.012, 0.006, 0.0)  # 松手后竖直抬起的高度,取第一个不碰炉腔顶的
RETREAT = 0.25  # 之后水平后退的距离


class GraspLostError(Exception):
    pass


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


def _tcp_dist(env, label):
    ee = _mat(env.get_arm_pose(ARM))
    tcp = ee[:3, 3] + env.robot(ARM).gripper_bias * ee[:3, 0]
    return float(np.linalg.norm(_center(env, label) - tcp))


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
    """微波炉位姿、缩放比例、缩放后的几何、炉腔朝里的水平方向(微波炉坐标系 +y)。"""
    mw = _mat(env.get_obj_pose("microwave"))
    k, geom = _microwave(env)
    inward = mw[:3, 1].copy()
    inward[2] = 0.0
    return mw, k, geom, inward / np.linalg.norm(inward)


class _Clearance:
    """用 cuRobo 机械臂模型的碰撞球 + 手指采样点检查一条规划好的路径会不会碰到微波炉。

    cuRobo 的碰撞球在它的基座坐标系里;基座 -> env 的变换用当前关节角的正运动学末端位姿和实测末端位姿标定,
    不依赖规划器内部的平台 / 偏置约定。
    """

    def __init__(self, env, mw, geom):
        self.env = env
        self.table_z = _table_z(env)
        self.planner = env.robot_manager.planner[env.robot(ARM).robot_name]
        self.mw_inv = np.linalg.inv(mw)
        self.shell = [geom.shell, *geom.panel]
        self.cav_x, self.cav_z = geom.cavity
        self.cav_y = geom.cavity_back
        self.opening_y = geom.opening_y
        spheres, ee = self._fk([env.get_arm_joints(ARM)])
        self.base_to_env = _mat(env.get_arm_pose(ARM)) @ np.linalg.inv(ee[0])

    def _fk(self, joints):
        """``joints`` 每行一组手臂关节角 -> (碰撞球 [B, N, 4](基座坐标系), ee_link 位姿 [B, 4, 4](基座坐标系))。"""
        p = self.planner
        q = np.stack([p._build_cspace_joint_values(j) for j in joints]).astype(np.float32)
        js = type(p._build_joint_state(joints[0])).from_position(
            torch.as_tensor(q, device=p.device_cfg.device), joint_names=p.cspace_joint_names
        )
        kin = p.motion_planner.compute_kinematics(js)
        spheres = kin.robot_spheres.reshape(len(joints), -1, 4).detach().cpu().numpy()
        pose = kin.tool_poses.to_dict()[p.ee_link]
        pos = pose.position.reshape(len(joints), 3).detach().cpu().numpy()
        quat = pose.quaternion.reshape(len(joints), 4).detach().cpu().numpy()
        ee = np.stack([_mat(np.concatenate([a, b])) for a, b in zip(pos, quat)])
        return spheres, ee

    def _hits(self, pts, r):
        """微波炉坐标系下的点 ``pts``(半径 ``r``)是否碰到微波炉:在外壳盒子(外扩 r)内,且不在炉口 / 炉腔的空腔内。"""
        r = r + CLEAR
        in_shell = np.zeros(len(pts), dtype=bool)
        for lo, hi in self.shell:
            in_shell |= np.all((pts > lo - r[:, None]) & (pts < hi + r[:, None]), axis=1)
        x, y, z = pts.T
        in_cavity = (
            (x - r > self.cav_x[0]) & (x + r < self.cav_x[1]) & (z - r > self.cav_z[0]) & (z + r < self.cav_z[1])
        ) & (y + r < self.cav_y)
        return in_shell & ~in_cavity

    def collides(self, plans, held=None, table="all", step=3):
        """``plans``(cuRobo 规划结果列表)的轨迹点(每 ``step`` 个取一个,加上终点)里,是否有碰到微波炉的;手指和
        ``held``(手中物体的采样点,ee_link 坐标系)还要高于桌面:``table="all"`` 全查,``"body"`` 不查指尖段(去抓取时:
        手指模型偏保守,抓矮物体时指尖会算到桌面以下 1 cm,实际抓得起来;但水平抓取时 6 cm 厚的夹爪根部压进桌面,
        一闭合物理就炸成 NaN)。返回第一个碰撞的描述,不碰时返回 None。"""
        joints = []
        for res in plans:
            traj = np.asarray(res["position"])
            joints += list(traj[::step]) + [traj[-1]]
        spheres, ee = self._fk(joints)
        held = np.zeros((0, 3)) if held is None else np.asarray(held, dtype=float)
        local = np.concatenate([FINGER_PTS, held])
        n_f = len(FINGER_PTS)
        for i in range(len(joints)):
            s = spheres[i][spheres[i][:, 3] > 0]
            world = np.concatenate(
                [
                    (np.c_[s[:, :3], np.ones(len(s))] @ self.base_to_env.T)[:, :3],
                    (np.c_[local, np.ones(len(local))] @ (self.base_to_env @ ee[i]).T)[:, :3],
                ]
            )
            pts = (np.c_[world, np.ones(len(world))] @ self.mw_inv.T)[:, :3]
            radii = np.concatenate([s[:, 3], np.zeros(n_f), np.full(len(held), OBJ_CLEAR - CLEAR)])
            hit = self._hits(pts, radii)
            if table:
                low = world[len(s) :, 2] < self.table_z + TABLE_CLEAR
                if table == "body":
                    low[:n_f] &= FINGER_PTS[:, 0] < FINGER_TIP
                    low[n_f:] = False
                hit[len(s) :] |= low
            if hit.any():
                j = int(np.argmax(hit))
                what = f"sphere {j}" if j < len(s) else ("finger" if j < len(s) + n_f else "held object")
                return f"waypoint {i}/{len(joints)}: {what} at {np.round(pts[j], 3).tolist()} (microwave frame)"
        return None

    def check_calibration(self):
        """标定误差:当前关节角的正运动学末端位置与实测末端位置之差(米);在标定之后关节动过时才有意义。"""
        _, ee = self._fk([self.env.get_arm_joints(ARM)])
        return float(np.linalg.norm((self.base_to_env @ ee[0])[:3, 3] - self.env.get_arm_pose(ARM)[:3]))


def _ik_near(planner, env, pose, seeds):
    """``pose``(ee_link)在关节限位内、离 ``seeds`` 中某个种子最近的 IK 解;都没有时返回 None。"""
    bounds = np.asarray(SIM_JOINT_BOUNDS.get(env.robot(ARM).robot_name, [np.pi + 1e-3] * 6))
    best, best_d = None, np.inf
    for seed in seeds:
        r = planner.solve_ik_to_joint(np.asarray(seed, dtype=float), list(pose), env.robot(ARM).entity_origin_pose)
        if r["status"] != "Success":
            continue
        q = (np.asarray(r["joint_value"], dtype=float) + np.pi) % (2 * np.pi) - np.pi
        if np.any(np.abs(q) > bounds):
            continue
        d = float(np.abs(q - np.asarray(seed)).max())
        if d < best_d:
            best, best_d = q, d
    return best


def _folds(plans):
    """路径中是否有腕部向上臂折回的构型(``J4_MAX`` / ``J2_PLUS_J4_MAX``);有时返回描述。"""
    q = np.concatenate([np.asarray(res["position"]) for res in plans])
    j4, j24 = float(np.abs(q[:, 3]).max()), float((q[:, 1] + q[:, 3]).max())
    if j4 > J4_MAX or j24 > J2_PLUS_J4_MAX:
        return f"folds the wrist onto the arm (|joint4| {j4:.2f}, joint2 + joint4 {j24:.2f})"
    return None


def _choose_clear(env, clearance, chains, what, held=None, table="all"):
    """``chains`` 中第一个能规划、并且整段不碰微波炉的候选链 ``(index, plans)``。"""
    reasons = []
    for i, poses in enumerate(chains):
        # 同 env.plan_chain,只是记下第一个规划不到的路点,失败时好定位;("joints", q) 表示关节空间规划到 q
        start, plans = env.get_arm_joints(ARM), []
        for pose in poses:
            if isinstance(pose, tuple):
                res = clearance.planner.plan_joint(start, pose[1])
                if res["status"] != "Success":
                    break
            else:
                try:
                    res = env.plan_to_pose(ARM, pose, start_joints=start)
                except PlanFailError:
                    break
            plans.append(res)
            start = np.asarray(res["position"][-1])
        if len(plans) < len(poses):
            j = len(plans)
            at = "joints" if isinstance(poses[j], tuple) else np.round(poses[j][:3], 3).tolist()
            reasons.append(f"unreachable at pose {j}/{len(poses)} {at}")
            continue
        fold = _folds(plans)
        if fold is not None:
            reasons.append(fold)
            continue
        hit = clearance.collides(plans, held, table)
        if hit is None:
            return i, plans
        reasons.append(hit)
    raise PlanFailError(f"{ARM}: no {what} path clears the microwave: " + "; ".join(reasons[:3] + reasons[-3:]))


def play_once(env, spec=EGG):
    try:
        _play(env, spec)
    except Exception:
        # 失败时也像成功时那样松开夹爪、回到初始位姿:手里拿着东西、停在别处结束的 episode 之后,
        # 同一进程的下一次 reset 会让机器人位姿变成 NaN(set_origin_endpose 里 LinAlgError)
        try:
            env.open_gripper(ARM)
            env.back_to_origin(ARM)
        except Exception as e:
            log.warning("cleanup after a failed attempt failed: %s: %s", type(e).__name__, e)
        raise


def _pose_from_axes(tcp, x, y, bias):
    """TCP 位置 + ee 的 x(接近方向)、y(手指方向)-> ee_link 位姿。"""
    x = x / np.linalg.norm(x)
    y = y - (y @ x) * x
    y = y / np.linalg.norm(y)
    R = np.stack([x, y, np.cross(x, y)], axis=1)
    return np.concatenate([tcp - bias * x, t3d.quaternions.mat2quat(R)])


def _horizontal(v):
    v = np.array([v[0], v[1], 0.0])
    return v / np.linalg.norm(v)


def _analytic_grasps(env, spec, inward):
    """``top`` / ``side`` 的手写抓取候选(ee_link 位姿),按优先级排序。"""
    bias = env.robot(ARM).gripper_bias
    pts = _bbox_points(env, spec.label)
    c = _center(env, spec.label)
    bottom, top = pts[:, 2].min(), pts[:, 2].max()
    up = np.array([0.0, 0.0, 1.0])
    out = []
    if spec.grasp == "top":
        # 指尖在 TCP 下方 1.3 cm:TCP 至少离桌面 TIP_CLEAR + 1.3 cm,并且不低于物体中心(汉堡夹在下层面包圆边上时,
        # 手指一合就把它挤出去)
        tcp = np.array([c[0], c[1], min(max(_table_z(env) + TIP_CLEAR + 0.013, c[2]), top - 0.005)])
        if spec.top_depth is not None:  # 指尖只伸到顶面下 top_depth + 1.3 cm:夹住上半部分,挤压只会把物体往桌上压
            tcp[2] = top - spec.top_depth
        for yaw in TOP_YAWS:
            finger = t3d.axangles.axangle2mat(up, np.radians(yaw)) @ np.cross(up, inward)
            for f in (finger, -finger):
                out.append(_pose_from_axes(tcp, -up, f, bias))
        return out
    base = env.robot(ARM).entity_origin_pose[:3]
    toward = _horizontal(c - np.asarray(base, dtype=float))
    if spec.box:
        m = _mat(env.get_obj_pose(spec.label))
        lo, hi = _bbox(env, spec.label)
        dirs = []
        for axis, other in ((0, 1), (1, 0)):
            if (hi - lo)[other] > FINGER_MAX:  # 手指夹住的是另一条轴方向上的两个侧面
                continue
            for sgn in (1.0, -1.0):
                a = _horizontal(sgn * m[:3, axis])
                if np.degrees(np.arccos(np.clip(a @ toward, -1, 1))) <= SIDE_MAX_YAW:
                    dirs.append(a)
        dirs.sort(key=lambda a: -(a @ toward))
    else:
        dirs = [t3d.axangles.axangle2mat(up, np.radians(yaw)) @ toward for yaw in SIDE_YAWS]
    tcp = np.array([c[0], c[1], bottom + spec.grip_height])
    for pitch in SIDE_PITCHES:
        p = np.radians(pitch)
        for a in dirs:
            finger = np.cross(up, a)
            for f in (finger, -finger):
                out.append(_pose_from_axes(tcp, np.cos(p) * a - np.sin(p) * up, f, bias))
    return out


def _grasp(env, spec, inward, clearance):
    """抓起 ``spec.label``;返回 ``(ee_pose, score)``(手写候选的 score 为 None)。
    ``graspgen`` 同 ``CollectEnv.grasp_actor``,只是多一条过滤:手指横跨物体短径。"""
    if spec.grasp == "graspgen":
        p = np.radians(spec.prefer_pitch)
        prefer = np.cos(p) * inward + np.sin(p) * np.array([0.0, 0.0, -1.0])
        long_axis = _mat(env.get_obj_pose(spec.label))[:3, 2]
        candidates = env.grasp_candidates(
            spec.label, ARM, prefer=prefer, approach_z=spec.approach_z, finger_z_max=FINGER_Z_MAX, max_candidates=1000
        )
        if spec.across_long_axis:
            candidates = [(g, sc) for g, sc in candidates if abs(_mat(g)[:3, 1] @ long_axis) < FINGER_ALONG_MAX]
        if spec.fingers_across_inward:
            # 手指与炉腔朝里的方向垂直:伸入时末端只需绕竖直轴转,不用绕接近轴拧,伸入起点才有 IK 解
            cross = np.cross([0.0, 0.0, 1.0], inward)
            candidates = [(g, sc) for g, sc in candidates if abs(_mat(g)[:3, 1] @ cross) > FINGER_ACROSS_MIN]
        candidates = candidates[:MAX_GRASPS]
    else:
        candidates = [(g, None) for g in _analytic_grasps(env, spec, inward)]
    if not candidates:
        raise PlanFailError(f"{ARM}: no {spec.grasp} grasp candidate for {spec.label}")
    # cuRobo 的碰撞世界里没有物体,从初始位姿直接规划到低处的预抓取位姿时手臂会扫过高的物体(14.6 cm 的罐装饮料);
    # 伸向物体时 link4 / link5 还会蹭到微波炉右侧的控制面板,机械臂和微波炉一起炸掉(物理 NaN)。
    # 候选链按优先级:(1) 先到预抓取位姿正上方的高点(手指尖高出物体顶面 ABOVE_TOP),竖直下到预抓取位姿,再直线接近;
    # (2) 同 (1),但先从初始位姿竖直抬起 START_LIFT;(3) 直接到预抓取位姿(物体旁边的高点 IK 常常无解,
    # 例如离基座很近的水平侧抓)。每条链都过微波炉碰撞检查。
    top = _bbox_points(env, spec.label)[:, 2].max()
    start = env.get_arm_pose(ARM)
    start[2] += START_LIFT
    bias = env.robot(ARM).gripper_bias
    via_above, via_lift, direct = [], [], []
    for g, _ in candidates:
        approach = _mat(g)[:3, 0]
        pre = g.copy()
        pre[:3] -= PRE_GRASP * approach
        tail = env._straight_poses(pre, g[:3] - pre[:3], 0.02)
        above = pre.copy()
        above[2] = max(pre[2], top + ABOVE_TOP + bias * max(0.0, -approach[2]))
        down = env._straight_poses(above, pre[:3] - above[:3], 0.03) if above[2] > pre[2] + 1e-3 else []
        via_above.append([above, *down, *tail])
        via_lift.append([start, above, *down, *tail])
        direct.append([pre, *tail])
    if spec.from_above:
        chains, candidates = via_above + via_lift + direct, candidates * 3
    else:  # 矮的物体(鸡蛋、罐头、汉堡)手臂扫不到:直接到预抓取位姿,同 heat_egg_microwave 采 20 条数据时的做法
        chains = direct
    i, plans = _choose_clear(env, clearance, chains, "grasp", table="body")
    env.open_gripper(ARM)
    for res in plans:
        env.execute_plan(ARM, res)
    if spec.gentle_close:
        # 只合到比物体窄 GENTLE_SQUEEZE、并且慢慢合:一下合到底时,圆鼓鼓的汉堡被两指一挤就弹飞
        fingers = _mat(candidates[i][0])[:3, 1]
        pts = _bbox_points(env, spec.label)[FACE_CENTERS]
        width = float(np.ptp(pts @ fingers))
        env.close_gripper(ARM, max(0.0, (width - GENTLE_SQUEEZE) / GRIPPER_OPEN), GENTLE_STEPS)
    else:
        env.close_gripper(ARM)
    env.delay(GRASP_SETTLE)
    return candidates[i]


def _play(env, spec):
    label = spec.label
    mw, k, geom, inward = _inward(env)
    clearance = _Clearance(env, mw, geom)
    grasp, score = _grasp(env, spec, inward, clearance)
    held0 = _tcp_dist(env, label)
    log.info(
        "%s grasp approach %s score %s; tcp-%s %.3f after closing",
        spec.grasp,
        np.round(_mat(grasp)[:3, 0], 2).tolist(),
        None if score is None else round(score, 3),
        label,
        held0,
    )

    def check_held(stage):
        d = _tcp_dist(env, label)
        if d > held0 + SLIP_TOL:
            raise GraspLostError(f"{label} slipped while {stage} (tcp-{label} {held0:.3f} -> {d:.3f})")

    env.move_up(ARM, LIFT)
    check_held("lifting")
    if spec.carry_to_front:
        # 从微波炉右侧抓起的物体:先搬到炉口正前方(TCP 在炉口外 CARRY_FRONT、转盘中心的横向位置,高度不变),
        # 从那里再转向、伸入;直接从右侧伸进去时手指 / 腕部扫过炉口右侧的控制面板
        here = env.get_arm_pose(ARM)
        tcp = here[:3] + env.robot(ARM).gripper_bias * _mat(here)[:3, 0]
        plate_xy = (mw @ [*geom.plate_xy, 0.0, 1.0])[:2]
        front = (mw @ [0.0, geom.opening_y, 0.0, 1.0])[:3] - CARRY_FRONT * inward
        goal = np.append(plate_xy + ((front[:2] - plate_xy) @ inward[:2]) * inward[:2], tcp[2])
        moves = [goal + [0.0, 0.0, dz] for dz in (0.0, 0.03, -0.02)]
        chains = [[np.concatenate([g - (tcp - here[:3]), here[3:]])] for g in moves]
        _, plans = _choose_clear(env, clearance, chains, "carry")
        for res in plans:
            env.execute_plan(ARM, res, INSERT_SLOW)
        check_held("carrying to the microwave")

    # TCP 停在转盘中心上方 h 处;炉口平面到 TCP 终点的水平距离 = 伸入深度。按 TCP 而不是物体中心对准转盘:
    # 物体中心在 TCP 后面 1 ~ 2 cm,按物体中心对准时夹爪要再往里伸,相机支架就碰到炉口上沿
    ee = _mat(env.get_arm_pose(ARM))
    ee_inv = np.linalg.inv(ee)
    pts = _bbox_points(env, label)
    if spec.round:
        pts = pts[FACE_CENTERS | (np.sum(GRID == 0.5, axis=1) == 3)]
    held = (np.c_[pts, np.ones(len(pts))] @ ee_inv.T)[:, :3]  # 手中物体的采样点,ee_link 坐标系
    floor_pts = held if spec.round else held[FACE_CENTERS]
    center_ee = (ee_inv @ np.append(_center(env, label), 1.0))[:3]
    bias = env.robot(ARM).gripper_bias
    plate = (mw @ [*geom.plate_xy, geom.plate_z, 1.0])[:3]
    opening = (mw @ [0.0, geom.opening_y, 0.0, 1.0])[:3]

    # 每个候选:炉口外远处的起点 + 每 2 cm 一段的直线伸入,整段可达且不碰微波炉才用
    candidates, chains = [], []
    heights = TCP_HEIGHTS_TALL if geom.cavity[1][1] - geom.plate_z >= TALL_CAVITY else TCP_HEIGHTS
    cavity_depth = float(geom.cavity_back - geom.opening_y)
    for h, pitch, out in [(h, p, o) for o in STAGE_OUT for h in heights for p in INSERT_PITCHES]:
        rot = _insert_rot(inward, ee[:3, 1], pitch)
        end = plate + [0.0, 0.0, h] - bias * rot[:, 0]  # TCP 在转盘中心正上方(物体中心在它前后 1 ~ 2 cm)
        inner = float(((end + held @ rot.T - opening) @ inward).max())  # 物体最里面离炉口平面多深
        end -= max(0.0, inner - (cavity_depth - BACK_CLEAR)) * inward
        bottom = end[2] + (floor_pts @ rot.T)[:, 2].min()
        if bottom < plate[2] + OBJ_FLOOR:
            continue
        depth = float((end + bias * rot[:, 0] - opening) @ inward)
        far = np.concatenate([end - (depth + out) * inward, t3d.quaternions.mat2quat(rot)])
        line = env._straight_poses(far, (depth + out) * inward, 0.02)
        short = round(float((plate - end - bias * rot[:, 0]) @ inward), 3)
        candidates.append((h, pitch, out, round(float(bottom - plate[2]), 3), short))
        chains.append([far, *line])
        # 同一条直线,但先在关节空间走到起点的"正手"IK 解:从微波炉右侧抓起物体时,按位姿规划到起点会落到腕部翻转的
        # 分支上,之后的直线伸入没有连续的 IK 解
        q = _ik_near(clearance.planner, env, far, INSERT_SEEDS)
        if q is not None:
            candidates.append((h, pitch, out, round(float(bottom - plate[2]), 3), short))
            chains.append([("joints", q), *line])
    if not chains:
        raise PlanFailError(f"{label} hangs too low in the gripper (center {np.round(center_ee, 3).tolist()} in ee)")
    i, plans = _choose_clear(env, clearance, chains, "insertion", held)
    log.info(
        "%s plate center %s, inward %s, insert (tcp height, pitch, stage out, bottom above plate, tcp short of plate"
        " center) %s",
        label,
        np.round(plate, 3).tolist(),
        np.round(inward[:2], 2).tolist(),
        candidates[i],
    )

    env.execute_plan(ARM, plans[0])
    check_held("turning toward the microwave")
    for res in plans[1:]:
        env.execute_plan(ARM, res, INSERT_SLOW)

    log.info(
        "before release: %s z axis . inward = %.2f, microwave moved %.3f m, FK prediction error %.4f m",
        label,
        abs(_mat(env.get_obj_pose(label))[:3, 2] @ inward),
        float(np.linalg.norm(env.get_obj_pose("microwave")[:3] - mw[:3, 3])),
        clearance.check_calibration(),
    )

    def moved(stage):
        d = env.get_obj_pose("microwave")[:3] - mw[:3, 3]
        log.info("microwave moved %s m after %s", np.round(d, 3).tolist(), stage)

    moved("inserting")
    env.open_gripper(ARM)
    env.delay(30)
    moved("releasing")
    # 退出:竖直抬起 lift(不碰炉腔顶)+ 每 2 cm 一段的水平后退,整段不碰微波炉
    here = env.get_arm_pose(ARM)
    chains = []
    for lift in LIFTS_OUT:
        up = here.copy()
        up[2] += lift
        chains.append(([up] if lift > 0 else []) + env._straight_poses(up, -RETREAT * inward, 0.02))
    i, plans = _choose_clear(env, clearance, chains, "retreat")
    log.info("retreat with a %.3f m lift", LIFTS_OUT[i])
    for res in plans:
        env.execute_plan(ARM, res)
    moved("retreating")
    env.back_to_origin(ARM)
    moved("returning to origin")
    mw_now = _mat(env.get_obj_pose("microwave"))
    end = np.linalg.solve(mw_now, np.append(_center(env, label), 1.0))[:3] / k
    log.info("%s center at the end, in unscaled microwave coordinates: %s", label, np.round(end, 3).tolist())
