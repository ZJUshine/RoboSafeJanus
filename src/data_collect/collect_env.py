"""采集环境:任务环境 + RoboTwin 风格的脚本化运动原语。

``create_collect_env`` 包装一个已注册的任务类(与 ``src/eval_client/eval_env.py`` 的做法相同),
但不带策略客户端。包装类提供专家脚本组合使用的原语 —— 基于 cuRobo 的 ``move`` / ``together_move``、
``open_gripper`` / ``close_gripper``、``move_by_displacement``、``back_to_origin``,以及用 GraspGenX
抓取缓存的 ``grasp_actor`` 和按放置点对齐的 ``place_actor`` ——
并在执行过程中每 ``collect_interval`` 个物理步录一帧观测。

位姿是手臂 ``ee_link`` 在 env 坐标系下的 ``[x, y, z, qw, qx, qy, qz]``,
与 ``robot_manager.get_real_endpose`` 同一坐标系。手臂用观测名称指代:
双臂机器人为 ``left_arm`` / ``right_arm``。
"""

from copy import deepcopy
import importlib
import json
import os

import numpy as np
import transforms3d as t3d

from env.global_configs import BENCHMARK
from env.observation_manager.obs_manager import ObsManager
from utils.cluttered_generator import UnStableError


class PlanFailError(Exception):
    """cuRobo 规划失败(无解、只有越界解或所有候选都不可达);``expert_collection.py`` 记为 ``plan_failed``。"""

    pass


# x5 的 URDF 给关节 1-5 设的是 ±10 rad 的占位限位,cuRobo 的路径可能超出仿真手臂能跟随的范围:
# 差了整整一圈的 IK 解(如 joint1 = -6.5)会在关节空间重新规划到折回 [-pi, pi] 的同一目标;
# 经过物理上到不了的关节角的路径(x5 腕关节 joint5 在约 ±1.56 rad 处自碰)会重新规划
# (每次调用的 IK 种子不同),多次失败后放弃。
JOINT_BOUND = np.pi + 1e-3
SIM_JOINT_BOUNDS = {"x5": [JOINT_BOUND] * 4 + [1.55, JOINT_BOUND]}  # 每个手臂关节允许的 |q| 上限
PLAN_TRIES = 3  # plan_to_pose 的最大重试次数
SETTLE_TOL = 0.02  # rad
SETTLE_STEPS = 300  # 物理步数(dt = 0.004 时为 1.2 s)
# GraspGenX 离线生成的抓取缓存(src/data_collect/grasp_gen.py);可用环境变量改位置
GRASP_CACHE_DIR = os.environ.get(
    "ROBODOJO_GRASP_CACHE", os.path.join(os.path.dirname(__file__), "..", "..", "data", "grasp_pose")
)
GRASP_GRIPPER = {"x5": "arx_x5"}  # 机器人名 -> GraspGenX 夹爪名
_GRASP_CACHE = {}  # 已读入的缓存文件,按路径记忆


def load_grasps(gripper, category, index):
    """读取一个资产的抓取缓存:``(ee_poses (N, 7) 物体坐标系, scores (N,), meta dict)``;没有缓存直接报错。"""
    path = os.path.normpath(os.path.join(GRASP_CACHE_DIR, gripper, category, f"{int(index):05d}.npz"))
    if path not in _GRASP_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no GraspGenX grasps for {category}/{index} ({path}); generate them with "
                f"bash scripts/gen_grasps.sh <task>"
            )
        with np.load(path) as f:
            _GRASP_CACHE[path] = (f["ee_poses"], f["scores"], json.loads(str(f["meta"])))
    return _GRASP_CACHE[path]


def _mat(pose):
    """[x, y, z, qw, qx, qy, qz] -> 4x4 齐次矩阵。"""
    m = np.eye(4)
    m[:3, :3] = t3d.quaternions.quat2mat(pose[3:7])
    m[:3, 3] = pose[:3]
    return m


def _pose(m):
    """4x4 齐次矩阵 -> [x, y, z, qw, qx, qy, qz]。"""
    return np.concatenate([m[:3, 3], t3d.quaternions.mat2quat(m[:3, :3])])


def _reuse_live_articulations(layout_manager):
    """让每个新布局中的关节物体复用场景里同一模型的现有对象。

    关节物体只生成一次,重载场景时不会重新生成(见 SceneManager.reload_env_scene),
    但 LayoutManager.load_saved_layout 会给它分配一个新的、指向空对象的实例名。
    采集每个 episode 都会重载布局,所以从第二个 episode 起关节物体(如烤面包机)会停在画面外。
    要求每个布局中该模型的 index 相同。
    """
    generate_paths = layout_manager._generate_object_paths
    load_layout = layout_manager.load_saved_layout

    def live_articulations(env_idx):
        scene_manager = layout_manager.scene_manager
        return scene_manager._articulation_objects[env_idx] if scene_manager is not None else {}

    def generate_object_paths(env_idx, cat_name, model_idx, type):
        # 关节物体:返回同模型、尚未被占用的现有 prim 路径和实例名
        if type == "articulation":
            taken = layout_manager.instance_type_by_env[env_idx]
            for name, obj in live_articulations(env_idx).items():
                if name not in taken and name.startswith(f"{cat_name}_{model_idx}_"):
                    return obj._prim_path, name
        return generate_paths(env_idx, cat_name, model_idx, type=type)

    def load_saved_layout(env_idx):
        # 把新布局中的位姿写回现有关节物体的默认位姿
        env_config = load_layout(env_idx)
        live = live_articulations(env_idx)
        for inst_list in (env_config or {}).get("Articulation", {}).values():
            for inst in inst_list:
                obj = live.get(inst["inst_name"])
                if obj is not None:
                    obj.default_pos = inst.get("default_pos", obj.default_pos)
                    obj.default_ori = inst.get("default_ori", obj.default_ori)
        return env_config

    layout_manager._generate_object_paths = generate_object_paths
    layout_manager.load_saved_layout = load_saved_layout


def create_collect_env(config, app, **kwargs):
    """按 ``config.collect_cfg["task_name"]`` 加载任务类,返回继承它的 ``CollectEnv`` 实例。"""
    task_name = config.collect_cfg["task_name"]
    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    task_name, task_class = task_registry.load_task_class(task_name)

    class CollectEnv(task_class):
        """任务环境 + 采集用的查询、规划、执行原语和录帧逻辑。"""

        def __init__(self, config, app, **kwargs):
            super().__init__(config, app, **kwargs)
            _reuse_live_articulations(self.scene_manager.layout_manager)
            self.collect_cfg = config.collect_cfg
            self.task_name = task_name
            self.obs_manager = ObsManager(
                obs_config=deepcopy(self.collect_cfg.get("observation", {})),
                num_envs=self.num_envs,
                dt=self.dt,
                task_name=self.task_name,
                description_cfg=self.collect_cfg.get("description", dict()),
                seeds_per_env=self.env_seed_list,
            )
            # 每多少个物理步录一帧(由 collect_freq 和 dt 决定)
            self.collect_interval = max(int(self.obs_manager.collect_interval), 1)
            self.success = [True] * self.num_envs
            self.end_flag = [False] * self.num_envs
            self.frames = []
            self.record = False
            self._step_cnt = 0

        def _post_setup_scene(self, sim):
            super()._post_setup_scene(sim)
            self.obs_manager.initialize(self)

        def close(self):
            self.obs_manager.reset()
            super().close()

        # ------------------------------------------------------------ episode 管理

        def reset_with_layout(self, layout, seed, record=True):
            """把 ``layout`` 加载到 env 0,等待物体沉降,开始新 episode;布局不稳时抛 ``UnStableError``。"""
            self.success = [True] * self.num_envs
            self.end_flag = [False] * self.num_envs
            self.scene_manager.layout_manager.set_saved_layout(0, layout)
            # 否则第一次 reset 会保留启动时构建场景用的布局
            # (SceneManager.reload_env_scene 只有在 setup_scene 被清掉后才会重载)。
            self.scene_manager.setup_scene = False
            super().reset(seed=[seed], options=None)
            self.obs_manager.reset()
            self.scene_manager.apply_saved_poses(env_idx_list=[0])
            stable, _ = self.scene_manager.layout_manager.check_layout_stability(self)
            if not stable:
                raise UnStableError(f"Layout for seed {seed} is unstable")
            # 预热渲染并空跑一段物理,让物体完全静止
            for _ in range(10):
                self.render()
            for _ in range(100):
                self.sim_step()
            # 重置机器人和判定状态,注册本任务的 run_reward 检查
            self.robot_manager.set_origin_endpose()
            self.robot_manager.set_robot_init_state()
            self.reward_manager.init_state()
            self.run_reward()
            if hasattr(self, "get_score"):
                self.get_score()
            self.plan_success = True
            self.grip_cmd = {}
            self.last_grasp = None  # 本 episode 最近一次 grasp_actor 选中的候选(调试用)
            self.frames = []
            self.record = record
            self._step_cnt = 0
            # 本 episode 的初始关节角 / 末端位姿,供 back_to_origin 和专家使用
            self.init_joints = {
                r.arm_name: np.asarray(self.robot_manager.get_joint(r, [0])[0]) for r in self._target_robots()
            }
            self.init_endpose = {
                r.arm_name: np.asarray(self.robot_manager.get_real_endpose(r, [0])[0]) for r in self._target_robots()
            }
            self._capture()

        def check_success(self, settle_steps=50):
            """先等物体静止,再运行任务自身的 run_reward 检查(所有检查须同时成立)。"""
            self.delay(settle_steps)
            reward = self.reward_manager.get_reward(final_check=True)[0]
            return reward > 1 - 1e-3

        def failed_checks(self):
            """当前未成立的 run_reward 检查项名称,用于失败日志。"""
            rm = self.reward_manager
            if not rm.func_parser._check_env_success(0):
                return ["env marked failed"]

            def name(check):
                return check[0] if isinstance(check, tuple) else "any-of group"

            failed = []
            if rm.check_list[0]:
                failed += [name(c) for c in rm.check_list[0][0] if not rm.check_once(c, 0)]
            # 仍在等待中的 trigger_check:触发条件发生时其检查从未通过(或从未触发)。
            for conditions, checks, mode, _ in rm.trigger_check_list[0]:
                failed.append(f"{'+'.join(map(name, checks))} (on {mode} of {'+'.join(map(name, conditions))})")
            return failed

        def instructions(self):
            """语言指令候选:任务 ``gen_instruction`` 的模板,其次 ObsManager 抽到的指令,最后用任务名。"""
            templates = list(self.gen_instruction(0)) if hasattr(self, "gen_instruction") else []
            sampled = self.obs_manager.instruction[0]
            return templates or ([sampled] if sampled else [self.task_name.replace("_", " ")])

        # ----------------------------------------------------------- 查询

        def _target_robots(self):
            """被控制的机器人(type == "target"),每只手臂一个。"""
            return [r for r in self.robot_manager.robot_list if r.type == "target"]

        def robot(self, arm):
            """``arm``(如 ``left_arm``)对应的机器人对象。"""
            return self.robot_manager.get_robot_by_arm_name(arm)

        def get_arm_pose(self, arm):
            """``ee_link`` 位姿 ``[x, y, z, qw, qx, qy, qz]``(env 坐标系)。"""
            return np.asarray(self.robot_manager.get_real_endpose(self.robot(arm), [0])[0], dtype=float)

        def get_arm_joints(self, arm):
            """手臂当前关节角。"""
            return np.asarray(self.robot_manager.get_joint(self.robot(arm), [0])[0], dtype=float)

        def get_obj_pose(self, label):
            """物体 ``label`` 的位姿 ``[x, y, z, qw, qx, qy, qz]``。"""
            pos, rot = self.scene_manager.layout_manager.get_instance_pose(env_idx=0, label=label)
            to_np = lambda v: v.detach().cpu().numpy().reshape(-1) if hasattr(v, "detach") else np.asarray(v)  # noqa: E731
            return np.concatenate([to_np(pos), to_np(rot)]).astype(float)

        def get_obj_extents(self, label):
            """物体 ``label`` 的 aligned bbox 尺寸。"""
            meta = self.scene_manager.layout_manager.get_instance_metadata(env_idx=0, label=label)
            return np.asarray(meta["geometry"]["aligned_bbox"]["extents"], dtype=float)

        def get_obj_asset(self, label):
            """``label`` 对应的资产:``(category, index, scale, metadata)``。"""
            lm = self.scene_manager.layout_manager
            meta = lm.get_instance_metadata(env_idx=0, label=label)
            if meta is None:
                raise KeyError(f"no object with label {label}")
            inst_name = lm.get_instance_name(0, label)
            scale = [1.0, 1.0, 1.0]
            for records in lm.object_records_by_type.values():
                for rec in records.layout_records_by_env[0]:
                    if rec.get("inst_name") == inst_name:
                        scale = rec.get("scale", scale)
            return meta["model_name"], int(meta["model_id"]), np.asarray(scale, dtype=float), meta

        def tcp_to_ee(self, arm, tcp_pose):
            """把夹爪中心(TCP)位姿转换成规划器需要的 ``ee_link`` 位姿。

            TCP 位于 ee x 轴(接近方向)上 ``gripper_bias`` 米处,与 RoboTwin 的本体配置一致。
            """
            tcp_pose = np.asarray(tcp_pose, dtype=float)
            axis = t3d.quaternions.quat2mat(tcp_pose[3:])[:, 0]
            return np.concatenate([tcp_pose[:3] - self.robot(arm).gripper_bias * axis, tcp_pose[3:]])

        # ---------------------------------------------------------- 规划

        def plan_to_pose(self, arm, pose, start_joints=None):
            """用 cuRobo 规划 ``arm`` 到 ee 位姿 ``pose`` 的路径(只规划不执行);失败抛 ``PlanFailError``。

            越出 ``JOINT_BOUND`` 的解改为关节空间规划到折回后的目标;仍越出 ``SIM_JOINT_BOUNDS`` 的解重试,
            最多 ``PLAN_TRIES`` 次。
            """
            robot = self.robot(arm)
            start = np.asarray(self.get_arm_joints(arm) if start_joints is None else start_joints, dtype=float)
            planner = self.robot_manager.planner[robot.robot_name]
            bounds = np.asarray(SIM_JOINT_BOUNDS.get(robot.robot_name, [JOINT_BOUND] * len(start)))
            for _ in range(PLAN_TRIES):
                res = planner.plan_path(start, list(pose), real_robot_pose=deepcopy(robot.entity_origin_pose))
                if res["status"] != "Success":
                    raise PlanFailError(
                        f"{arm}: no path to {np.round(pose, 3).tolist()} from joints {np.round(start, 3).tolist()}"
                    )
                if np.abs(np.asarray(res["position"])).max() > JOINT_BOUND:
                    goal = np.asarray(res["position"][-1], dtype=float)
                    res = planner.plan_joint(start, (goal + np.pi) % (2 * np.pi) - np.pi)
                    if res["status"] != "Success":
                        continue
                if (np.abs(np.asarray(res["position"])) <= bounds).all():
                    return res
            raise PlanFailError(f"{arm}: only paths outside the reachable joint range to {np.round(pose, 3).tolist()}")

        def plan_chain(self, arm, poses):
            """依次规划经过 ``poses`` 的多段路径,每段从上一段终点出发;任一段失败返回 None。"""
            start, plans = self.get_arm_joints(arm), []
            for pose in poses:
                try:
                    res = self.plan_to_pose(arm, pose, start_joints=start)
                except PlanFailError:
                    return None
                plans.append(res)
                start = np.asarray(res["position"][-1])
            return plans

        def choose_chain(self, arm, candidates):
            """仿 RoboTwin 的 choose_grasp_pose:返回第一个能规划成功的候选链 (index, plans)。"""
            for i, poses in enumerate(candidates):
                plans = self.plan_chain(arm, poses)
                if plans is not None:
                    return i, plans
            raise PlanFailError(f"{arm}: none of {len(candidates)} candidate chains is reachable")

        def execute_plan(self, arm, res, slow=1.0):
            """执行一个规划结果并等手臂到位;``slow`` > 1 时按比例放慢。"""
            self.execute(self._arm_seq(arm, res, slow))
            self.settle(arm, res["position"][-1])

        def settle(self, arm, joints, tol=SETTLE_TOL, max_steps=SETTLE_STEPS):
            """等待 ``arm`` 到达 ``joints``:长距离运动时关节驱动会落后于 cuRobo 的时间轴,
            下一个原语(如闭合夹爪)不能在手臂还在路上时开始。"""
            joints = np.asarray(joints, dtype=float)
            for _ in range(0, max_steps, 10):
                if np.abs(self.get_arm_joints(arm) - joints).max() < tol:
                    return
                self.delay(10)

        def is_held(self, arm, label, tol=0.08):
            """``label`` 是否仍在夹爪中心(TCP)``tol`` 米范围内。"""
            ee = self.get_arm_pose(arm)
            tcp = ee[:3] + self.robot(arm).gripper_bias * t3d.quaternions.quat2mat(ee[3:])[:, 0]
            return float(np.linalg.norm(self.get_obj_pose(label)[:3] - tcp)) < tol

        def plan_to_joints(self, arm, joints):
            """关节空间规划到 ``joints``;失败抛 ``PlanFailError``。"""
            robot = self.robot(arm)
            planner = self.robot_manager.planner[robot.robot_name]
            res = planner.plan_joint(self.get_arm_joints(arm), np.asarray(joints, dtype=float))
            if res["status"] != "Success":
                raise PlanFailError(f"{arm}: no joint-space path")
            return res

        def _arm_seq(self, arm, res, slow=1.0):
            """把关节路点转成控制字典序列;``slow`` > 1 时在时间上拉伸路径(如搬运时)。"""
            key = self.robot_manager.process_name(arm)
            pos, vel = np.asarray(res["position"]), np.asarray(res["velocity"])
            if slow > 1.0 and len(pos) > 1:
                t_old = np.arange(len(pos))
                t_new = np.linspace(0, len(pos) - 1, int(round(len(pos) * slow)))
                pos = np.stack([np.interp(t_new, t_old, pos[:, j]) for j in range(pos.shape[1])], axis=1)
                vel = np.stack([np.interp(t_new, t_old, vel[:, j]) for j in range(vel.shape[1])], axis=1) / slow
            return [{key: {"position": p.tolist(), "velocity": v.tolist()}} for p, v in zip(pos, vel)]

        def _gripper_seq(self, arm, val, steps):
            """夹爪在 ``steps`` 步内运动到归一化开度 ``val`` 的控制序列。"""
            return self.robot_manager.plan_endeffector_joint(
                env_idx=0, arm_tag=arm, target_val=val, control_step_num=steps
            )

        # --------------------------------------------------------- 执行

        def execute(self, *seqs):
            """并行执行多条控制序列(每个手臂 / 夹爪一条),较短的序列在末尾保持不动。

            每个物理步都会推进仿真,每 ``collect_interval`` 步录一帧;结束后更新一次 reward_manager。
            """
            n = max((len(s) for s in seqs), default=0)
            merged = [dict() for _ in range(n)]
            for seq in seqs:
                for i, ctrl in enumerate(seq):
                    merged[i].update(ctrl)
            if not merged:
                return
            self.robot_manager.control_manager.push([0], [merged])
            while not self.robot_manager.control_manager.control_queue[0].is_empty():
                meta = self.robot_manager.control_manager.pop([0])
                super().step(meta_control_list=meta)
                self.sim_step(render=False)
                self._step_cnt += 1
                if self._step_cnt % self.collect_interval == 0:
                    self._capture()
            self.reward_manager.step(env_idx_list=[0])

        def _capture(self):
            """录制状态下渲染并追加一帧观测到 ``self.frames``。"""
            if not self.record:
                return
            self.obs_manager.render_for_capture()
            obs = self.obs_manager.get_obs(env_idx_list=[0])[0]
            self.frames.append(obs)

        # --------------------------------------------------------- 运动原语

        def move(self, arm, pose):
            """沿 cuRobo 的无碰撞路径把一只手臂的 ee_link 移到 ``pose``。"""
            self.execute_plan(arm, self.plan_to_pose(arm, pose))

        def together_move(self, left_pose, right_pose, left_arm="left_arm", right_arm="right_arm"):
            """双臂同时移动到各自的目标位姿。"""
            self.execute(
                self._arm_seq(left_arm, self.plan_to_pose(left_arm, left_pose)),
                self._arm_seq(right_arm, self.plan_to_pose(right_arm, right_pose)),
            )

        def move_by_displacement(self, arm, x=0.0, y=0.0, z=0.0, move_axis="world", straight=False, slow=1.0):
            """末端平移,姿态不变(同 RoboTwin)。

            ``move_axis="world"``:按世界坐标 (x, y, z) 平移;``"arm"``:沿接近方向(ee x 轴)后退 ``z`` 米
            (``z`` 为负则前进),``x`` / ``y`` 忽略。``straight=True`` 时按直线分段执行(见 ``move_straight``)。
            """
            pose = self.get_arm_pose(arm)
            if move_axis == "arm":
                delta = -z * t3d.quaternions.quat2mat(pose[3:])[:, 0]
            elif move_axis == "world":
                delta = np.array([x, y, z], dtype=float)
            else:
                raise ValueError(f"move_axis must be 'world' or 'arm', not {move_axis}")
            if straight:
                self.move_straight(arm, delta, slow=slow)
            else:
                pose[:3] += delta
                self.execute_plan(arm, self.plan_to_pose(arm, pose), slow)

        def _straight_poses(self, pose, delta, step):
            """从 ``pose`` 沿 ``delta`` 直线移动的分段目标位姿(每段不超过 ``step`` 米,姿态不变)。"""
            n = max(int(np.ceil(np.linalg.norm(delta) / step)), 1)
            poses = []
            for i in range(1, n + 1):
                p = np.array(pose, dtype=float)
                p[:3] += np.asarray(delta, dtype=float) * i / n
                poses.append(p)
            return poses

        def move_straight(self, arm, delta, step=0.01, slow=1.0):
            """末端沿 ``delta`` 走直线:分成每段 ``step`` 米分别规划。一次规划完整段距离时 cuRobo 可能把手腕
            甩向一侧,而规划器不知道桌上的物体。任一段规划失败抛 ``PlanFailError``。"""
            plans = self.plan_chain(arm, self._straight_poses(self.get_arm_pose(arm), delta, step))
            if plans is None:
                raise PlanFailError(f"{arm}: no straight path along {np.round(delta, 3).tolist()}")
            for res in plans:
                self.execute_plan(arm, res, slow)

        def move_up(self, arm, heights, slow=1.0):
            """竖直抬升 ``heights`` 中第一个可达的高度(高的放前面)。"""
            base = self.get_arm_pose(arm)
            chains = []
            for dz in heights:
                pose = base.copy()
                pose[2] += dz
                chains.append([pose])
            _, (res,) = self.choose_chain(arm, chains)
            self.execute_plan(arm, res, slow)

        def set_gripper(self, arm, val, steps=60):
            """``val`` 为归一化开度:1 = 张开,0 = 闭合。

            从上一次的*指令值*线性过渡。若从实测开度开始,手指被抓着的物体挡住时,
            夹紧力会在一步内从满力降到零,松手瞬间会把物体弹开。
            """
            start = self.grip_cmd.get(arm)
            if start is None:
                seq = self._gripper_seq(arm, val, steps)
            else:
                seq = [self._gripper_seq(arm, v, 1)[0] for v in np.linspace(start, val, max(steps, 1))]
            self.execute(seq)
            self.grip_cmd[arm] = float(val)

        def open_gripper(self, arm, val=1.0, steps=60):
            self.set_gripper(arm, val, steps)

        def close_gripper(self, arm, val=0.0, steps=60):
            self.set_gripper(arm, val, steps)

        def back_to_origin(self, *arms):
            """关节空间回到本 episode 的初始关节角(默认所有手臂同时)。"""
            arms = arms or tuple(r.arm_name for r in self._target_robots())
            self.execute(*[self._arm_seq(a, self.plan_to_joints(a, self.init_joints[a])) for a in arms])

        def delay(self, steps):
            """空等 ``steps`` 个物理步(同样会录帧)。"""
            self.execute([{} for _ in range(steps)])

        # ------------------------------------------------------ 抓取 / 放置(参照 RoboTwin)

        def grasp_candidates(
            self,
            label,
            arm,
            prefer=None,
            prefer_weight=1.0,
            approach_z=(-1.0, 0.3),
            finger_z_max=None,
            obj_z=None,
            max_candidates=40,
        ):
            """``label`` 的 GraspGenX 抓取候选(env 坐标系的 ee 位姿),过滤后按代价从低到高排序。

            过滤条件:
              - ``approach_z``:接近方向单位向量 z 分量的范围,默认排除从下往上抓;
                自上而下约为 ``(-1, -0.7)``,水平侧抓约为 ``(-0.3, 0.1)``;
              - ``finger_z_max``:手指方向 |z| 上限,例如要求手指水平(便于之后绕接近轴转动);
              - ``obj_z``:抓取中心(TCP)在物体坐标系下的高度范围,用来限定抓哪个部位(如瓶颈)。
            排序(参照 RoboTwin ``choose_grasp_pose``):代价 = ``prefer_weight`` x 接近方向与 ``prefer`` 的夹角
            (弧度 / pi) - 判别器分数;``prefer`` 为空时只按分数。夹爪对称时,每个抓取绕接近轴转 180 度的
            版本也作为候选。返回 ``[(ee_pose, score), ...]``,最多 ``max_candidates`` 个。
            """
            robot = self.robot(arm)
            gripper = GRASP_GRIPPER.get(robot.robot_name)
            if gripper is None:
                raise ValueError(f"no GraspGenX gripper registered for robot {robot.robot_name}")
            category, index, scale, _ = self.get_obj_asset(label)
            if not np.allclose(scale, 1.0):
                raise ValueError(f"{label} is scaled by {scale.tolist()}; grasp caches are for unscaled meshes")
            ee_local, scores, meta = load_grasps(gripper, category, index)
            obj_m = _mat(self.get_obj_pose(label))
            flip = np.diag([1.0, -1.0, -1.0, 1.0])  # 绕 x 轴(接近方向)转 180 度
            prefer = None if prefer is None else np.asarray(prefer, dtype=float) / np.linalg.norm(prefer)
            out = []
            for local, score in zip(ee_local, scores):
                for m_local in [_mat(local)] + ([_mat(local) @ flip] if meta.get("symmetric") else []):
                    approach = m_local[:3, 0]
                    if obj_z is not None:
                        tcp_z = m_local[2, 3] + robot.gripper_bias * approach[2]
                        if not obj_z[0] <= tcp_z <= obj_z[1]:
                            continue
                    world = obj_m @ m_local
                    approach_w, finger_w = world[:3, 0], world[:3, 1]
                    if not approach_z[0] <= approach_w[2] <= approach_z[1]:
                        continue
                    if finger_z_max is not None and abs(finger_w[2]) > finger_z_max:
                        continue
                    cost = -float(score)
                    if prefer is not None:
                        cost += prefer_weight * np.arccos(np.clip(approach_w @ prefer, -1.0, 1.0)) / np.pi
                    out.append((cost, _pose(world), float(score)))
            out.sort(key=lambda c: c[0])
            return [(pose, score) for _, pose, score in out[:max_candidates]]

        def grasp_actor(self, label, arm, pre_dis=0.08, close_val=0.0, settle=30, slow=1.0, **filters):
            """用 GraspGenX 抓取候选抓起 ``label``:选第一个能规划到的候选,张开夹爪、移动到预抓取位姿
            (沿接近方向后退 ``pre_dis``),再直线接近、闭合夹爪。

            ``filters`` 传给 ``grasp_candidates``。没有缓存、没有通过过滤的候选、或全部不可达时都直接报错,
            不回退到其他抓取方式。返回选中候选的 ``(ee_pose, score)``。
            """
            candidates = self.grasp_candidates(label, arm, **filters)
            if not candidates:
                raise PlanFailError(f"{arm}: no GraspGenX grasp of {label} passes the filters {filters}")
            chains = []
            for pose, _ in candidates:
                approach = t3d.quaternions.quat2mat(pose[3:])[:, 0]
                pre = pose.copy()
                pre[:3] -= pre_dis * approach
                # 预抓取位姿 + 沿接近方向每 2 cm 一段的直线接近,都可达才算这个候选可用
                chains.append([pre, *self._straight_poses(pre, pose[:3] - pre[:3], 0.02)])
            i, plans = self.choose_chain(arm, chains)
            pose, score = candidates[i]
            self.last_grasp = {"label": label, "index": i, "score": score, "ee_pose": pose}
            self.open_gripper(arm)
            for res in plans:
                self.execute_plan(arm, res, slow)
            self.close_gripper(arm, close_val)
            self.delay(settle)
            return pose, score

        def _place_point(self, meta, point):
            """物体坐标系下的放置对齐点:``"place"`` = ``active.place.up.contact_circle`` 中心(底面接触点),
            ``None`` = 物体原点,也可直接给 (3,) 向量。"""
            if point is None:
                return np.zeros(3)
            if isinstance(point, str):
                if point != "place":
                    raise ValueError(f"unknown place point {point}")
                return np.asarray(meta["active"]["place"]["up"]["contact_circle"]["center"][:3], dtype=float)
            return np.asarray(point, dtype=float)

        def place_actor(
            self,
            label,
            arm,
            targets,
            point="place",
            pre_dis=0.06,
            yaw_offsets=(0.0,),
            upright=True,
            slow=1.0,
            place_slow=None,
            release=True,
            open_steps=60,
            settle=60,
        ):
            """把手中的 ``label`` 放到 ``targets`` 之一(参照 RoboTwin ``place_actor``)。

            ``targets``:一个或多个 env 坐标系的 (3,) 位置,物体的对齐点 ``point``(默认底面接触点)会落在
            这里;按顺序尝试,取第一个可达的。用调用时实测的 ee->物体 变换反推 ee 位姿,所以抓取略有滑动
            也能放准。``upright=True`` 时去掉物体当前的倾斜、只保留偏航,再叠加 ``yaw_offsets``(度)中的
            偏航作为候选。先到目标正上方 ``pre_dis`` 处,停 ``settle`` 步让物体停止晃动,再按**重新测量**的
            ee->物体 变换算下放位姿:捏着物体一处(如瓶颈)时它会绕手指转动,只用开始时测的变换会放歪。
            ``release`` 时松开夹爪。返回 ``(目标序号, 偏航偏移)``。
            """
            targets = np.asarray(targets, dtype=float).reshape(-1, 3)
            if len(targets) == 0:
                raise PlanFailError(f"{arm}: no place target for {label}")
            _, _, scale, meta = self.get_obj_asset(label)
            local_point = self._place_point(meta, point) * scale
            R_up = t3d.quaternions.quat2mat(meta["active"]["place"]["up"]["orientation"])

            def measure():
                """当前的 ee->物体 变换,以及物体的偏航(去掉倾斜后)和姿态。"""
                obj_m = _mat(self.get_obj_pose(label))
                tilted = obj_m[:3, :3] @ R_up.T  # 物体相对“正立”姿态的旋转,只保留其偏航
                yaw = np.arctan2(tilted[1, 0], tilted[0, 0])
                return np.linalg.inv(_mat(self.get_arm_pose(arm))) @ obj_m, yaw, obj_m[:3, :3]

            def object_pose(target, yaw, R_obj, off):
                """对齐点落在 ``target``、偏航加 ``off`` 度时物体的位姿矩阵。"""
                if upright:
                    R = t3d.euler.euler2mat(0.0, 0.0, yaw + np.radians(off)) @ R_up
                else:
                    R = t3d.euler.euler2mat(0.0, 0.0, np.radians(off)) @ R_obj
                m = np.eye(4)
                m[:3, :3] = R
                m[:3, 3] = target - R @ local_point
                return m

            ee_T_obj, yaw0, R_obj = measure()
            inv = np.linalg.inv(ee_T_obj)
            chains, keys = [], []
            for t_idx, target in enumerate(targets):
                for off in yaw_offsets:
                    place = object_pose(target, yaw0, R_obj, off)
                    above = place.copy()
                    above[2, 3] += pre_dis
                    chains.append([_pose(above @ inv), _pose(place @ inv)])
                    keys.append((t_idx, off))
            i, (to_above, to_place) = self.choose_chain(arm, chains)
            self.execute_plan(arm, to_above, slow)
            self.delay(settle)
            # 用重新测量的变换算下放位姿(物体在手中转偏时手腕会把它摆正);规划不到就用原来的计划
            t_idx, off = keys[i]
            ee_T_obj, _, R_obj = measure()
            place = object_pose(targets[t_idx], yaw0, R_obj, off)
            try:
                to_place = self.plan_to_pose(arm, _pose(place @ np.linalg.inv(ee_T_obj)))
            except PlanFailError:
                pass
            self.execute_plan(arm, to_place, slow if place_slow is None else place_slow)
            if release:
                self.open_gripper(arm, steps=open_steps)
            return keys[i]

        # ------------------------------------------------------ 抓取辅助

        @staticmethod
        def top_down_quat(yaw):
            """自上而下抓取的 ee 姿态:x 轴(接近方向)朝下,y 轴(手指方向)在桌面内、朝向角为 ``yaw``。"""
            x = np.array([0.0, 0.0, -1.0])
            y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            z = np.cross(x, y)
            return t3d.quaternions.mat2quat(np.stack([x, y, z], axis=1))

    return CollectEnv(config, app, **kwargs)
