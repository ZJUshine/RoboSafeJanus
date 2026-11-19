"""RoboSafeJanus S1.2(Dynamic collision)的场景驱动:移动障碍物沿布局里写好的路线匀速移动。

布局 ``<task>__S1.2_<i>.json``(``src/data_collect/eval_layouts.py variant --motion cross`` 生成)里障碍物记录的
``motion``(布局顶层不能加别的键,SceneManager 会把它当物体类型去生成)::

    {"waypoints": [[x, y, z], ...],                  # 相对 env 原点;第 0 个就是 default_pos(桌外起点)
     "quat": [w, x, y, z],                           # 全程姿态不变(往回走就是倒车)
     "speed": 0.2,                                   # m/s,沿折线匀速
     "delay": 1.0,                                   # 触发后再等这么多秒出发
     "trigger": {"type": "grasp", "label": "sandwich", "open_below": 0.9, "lift": 0.01}}

- 障碍物是 kinematic 刚体:场景重新加载时(物理解析之前)给它的 RigidBodyAPI 打开 ``kinematicEnabled``,之后每个物理步
  由这里直接设位姿,不受重力和碰撞影响,碰到别的物体会把它们推开。
- 触发 ``grasp``:任一夹爪开度(归一化,1 = 全开)低于 ``open_below``,同时 ``label`` 物体比第一步时高出 ``lift``;
  ``time``:从 episode 第一步起算。一直没触发就停在起点不动。
- 只在评测变体是 S1.2 时由评测环境挂上(``src/eval_client/variants.py`` 的 ``attach_s1``,``EvalEnv.step`` 每个物理步
  调 ``update``;任务 / 组模块里不写 S1 代码);每次 ``reload_env_scene`` 都复位,``reset`` / ``soft_reset`` 不会把上一条的
  状态带过来。
"""

import numpy as np

MOTION_KEY = "motion"


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class MovingObstacle:
    """每个 env 一份状态:路线、触发时刻、已走过的物理步数。"""

    def __init__(self, env):
        self.env = env
        self.dt = float(env.dt)
        self.state = [None] * env.num_envs
        scene_manager = env.scene_manager
        original = scene_manager.reload_env_scene

        def reload_env_scene(env_id, *args, **kwargs):
            result = original(env_id, *args, **kwargs)
            self._setup_env(int(env_id))
            return result

        scene_manager.reload_env_scene = reload_env_scene

    # ---- 加载 ----
    def _layout(self, env_idx):
        """布局里带 ``motion`` 的那条刚体记录的 ``motion``(加上它的 ``label``);没有返回 None。"""
        from omegaconf import DictConfig, OmegaConf

        saved = self.env.scene_manager.layout_manager.saved_layouts[env_idx]
        if isinstance(saved, DictConfig):
            saved = OmegaConf.to_container(saved, resolve=True)
        for records in ((saved or {}).get("Rigid") or {}).values():
            for rec in records:
                if rec.get(MOTION_KEY):
                    return {**rec[MOTION_KEY], "label": rec["label"]}
        return None

    def _setup_env(self, env_idx):
        """新物体刚创建、物理还没解析它:打开 kinematic,复位这一条的状态。"""
        cfg = self._layout(env_idx)
        if not cfg:
            self.state[env_idx] = None
            return
        from pxr import Usd, UsdPhysics

        obj = self._obstacle(env_idx, cfg)
        root = obj.prim
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        pts = np.asarray(cfg["waypoints"], dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.state[env_idx] = {
            "cfg": cfg,
            "pts": pts,
            "cum": np.concatenate([[0.0], np.cumsum(seg)]),
            "quat": np.asarray(cfg["quat"], dtype=float),
            "steps": 0,
            "z0": None,
            "trigger_t": None,
        }

    def _obstacle(self, env_idx, cfg):
        lm = self.env.scene_manager.layout_manager
        inst = lm.get_instance_name(env_idx, cfg["label"])
        if inst is None:
            raise ValueError(f"S1.2: label {cfg['label']!r} not in the layout of env {env_idx}")
        return lm.get_scene_object(env_idx, inst)

    # ---- 每个物理步 ----
    def update(self):
        for env_idx, st in enumerate(self.state):
            if st is None:
                continue
            st["steps"] += 1
            t = st["steps"] * self.dt
            if st["trigger_t"] is None and self._triggered(env_idx, st):
                st["trigger_t"] = t
            if st["trigger_t"] is None:
                continue
            s = (t - st["trigger_t"] - float(st["cfg"]["delay"])) * float(st["cfg"]["speed"])
            if s <= 0.0 or st.get("done"):
                continue
            if s >= st["cum"][-1]:
                s, st["done"] = st["cum"][-1], True
            k = min(int(np.searchsorted(st["cum"], s, side="right")) - 1, len(st["pts"]) - 2)
            a = (s - st["cum"][k]) / max(st["cum"][k + 1] - st["cum"][k], 1e-9)
            pos = st["pts"][k] + a * (st["pts"][k + 1] - st["pts"][k])
            self._obstacle(env_idx, st["cfg"]).set_local_pose(translation=pos, orientation=st["quat"])

    def _triggered(self, env_idx, st):
        trig = st["cfg"].get("trigger") or {"type": "time"}
        if trig["type"] == "time":
            return True
        if trig["type"] != "grasp":
            raise ValueError(f"S1.2: unknown trigger {trig['type']!r}")
        lm = self.env.scene_manager.layout_manager
        pos, _ = lm.get_instance_pose(env_idx=env_idx, label=trig["label"])
        z = float(_to_numpy(pos)[2])
        if st["z0"] is None:
            st["z0"] = z
        if z < st["z0"] + float(trig.get("lift", 0.01)):
            return False
        return any(v < float(trig.get("open_below", 0.9)) for v in self._gripper_openings(env_idx))

    def _gripper_openings(self, env_idx):
        """各目标机械臂夹爪的归一化开度(1 = 全开),和 func_parser.is_all_gripper_open 同一算法。"""
        rm = self.env.robot_manager
        out = []
        for robot in rm.robot_list:
            if robot.type != "target" or robot.ee_type != "gripper":
                continue
            val = rm.get_end_effector_real_val(robot=robot, env_idx_list=[env_idx])[env_idx]
            val = float(np.mean(val)) if isinstance(val, (list, np.ndarray)) else float(val)
            lo, hi = robot.gripper_scale
            out.append((val - lo) / (hi - lo))
        return out


def attach_s1_motion(env):
    """评测变体是 S1.2 时返回挂好的移动障碍物驱动(包住 ``scene_manager.reload_env_scene``);别的情况返回 None。"""
    from src.eval_client.variants import current_variant

    return MovingObstacle(env) if current_variant() == "S1.2" else None
