"""离线可行性扫描:不跑物理执行,只看脚本专家的规划在一批布局上走得通多少。

对每个 seed:按采集同样的方式采样布局、``reset_with_layout``(要用 Isaac 载入场景、沉降物体),然后

1. **漏斗**(``--funnel_grasps`` > 0,只对用 ``common._pick_place_direct`` 的专家):截获专家传给
   ``_pick_place_direct`` 的碰撞世界 / 抓取候选 / 松手位姿等参数,对前 K 个抓取候选逐段规划,记录每个候选走到哪一段:

   ``grasp``(初始 -> 抓取)-> ``release_pred``(抓取 -> 松手位姿,专家选候选用的就是这一条)-> ``prepare``(抬起 / 转成
   伸入姿态;没有准备段时直接算过)-> ``release``(准备段终点 -> 松手位姿)-> ``retreat``(沿接近方向退出)-> ``return``
   (回初始关节)。另外对每个松手位姿单独解一次带碰撞的 IK(``release_ik``),区分"松手位姿本身就到不了"和"到得了但
   路径规划不出来"。物体按刚性夹持处理:手中位姿 = 抓取时预测的 ee->物体 变换。

2. **干跑专家**(``play_once`` 原样运行):``execute_plan`` 不推进物理,只把虚拟关节角 / 末端位姿设成规划终点;
   夹爪合拢时把最近的物体挂到 TCP 上,张开时放下。专家跑完不报错就记 ``plan_ok``。这是"物理完美时专家能不能走完",
   与真实采集的差距就是物理问题(滑动、物体下荡、推动微波炉)。

输出 ``<out_dir>/scan.jsonl``(每个 seed 一行,已有的 seed 跳过,可续跑),结束时打印汇总;``--summarize`` 只读
jsonl 汇总,不启动 Isaac。用 ``scripts/feasibility_scan.sh`` 分块运行(Isaac 同一进程反复载入场景约 20 次后会卡死)。

只支持通过 ``execute_plan`` / ``move*`` / ``back_to_origin`` 运动的专家;直接调用 ``env.execute`` 发关节序列的
(``pour_common``)干跑时这些运动不会更新虚拟状态,结果不可信。
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
import re
import sys
import time

import numpy as np

STAGES = ("grasp", "release_pred", "prepare", "release", "retreat", "return")
# 干跑失败信息 -> 失败段(按顺序匹配,见 common._pick_place_direct 的 PlanFailError)
FAIL_PATTERNS = (
    (r"no .*grasp candidate|passes the filters", "no_grasp_candidate"),
    (r"no cavity geometry", "asset_geometry"),
    (r"no grasp with a reachable release", "grasp_or_release_pred"),
    (r"after lifting", "release_after_prepare"),
    (r"with the measured grasp", "release_measured"),
    (r"cannot back", "retreat"),
    (r"no way back", "return"),
)


def fail_stage(reason):
    for pat, name in FAIL_PATTERNS:
        if re.search(pat, reason):
            return name
    return "other"


# ------------------------------------------------------------------ 汇总(不需要 Isaac)


def load_records(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records):
    """打印汇总,返回汇总 dict。"""
    total = len(records)
    unstable = sum(r["status"] == "unstable" for r in records)
    errors = sum(r["status"] == "error" for r in records)
    runs = [r for r in records if r["status"] == "ok"]
    out = {"seeds": total, "unstable": unstable, "error": errors, "scanned": len(runs)}
    print(f"\n==== feasibility: {total} seeds, {len(runs)} scanned, {unstable} unstable, {errors} errors ====")
    if not runs:
        return out

    # 干跑
    dry = [r["dry"] for r in runs if r.get("dry")]
    if dry:
        ok = sum(d["ok"] for d in dry)
        stages = Counter(d["fail_stage"] for d in dry if not d["ok"])
        out["dry"] = {"ok": ok, "n": len(dry), "fail_stages": dict(stages)}
        print(f"\n[dry-run] expert plans end to end: {ok}/{len(dry)} = {ok / len(dry):.0%}")
        for name, n in stages.most_common():
            print(f"  fail at {name:<24} {n:>4}  ({n / len(dry):.0%})")
        t = [d["seconds"] for d in dry]
        print(f"  dry-run time per seed: median {np.median(t):.1f} s, max {np.max(t):.1f} s")

    # 漏斗
    fun = [r["funnel"] for r in runs if r.get("funnel")]
    if fun:
        print(f"\n[funnel] first <= {max(f['k'] for f in fun)} grasp candidates per seed, {len(fun)} seeds")
        print(f"  {'stage':<14}{'seeds with >=1 grasp':>22}{'grasps passing':>18}")
        out["funnel"] = {}
        n_grasps = sum(len(f["grasps"]) for f in fun)
        for s in STAGES:
            seeds = sum(any(g.get(s) for g in f["grasps"]) for f in fun)
            grasps = sum(bool(g.get(s)) for f in fun for g in f["grasps"])
            out["funnel"][s] = {"seeds": seeds, "grasps": grasps}
            print(f"  {s:<14}{seeds:>12}/{len(fun):<4} {seeds / len(fun):>4.0%}{grasps:>10}/{n_grasps:<5}")
        cand = [f["n_candidates"] for f in fun]
        print(
            f"  grasp candidates after filtering: median {np.median(cand):.0f}, min {min(cand)}, zero in "
            f"{sum(c == 0 for c in cand)} seeds"
        )
        # 松手位姿:几何候选数 / 带碰撞 IK 有解的个数(只统计走过 grasp 段的候选)
        gs = [g for f in fun for g in f["grasps"] if g.get("grasp")]
        if gs:
            n_rel = [g.get("n_release", 0) for g in gs]
            n_ik = [g.get("release_ik", 0) for g in gs]
            print(
                f"  release poses per grasp: median {np.median(n_rel):.0f} (0 in {sum(n == 0 for n in n_rel)}/"
                f"{len(gs)} grasps); with collision-free IK: median {np.median(n_ik):.0f} "
                f"(0 in {sum(n == 0 for n in n_ik)}/{len(gs)})"
            )
            ik_seeds = sum(any(g.get("release_ik", 0) > 0 for g in f["grasps"]) for f in fun)
            print(f"  seeds where some release pose has IK: {ik_seeds}/{len(fun)}")
        # 去掉位姿 / 关节角数字,同一类失败归到一起
        reasons = Counter(re.sub(r"\[[^\]]*\]?", "[..]", g["fail"]) for f in fun for g in f["grasps"] if g.get("fail"))
        if reasons:
            print("  most common per-grasp failures:")
            for reason, n in reasons.most_common(6):
                print(f"    {n:>5}  {reason}")

    # 位置:通过 / 不通过的 seed 里各物体的 xy 范围(世界坐标),看可行区域
    if dry:
        print("\n[layout] object xy (world) for plan_ok vs failed seeds:")
        by = defaultdict(lambda: {True: [], False: []})
        for r in runs:
            if r.get("dry"):
                for label, p in r.get("objects", {}).items():
                    by[label][r["dry"]["ok"]].append(p[:3])
        for label, groups in by.items():
            for ok_flag in (True, False):
                pts = np.asarray(groups[ok_flag])
                if len(pts):
                    lo, hi, mean = pts.min(0), pts.max(0), pts.mean(0)
                    print(
                        f"  {label:<18}{'ok' if ok_flag else 'fail':<5} n={len(pts):<4} "
                        f"x [{lo[0]:+.3f}, {hi[0]:+.3f}] mean {mean[0]:+.3f}   "
                        f"y [{lo[1]:+.3f}, {hi[1]:+.3f}] mean {mean[1]:+.3f}"
                    )
    return out


def _parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--expert", type=str, default=None, help="experts/<name>.py (default: task_name)")
    parser.add_argument("--env_cfg_type", type=str, default="arx_x5")
    parser.add_argument("--start_seed", type=int, default=100000, help="same default as expert_collection.py")
    parser.add_argument("--num_seeds", type=int, default=50)
    parser.add_argument("--max_seeds_this_run", type=int, default=0, help="stop after this many new seeds (chunking)")
    parser.add_argument("--layout_source", choices=("random", "eval"), default="random")
    parser.add_argument("--layout_template", type=str, default=None)
    parser.add_argument("--funnel_grasps", type=int, default=10, help="funnel: grasp candidates per seed (0 = off)")
    parser.add_argument("--no_dry", action="store_true", help="skip the dry run of play_once")
    parser.add_argument("--max_grasps", type=int, default=0, help="dry run: cap grasp candidates (0 = as the expert)")
    parser.add_argument("--out_dir", type=str, default=None, help="default data/debug/feasibility/<task>")
    parser.add_argument("--summarize", action="store_true", help="only summarize an existing scan.jsonl")
    return parser


if __name__ == "__main__":
    _pre, _ = _parse().parse_known_args()
    _out_dir = _pre.out_dir or os.path.join("data", "debug", "feasibility", _pre.task_name)
    if _pre.summarize:
        _recs = load_records(os.path.join(_out_dir, "scan.jsonl"))
        _summary = summarize(_recs)
        with open(os.path.join(_out_dir, "summary.json"), "w") as _f:
            json.dump(_summary, _f, indent=2)
        sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

parser = _parse()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from env.camera_manager.capture.render_sync import add_zero_delay_kit_args  # noqa: E402

add_zero_delay_kit_args(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from env.global_configs import BENCHMARK  # noqa: E402
from src.data_collect.collect_env import PlanFailError, create_collect_env  # noqa: E402
from src.data_collect.common import build_env_cfg, make_layout_getter  # noqa: E402
from utils.cluttered_generator import UnStableError  # noqa: E402

common = __import__(f"task.{BENCHMARK}.experts.common", fromlist=["common"])


# ------------------------------------------------------------------ 干跑:虚拟机器人 / 物体状态


class DryRun:
    """把采集环境的执行原语换成只更新虚拟状态的版本(规划照常用 cuRobo)。``reset_with_layout`` 后调用 ``sync``。"""

    ATTACH_TOL = 0.08  # 合拢时离 TCP 这么近的物体挂到夹爪上(同 CollectEnv.is_held)

    def __init__(self, env):
        self.env = env
        self.q, self.ee = {}, {}
        self.obj = {}  # label -> 覆盖的位姿(挂在夹爪上 / 放下后)
        self.held = {}  # arm -> (label, ee_T_obj)
        self.seen = {}  # 本 seed 查询过的物体的初始位姿
        self._real_obj = env.get_obj_pose
        self._real_plan_to_pose = env.plan_to_pose
        self._install()

    def sync(self):
        env = self.env
        self.q = {a: np.asarray(j, float).copy() for a, j in env.init_joints.items()}
        self.ee = {a: np.asarray(p, float).copy() for a, p in env.init_endpose.items()}
        self.obj, self.held, self.seen = {}, {}, {}

    # -- 查询
    def get_arm_joints(self, arm):
        return self.q[arm].copy()

    def get_arm_pose(self, arm):
        return self.ee[arm].copy()

    def get_obj_pose(self, label):
        for arm, (held, ee_T_obj) in self.held.items():
            if held == label:
                return common._pose(common._mat(self.ee[arm]) @ ee_T_obj)
        if label in self.obj:
            return self.obj[label].copy()
        pose = self._real_obj(label)
        self.seen.setdefault(label, pose.tolist())
        return pose

    # -- 规划结果带上终点末端位姿
    def plan_to_pose(self, arm, pose, start_joints=None):
        res = self._real_plan_to_pose(arm, pose, start_joints=start_joints)
        res["_ee"] = np.asarray(pose, float).copy()
        return res

    def _wrap_plan_joint(self, planner):
        real = planner.plan_joint

        def plan_joint(start, goal):
            res = real(start, goal)
            for arm, q0 in self.env.init_joints.items():
                if res.get("status") == "Success" and np.allclose(np.asarray(goal, float), q0, atol=1e-3):
                    res["_ee"] = np.asarray(self.env.init_endpose[arm], float).copy()
            return res

        planner.plan_joint = plan_joint

    # -- 执行
    def execute_plan(self, arm, res, slow=1.0):
        if "_ee" not in res:
            raise RuntimeError("dry run: plan without a known end-effector pose (joint goal other than the start)")
        self.q[arm] = np.asarray(res["position"][-1], float).copy()
        self.ee[arm] = res["_ee"].copy()

    def set_gripper(self, arm, val, steps=60):
        self.env.grip_cmd[arm] = float(val)
        if val < 0.5 and arm not in self.held:
            m = common._mat(self.ee[arm])
            tcp = m[:3, 3] + self.env.robot(arm).gripper_bias * m[:3, 0]
            best = None
            for label in self.seen:
                d = float(np.linalg.norm(self.get_obj_pose(label)[:3] - tcp))
                if d < self.ATTACH_TOL and (best is None or d < best[0]):
                    best = d, label
            if best is not None:
                label = best[1]
                self.held[arm] = label, np.linalg.inv(m) @ common._mat(self.get_obj_pose(label))
        elif val >= 0.5 and arm in self.held:
            label, _ = self.held[arm]
            pose = self.get_obj_pose(label)
            del self.held[arm]
            self.obj[label] = pose

    def back_to_origin(self, *arms):
        env = self.env
        arms = arms or tuple(r.arm_name for r in env._target_robots())
        for a in arms:
            env.plan_to_joints(a, env.init_joints[a])
            self.q[a] = np.asarray(env.init_joints[a], float).copy()
            self.ee[a] = np.asarray(env.init_endpose[a], float).copy()

    def together_move(self, left_pose, right_pose, left_arm="left_arm", right_arm="right_arm"):
        for arm, pose in ((left_arm, left_pose), (right_arm, right_pose)):
            self.execute_plan(arm, self.plan_to_pose(arm, pose))

    def _install(self):
        env = self.env
        for name in (
            "get_arm_joints",
            "get_arm_pose",
            "get_obj_pose",
            "plan_to_pose",
            "execute_plan",
            "set_gripper",
            "back_to_origin",
            "together_move",
        ):
            setattr(env, name, getattr(self, name))
        env.open_gripper = lambda arm, val=1.0, steps=60: self.set_gripper(arm, val, steps)
        env.close_gripper = lambda arm, val=0.0, steps=60: self.set_gripper(arm, val, steps)
        env.execute = lambda *seqs: None
        env.delay = lambda steps: None
        env.settle = lambda *a, **k: None
        for planner in {id(p): p for p in env.robot_manager.planner.values()}.values():
            self._wrap_plan_joint(planner)


# ------------------------------------------------------------------ 漏斗


def _end(res):
    return np.asarray(res["position"][-1], float)


def funnel(
    env,
    dry,
    k,
    label,
    *,
    set_world,
    grasps,
    open_tool,
    held_tool,
    releases,
    lift=0.0,
    prepare=None,
    retreat=0.0,
    return_fallback=False,
    **_,
):
    """``_pick_place_direct`` 的逐段可行性,参数同它(见模块说明第 1 条)。"""
    arm = common.ARM
    planner = env.robot_manager.planner[env.robot(arm).robot_name]
    root = env.robot(arm).entity_origin_pose
    q0 = np.asarray(env.init_joints[arm], float)
    home = np.asarray(env.init_endpose[arm], float)
    obj0 = common._mat(env.get_obj_pose(label))
    lifts = tuple(lift) if isinstance(lift, (tuple, list)) else (lift,)
    dists = tuple(retreat) if isinstance(retreat, (tuple, list)) else ((retreat,) if retreat > 0 else ())

    def first_release(ee, ee_T_obj, start):
        set_world(False, held_tool(ee_T_obj))
        for rel in releases(ee, ee_T_obj):
            res = common._plan_once(env, rel, start=start)
            if res is not None:
                return rel, res
        return None, None

    out = []
    for i, (g, score) in enumerate(grasps[:k]):
        rec = {"i": i, "score": float(score)}
        out.append(rec)
        gm = common._mat(g)
        ee_T_obj = np.linalg.inv(gm) @ obj0
        try:
            set_world("grasp", open_tool)
            r1 = common._plan_once(env, g, start=q0)
            if r1 is None:
                rec["fail"] = f"grasp: {common.LAST_PLAN_FAIL}"
                continue
            rec["grasp"] = True
            q1 = _end(r1)
            set_world(False, held_tool(ee_T_obj))
            rels = releases(gm, ee_T_obj)
            rec["n_release"] = len(rels)
            rec["release_ik"] = sum(
                planner.solve_ik_to_joint(q1, list(rel), real_robot_pose=deepcopy(root))["status"] == "Success"
                for rel in rels
            )
            rel_pred, r2p = first_release(gm, ee_T_obj, q1)
            rec["release_pred"] = r2p is not None

            # 准备段(抬起 / 转成伸入姿态)
            ee, qp, lifted = gm, q1, False
            if prepare is not None:
                set_world(False, held_tool(ee_T_obj))
                for target in prepare(gm):
                    res = common._plan_once(env, target, start=q1)
                    if res is not None:
                        ee, qp, lifted = common._mat(target), _end(res), True
                        break
            if not lifted:  # 准备段规划不到时同 _pick_place_direct,退回竖直抬升
                for dist in lifts:
                    if dist <= 0:
                        break
                    up = gm.copy()
                    up[2, 3] += float(dist)
                    set_world(False, held_tool(ee_T_obj))
                    res = common._plan_linear(env, common._pose(up), start=q1) or common._plan_once(
                        env, common._pose(up), start=q1
                    )
                    if res is not None:
                        ee, qp, lifted = up, _end(res), True
                        break
            rec["prepare"] = lifted or not (prepare is not None or any(d > 0 for d in lifts))

            # 松手位姿:准备过就从准备段终点规划,否则沿用预测的那条(同 _pick_place_direct)
            rel, r2 = first_release(ee, ee_T_obj, qp) if lifted else (rel_pred, r2p)
            if r2 is None:
                rec["fail"] = f"release: {common.LAST_PLAN_FAIL or 'no release pose reachable'}"
                continue
            rec["release"] = True
            q2 = _end(r2)
            rm = common._mat(rel)
            dry.obj[label] = common._pose(rm @ ee_T_obj)  # 松手后物体留在松手位置

            # 退出
            q3, tries = q2, ("return", False)
            if dists:
                rb = None
                for dist in dists:
                    back = rm.copy()
                    back[:3, 3] -= dist * rm[:3, 0]
                    for with_obj in ("return_tight", False):
                        set_world(with_obj, open_tool)
                        rb = common._plan_linear(env, common._pose(back), start=q2)
                        if rb is not None:
                            break
                    if rb is not None:
                        break
                if rb is None:
                    rec["fail"] = f"retreat: cannot back {max(dists):.2f} m"
                    continue
                q3 = _end(rb)
                tries = ("return", "return_tight") if return_fallback else ("return",)
            rec["retreat"] = True

            # 回初始
            ok = False
            for with_obj in tries:
                set_world(with_obj, open_tool)
                res = planner.plan_joint(q3, q0)
                if res.get("status") == "Success" and common._folds([res]) is None:
                    ok = True
                    break
                if common._plan_once(env, home, start=q3) is not None:
                    ok = True
                    break
            rec["return"] = ok
            if not ok:
                rec["fail"] = "return: no way back"
        except PlanFailError as e:
            rec["fail"] = f"error: {str(e)[:80]}"
        finally:
            dry.obj.pop(label, None)
    planner.set_obstacles({}, env.robot(arm).entity_origin_pose)
    planner.set_tool_spheres(None)
    return out


# ------------------------------------------------------------------ 主循环


def main():
    task_name = args_cli.task_name
    out_dir = args_cli.out_dir or os.path.join("data", "debug", "feasibility", task_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "scan.jsonl")
    # 上一个进程在某个 seed 上卡死 / 崩溃(被 .sh 强杀)时把它记为 error,免得每个新进程都卡在同一个 seed 上
    inflight = os.path.join(out_dir, "inflight.txt")
    if os.path.exists(inflight):
        seed = int(open(inflight).read())
        if seed not in {r["seed"] for r in load_records(path)}:
            with open(path, "a") as f:
                f.write(
                    json.dumps({"seed": seed, "task": task_name, "status": "error", "reason": "process died"}) + "\n"
                )
        os.remove(inflight)
    done = {r["seed"] for r in load_records(path)}
    seeds = [s for s in range(args_cli.start_seed, args_cli.start_seed + args_cli.num_seeds) if s not in done]
    if args_cli.max_seeds_this_run:
        seeds = seeds[: args_cli.max_seeds_this_run]
    print(f"[scan] task={task_name} {len(done)} seeds done, {len(seeds)} to scan -> {path}")

    module = __import__(f"task.{BENCHMARK}.experts.{args_cli.expert or task_name}", fromlist=["play_once"])
    play_once = module.play_once
    env_cfg, collect_cfg = build_env_cfg(task_name, args_cli.env_cfg_type)
    get_layout = make_layout_getter(
        env_cfg, collect_cfg, task_name, args_cli.layout_source, template=args_cli.layout_template
    )
    env = create_collect_env(env_cfg, simulation_app)
    dry = None

    # 截获 _pick_place_direct:先跑漏斗,再(可选地截断抓取候选后)交给原函数干跑
    real_ppd = common._pick_place_direct
    state = {}

    def pick_place_direct(env_, label, **kw):
        if args_cli.funnel_grasps > 0 and "funnel" not in state:
            t = time.time()
            grasps = list(kw["grasps"])
            state["funnel"] = {
                "k": args_cli.funnel_grasps,
                "label": label,
                "n_candidates": len(grasps),
                "grasps": funnel(env_, dry, args_cli.funnel_grasps, label, **{**kw, "grasps": grasps}),
                "seconds": round(time.time() - t, 1),
            }
            kw["grasps"] = grasps
        if args_cli.no_dry:
            raise _SkipDry()
        if args_cli.max_grasps:
            kw["grasps"] = list(kw["grasps"])[: args_cli.max_grasps]
        return real_ppd(env_, label, **kw)

    common._pick_place_direct = pick_place_direct

    for seed in seeds:
        rec = {"seed": seed, "task": task_name}
        state.clear()
        with open(inflight, "w") as f:
            f.write(str(seed))
        if dry is not None:  # reset 期间查询物体位姿时不能用上一个 seed 的覆盖
            dry.obj, dry.held = {}, {}
        try:
            layout = get_layout(seed)
            env.reset_with_layout(layout, seed, record=False)
            if dry is None:
                dry = DryRun(env)
            dry.sync()
            rec["status"] = "ok"
            t = time.time()
            try:
                play_once(env)
                rec["dry"] = {"ok": True}
            except _SkipDry:
                pass
            except PlanFailError as e:
                rec["dry"] = {"ok": False, "fail_stage": fail_stage(str(e)), "reason": str(e)[:300]}
            if "dry" in rec:
                rec["dry"]["seconds"] = round(time.time() - t, 1)
            rec["objects"] = {k: [round(v, 4) for v in p] for k, p in dry.seen.items()}
            if "funnel" in state:
                rec["funnel"] = state["funnel"]
        except UnStableError as e:
            rec["status"], rec["reason"] = "unstable", str(e)
        except Exception as e:  # 专家 / 场景异常:记下来,继续下一个 seed
            rec["status"], rec["reason"] = "error", f"{type(e).__name__}: {e}"[:300]
            import traceback

            traceback.print_exc()
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        os.remove(inflight)
        d = rec.get("dry", {})
        fun = rec.get("funnel", {}).get("grasps", [])
        far = max((sum(bool(g.get(s)) for s in STAGES) for g in fun), default=0)
        print(
            f"[scan] seed={seed} {rec['status']} dry={'-' if not d else ('ok' if d['ok'] else d['fail_stage'])} "
            f"funnel_best={STAGES[far - 1] if far else 'none'}"
        )

    if not args_cli.max_seeds_this_run:  # 分块运行时由 .sh 在最后统一汇总
        summary = summarize(load_records(path))
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    env.close()
    simulation_app.close()


class _SkipDry(Exception):
    pass


if __name__ == "__main__":
    main()
