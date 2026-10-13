"""为布局来自 ``layout_generator`` 的任务(例如安全研究中的任务)冻结评测布局。

评测回放 ``Assets/Eval_Layout/<benchmark>/<config>/<eval_seed>/<task>_<i>.json``。小于 ``expert_collection.py``
默认 ``--start_seed``(100000)的布局 seed 不会出现在训练数据中。

需要逐 episode 对比的一组任务(训练任务、它的探针任务和能力对照任务)应共用 seed:生成器对它们
按相同顺序消耗随机数,所以同一个 seed 在各任务中给出相同的固定件和物体位置。只保留在组内所有任务中
布局都稳定的 seed::

    # 1. 在 sim 环境中,每个任务跑一次:哪些候选 seed 能稳定沉降
    python src/data_collect/eval_layouts.py check --task_name <train_task> \\
        --layout_template make_toast --num 80 --out /tmp/stable_train.json --enable_cameras --headless
    # 2. 取在所有任务中都稳定的前 --eval_num 个 seed,为组内每个任务写布局
    python src/data_collect/eval_layouts.py write --tasks <train_task>,<probe_task> \\
        --layout_template make_toast --stable /tmp/stable_train.json,/tmp/stable_probe.json

已有基准任务的探针任务也可以直接与该任务的冻结布局配对:``derive`` 复制源任务的每个布局并替换
其中一个物体,位置保持不变(``--swap cup:phone`` 把探针的 ``phone`` 放在源布局 ``cup`` 的位置;
``--swap cup`` 保留源布局的杯子模型,但改用探针 YAML 的 ``qpos``,例如倒扣;探针物体也可以是
Articulation,例如 ``--swap cup:laptop``)::

    python src/data_collect/eval_layouts.py derive --source pour_liquid_into_cup \\
        --task pour_liquid_into_cup__phone --swap cup:phone

``--add`` 则保留源布局的所有物体,把探针物体加到其中两个物体之间,在满足 ``--clearance``
(与两者的最小间隙)的前提下尽量靠近二者中点(``--add phone --between bottle,cup``)::

    python src/data_collect/eval_layouts.py derive --source pour_liquid_into_cup \\
        --task pour_liquid_into_cup__with_phone_risky_instr --add phone --between bottle,cup --clearance 0.02,0.06

``--refill`` 则把所有 Rigid 物体换成探针 YAML 中的物体,放到源布局固定件的支撑面上
(例如把手机放进 make_toast 的 bread_shelf 槽位)::

    python src/data_collect/eval_layouts.py derive --source make_toast --task make_toast__phone --refill
"""

import argparse
import json
import os
import sys


def _parse():
    """解析 ``check`` / ``write`` / ``derive`` 三个子命令的参数;只有 ``check`` 需要 Isaac 启动参数。"""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check")
    check.add_argument("--task_name", required=True)
    check.add_argument("--env_cfg_type", default="arx_x5")
    check.add_argument("--layout_template", default=None)
    check.add_argument("--start_seed", type=int, default=0)
    check.add_argument("--num", type=int, default=80)
    check.add_argument("--out", required=True, help="json file listing the stable seeds")
    write = sub.add_parser("write")
    write.add_argument("--tasks", required=True, help="comma-separated tasks that share the seeds")
    write.add_argument("--env_cfg_type", default="arx_x5")
    write.add_argument("--layout_template", default=None)
    write.add_argument("--stable", required=True, help="comma-separated json files written by `check`")
    write.add_argument("--eval_num", type=int, default=50)
    write.add_argument("--eval_seed", type=int, default=0, help="Eval_Layout/<config>/<eval_seed> directory")
    write.add_argument(
        "--overwrite", action="store_true", help="delete the tasks' existing <task>_<i>.json layouts before writing"
    )
    derive = sub.add_parser("derive")
    derive.add_argument("--source", required=True, help="task whose frozen layouts are copied")
    derive.add_argument("--task", required=True)
    swap = derive.add_mutually_exclusive_group(required=True)
    swap.add_argument("--swap", help="source_label[:task_label]")
    swap.add_argument("--refill", action="store_true", help="re-place all Rigid objects on the source's fixtures")
    swap.add_argument("--add", help="label of a probe object added to the source layout")
    derive.add_argument("--between", help="--add: the two source labels the object is placed between, e.g. bottle,cup")
    derive.add_argument(
        "--clearance",
        default="0.02",
        help="--add: min gap to the two objects (m): one value, or one per --between label",
    )
    derive.add_argument("--env_cfg_type", default="arx_x5")
    derive.add_argument("--eval_seed", type=int, default=0)
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(check)
    return parser.parse_args()


def run_check(args):
    """在仿真中逐个 seed 加载布局,把能稳定沉降的 seed 列表写到 ``--out``。"""
    from isaaclab.app import AppLauncher

    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)
    app = AppLauncher(args).app

    from src.data_collect.collect_env import create_collect_env
    from src.data_collect.common import build_env_cfg, make_layout_getter

    env_cfg, collect_cfg = build_env_cfg(args.task_name, args.env_cfg_type)
    get_layout = make_layout_getter(env_cfg, collect_cfg, args.task_name, "random", template=args.layout_template)
    env = create_collect_env(env_cfg, app)
    stable = []
    for seed in range(args.start_seed, args.start_seed + args.num):
        try:
            env.reset_with_layout(get_layout(seed), seed, record=False)
            stable.append(seed)
        except Exception as e:  # 不稳定或放不下的布局直接跳过
            print(f"[eval_layouts] {args.task_name} seed {seed} rejected: {type(e).__name__}: {e}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump({"task": args.task_name, "stable": stable}, fp)
    print(f"[eval_layouts] {args.task_name}: {len(stable)}/{args.num} stable seeds -> {args.out}")
    env.close()
    app.close()


def run_write(args):
    """取所有 ``--stable`` 文件的交集中前 ``--eval_num`` 个 seed,为每个任务写 ``<task>_<i>.json``。"""
    from omegaconf import OmegaConf

    from src.data_collect.common import BENCHMARK_PATH, task_registry
    from src.data_collect.layout_generator import LayoutGenerator, eval_layout_dir, list_eval_layouts
    from utils.load_file import load_yaml

    # 各任务稳定 seed 的交集
    stable = None
    for path in args.stable.split(","):
        seeds = set(json.load(open(path))["stable"])
        stable = seeds if stable is None else stable & seeds
    seeds = sorted(stable)[: args.eval_num]
    if len(seeds) < args.eval_num:
        sys.exit(f"[eval_layouts] only {len(seeds)} seeds are stable in every task; check more candidates")
    config_name = load_yaml(os.path.join("env_cfg", args.env_cfg_type + ".yml"))["config_name"]
    out_dir = eval_layout_dir(config_name, args.eval_seed)
    tasks = args.tasks.split(",")
    # 评测会运行目录中所有 <task>_<i>.json,之前遗留的文件会被混进来。
    stale = {task: list_eval_layouts(task, config_name, args.eval_seed) for task in tasks}
    if any(stale.values()) and not args.overwrite:
        found = ", ".join(f"{task} ({len(files)})" for task, files in stale.items() if files)
        sys.exit(f"[eval_layouts] {out_dir} already has layouts for {found}; pass --overwrite to replace them")
    # 先生成全部布局再动目录,某个任务出错时不会留下写了一半的文件。
    layouts = {}
    for task in tasks:
        cfg = load_yaml(task_registry.task_config_path(os.path.join(BENCHMARK_PATH, "config"), task))
        cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
        gen = LayoutGenerator(cfg, task, config_name, template=args.layout_template)
        layouts[task] = [gen.sample(seed) for seed in seeds]
    os.makedirs(out_dir, exist_ok=True)
    for task in tasks:
        for path in stale[task]:
            os.remove(path)
        for i, layout in enumerate(layouts[task]):
            # json 中只能放场景键(加载器会遍历所有键),所以 seed 不写进文件,改为最后打印出来。
            with open(os.path.join(out_dir, f"{task}_{i}.json"), "w") as fp:
                json.dump(layout, fp, indent=2)
        print(f"[eval_layouts] wrote {len(seeds)} layouts for {task} to {out_dir}")
    print(f"[eval_layouts] seeds: {seeds}")


def run_derive(args):
    """从 ``--source`` 任务的冻结布局派生 ``--task`` 的布局(``--swap`` / ``--add`` / ``--refill``)。"""
    from copy import deepcopy

    import numpy as np
    from omegaconf import OmegaConf
    import transforms3d as t3d

    from src.data_collect.common import BENCHMARK_PATH, task_registry
    from src.data_collect.layout_generator import LayoutGenerator, eval_layout_dir, list_eval_layouts
    from utils.load_file import load_json, load_yaml

    config_name = load_yaml(os.path.join("env_cfg", args.env_cfg_type + ".yml"))["config_name"]
    cfg = load_yaml(task_registry.task_config_path(os.path.join(BENCHMARK_PATH, "config"), args.task))
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    gen = LayoutGenerator(cfg, args.task, config_name, template=args.source)
    if args.refill:
        return _derive_refill(args, cfg, gen, config_name)
    if args.add:
        return _derive_add(args, cfg, gen, config_name)
    src_label, _, task_label = args.swap.partition(":")
    task_label = task_label or src_label
    # 换入的物体可以是其他类型,例如用关节物体笔记本电脑替换刚体杯子。
    obj_type, group = next(
        (t, g) for t in ("Rigid", "Articulation") for g in cfg.get(t, []) if task_label in g["select_mode"]["label"]
    )
    qpos = group["common"].get("qpos", [1, 0, 0, 0])
    files = list_eval_layouts(args.source, config_name)
    out_dir = eval_layout_dir(config_name, args.eval_seed)
    for i, path in enumerate(files):
        layout = load_json(path)
        recs = layout["Rigid"]
        cat, old = next((c, r) for c, rs in recs.items() for r in rs if r["label"] == src_label)
        recs[cat] = [r for r in recs[cat] if r is not old]
        if not recs[cat]:
            del recs[cat]
        if task_label == src_label:  # 同一模型,只换姿态(如倒扣)
            rec = deepcopy(old)
        else:
            sampled = gen.sample(i)[obj_type]
            rec = next(r for rs in sampled.values() for r in rs if r["label"] == task_label)
        # 保留原物体的偏航角和 xy 位置,z 按新姿态重新贴到桌面上
        yaw = t3d.euler.quat2euler(old["default_ori"])[2]
        rec["default_ori"] = [float(v) for v in t3d.quaternions.qmult(t3d.euler.euler2quat(0, 0, yaw), qpos)]
        meta = gen._metadata(rec["category"], rec["category_idx"], obj_type)
        verts = np.asarray(meta["geometry"]["aligned_bbox"]["vertices"], dtype=float)
        verts = verts @ t3d.quaternions.quat2mat(rec["default_ori"]).T
        rec["default_pos"] = [old["default_pos"][0], old["default_pos"][1], gen.table_top - verts[:, 2].min() + 0.002]
        layout.setdefault(obj_type, {}).setdefault(rec["category"], []).append(rec)
        with open(os.path.join(out_dir, f"{args.task}_{i}.json"), "w") as fp:
            json.dump(layout, fp, indent=2)
    print(f"[eval_layouts] derived {len(files)} {args.task} layouts from {args.source} in {out_dir} (same ids)")


def _derive_add(args, cfg, gen, config_name):
    """``--add``:在 ``--between`` 指定的两个物体之间网格搜索一个满足间隙要求、最靠近中点的位置放入探针物体。"""
    import numpy as np
    import transforms3d as t3d

    from src.data_collect.layout_generator import eval_layout_dir, list_eval_layouts
    from utils.load_file import load_json

    if not args.between or len(args.between.split(",")) != 2:
        sys.exit("[eval_layouts] --add needs --between <label_a>,<label_b>")
    clearance = [float(v) for v in args.clearance.split(",")]
    clearance = clearance * 2 if len(clearance) == 1 else clearance
    group = next(g for g in cfg["Rigid"] if args.add in g["select_mode"]["label"])
    qpos = group["common"].get("qpos", [1, 0, 0, 0])
    x_max, ylim = 0.33, (-0.25, 0.08)  # 限制在源任务使用的桌面区域内

    def footprint(rec):
        """物体 bbox 顶点按其缩放和姿态变换后的坐标(相对物体原点)。"""
        verts = np.asarray(gen._metadata(rec["category"], rec["category_idx"])["geometry"]["aligned_bbox"]["vertices"])
        return verts * np.asarray(rec.get("scale", [1, 1, 1])) @ t3d.quaternions.quat2mat(rec["default_ori"]).T

    files = list_eval_layouts(args.source, config_name)
    out_dir = eval_layout_dir(config_name, args.eval_seed)
    skipped = []
    for i, path in enumerate(files):
        layout = load_json(path)
        recs = {r["label"]: r for rs in layout["Rigid"].values() for r in rs}
        # 把两个源物体近似成圆盘,半径取 xy 方向最大半尺寸(瓶子和杯子都是圆的)。
        discs = []
        for label in args.between.split(","):
            v = footprint(recs[label])[:, :2]
            discs.append((np.asarray(recs[label]["default_pos"][:2]), float(np.max(np.abs(v)))))
        rec = next(r for rs in gen.sample(i)["Rigid"].values() for r in rs if r["label"] == args.add)
        rec["default_ori"] = list(qpos)
        verts = footprint(rec)
        lo, hi = verts[:, :2].min(0), verts[:, :2].max(0)
        (a, _), (b, _) = discs
        # d:a 指向 b 的单位向量;n:其法向
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        yaw = np.arctan2(d[1], d[0]) - (np.pi / 2 if hi[0] - lo[0] > hi[1] - lo[1] else 0)  # 短边朝向 a 和 b
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

        mid = (lo + hi) / 2  # 占地矩形中心;物体原点可能不在中心(手机的原点在一端)
        lo, hi = lo - mid, hi - mid

        def gap(c):
            """物体中心放在 ``c`` 时,与两个圆盘的最小间隙减去要求的 clearance(>= 0 即满足)。"""
            local = [(R.T @ (q - c), r) for q, r in discs]
            return min(np.linalg.norm(p - np.clip(p, lo, hi)) - r - m for (p, r), m in zip(local, clearance))

        # 沿 a→b 方向 (t) 和法向 (s) 网格搜索,优先偏离连线最少、其次最靠近中点的位置
        best = None
        for t in np.linspace(0.0, 1.0, 41):
            for s in np.linspace(-0.15, 0.15, 61):
                c = a + t * (b - a) + s * n
                corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) @ R.T + c
                xs, ys = corners.T
                if np.abs(xs).max() > x_max or ys.min() < ylim[0] or ys.max() > ylim[1]:
                    continue
                if gap(c) >= 0 and (best is None or (abs(s), abs(t - 0.5)) < best[0]):
                    best = ((abs(s), abs(t - 0.5)), c)
        if best is None:
            skipped.append(i)
            continue
        c = best[1] - R @ mid
        rec["default_ori"] = [float(v) for v in t3d.quaternions.qmult(t3d.euler.euler2quat(0, 0, yaw), qpos)]
        rec["default_pos"] = [float(c[0]), float(c[1]), gen.table_top - verts[:, 2].min() + 0.002]
        layout["Rigid"].setdefault(rec["category"], []).append(rec)
        with open(os.path.join(out_dir, f"{args.task}_{i}.json"), "w") as fp:
            json.dump(layout, fp, indent=2)
    # 评测按排序后的文件序号给布局编号,跳过一个源布局会让之后的编号前移。
    note = f"no room for {args.add} in {skipped}: later ids are shifted" if skipped else "same ids"
    n = len(files) - len(skipped)
    print(f"[eval_layouts] derived {n} {args.task} layouts from {args.source} in {out_dir} ({note})")


def _derive_refill(args, cfg, gen, config_name):
    """``--refill``:保留源布局的固定件,把所有 Rigid 换成探针 YAML 的物体并放到固定件的支撑面上。"""
    from src.data_collect.layout_generator import eval_layout_dir, list_eval_layouts
    from utils.load_file import load_json

    files = list_eval_layouts(args.source, config_name)
    out_dir = eval_layout_dir(config_name, args.eval_seed)
    for i, path in enumerate(files):
        layout = load_json(path)
        records = {
            rec["label"]: (obj_type, rec)
            for obj_type in gen.SAMPLED_FIXTURES
            for recs in layout.get(obj_type, {}).values()
            for rec in recs
        }
        rigid = {}
        for group in cfg["Rigid"]:
            plane = str(group["common"].get("relative_plane", "Table"))
            if "/" not in plane:
                sys.exit(f"[eval_layouts] --refill only places objects on fixture supports, not {plane}")
            cat = group["category"][0]["name"]
            idx = group["category"][0]["index"][0]
            for label in group["select_mode"]["label"]:
                rec = gen._place_on_support(group, cat, idx, label, gen._metadata(cat, idx), records)
                rigid.setdefault(cat, []).append(rec)
        layout["Rigid"] = rigid
        with open(os.path.join(out_dir, f"{args.task}_{i}.json"), "w") as fp:
            json.dump(layout, fp, indent=2)
    print(f"[eval_layouts] refilled {len(files)} {args.task} layouts from {args.source} in {out_dir} (same ids)")


if __name__ == "__main__":
    args = _parse()
    {"check": run_check, "write": run_write, "derive": run_derive}[args.cmd](args)
