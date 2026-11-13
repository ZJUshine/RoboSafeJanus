"""RoboSafeJanus-Benchmark 数据目录布局。

``data/RoboSafeJanus-Benchmark`` 按任务定义(入口 ``task/robosafejanus_tasks.json``,见 ``load_tasks``)
的组与 safe / risky 细分::

    <save_dir>/<group>/<safe_tasks|risky_tasks>/<task>/<config>/

只有 ``save_dir`` 本身是 ``RoboSafeJanus-Benchmark`` 根目录、且任务在 JSON 中时才嵌套;其他目录
(``_shards/<task>/w<k>``、``data/debug/...`` 等)以及 JSON 之外的任务仍是 ``<save_dir>/<task>/``。

命令行(供 shell 脚本使用)::

    python src/data_collect/benchmark_layout.py <save_dir> <task_name>   # 打印任务目录
    python src/data_collect/benchmark_layout.py --flat_view              # 重建扁平软链接视图

扁平视图 ``data/RoboSafeJanus-Benchmark-flat/<task>`` 软链接到嵌套目录,供要求
``data/<bench>/<task>/<config>/data`` 结构的 XPolicyLab 转换 / 训练脚本使用(bench 名为
``RoboSafeJanus-Benchmark-flat``)。

远端仓库(``upload_to_modelscope.py``)沿用扁平布局 ``RoboSafeJanus/<task>[_failed]/...``:
``upload_files`` 给出远端路径到本地文件的映射,``build_upload_view`` 在 ``<save_dir 的父目录>/.cache/upload_view``
下按远端路径建硬链接,供 SDK 整目录上传(硬链接与原文件 mtime / size 相同,上传缓存可复用)。
"""

import argparse
from functools import lru_cache
import json
import os
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# 任务定义:
#   task/robosafejanus_tasks.json  任务合集(入口):{组: [子任务, ...]},决定有哪些组 / 任务及其顺序
#   task/robosafejanus_tasks/      设计,按组各一个 <group>.json:指令、S2 题目、S1 变体(只写物体名);
#                                  S2.3 的 ground_truth 决定 safe / risky
#   task/robosafejanus_sim/        仿真细节,按组各一个 <group>.json:资产路径、setup、success,以及 S1.x 新增的
#                                  资产 / 布置 / 安全判定
# 三处的组和任务必须一一对应。
INDEX_JSON = os.path.join(ROOT_DIR, "task", "robosafejanus_tasks.json")
DESIGN_DIR = os.path.join(ROOT_DIR, "task", "robosafejanus_tasks")
SIM_DIR = os.path.join(ROOT_DIR, "task", "robosafejanus_sim")
TASKS_JSON = INDEX_JSON  # 旧名,报错信息 / 看板里指向任务定义入口
BENCHMARK_NAME = "RoboSafeJanus-Benchmark"
DEFAULT_SAVE_DIR = os.path.join("data", BENCHMARK_NAME)
FLAT_VIEW_DIR = os.path.join(ROOT_DIR, "data", f"{BENCHMARK_NAME}-flat")
LEVELS = ("safe_tasks", "risky_tasks")
REMOTE_PREFIX = "RoboSafeJanus"


def _read_group_dir(path):
    groups = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".json"):
            with open(os.path.join(path, name), encoding="utf-8") as fp:
                data = json.load(fp)
            groups[data["task_name"]] = data
    return groups


def load_index():
    """任务合集 ``task/robosafejanus_tasks.json``:``{group: [task, ...]}``(顺序即看板 / 评测中的顺序)。"""
    with open(INDEX_JSON, encoding="utf-8") as fp:
        index = json.load(fp)
    if not isinstance(index, dict):
        raise ValueError(f"{INDEX_JSON}: expected {{group: [task, ...]}}")
    for group, tasks in index.items():
        if not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
            raise ValueError(f"{INDEX_JSON}: {group!r} must map to a list of task names")
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"{INDEX_JSON}: {group!r} lists a task more than once")
    return index


def _level(design_task):
    return "risky_tasks" if design_task["S2"]["S2.3"]["ground_truth"] == "unsafe" else "safe_tasks"


def _check_shared(group, task, owner, design, sim):
    """``sim`` 里写 ``{"same_as": <owner 组>}`` 的任务:资产 / success / S1 细节取自 owner 组,数据只存在 owner 组下。

    两组的设计可以各写各的 S2(场景物体不同),但指令、S1 指令和 safe / risky 必须一致。
    """
    where = f"{group}/{task}: same_as {owner!r}"
    if owner == group or owner not in sim:
        raise ValueError(f"{where} is not another group")
    if task not in sim[owner]["tasks"] or task not in design[owner]["tasks"]:
        raise ValueError(f"{where} does not define this task")
    if "same_as" in sim[owner]["tasks"][task]:
        raise ValueError(f"{where} is itself a same_as reference")
    d, d_owner = design[group]["tasks"][task], design[owner]["tasks"][task]
    for name, get in (
        ("instruction", lambda t: t["instruction"]),
        ("S1.instruction", lambda t: t.get("S1", {}).get("instruction")),
        ("level", _level),
    ):
        if get(d) != get(d_owner):
            raise ValueError(f"{where}: {name} differs ({get(d)!r} vs {get(d_owner)!r})")


def load_tasks(include_shared=False):
    """``{group: {"safe_tasks" | "risky_tasks": {task: {"instruction", "assets", ["setup"], "success", "S1"}}}}``。

    组和任务(及顺序)来自任务合集 ``task/robosafejanus_tasks.json``,内容由 ``task/robosafejanus_tasks/<group>.json``
    (指令、safe / risky)和 ``task/robosafejanus_sim/<group>.json``(资产、setup、success、S1 细节)合成,``assets`` 为
    ``{label: {"asset": path}}``。三处的组和任务必须一一对应,否则报错。

    一个任务只归属一个组(数据目录、上传、success 写回都按归属组)。别的组要把它列进来时,在自己的 sim 文件里
    写 ``{"same_as": <归属组>}``(见 ``_check_shared``):默认不出现在本组里;``include_shared=True`` 时也列在
    本组,内容取自归属组,并多一个 ``"same_as"`` 字段(任务目录页用)。
    """
    index = load_index()
    design, sim = _read_group_dir(DESIGN_DIR), _read_group_dir(SIM_DIR)
    for where, groups in ((DESIGN_DIR, design), (SIM_DIR, sim)):
        if set(groups) != set(index):
            missing, extra = sorted(set(index) - set(groups)), sorted(set(groups) - set(index))
            raise ValueError(f"groups differ from {INDEX_JSON}: missing in {where} {missing}, not in index {extra}")
    out = {}
    for group, names in index.items():
        d_tasks, s_tasks = design[group]["tasks"], sim[group]["tasks"]
        for where, tasks in ((DESIGN_DIR, d_tasks), (SIM_DIR, s_tasks)):
            if set(tasks) != set(names):
                missing, extra = sorted(set(names) - set(tasks)), sorted(set(tasks) - set(names))
                raise ValueError(
                    f"{group}: tasks differ from {INDEX_JSON}: missing in {where} {missing}, not in index {extra}"
                )
        levels = {level: {} for level in LEVELS}
        for task in names:
            d = d_tasks[task]
            s = s_tasks[task]
            owner = s.get("same_as")
            if owner is not None:
                _check_shared(group, task, owner, design, sim)
                if not include_shared:
                    continue
                s = sim[owner]["tasks"][task]
            spec = {"instruction": d["instruction"], "assets": {k: {"asset": v} for k, v in s["assets"].items()}}
            if "setup" in s:
                spec["setup"] = s["setup"]
            spec["success"] = s["success"]
            spec["S1"] = s.get("S1", {})
            if owner is not None:
                spec["same_as"] = owner
            levels[_level(d)][task] = spec
        out[group] = levels
    return out


def save_success(task_name, success):
    """把 ``success`` 写回 ``task/robosafejanus_sim/<group>.json`` 中的该任务(其余内容不动)。

    ``same_as`` 引用不算:写回任务的归属组。
    """
    for name in sorted(os.listdir(SIM_DIR)):
        path = os.path.join(SIM_DIR, name)
        if not name.endswith(".json"):
            continue
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        if task_name in data["tasks"] and "same_as" not in data["tasks"][task_name]:
            data["tasks"][task_name]["success"] = success
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            return path
    raise KeyError(f"{task_name} not in {SIM_DIR}")


@lru_cache(maxsize=1)
def instruction_candidates():
    """设计上的指令候选 ``{task: (指令, ...)}``:设计文件里的 ``S1.instruction`` 列表,没写时只有顶层 ``instruction``。

    运行时的候选在任务代码里(``INSTRUCTION`` / ``gen_instruction``);``check_task_alignment.py`` 用这里核对两者。
    ``same_as`` 共享任务两组的 S1 指令必须一致(见 ``_check_shared``),取哪组都一样。
    """
    out = {}
    for data in _read_group_dir(DESIGN_DIR).values():
        for task, d in data["tasks"].items():
            cands = (d.get("S1") or {}).get("instruction") or [d["instruction"]]
            out[task] = (cands,) if isinstance(cands, str) else tuple(cands)
    return out


@lru_cache(maxsize=1)
def task_groups():
    """``{task_name: (group, level)}``,来自任务定义(``load_tasks``)。

    组是任务的归属组;别的组里的 ``same_as`` 引用不在这里(同一任务只有一个数据目录)。
    """
    mapping = {}
    for group, levels in load_tasks().items():
        for level in LEVELS:
            for task_name in levels.get(level, {}):
                if task_name in mapping:
                    raise ValueError(f"task {task_name!r} appears in more than one group")
                mapping[task_name] = (group, level)
    return mapping


def is_benchmark_root(save_dir):
    return os.path.basename(os.path.normpath(save_dir)) == BENCHMARK_NAME


def task_dir(save_dir, task_name):
    """任务数据目录(不含 ``<config>``)。失败尝试放在同级的 ``task_dir(...) + "_failed"``。"""
    if is_benchmark_root(save_dir) and task_name in task_groups():
        return os.path.join(save_dir, *task_groups()[task_name], task_name)
    return os.path.join(save_dir, task_name)


def build_flat_view(benchmark_dir=os.path.join(ROOT_DIR, DEFAULT_SAVE_DIR), view_dir=FLAT_VIEW_DIR):
    """为嵌套目录下已有的每个任务(含 ``<task>_failed``)建立 ``<view_dir>/<task>`` 相对软链接。"""
    os.makedirs(view_dir, exist_ok=True)
    for name in os.listdir(view_dir):
        path = os.path.join(view_dir, name)
        if os.path.islink(path):
            os.unlink(path)
    linked = []
    for task_name in task_groups():
        for name in (task_name, f"{task_name}_failed"):
            src = task_dir(benchmark_dir, task_name) + name[len(task_name):]
            if os.path.isdir(src):
                os.symlink(os.path.relpath(src, view_dir), os.path.join(view_dir, name))
                linked.append(name)
    return linked


def upload_files(benchmark_dir, prefix=REMOTE_PREFIX):
    """``{远端路径: 本地文件}``:``<group>/<level>/<task>[_failed]/...`` → ``<prefix>/<task>[_failed]/...``。

    只包含 JSON 中的任务;``_shards/`` 等其他目录不上传。
    """
    files = {}
    for task_name in task_groups():
        for name in (task_name, f"{task_name}_failed"):
            src = task_dir(benchmark_dir, task_name) + name[len(task_name):]
            for dirpath, _, filenames in os.walk(src):
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    rel = os.path.relpath(path, src).replace(os.sep, "/")
                    files[f"{prefix}/{name}/{rel}"] = path
    return files


def upload_view_dir(benchmark_dir):
    return os.path.join(os.path.dirname(os.path.abspath(benchmark_dir)), ".cache", "upload_view")


def build_upload_view(files, view_dir):
    """在 ``view_dir`` 下按远端路径为 ``files`` 建硬链接;每次先删掉旧的 ``<prefix>/`` 再重建。"""
    for top in {path.split("/", 1)[0] for path in files}:
        shutil.rmtree(os.path.join(view_dir, top), ignore_errors=True)
    for remote_path, src in files.items():
        dst = os.path.join(view_dir, *remote_path.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.link(src, dst)
    return view_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("save_dir", nargs="?")
    parser.add_argument("task_name", nargs="?")
    parser.add_argument("--flat_view", action="store_true", help="rebuild data/RoboSafeJanus-Benchmark-flat")
    args = parser.parse_args()
    if args.flat_view:
        linked = build_flat_view()
        print(f"[benchmark_layout] linked {len(linked)} task dirs under {FLAT_VIEW_DIR}")
    elif args.save_dir and args.task_name:
        print(task_dir(args.save_dir, args.task_name))
    else:
        parser.error("give <save_dir> <task_name>, or --flat_view")


if __name__ == "__main__":
    main()
