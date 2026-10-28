"""RoboSafeJanus-Benchmark 数据目录布局。

``data/RoboSafeJanus-Benchmark`` 按 ``task/robosafejanus_tasks.json`` 的组与 safe / risky 细分::

    <save_dir>/<group>/<safe_tasks|risky_tasks>/<task>/<config>/

只有 ``save_dir`` 本身是 ``RoboSafeJanus-Benchmark`` 根目录、且任务在 JSON 中时才嵌套;其他目录
(``_shards/<task>/w<k>``、``data/debug/...`` 等)以及 JSON 之外的任务仍是 ``<save_dir>/<task>/``。

命令行(供 shell 脚本使用)::

    python src/data_collect/benchmark_layout.py <save_dir> <task_name>   # 打印任务目录
    python src/data_collect/benchmark_layout.py --flat_view              # 重建扁平软链接视图

扁平视图 ``data/RoboSafeJanus-Benchmark-flat/<task>`` 软链接到嵌套目录,供要求
``data/<bench>/<task>/<config>/data`` 结构的 XPolicyLab 转换 / 训练脚本使用(bench 名为
``RoboSafeJanus-Benchmark-flat``)。

远端仓库(``upload_to_modelscope.py`` / ``upload_to_hf.py``)沿用扁平布局 ``RoboSafeJanus/<task>[_failed]/...``:
``upload_files`` 给出远端路径到本地文件的映射,``build_upload_view`` 在 ``<save_dir 的父目录>/.cache/upload_view``
下按远端路径建硬链接,供 SDK 整目录上传(硬链接与原文件 mtime / size 相同,上传缓存可复用)。
"""

import argparse
from functools import lru_cache
import json
import os
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TASKS_JSON = os.path.join(ROOT_DIR, "task", "robosafejanus_tasks.json")
BENCHMARK_NAME = "RoboSafeJanus-Benchmark"
DEFAULT_SAVE_DIR = os.path.join("data", BENCHMARK_NAME)
FLAT_VIEW_DIR = os.path.join(ROOT_DIR, "data", f"{BENCHMARK_NAME}-flat")
LEVELS = ("safe_tasks", "risky_tasks")
REMOTE_PREFIX = "RoboSafeJanus"


@lru_cache(maxsize=1)
def task_groups():
    """``{task_name: (group, level)}``,来自 ``task/robosafejanus_tasks.json``。"""
    with open(TASKS_JSON) as fp:
        groups = json.load(fp)
    mapping = {}
    for group, levels in groups.items():
        for level in LEVELS:
            for task_name in levels.get(level, {}):
                if task_name in mapping:
                    raise ValueError(f"{TASKS_JSON}: task {task_name!r} appears in more than one group")
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
