"""把已采集 hdf5 中的 ``instruction`` 按任务设计文件重新分配(原地修改)。

每个任务的候选指令取自任务代码(``INSTRUCTION`` 列表或 ``gen_instruction`` 中的字符串,静态解析,与
``check_task_alignment.py`` 相同),与采集时 ``CollectEnv.instructions()`` 一致。每条 episode 的目标指令与采集时相同:
``random.Random(seed).choice(候选)``(``common.save_episode``)。seed 取自 hdf5 的 ``meta/seed``(失败尝试),
否则按 episode 编号查同目录的 ``seed.txt``;都查不到时,原指令在候选内就保留,否则用第一条候选。

只改 ``instruction`` 这一个 dataset,轨迹、图像等其他内容不动。改过的文件 mtime 和大小会变,
下次上传会重传这些 hdf5。默认只列出要改的文件,加 ``--apply`` 才写入::

    python src/data_collect/sync_instructions.py --group heat_sealed_food_microwave            # 预览
    python src/data_collect/sync_instructions.py --group heat_sealed_food_microwave --apply
    python src/data_collect/sync_instructions.py --task heat_egg_microwave --apply

需要 h5py(在 RoboDojo 环境中运行)。成功数据与 ``<task>_failed`` 下的失败尝试都会处理。
"""

import argparse
from collections import Counter
from functools import cache
import glob
import os
import random
import re
import sys

import h5py

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from src.data_collect.benchmark_layout import DEFAULT_SAVE_DIR, LEVELS, TASKS_JSON, load_tasks, task_dir  # noqa: E402

sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "internal"))
from check_task_alignment import TaskSource  # noqa: E402


def _string(text):
    """与 ``hdf5_writer._string`` 相同:纯 ASCII 用 ascii 字符串类型,否则 utf-8。"""
    try:
        text.encode("ascii")
        return h5py.string_dtype(encoding="ascii")
    except UnicodeEncodeError:
        return h5py.string_dtype(encoding="utf-8")


def task_group_names():
    """``{task: group}``,只含归属本组的任务(``same_as`` 引用不算,数据在归属组下)。"""
    return {name: group for group, levels in load_tasks().items() for level in LEVELS for name in levels.get(level, {})}


def read_instruction(f):
    value = f["instruction"][()]
    return value.decode() if isinstance(value, bytes) else str(value)


@cache
def _seed_list(config_dir):
    path = os.path.join(config_dir, "seed.txt")
    return [int(x) for x in open(path).read().split()] if os.path.exists(path) else []


def episode_seed(path, f):
    """该 episode 的布局 seed:``meta/seed``,其次同目录 ``seed.txt`` 中按 episode 编号的那一项;查不到为 None。"""
    if "meta" in f and "seed" in f["meta"]:
        return int(f["meta"]["seed"][()])
    m = re.fullmatch(r"episode_(\d+)\.hdf5", os.path.basename(path))
    seeds = _seed_list(os.path.dirname(os.path.dirname(path)))
    if m and int(m.group(1)) < len(seeds):
        return seeds[int(m.group(1))]
    return None


def target_instruction(seed, current, candidates):
    if seed is not None:
        return random.Random(seed).choice(candidates)
    return current if current in candidates else candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", default=None, help="组名,逗号分隔")
    parser.add_argument("--task", default=None, help="任务名,逗号分隔")
    parser.add_argument("--save_dir", default=os.path.join(ROOT_DIR, DEFAULT_SAVE_DIR))
    parser.add_argument("--apply", action="store_true", help="真正写入(默认只预览)")
    args = parser.parse_args()

    tasks = task_group_names()
    source = TaskSource()
    groups = {g.strip() for g in (args.group or "").split(",") if g.strip()}
    names = {t.strip() for t in (args.task or "").split(",") if t.strip()}
    unknown = (groups - set(tasks.values())) | (names - set(tasks))
    if unknown:
        parser.error(f"not in {os.path.basename(TASKS_JSON)}: {', '.join(sorted(unknown))}")
    selected = sorted(n for n, g in tasks.items() if g in groups or n in names)
    if not selected:
        parser.error("give --group and/or --task")

    total = 0
    for name in selected:
        cands = source.instructions(name)
        if not cands:
            print(f"{name}: no instructions parsed from task code, skipped")
            continue
        for suffix in ("", "_failed"):
            files = sorted(glob.glob(os.path.join(task_dir(args.save_dir, name) + suffix, "*", "data", "*.hdf5")))
            if not files:
                continue
            stale, targets, no_seed = [], Counter(), 0
            for path in files:
                with h5py.File(path, "r") as f:
                    current, seed = read_instruction(f), episode_seed(path, f)
                no_seed += seed is None
                target = target_instruction(seed, current, cands)
                targets[target] += 1
                if current != target:
                    stale.append((path, target))
            print(f"{name}{suffix}: {len(stale)}/{len(files)} files to update, {len(cands)} candidates"
                  + (f", {no_seed} without seed" if no_seed else ""))
            for text, n in targets.most_common():
                print(f"    {n:5d}  {text}")
            total += len(stale)
            if not args.apply:
                continue
            for path, target in stale:
                with h5py.File(path, "r+") as f:
                    del f["instruction"]
                    f.create_dataset("instruction", data=target, dtype=_string(target))
                with h5py.File(path, "r") as f:
                    if read_instruction(f) != target:
                        sys.exit(f"[sync_instructions] verification failed: {path}")
    action = "updated" if args.apply else "would update (dry run; add --apply)"
    print(f"[sync_instructions] {action} {total} files")


if __name__ == "__main__":
    main()
