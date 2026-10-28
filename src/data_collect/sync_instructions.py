"""把已采集 hdf5 中的 ``instruction`` 改成 ``task/robosafejanus_tasks.json`` 中的指令(原地修改)。

只改 ``instruction`` 这一个 dataset,轨迹、图像等其他内容不动。改过的文件 mtime 和大小会变,
下次上传会重传这些 hdf5。默认只列出要改的文件,加 ``--apply`` 才写入::

    python src/data_collect/sync_instructions.py --group heat_X_microwave            # 预览
    python src/data_collect/sync_instructions.py --group heat_X_microwave --apply
    python src/data_collect/sync_instructions.py --task heat_egg_microwave --apply

需要 h5py(在 RoboDojo 环境中运行)。成功数据与 ``<task>_failed`` 下的失败尝试都会处理。
"""

import argparse
import glob
import json
import os
import sys

import h5py

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from src.data_collect.benchmark_layout import DEFAULT_SAVE_DIR, LEVELS, TASKS_JSON, task_dir  # noqa: E402


def _string(text):
    """与 ``hdf5_writer._string`` 相同:纯 ASCII 用 ascii 字符串类型,否则 utf-8。"""
    try:
        text.encode("ascii")
        return h5py.string_dtype(encoding="ascii")
    except UnicodeEncodeError:
        return h5py.string_dtype(encoding="utf-8")


def json_instructions():
    with open(TASKS_JSON, encoding="utf-8") as fp:
        groups = json.load(fp)
    return {
        name: (group, spec["instruction"])
        for group, levels in groups.items()
        for level in LEVELS
        for name, spec in levels.get(level, {}).items()
    }


def read_instruction(f):
    value = f["instruction"][()]
    return value.decode() if isinstance(value, bytes) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", default=None, help="组名,逗号分隔")
    parser.add_argument("--task", default=None, help="任务名,逗号分隔")
    parser.add_argument("--save_dir", default=os.path.join(ROOT_DIR, DEFAULT_SAVE_DIR))
    parser.add_argument("--apply", action="store_true", help="真正写入(默认只预览)")
    args = parser.parse_args()

    tasks = json_instructions()
    groups = {g.strip() for g in (args.group or "").split(",") if g.strip()}
    names = {t.strip() for t in (args.task or "").split(",") if t.strip()}
    unknown = (groups - {g for g, _ in tasks.values()}) | (names - set(tasks))
    if unknown:
        parser.error(f"not in {os.path.basename(TASKS_JSON)}: {', '.join(sorted(unknown))}")
    selected = sorted(n for n, (g, _) in tasks.items() if g in groups or n in names)
    if not selected:
        parser.error("give --group and/or --task")

    total = 0
    for name in selected:
        instruction = tasks[name][1]
        for suffix in ("", "_failed"):
            files = sorted(glob.glob(os.path.join(task_dir(args.save_dir, name) + suffix, "*", "data", "*.hdf5")))
            stale = []
            for path in files:
                with h5py.File(path, "r") as f:
                    if read_instruction(f) != instruction:
                        stale.append((path, read_instruction(f)))
            if not files:
                continue
            olds = sorted({old for _, old in stale})
            print(f"{name}{suffix}: {len(stale)}/{len(files)} files to update -> {instruction!r}"
                  + (f" (was {olds})" if olds else ""))
            total += len(stale)
            if not args.apply:
                continue
            for path, _ in stale:
                with h5py.File(path, "r+") as f:
                    del f["instruction"]
                    f.create_dataset("instruction", data=instruction, dtype=_string(instruction))
                with h5py.File(path, "r") as f:
                    if read_instruction(f) != instruction:
                        sys.exit(f"[sync_instructions] verification failed: {path}")
    action = "updated" if args.apply else "would update (dry run; add --apply)"
    print(f"[sync_instructions] {action} {total} files")


if __name__ == "__main__":
    main()
