"""``upload_to_hf.py`` / ``upload_to_modelscope.py`` 共用:本地 → 远端路径映射与同步计划。

远端布局为扁平的 ``<prefix>/<task>[_failed]/<config>/...``(默认 prefix ``RoboSafeJanus``),本地是
``data/RoboSafeJanus-Benchmark/<group>/<level>/<task>/...``,映射见 ``benchmark_layout.upload_files``。
"""

from collections import defaultdict
import os

from src.data_collect.benchmark_layout import task_groups, upload_files


def local_upload_files(local_dir, prefix, keep):
    """``{远端路径: 本地文件}``,只保留 ``keep(远端路径)`` 为真的文件。"""
    return {path: src for path, src in upload_files(local_dir, prefix).items() if keep(path)}


def unmapped_dirs(local_dir):
    """``local_dir`` 下不会上传的目录:不是 JSON 中的组(``_shards`` 与隐藏目录除外)。"""
    groups = {group for group, _ in task_groups().values()}
    return sorted(
        name for name in os.listdir(local_dir)
        if os.path.isdir(os.path.join(local_dir, name)) and name not in groups
        and name != "_shards" and not name.startswith(".")
    )


def _task_of(path):
    return path.split("/")[1] if path.count("/") >= 2 else path


def print_plan(tag, local_sizes, remote_sizes):
    """按任务汇总:新增 / 大小变化(要上传)、远端多出(要删除)、不变。返回远端多出的路径(已排序)。"""
    new = sorted(set(local_sizes) - set(remote_sizes))
    changed = sorted(p for p in set(local_sizes) & set(remote_sizes) if local_sizes[p] != remote_sizes[p])
    stale = sorted(set(remote_sizes) - set(local_sizes))
    per_task = defaultdict(lambda: {"new": 0, "changed": 0, "bytes": 0, "delete": 0})
    for path in new + changed:
        per_task[_task_of(path)]["changed" if path in remote_sizes else "new"] += 1
        per_task[_task_of(path)]["bytes"] += local_sizes[path]
    for path in stale:
        per_task[_task_of(path)]["delete"] += 1
    local_tasks = {_task_of(p) for p in local_sizes}
    unchanged = sorted(local_tasks - {t for t, c in per_task.items() if c["new"] or c["changed"]})
    print(f"[{tag}] local: {len(local_tasks)} tasks, {len(local_sizes)} files; "
          f"remote: {len({_task_of(p) for p in remote_sizes})} tasks, {len(remote_sizes)} files")
    for task, c in sorted(per_task.items()):
        if c["new"] or c["changed"]:
            print(f"  upload  {task:<42} {c['new']:>5} new, {c['changed']:>5} changed ({c['bytes'] / 1e9:.2f} GB)")
    for task, c in sorted(per_task.items()):
        if c["delete"]:
            print(f"  delete  {task:<42} {c['delete']:>5} files")
    print(f"  unchanged: {len(unchanged)} tasks" + (f" ({', '.join(unchanged)})" if unchanged else ""))
    print(f"[{tag}] total: upload {len(new)} new + {len(changed)} changed files "
          f"({sum(local_sizes[p] for p in new + changed) / 1e9:.2f} GB), delete {len(stale)} remote files")
    return stale
