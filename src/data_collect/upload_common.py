"""上传脚本共用:本地 → 远端映射、内容级同步计划、断点缓存维护。

远端布局为扁平的 ``<prefix>/<task>[_failed]/<config>/...``(默认 prefix ``RoboSafeJanus``),本地是
``data/RoboSafeJanus-Benchmark/<group>/<level>/<task>/...``,映射见 ``benchmark_layout.upload_files``。

同步判定以 **sha256 内容哈希** 为准(远端 ``list_repo_files`` 的 ``sha256`` 字段与本地 ``sha256sum``
一致),不再依赖 ``路径|mtime|size``:本地内容变了但文件名、大小、修改时间都没变时也会重传。
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os

from src.data_collect.benchmark_layout import task_groups, upload_files

# 本地哈希缓存:只用来省去重复读盘,不参与"要不要上传"的判定。键里带 ctime,
# 用户态工具(cp -p / rsync -a / touch -r)无法把它改回去,所以内容变了缓存一定会失效。
HASH_CACHE_VERSION = 1
HASH_CHUNK = 4 * 1024 * 1024


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


def group_tasks(groups):
    """``groups`` 中各组的自有任务(``same_as`` 引用归属别的组,不算),未知组报错退出。"""
    mapping = task_groups()
    known = {group for group, _ in mapping.values()}
    unknown = sorted(set(groups) - known)
    if unknown:
        raise SystemExit(f"unknown group(s) {unknown}; groups with tasks: {sorted(known)}")
    return sorted(task for task, (group, _) in mapping.items() if group in set(groups))


def scoped_tasks(groups=(), tasks=()):
    """``--group`` / ``--task`` 选中的自有任务(组展开成它的自有任务);未知的组 / 任务报错退出。"""
    mapping = task_groups()
    unknown = sorted(set(tasks) - set(mapping))
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}")
    return sorted(set(group_tasks(groups) if groups else []) | set(tasks))


def require_pipeline_approval(tasks, tag):
    """``--group`` / ``--task`` 真正上传前:每个任务的流水线 expert 阶段必须已批准且仍有效
    (``rsj_pipeline.py gate <group> expert --task <task> --approved``;上传本身不审核)。"""
    import subprocess
    import sys

    script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "scripts", "internal", "rsj_pipeline.py")
    mapping = task_groups()
    blocked = []
    for task in tasks:
        group = mapping[task][0]
        result = subprocess.run([sys.executable, script, "gate", group, "expert", "--task", task, "--approved"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            blocked.append(f"{group}/{task}: {(result.stdout + result.stderr).strip()}")
    if blocked:
        raise SystemExit(f"[{tag}] 流水线 expert 阶段未批准,拒绝上传(--dry_run 不受限):\n  " + "\n  ".join(blocked))


def _task_of(path):
    return path.split("/")[1] if path.count("/") >= 2 else path


# ---------------------------------------------------------------------------
# 内容哈希
# ---------------------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_key(stat_result):
    return [stat_result.st_size, stat_result.st_mtime_ns, stat_result.st_ctime_ns]


def _load_hash_cache(cache_path):
    try:
        with open(cache_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != HASH_CACHE_VERSION:
        return {}
    files = data.get("files")
    return files if isinstance(files, dict) else {}


def _save_hash_cache(cache_path, files):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = f"{cache_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"version": HASH_CACHE_VERSION, "files": files}, handle, separators=(",", ":"))
    os.replace(tmp, cache_path)


def _hash_entry(item):
    path, src = item
    try:
        before = os.stat(src)
        digest = sha256_file(src)
        after = os.stat(src)
    except OSError as error:
        return path, None, None, str(error)
    # 读的过程中文件被改写/替换:哈希不可信,不写缓存;调用方会按"不同"处理。
    stable = _stat_key(before) == _stat_key(after)
    return path, digest, (after if stable else None), None


def local_sha256_map(files, cache_path, workers=None):
    """``{远端路径: sha256}``,读盘结果按 (size, mtime_ns, ctime_ns) 缓存在 ``cache_path``。

    缓存命中时仍会 ``stat`` 一次确认文件没变,所以只是省读盘,不会把"内容改了"误判成没改。
    """
    cache = _load_hash_cache(cache_path) if cache_path else {}
    workers = workers or min(8, os.cpu_count() or 4)
    hashes = {}
    refreshed = {}
    pending = []
    errors = []
    for path, src in files.items():
        key = str(src)
        try:
            stat_result = os.stat(src)
        except OSError as error:
            errors.append(f"{path}: {error}")
            continue
        record = cache.get(key)
        if isinstance(record, dict) and record.get("stat") == _stat_key(stat_result) and record.get("sha256"):
            hashes[path] = record["sha256"]
            refreshed[key] = record
        else:
            pending.append((path, src))

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for path, digest, stat_result, error in pool.map(_hash_entry, pending):
                if error or not digest:
                    errors.append(f"{path}: {error or 'hash failed'}")
                    continue
                hashes[path] = digest
                if stat_result is not None:
                    refreshed[str(files[path])] = {"stat": _stat_key(stat_result), "sha256": digest}

    if cache_path:
        _save_hash_cache(cache_path, refreshed)
    if errors:
        # 有文件读不了时不能猜:继续跑会把它当成"远端多出的"删掉,或者静默漏传
        shown = "\n  ".join(errors[:10])
        more = f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise RuntimeError(f"{len(errors)} local file(s) could not be hashed:\n  {shown}{more}")
    return hashes


# ---------------------------------------------------------------------------
# 远端状态与同步计划
# ---------------------------------------------------------------------------
def remote_file_map(api, repo_id, repo_type, keep):
    """``{路径: FileInfo}``(含 ``size`` / ``sha256``);仓库不存在时返回空。"""
    if not api.repo_exists(repo_id, repo_type):
        return {}
    return {
        item.path: item
        for item in api.list_repo_files(repo_id, repo_type, recursive=True)
        if not item.is_dir and keep(item.path)
    }


def plan_sync(local_files, local_sizes, local_hashes, remote_map):
    """按内容比对,返回 ``({路径: 原因}, 远端多出的路径列表)``。

    原因取值:``new``(远端没有)、``size``(大小不同)、``content``(sha256 不同)、
    ``unverified``(远端没给 sha256,保守重传)。
    """
    to_upload = {}
    for path in local_files:
        remote = remote_map.get(path)
        if remote is None:
            to_upload[path] = "new"
        elif remote.size != local_sizes[path]:
            to_upload[path] = "size"
        elif remote.sha256 is None:
            to_upload[path] = "unverified"
        elif remote.sha256 != local_hashes.get(path):
            to_upload[path] = "content"
    return to_upload, sorted(set(remote_map) - set(local_files))


def print_sync_plan(tag, to_upload, local_sizes, stale, remote_map, local_files):
    """按任务汇总:新增 / 内容变化(要上传)、远端多出(要删除)、同步。"""
    per_task = defaultdict(lambda: {"new": 0, "content": 0, "bytes": 0, "delete": 0})
    for path, reason in to_upload.items():
        bucket = per_task[_task_of(path)]
        bucket["new" if reason in ("new", "unverified") else "content"] += 1
        bucket["bytes"] += local_sizes.get(path, 0)
    for path in stale:
        per_task[_task_of(path)]["delete"] += 1
    changed_tasks = {t for t, c in per_task.items() if c["new"] or c["content"]}
    local_tasks = {_task_of(p) for p in local_files}
    print(f"[{tag}] local: {len(local_tasks)} tasks, {len(local_files)} files; "
          f"remote: {len({_task_of(p) for p in remote_map})} tasks, {len(remote_map)} files")
    for task, c in sorted(per_task.items()):
        if c["new"] or c["content"]:
            print(f"  upload  {task:<42} {c['new']:>5} new, {c['content']:>5} changed "
                  f"({c['bytes'] / 1e9:.2f} GB)")
    for task, c in sorted(per_task.items()):
        if c["delete"]:
            print(f"  delete  {task:<42} {c['delete']:>5} files")
    unchanged = sorted(local_tasks - changed_tasks)
    print(f"  unchanged: {len(unchanged)} tasks" + (f" ({', '.join(unchanged)})" if unchanged else ""))
    new = sum(1 for reason in to_upload.values() if reason in ("new", "unverified"))
    content = len(to_upload) - new
    print(f"[{tag}] total: upload {new} new + {content} changed files "
          f"({sum(local_sizes.get(p, 0) for p in to_upload) / 1e9:.2f} GB), "
          f"delete {len(stale)} remote files")


def verify_sync(local_files, local_sizes, local_hashes, remote_map):
    """上传后校验:远端每个文件都必须存在、大小和 sha256 与本地一致。返回问题列表。"""
    problems = []
    for path in sorted(local_files):
        remote = remote_map.get(path)
        if remote is None:
            problems.append((path, "missing on remote"))
        elif remote.size != local_sizes[path]:
            problems.append((path, f"size {remote.size} != {local_sizes[path]}"))
        elif remote.sha256 is None:
            problems.append((path, "remote sha256 unavailable"))
        elif remote.sha256 != local_hashes.get(path):
            problems.append((path, "sha256 differs"))
    return problems


# ---------------------------------------------------------------------------
# 断点缓存
# ---------------------------------------------------------------------------
def purge_tracker(tracker_path, paths):
    """从 modelscope_hub 的断点缓存里删掉这些路径的所有条目,避免按 mtime/size 命中后跳过。

    缓存 key 是 ``路径|mtime|size``,同一路径可能有多个历史条目,全部删掉。
    """
    if not tracker_path or not os.path.exists(tracker_path):
        return 0
    try:
        with open(tracker_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return 0
    entries = data.get("files") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return 0
    targets = set(paths)
    # key 形如 ``路径|mtime|size``;路径里不含 ``|``,右边切两刀即得路径
    doomed = [key for key in entries if key.rsplit("|", 2)[0] in targets]
    for key in doomed:
        del entries[key]
    if doomed:
        tmp = f"{tracker_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, separators=(",", ":"))
        os.replace(tmp, tracker_path)
    return len(doomed)
