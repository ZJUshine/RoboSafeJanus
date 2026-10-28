#!/usr/bin/env python3
"""Generate a live sync-status table for local data, ModelScope, and LeRobot exports.

Writes three files under ``data/`` by default:

    data/SYNC_STATUS.md     human-readable Markdown table
    data/SYNC_STATUS.html   self-contained dashboard (auto-refreshes the page)
    data/SYNC_STATUS.json   machine-readable snapshot

Usage::

    python scripts/internal/sync_status.py                     # one snapshot
    python scripts/internal/sync_status.py --watch 30          # refresh every 30 s
    python scripts/internal/sync_status.py --no_remote         # local + LeRobot only
    python scripts/internal/sync_status.py --repo_id user/repo
    python scripts/internal/sync_status.py --lerobot_repo_id user/lerobot-repo

The HDF5 repository defaults to ``MS_DATA_REPO_ID``; if that is unset, it is
inferred from ``data/.cache/modelscope/upload_<user>__<repo>.json``. The LeRobot
repository defaults to ``MS_LEROBOT_REPO_ID`` or ``<repo_id>-Lerobot``. ModelScope
access needs ``modelscope_hub`` (available in the ``RoboDojo`` environment).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from src.data_collect.benchmark_layout import (  # noqa: E402
    DEFAULT_SAVE_DIR,
    LEVELS,
    REMOTE_PREFIX,
    TASKS_JSON,
    task_dir,
    task_groups,
)

DEFAULT_LOCAL = ROOT_DIR / DEFAULT_SAVE_DIR
DEFAULT_LEROBOT = ROOT_DIR / "data" / "RoboSafeJanus-Benchmark-Lerobot"
DEFAULT_OUT = ROOT_DIR / "data"
LEROBOT_VERSIONS = ("v30", "v21")
LEVEL_ORDER = {"safe_tasks": 0, "risky_tasks": 1}

# 脚本专家/任务实现所在目录:三类文件齐全才算"可采集"。
CODE_ROOT = ROOT_DIR / "task" / "RoboSafeJanus"
CODE_DIRS = {
    "impl": ("tasks", ".py"),
    "config": ("config", ".yml"),
    "expert": ("experts", ".py"),
}
CODE_LABELS = {"impl": "实现", "config": "配置", "expert": "专家"}


def human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1000
    return f"{value:.2f} TB"


def signed_bytes(value: float) -> str:
    return ("+" if value >= 0 else "-") + human_bytes(abs(value))


def human_time(timestamp: float | None) -> str:
    if not timestamp:
        return "—"
    return datetime.fromtimestamp(timestamp).strftime("%m-%d %H:%M")


LOCAL_IGNORE_NAMES = {".ms_upload_cache", ".ms_upload_progress", ".DS_Store"}


def dir_size(path: Path) -> tuple[int, int, float]:
    """(file count, total bytes, newest mtime) under *path*, skipping upload cache files."""
    files = 0
    size = 0
    newest = 0.0
    if not path.is_dir():
        return files, size, newest
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            if name in LOCAL_IGNORE_NAMES:
                continue
            try:
                stat = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue
            files += 1
            size += stat.st_size
            newest = max(newest, stat.st_mtime)
    return files, size, newest


def count_episodes(task_path: Path) -> int:
    return sum(1 for _ in task_path.glob("*/data/*.hdf5"))


def infer_repo_id(local_dir: Path) -> str | None:
    repo = os.environ.get("MS_DATA_REPO_ID")
    if repo:
        return repo
    cache = local_dir.parent / ".cache" / "modelscope"
    matches = sorted(cache.glob("upload_*.json")) if cache.is_dir() else []
    if len(matches) == 1:
        return matches[0].stem[len("upload_") :].replace("__", "/")
    return None


def collect_local(local_dir: Path, json_tasks: dict[str, tuple[str, str]]) -> dict[str, dict]:
    """Scan nested task directories, including tasks no longer in the JSON."""
    result: dict[str, dict] = {}
    if not local_dir.is_dir():
        return result
    for group_dir in sorted(p for p in local_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for level_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            if level_dir.name not in LEVELS:
                continue
            for task_path in sorted(p for p in level_dir.iterdir() if p.is_dir()):
                name = task_path.name.removesuffix("_failed")
                failed = task_path.name.endswith("_failed")
                if failed:
                    continue
                files, size, newest = dir_size(task_path)
                if files == 0:
                    continue
                spec = json_tasks.get(name)
                result[name] = {
                    "name": name,
                    "group": spec[0] if spec else group_dir.name,
                    "level": spec[1] if spec else level_dir.name,
                    "group_actual": group_dir.name,
                    "level_actual": level_dir.name,
                    "in_json": spec is not None,
                    "path": str(task_path.relative_to(ROOT_DIR)),
                    "episodes": count_episodes(task_path),
                    "files": files,
                    "size": size,
                    "mtime": newest,
                }
    return result


def collect_remote(repo_id: str | None, repo_type: str, prefix: str) -> tuple[dict[str, dict], str | None]:
    if not repo_id:
        return {}, "未配置 repo_id"
    try:
        from modelscope_hub import HubApi
    except ImportError:
        return {}, "未安装 modelscope_hub"
    try:
        files = HubApi().list_repo_files(repo_id, repo_type, recursive=True)
    except Exception as error:  # noqa: BLE001 - a dashboard must survive API failures
        return {}, f"{type(error).__name__}: {error}"
    prefix_parts = prefix.strip("/").split("/") if prefix else []
    result: dict[str, dict] = {}
    for item in files:
        if getattr(item, "type", "blob") == "tree":
            continue
        parts = item.path.split("/")
        if len(parts) <= len(prefix_parts) or parts[: len(prefix_parts)] != prefix_parts:
            continue
        task = parts[len(prefix_parts)]
        entry = result.setdefault(
            task,
            {"name": task, "files": 0, "size": 0, "episodes": 0, "failed": task.endswith("_failed")},
        )
        entry["files"] += 1
        entry["size"] += getattr(item, "size", 0) or 0
        if item.path.endswith(".hdf5"):
            entry["episodes"] += 1
    for entry in result.values():
        entry["base"] = entry["name"].removesuffix("_failed")
    return result, None


def lerobot_episodes(task_path: Path) -> int | None:
    info = task_path / "meta" / "info.json"
    if info.is_file():
        try:
            data = json.loads(info.read_text(encoding="utf-8"))
            value = data.get("total_episodes")
            if isinstance(value, int):
                return value
        except (OSError, json.JSONDecodeError):
            pass
    episodes_file = task_path / "meta" / "episodes.jsonl"
    if episodes_file.is_file():
        try:
            return sum(1 for line in episodes_file.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            pass
    return None


def collect_lerobot(lerobot_dir: Path) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {version: {} for version in LEROBOT_VERSIONS}
    for version in LEROBOT_VERSIONS:
        root = lerobot_dir / version
        if not root.is_dir():
            continue
        for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for level_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                if level_dir.name not in LEVELS:
                    continue
                for task_path in sorted(p for p in level_dir.iterdir() if p.is_dir()):
                    files, size, newest = dir_size(task_path)
                    if files == 0:
                        continue
                    result[version][task_path.name] = {
                        "path": str(task_path.relative_to(ROOT_DIR)),
                        "group": group_dir.name,
                        "level": level_dir.name,
                        "episodes": lerobot_episodes(task_path),
                        "files": files,
                        "size": size,
                        "mtime": newest,
                    }
    return result


def resolve_task_name(raw: str, known: set[str]) -> str:
    """LeRobot 远端目录名可能是 put_/place_ 旧命名,优先映射到当前任务名。"""
    if raw in known:
        return raw
    for alias in code_aliases(raw):
        if alias in known:
            return alias
    return raw


def lerobot_remote_meta(api, repo_id: str, repo_type: str, directory: str, info_size: int | None,
                        cache: dict, cache_dir: Path) -> dict:
    """读取远端 ``<dataset>/meta/info.json``,按 info.json 大小缓存,避免每轮刷新都下载。"""
    key = f"{repo_id}/{directory}"
    entry = cache.get(key)
    if info_size is None:
        return entry or {}
    if entry and entry.get("info_size") == info_size:
        return entry
    version = None
    episodes = None
    try:
        path = api.download_file(
            repo_id, repo_type, f"{directory}/meta/info.json",
            cache_dir=str(cache_dir), force=True,
        )
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        codebase = str(data.get("codebase_version", ""))
        if codebase.startswith("v3"):
            version = "v30"
        elif codebase.startswith("v2"):
            version = "v21"
        if isinstance(data.get("total_episodes"), int):
            episodes = data["total_episodes"]
    except Exception:  # noqa: BLE001 - 看板要能在远端异常时继续
        pass
    entry = {"info_size": info_size, "version": version, "episodes": episodes}
    cache[key] = entry
    return entry


def collect_lerobot_remote(
    repo_id: str | None,
    repo_type: str,
    prefix: str,
    known_tasks: set[str],
    json_tasks: dict[str, tuple[str, str]],
    cache_path: Path,
) -> tuple[dict[str, dict[str, dict]], str | None]:
    """扫描 ModelScope 上的 LeRobot 数据集,支持两种布局。

    - 镜像布局:``<v30|v21>/<group>/<level>/<task>/...``(也接受 ``<v30|v21>/<task>/...``)
    - 旧版平铺布局:``<prefix>-<task>-<robot>-<action>/...``,版本看 ``meta/info.json``
    """
    empty = {version: {} for version in LEROBOT_VERSIONS}
    if not repo_id:
        return empty, "未配置 LeRobot repo_id"
    try:
        from modelscope_hub import HubApi
    except ImportError:
        return empty, "未安装 modelscope_hub"
    try:
        files = HubApi().list_repo_files(repo_id, repo_type, recursive=True)
    except Exception as error:  # noqa: BLE001 - 看板要能在远端异常时继续
        return empty, f"{type(error).__name__}: {error}"

    datasets: dict[str, dict] = {}
    info_sizes: dict[str, int] = {}
    for item in files:
        if getattr(item, "type", "blob") == "tree":
            continue
        parts = item.path.split("/")
        if len(parts) < 2:
            continue
        version = group = level = task = None
        if parts[0] in LEROBOT_VERSIONS:
            version = parts[0]
            if len(parts) >= 5 and parts[2] in LEVELS:
                group, level, task = parts[1], parts[2], parts[3]
                directory = "/".join(parts[:4])
            else:
                task = parts[1]
                directory = "/".join(parts[:2])
        else:
            match = re.match(rf"^{re.escape(prefix)}-(.+)-[^-]+-[^-]+$", parts[0])
            if not match:
                continue
            task = match.group(1)
            directory = parts[0]
        entry = datasets.setdefault(
            directory,
            {"version": version, "task": task, "group": group, "level": level, "files": 0, "size": 0},
        )
        entry["files"] += 1
        entry["size"] += getattr(item, "size", 0) or 0
        if item.path == f"{directory}/meta/info.json":
            info_sizes[directory] = getattr(item, "size", 0) or 0

    cache = load_json_cache(cache_path)
    cache_dir = cache_path.parent / "lerobot_remote"
    result: dict[str, dict[str, dict]] = {version: {} for version in LEROBOT_VERSIONS}
    api = HubApi()
    for directory, entry in datasets.items():
        version = entry["version"]
        episodes = None
        info = lerobot_remote_meta(
            api, repo_id, repo_type, directory, info_sizes.get(directory), cache, cache_dir
        )
        version = version or info.get("version")
        episodes = info.get("episodes")
        if version not in LEROBOT_VERSIONS:
            continue
        task = resolve_task_name(entry["task"], known_tasks)
        spec = json_tasks.get(task)
        group = entry["group"] or (spec[0] if spec else "—")
        level = entry["level"] or (spec[1] if spec else "—")
        result[version][task] = {
            "task": task,
            "path": directory,
            "group": group,
            "level": level,
            "episodes": episodes,
            "files": entry["files"],
            "size": entry["size"],
        }
    write_json_cache(cache_path, cache)
    return result, None


def load_json_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json_cache(path: Path, cache: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        pass


def code_catalog() -> dict[str, set[str]]:
    """``{kind: {task name}}``,分别来自 tasks/config/experts 目录。"""
    catalog: dict[str, set[str]] = {}
    for kind, (subdir, suffix) in CODE_DIRS.items():
        path = CODE_ROOT / subdir
        catalog[kind] = (
            {p.stem for p in path.glob(f"*{suffix}") if not p.name.startswith("_")}
            if path.is_dir()
            else set()
        )
    return catalog


def code_aliases(name: str) -> list[str]:
    """``put_*`` / ``place_*`` 是同一批任务的两套命名,按顺序尝试。"""
    names = [name]
    for prefix, other in (("put_", "place_"), ("place_", "put_")):
        if name.startswith(prefix):
            alias = other + name[len(prefix) :]
            if alias not in names:
                names.append(alias)
    return names


def code_readiness(name: str, catalog: dict[str, set[str]]) -> dict:
    """任务实现、配置、脚本专家是否齐全;允许 put_/place_ 别名。"""
    best = None
    for candidate in code_aliases(name):
        present = {kind: candidate in files for kind, files in catalog.items()}
        score = sum(present.values())
        if best is None or score > best[0]:
            best = (score, candidate, present)
    _, code_name, present = best
    missing = [kind for kind, ok in present.items() if not ok]
    if not missing:
        state = "ready"
    elif len(missing) == len(present):
        state = "none"
    else:
        state = "partial"
    return {"state": state, "code_name": code_name, "present": present, "missing": missing}


def code_text(code: dict, name: str | None = None) -> tuple[str, str]:
    """(简短标签, 明细),Markdown 和 HTML 共用。"""
    parts = " ".join(
        f"{CODE_LABELS[kind]}{'✅' if ok else '❌'}" for kind, ok in code["present"].items()
    )
    if code["state"] == "ready":
        label = "✅ 就绪"
        if name and code["code_name"] != name:
            parts += f" · 代码名 {code['code_name']}"
        return label, parts
    if code["state"] == "partial":
        return "⚠️ 缺" + "/".join(CODE_LABELS[kind] for kind in code["missing"]), parts
    return "❌ 无代码", parts


def local_status(item: dict | None) -> str:
    if not item:
        return "missing"
    if item["episodes"] == 0:
        return "empty"
    return "ok" if item["in_json"] else "orphan"


def remote_status(local: dict | None, remote: dict | None) -> str:
    if not local and not remote:
        return "missing"
    if not local and remote:
        return "remote_only"
    if local and not remote:
        return "missing"
    if local["episodes"] == remote["episodes"] and local["size"] == remote["size"]:
        return "synced"
    return "diff"


def lerobot_status(local: dict | None, export: dict | None) -> str:
    if not local:
        return "missing"
    if not export:
        return "missing"
    if export["episodes"] is None:
        return "unknown"
    if export["episodes"] >= local["episodes"]:
        return "synced"
    return "partial"


def lerobot_remote_status(local: dict | None, remote: dict | None) -> str:
    if not local and not remote:
        return "missing"
    if local and not remote:
        return "local_only"
    if not local and remote:
        return "remote_only"
    if local["episodes"] is None or remote["episodes"] is None:
        return "unknown"
    if local["episodes"] == remote["episodes"]:
        return "synced"
    return "diff"


def json_alignment(
    name: str,
    spec: tuple[str, str] | None,
    local: dict | None,
    remote: dict | None,
    lerobot: dict[str, dict | None],
) -> tuple[str, str]:
    """Compare a task row against ``task/robosafejanus_tasks.json``."""
    if spec is None:
        sources = []
        if local:
            sources.append(f"本地 {local['group_actual']}/{local['level_actual']}")
        if remote:
            sources.append("远端")
        if any(lerobot.values()):
            sources.append("LeRobot")
        return "extra", "、".join(sources) + " 不在 JSON"

    group, level = spec
    mismatches = []
    if local and (local["group_actual"], local["level_actual"]) != (group, level):
        mismatches.append(f"本地在 {local['group_actual']}/{local['level_actual']}")
    for version, export in lerobot.items():
        if export and (export["group"], export["level"]) != (group, level):
            mismatches.append(f"LeRobot {version} 在 {export['group']}/{export['level']}")
    if not local and not remote and not any(lerobot.values()):
        return "definition_only", "只在 JSON 中定义"
    if mismatches:
        return "mismatch", "; ".join(mismatches)
    return "aligned", ""


def status_badge(status: str) -> str:
    return {
        "ok": "✅ 已采集",
        "empty": "○ 空目录",
        "orphan": "⚠️ JSON 外",
        "missing": "❌ 无",
        "remote_only": "⬇️ 仅远端",
        "synced": "✅ 同步",
        "diff": "⚠️ 不一致",
        "partial": "⚠️ 未导全",
        "unknown": "❔ 未知",
    }.get(status, status)


def json_order(json_tasks: dict[str, tuple[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    """JSON 里的组顺序和任务顺序(采集布局和上传都以 JSON 为准)。"""
    group_order: dict[str, int] = {}
    task_order: dict[str, int] = {}
    for index, (name, (group, _level)) in enumerate(json_tasks.items()):
        group_order.setdefault(group, len(group_order))
        task_order[name] = index
    return group_order, task_order


def build_snapshot(
    local_dir: Path,
    lerobot_dir: Path,
    repo_id: str | None,
    repo_type: str,
    prefix: str,
    lerobot_repo_id: str | None = None,
) -> dict:
    task_groups.cache_clear()
    json_tasks = task_groups()
    group_order, task_order = json_order(json_tasks)
    local = collect_local(local_dir, json_tasks)
    remote, remote_error = collect_remote(repo_id, repo_type, prefix)
    lerobot = collect_lerobot(lerobot_dir)
    known_tasks = (
        set(json_tasks)
        | set(local)
        | {v["base"] for v in remote.values()}
        | set(lerobot["v30"])
        | set(lerobot["v21"])
    )
    lerobot_remote, lerobot_remote_error = (
        collect_lerobot_remote(
            lerobot_repo_id, repo_type, prefix, known_tasks, json_tasks,
            local_dir.parent / ".cache" / "lerobot_remote.json",
        )
        if lerobot_repo_id
        else ({version: {} for version in LEROBOT_VERSIONS}, None)
    )
    catalog = code_catalog()

    names = set(known_tasks)
    rows = []
    for name in names:
        local_item = local.get(name)
        spec = json_tasks.get(name)
        group = spec[0] if spec else (local_item or lerobot["v30"].get(name) or lerobot["v21"].get(name) or {}).get("group", "—")
        level = spec[1] if spec else (local_item or lerobot["v30"].get(name) or lerobot["v21"].get(name) or {}).get("level", "—")
        remote_item = remote.get(name) or remote.get(f"{name}_failed")
        lerobot_items = {version: lerobot[version].get(name) for version in LEROBOT_VERSIONS}
        lerobot_remote_items = {version: lerobot_remote[version].get(name) for version in LEROBOT_VERSIONS}
        json_state, json_detail = json_alignment(name, spec, local_item, remote_item, lerobot_items)
        statuses = {
            "local": local_status(local_item),
            "remote": remote_status(local_item, remote_item),
            "v30": lerobot_status(local_item, lerobot_items["v30"]),
            "v21": lerobot_status(local_item, lerobot_items["v21"]),
            "v30_remote": lerobot_remote_status(lerobot_items["v30"], lerobot_remote_items["v30"]),
            "v21_remote": lerobot_remote_status(lerobot_items["v21"], lerobot_remote_items["v21"]),
        }
        if statuses["local"] == "orphan":
            overall = "orphan"
        elif statuses["local"] == "missing" and statuses["remote"] == "remote_only":
            overall = "remote_only"
        elif statuses["local"] != "ok":
            overall = "not_collected"
        elif statuses["remote"] != "synced":
            overall = "to_upload"
        elif statuses["v30"] != "synced" or statuses["v21"] != "synced":
            overall = "to_export"
        elif statuses["v30_remote"] != "synced" or statuses["v21_remote"] != "synced":
            overall = "to_upload_lerobot"
        else:
            overall = "done"
        rows.append(
            {
                "name": name,
                "group": group,
                "level": level,
                "in_json": spec is not None,
                "json": {"state": json_state, "detail": json_detail},
                "code": code_readiness(name, catalog),
                "local": local_item,
                "remote": remote_item,
                "lerobot": lerobot_items,
                "lerobot_remote": lerobot_remote_items,
                "status": statuses,
                "overall": overall,
            }
        )
    # remote tasks that only exist under an old/extra name still need a row
    local_names = set(local)
    for remote_item in remote.values():
        base = remote_item["base"]
        if base not in local_names and base not in json_tasks:
            if not any(row["name"] == base for row in rows):
                rows.append(
                    {
                        "name": base,
                        "group": "—",
                        "level": "—",
                        "in_json": False,
                        "json": {"state": "extra", "detail": "远端 不在 JSON"},
                        "code": code_readiness(base, catalog),
                        "local": None,
                        "remote": remote_item,
                        "lerobot": {version: None for version in LEROBOT_VERSIONS},
                        "lerobot_remote": {
                            version: lerobot_remote[version].get(base) for version in LEROBOT_VERSIONS
                        },
                        "status": {
                            "local": "missing", "remote": "remote_only",
                            "v30": "missing", "v21": "missing",
                            "v30_remote": lerobot_remote_status(None, lerobot_remote["v30"].get(base)),
                            "v21_remote": lerobot_remote_status(None, lerobot_remote["v21"].get(base)),
                        },
                        "overall": "remote_only",
                    }
                )

    def row_order(row: dict) -> tuple:
        if row["group"] in group_order:
            return (
                0,
                group_order[row["group"]],
                LEVEL_ORDER.get(row["level"], 9),
                task_order.get(row["name"], len(task_order)),
                row["name"],
            )
        return (1, 0, 0, 0, f'{row["group"]}/{row["name"]}')

    rows.sort(key=row_order)
    summary = {
        "local_tasks": sum(1 for r in rows if (r["local"] or {}).get("episodes")),
        "local_episodes": sum((r["local"] or {}).get("episodes", 0) for r in rows),
        "local_bytes": sum((r["local"] or {}).get("size", 0) for r in rows),
        "remote_tasks": len({r["remote"]["name"] for r in rows if r["remote"]}),
        "remote_files": sum((r["remote"] or {}).get("files", 0) for r in rows),
        "remote_bytes": sum((r["remote"] or {}).get("size", 0) for r in rows),
        "lerobot": {
            version: {
                "tasks": sum(1 for r in rows if r["lerobot"][version]),
                "episodes": sum((r["lerobot"][version] or {}).get("episodes") or 0 for r in rows),
                "bytes": sum((r["lerobot"][version] or {}).get("size", 0) for r in rows),
            }
            for version in LEROBOT_VERSIONS
        },
        "lerobot_remote": {
            version: {
                "tasks": sum(1 for r in rows if r["lerobot_remote"][version]),
                "episodes": sum((r["lerobot_remote"][version] or {}).get("episodes") or 0 for r in rows),
                "bytes": sum((r["lerobot_remote"][version] or {}).get("size", 0) for r in rows),
            }
            for version in LEROBOT_VERSIONS
        },
        "states": dict(sorted((state, sum(1 for r in rows if r["overall"] == state)) for state in {
            "done", "to_upload", "to_export", "to_upload_lerobot", "not_collected", "orphan", "remote_only"
        })),
        "json": dict(sorted((state, sum(1 for r in rows if r["json"]["state"] == state)) for state in {
            "aligned", "extra", "mismatch", "definition_only"
        })),
    }
    json_code = [code_readiness(name, catalog) for name in json_tasks]
    covered = (
        set(local)
        | {item["base"] for item in remote.values()}
        | {name for version in LEROBOT_VERSIONS for name in lerobot[version]}
    )
    ready_names = set.intersection(*catalog.values())
    collectable = []
    for code_name in ready_names:
        aliases = code_aliases(code_name)
        if any(alias in covered for alias in aliases):
            continue
        collectable.append({
            "name": code_name,
            "in_json": any(alias in json_tasks for alias in aliases),
        })

    def collectable_order(item: dict) -> tuple:
        ranks = [task_order[alias] for alias in code_aliases(item["name"]) if alias in task_order]
        return (0, min(ranks), item["name"]) if ranks else (1, 0, item["name"])

    collectable.sort(key=collectable_order)
    summary["code"] = {
        "total": len(json_code),
        "ready": sum(1 for item in json_code if item["state"] == "ready"),
        "partial": sum(1 for item in json_code if item["state"] == "partial"),
        "missing": sum(1 for item in json_code if item["state"] == "none"),
        "collectable": collectable,
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "prefix": prefix,
        "remote_error": remote_error,
        "lerobot_repo_id": lerobot_repo_id,
        "lerobot_remote_error": lerobot_remote_error,
        "tasks_json": str(Path(TASKS_JSON).resolve().relative_to(ROOT_DIR)),
        "summary": summary,
        "tasks": rows,
    }


def markdown(snapshot: dict, refresh: int) -> str:
    lines = [
        "# RoboSafeJanus 数据同步状态",
        "",
        f"- 生成时间：`{snapshot['generated_at']}`",
        f"- ModelScope：`{snapshot['repo_id'] or '未配置'}`"
        + (f"（{snapshot['remote_error']}）" if snapshot["remote_error"] else ""),
        f"- LeRobot ModelScope：`{snapshot['lerobot_repo_id'] or '未配置'}`"
        + (f"（{snapshot['lerobot_remote_error']}）" if snapshot["lerobot_remote_error"] else ""),
        f"- 任务定义：`{snapshot['tasks_json']}`",
    ]
    if refresh:
        lines.append(f"- 自动刷新：每 {refresh} 秒重新生成本文件")
    summary = snapshot["summary"]
    lines += [
        "",
        "## 汇总",
        "",
        "| 数据源 | 任务数 | Episode | 文件大小 |",
        "| --- | ---: | ---: | ---: |",
        f"| HDF5 本地 | {summary['local_tasks']} | {summary['local_episodes']} | {human_bytes(summary['local_bytes'])} |",
        f"| ModelScope HDF5 | {summary['remote_tasks']} | — | {human_bytes(summary['remote_bytes'])} |",
    ]
    for version in LEROBOT_VERSIONS:
        item = summary["lerobot"][version]
        remote = summary["lerobot_remote"][version]
        lines.append(
            f"| LeRobot {version} 本地 | {item['tasks']} | {item['episodes']} | {human_bytes(item['bytes'])} |"
        )
        lines.append(
            f"| LeRobot {version} 远端 | {remote['tasks']} | {remote['episodes']} | {human_bytes(remote['bytes'])} |"
        )
    states = summary["states"]
    json_states = summary["json"]
    code_states = summary["code"]
    lines += [
        "",
        "状态统计："
        + "，".join(
            f"{label} {states[key]}"
            for key, label in (
                ("done", "全同步"),
                ("to_upload", "HDF5 待上传"),
                ("to_export", "待 LeRobot 导出"),
                ("to_upload_lerobot", "LeRobot 待上传"),
                ("not_collected", "未采集"),
                ("orphan", "JSON 外"),
                ("remote_only", "仅远端"),
            )
        ),
        "",
        "JSON 对齐："
        + "，".join(
            f"{label} {json_states[key]}"
            for key, label in (
                ("aligned", "一致"),
                ("extra", "JSON 外"),
                ("mismatch", "位置不符"),
                ("definition_only", "仅定义"),
            )
        ),
        "",
        f"专家脚本：就绪 {code_states['ready']}/{code_states['total']}"
        f"，代码不全 {code_states['partial']}，无代码 {code_states['missing']}",
        "",
        "## 可采集但未采集",
        "",
    ]
    if code_states["collectable"]:
        lines += ["| 任务 | 在 JSON | 说明 |", "| --- | --- | --- |"]
        for item in code_states["collectable"]:
            note = "可直接采集" if item["in_json"] else "不在 robosafejanus_tasks.json，需先补定义"
            lines.append(f"| `{item['name']}` | {'✅' if item['in_json'] else '⚠️ 否'} | {note} |")
    else:
        lines.append("无。")
    lines += [
        "",
        "## 任务明细",
        "",
        "| 组 | 任务 | JSON 对齐 | 专家脚本 | HDF5 本地 | ModelScope HDF5 | LeRobot v30 | LeRobot v21 | 总状态 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in snapshot["tasks"]:
        local = row["local"]
        remote = row["remote"]
        local_text = (
            f"{local['episodes']} 条 / {human_bytes(local['size'])}"
            if local and local["episodes"]
            else "—"
        )
        if not remote:
            remote_text = "❌ 无"
        elif row["status"]["remote"] == "synced":
            remote_text = f"✅ 同步 / {human_bytes(remote['size'])}"
        elif row["status"]["remote"] == "remote_only":
            remote_text = f"⬇️ 仅远端 / {human_bytes(remote['size'])}"
        else:
            delta = remote["size"] - (local["size"] if local else 0)
            remote_text = f"⚠️ {remote['episodes']} 条 / {human_bytes(remote['size'])}（{signed_bytes(delta)}）"
        lerobot_text = {}
        for version in LEROBOT_VERSIONS:
            export = row["lerobot"][version]
            remote_export = row["lerobot_remote"][version]
            status = row["status"][version]
            remote_status = row["status"][f"{version}_remote"]
            if not export:
                local_part = "本地 ❌ 无"
            else:
                episodes = export["episodes"]
                local_part = (
                    f"本地 {'✅' if status == 'synced' else '⚠️'} "
                    + (f"{episodes} 条" if episodes is not None else "未知")
                    + f" / {human_bytes(export['size'])}"
                )
            remote_label = {
                "synced": "✅ 同步",
                "diff": "⚠️ 不一致",
                "remote_only": "⬇️ 仅远端",
                "local_only": "⬆️ 待上传",
                "unknown": "❔ 未知",
            }.get(remote_status, "❌ 无")
            remote_part = f"远端 {remote_label}"
            if remote_export:
                episodes = f"{remote_export['episodes']} 条" if remote_export["episodes"] is not None else "未知"
                remote_part += f" / {episodes} / {human_bytes(remote_export['size'])}"
            lerobot_text[version] = f"{local_part}；{remote_part}"
        overall = {
            "done": "✅ 全同步",
            "to_upload": "⬆️ HDF5 待上传",
            "to_export": "🔄 待导出",
            "to_upload_lerobot": "⬆️ LeRobot 待上传",
            "not_collected": "○ 未采集",
            "orphan": "⚠️ JSON 外",
            "remote_only": "⬇️ 仅远端",
        }.get(row["overall"], row["overall"])
        json_label = {
            "aligned": "✅ 一致",
            "extra": "⚠️ JSON 外",
            "mismatch": "⚠️ 位置不符",
            "definition_only": "○ 仅定义",
        }.get(row["json"]["state"], row["json"]["state"])
        json_text = json_label + (f"（{row['json']['detail']}）" if row["json"]["detail"] else "")
        code_label, code_detail = code_text(row["code"], row["name"])
        code_cell = code_label + (f"（{code_detail}）" if code_detail else "")
        lines.append(
            f"| `{row['group']}` | `{row['name']}` | {json_text} | {code_cell} | {local_text} | {remote_text} | "
            f"{lerobot_text['v30']} | {lerobot_text['v21']} | {overall} |"
        )
    lines += [
        "",
        "> `✅ 同步`：HDF5 本地 episode 数、总大小与 ModelScope HDF5 一致。`⚠️ 不一致`：远端存在但条数或大小不同。",
        "> LeRobot 列前半是本地导出，后半是 ModelScope 上的 LeRobot 数据集：`本地 ❌ 无` 未导出、`远端 ⬆️ 待上传` 未上传、`远端 ✅ 同步` 条数一致。",
        "> `JSON 外`：本地有数据但不在 `task/robosafejanus_tasks.json`，上传脚本会跳过。",
        "> `✅ 就绪`：`task/RoboSafeJanus/{tasks,config,experts}/<task>` 三件齐全；`put_*` / `place_*` 别名视为同一任务。",
        "> LeRobot 路径：`data/RoboSafeJanus-Benchmark-Lerobot/<v30|v21>/<group>/<level>/<task>`。",
        "",
    ]
    return "\n".join(lines)


def html_page(snapshot: dict, refresh: int) -> str:
    summary = snapshot["summary"]
    rows = []
    for row in snapshot["tasks"]:
        local = row["local"]
        remote = row["remote"]
        local_text = f"{local['episodes']} 条<br><small>{human_bytes(local['size'])}</small>" if local and local["episodes"] else "—"
        if not remote:
            remote_text = '<span class="badge bad">❌ 无</span>'
        elif row["status"]["remote"] == "synced":
            remote_text = f'<span class="badge good">✅ 同步</span><br><small>{human_bytes(remote["size"])}</small>'
        elif row["status"]["remote"] == "remote_only":
            remote_text = f'<span class="badge warn">⬇️ 仅远端</span><br><small>{human_bytes(remote["size"])}</small>'
        else:
            delta = remote["size"] - (local["size"] if local else 0)
            remote_text = (
                f'<span class="badge warn">⚠️ 不一致</span><br>'
                f'<small>{remote["episodes"]} 条 / {human_bytes(remote["size"])}（{signed_bytes(delta)}）</small>'
            )
        lerobot_cells = []
        for version in LEROBOT_VERSIONS:
            export = row["lerobot"][version]
            remote_export = row["lerobot_remote"][version]
            status = row["status"][version]
            remote_status = row["status"][f"{version}_remote"]
            if not export:
                local_html = '<span class="badge bad">本地 ❌ 无</span>'
            else:
                klass = "good" if status == "synced" else "warn"
                label = "✅" if status == "synced" else "⚠️"
                episodes = f"{export['episodes']} 条" if export["episodes"] is not None else "未知"
                local_html = (
                    f'<span class="badge {klass}">本地 {label} {episodes}</span>'
                    f'<br><small>{human_bytes(export["size"])}</small>'
                )
            remote_label = {
                "synced": "远端 ✅ 同步",
                "diff": "远端 ⚠️ 不一致",
                "remote_only": "远端 ⬇️ 仅远端",
                "local_only": "远端 ⬆️ 待上传",
                "unknown": "远端 ❔ 未知",
            }.get(remote_status, "远端 ❌ 无")
            remote_class = "good" if remote_status == "synced" else ("bad" if remote_status == "missing" else "warn")
            remote_html = f'<span class="badge {remote_class}">{remote_label}</span>'
            if remote_export:
                episodes = f"{remote_export['episodes']} 条" if remote_export["episodes"] is not None else "未知"
                remote_html += f'<br><small>{episodes} / {human_bytes(remote_export["size"])}</small>'
            lerobot_cells.append(local_html + "<br>" + remote_html)
        overall_label = {
            "done": "✅ 全同步",
            "to_upload": "⬆️ HDF5 待上传",
            "to_export": "🔄 待导出",
            "to_upload_lerobot": "⬆️ LeRobot 待上传",
            "not_collected": "○ 未采集",
            "orphan": "⚠️ JSON 外",
            "remote_only": "⬇️ 仅远端",
        }.get(row["overall"], row["overall"])
        overall_class = {
            "done": "good", "to_upload": "warn", "to_export": "warn",
            "to_upload_lerobot": "warn", "orphan": "warn", "remote_only": "warn",
        }.get(row["overall"], "bad")
        json_label = {
            "aligned": "✅ 一致",
            "extra": "⚠️ JSON 外",
            "mismatch": "⚠️ 位置不符",
            "definition_only": "○ 仅定义",
        }.get(row["json"]["state"], row["json"]["state"])
        json_class = {"aligned": "good", "extra": "warn", "mismatch": "warn"}.get(row["json"]["state"], "bad")
        json_cell = f'<span class="badge {json_class}">{json_label}</span>'
        if row["json"]["detail"]:
            json_cell += f'<br><small>{html.escape(row["json"]["detail"])}</small>'
        code_label, code_detail = code_text(row["code"], row["name"])
        code_state = row["code"]["state"]
        code_class = "good" if code_state == "ready" else ("warn" if code_state == "partial" else "bad")
        code_cell = f'<span class="badge {code_class}">{html.escape(code_label)}</span>'
        if code_detail:
            code_cell += f'<br><small>{html.escape(code_detail)}</small>'
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['group'])}</code></td>"
            f"<td><code>{html.escape(row['name'])}</code></td>"
            f"<td>{json_cell}</td>"
            f"<td>{code_cell}</td>"
            f"<td>{local_text}</td><td>{remote_text}</td>"
            f"<td>{lerobot_cells[0]}</td><td>{lerobot_cells[1]}</td>"
            f'<td><span class="badge {overall_class}">{overall_label}</span></td>'
            "</tr>"
        )
    code_summary = summary["code"]
    if code_summary["collectable"]:
        collectable_cells = []
        for item in code_summary["collectable"]:
            if item["in_json"]:
                json_badge = '<span class="badge good">✅ 在 JSON</span>'
                note = "可直接采集"
            else:
                json_badge = '<span class="badge warn">⚠️ 不在 JSON</span>'
                note = "需先加入 robosafejanus_tasks.json，否则按扁平目录落盘且不会上传"
            collectable_cells.append(
                "<tr>"
                f"<td><code>{html.escape(item['name'])}</code></td>"
                f"<td>{json_badge}</td>"
                f"<td><small>{note}</small></td>"
                "</tr>"
            )
        collectable_html = (
            "<h2>可采集但未采集</h2>"
            "<table><thead><tr><th>任务</th><th>在 JSON</th><th>说明</th></tr></thead>"
            f"<tbody>{''.join(collectable_cells)}</tbody></table>"
        )
    else:
        collectable_html = '<h2>可采集但未采集</h2><p class="empty">无。</p>'
    cards = [
        ("HDF5 本地", f"{summary['local_episodes']} episodes", f"{summary['local_tasks']} tasks · {human_bytes(summary['local_bytes'])}"),
        ("ModelScope HDF5", f"{summary['remote_tasks']} tasks", f"{summary['remote_files']} files · {human_bytes(summary['remote_bytes'])}"),
    ]
    for version in LEROBOT_VERSIONS:
        item = summary["lerobot"][version]
        remote = summary["lerobot_remote"][version]
        cards.append((
            f"LeRobot {version}",
            f"{item['episodes']} episodes",
            f"本地 {item['tasks']} tasks · 远端 {remote['tasks']} tasks",
        ))
    code_summary = summary["code"]
    cards.append((
        "专家脚本",
        f"{code_summary['ready']}/{code_summary['total']} tasks",
        f"代码不全 {code_summary['partial']} · 无代码 {code_summary['missing']} · 可采未采 {len(code_summary['collectable'])}",
    ))
    cards_html = "".join(
        f'<div class="card"><div class="label">{html.escape(title)}</div><div class="value">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>'
        for title, value, sub in cards
    )
    refresh_tag = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    remote_note = f" · <span class='bad'>{html.escape(snapshot['remote_error'])}</span>" if snapshot["remote_error"] else ""
    lerobot_remote_note = (
        f" · <span class='bad'>{html.escape(snapshot['lerobot_remote_error'])}</span>"
        if snapshot["lerobot_remote_error"] else ""
    )
    states = " · ".join(
        f"{label} {summary['states'][key]}"
        for key, label in (
            ("done", "全同步"),
            ("to_upload", "HDF5 待上传"),
            ("to_export", "待导出"),
            ("to_upload_lerobot", "LeRobot 待上传"),
            ("not_collected", "未采集"),
            ("orphan", "JSON 外"),
            ("remote_only", "仅远端"),
        )
    )
    json_states_text = " · ".join(
        f"{label} {summary['json'][key]}"
        for key, label in (
            ("aligned", "JSON 一致"),
            ("extra", "JSON 外"),
            ("mismatch", "位置不符"),
            ("definition_only", "仅定义"),
        )
    )
    code_states_text = (
        f"专家脚本 就绪 {code_summary['ready']}/{code_summary['total']}"
        f" · 代码不全 {code_summary['partial']} · 无代码 {code_summary['missing']}"
    )
    refresh_script = """
<script>
(() => {
  const btn = document.getElementById('refresh-btn');
  const hint = document.getElementById('refresh-hint');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    document.querySelectorAll('meta[http-equiv="refresh"]').forEach((meta) => meta.remove());
    btn.disabled = true;
    hint.textContent = '正在刷新…';
    const reload = () => setTimeout(() => location.reload(), 300);
    try {
      const res = await fetch('api/refresh', { method: 'POST', cache: 'no-store' });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      hint.textContent = '已重新生成到 ' + data.generated_at;
    } catch (error) {
      hint.textContent = '静态页面：已重新加载（数据由 --watch 定时生成）';
    }
    reload();
  });
})();
</script>
"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
{refresh_tag}
<title>RoboSafeJanus 数据同步状态</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 24px; background: #0d1117; color: #e6edf3; }}
  h1 {{ margin: 0 0 4px; font-size: 24px; }}
  .meta {{ color: #8b949e; font-size: 13px; margin-bottom: 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 18px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }}
  .card .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
  .card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .card .sub {{ color: #8b949e; font-size: 12px; margin-top: 4px; }}
  .states {{ color: #8b949e; font-size: 13px; margin-bottom: 14px; }}
  .toolbar {{ margin: 0 0 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .toolbar button {{ background: #1f6f43; color: #d2f8e2; border: 1px solid #2ea043; border-radius: 6px; padding: 7px 12px; font-size: 13px; cursor: pointer; }}
  .toolbar button:hover {{ background: #2a8a55; }}
  .toolbar button:disabled {{ opacity: .6; cursor: progress; }}
  .toolbar .hint {{ color: #8b949e; font-size: 12px; }}
  h2 {{ font-size: 16px; margin: 22px 0 8px; }}
  .empty {{ color: #8b949e; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 9px 10px; border-bottom: 1px solid #21262d; text-align: left; font-size: 13px; vertical-align: top; }}
  th {{ background: #1c2128; color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; position: sticky; top: 0; }}
  tr:hover td {{ background: #1c2128; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  small {{ color: #8b949e; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 12px; font-weight: 500; white-space: nowrap; }}
  .good {{ background: #1f6f43; color: #d2f8e2; }}
  .warn {{ background: #7a5b12; color: #fff3c4; }}
  .bad {{ background: #7d2b2b; color: #ffdcdc; }}
</style>
</head>
<body>
<h1>RoboSafeJanus 数据同步状态</h1>
<div class="meta">生成时间 {html.escape(snapshot['generated_at'])} · ModelScope HDF5 <code>{html.escape(snapshot['repo_id'] or '未配置')}</code>{remote_note}
· LeRobot ModelScope <code>{html.escape(snapshot['lerobot_repo_id'] or '未配置')}</code>{lerobot_remote_note}
{f'· 每 {refresh} 秒刷新' if refresh else ''}</div>
<div class="toolbar">
  <button id="refresh-btn" type="button">🔄 手动刷新</button>
  <span id="refresh-hint" class="hint">用 --serve 打开时会在服务端重新生成；静态页面则重新读取当前文件</span>
</div>
<div class="cards">{cards_html}</div>
<div class="states">{states}<br>{json_states_text}<br>{code_states_text}</div>
<table>
<thead><tr><th>组</th><th>任务</th><th>JSON 对齐</th><th>专家脚本</th><th>HDF5 本地</th><th>ModelScope HDF5</th><th>LeRobot v30</th><th>LeRobot v21</th><th>总状态</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
{collectable_html}
<p class="empty">专家脚本：<code>tasks/&lt;task&gt;.py</code>、<code>config/&lt;task&gt;.yml</code>、<code>experts/&lt;task&gt;.py</code> 三件齐全为 ✅ 就绪；<code>put_*</code> / <code>place_*</code> 别名视为同一任务。</p>
{refresh_script}
</body>
</html>
"""


class StatusRequestHandler(SimpleHTTPRequestHandler):
    """静态文件 + ``POST/GET /api/refresh`` 立即重新生成快照。"""

    refresh_callback = None

    def do_GET(self):
        if self.path.rstrip("/") == "/api/refresh":
            return self._refresh()
        if self.path in ("", "/"):
            self.send_response(302)
            self.send_header("Location", "/SYNC_STATUS.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") == "/api/refresh":
            return self._refresh()
        self.send_error(404)

    def _refresh(self):
        try:
            snapshot = type(self).refresh_callback()
        except Exception as error:  # noqa: BLE001 - 把异常回传给页面
            payload = json.dumps({"error": str(error)}).encode("utf-8")
            self.send_response(500)
        else:
            payload = json.dumps({"generated_at": snapshot["generated_at"]}).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):  # noqa: A002 - 与父类签名一致
        pass


def write_outputs(snapshot: dict, out_dir: Path, refresh: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SYNC_STATUS.md").write_text(markdown(snapshot, refresh), encoding="utf-8")
    (out_dir / "SYNC_STATUS.html").write_text(html_page(snapshot, refresh), encoding="utf-8")
    (out_dir / "SYNC_STATUS.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[sync_status] {snapshot['generated_at']} -> {out_dir / 'SYNC_STATUS.md'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=str(DEFAULT_LOCAL))
    parser.add_argument("--lerobot_dir", default=str(DEFAULT_LEROBOT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--repo_id", default=None, help="ModelScope HDF5 repo; default MS_DATA_REPO_ID or upload cache")
    parser.add_argument("--lerobot_repo_id", default=None,
                        help="LeRobot ModelScope repo; default MS_LEROBOT_REPO_ID or <repo_id>-Lerobot")
    parser.add_argument("--repo_type", default="dataset", choices=("dataset", "model"))
    parser.add_argument("--prefix", default=REMOTE_PREFIX)
    parser.add_argument("--no_remote", action="store_true", help="skip ModelScope")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="regenerate every N seconds")
    parser.add_argument("--serve", type=int, default=0, metavar="PORT",
                        help="serve out_dir over HTTP; POST /api/refresh regenerates immediately")
    parser.add_argument("--host", default="127.0.0.1", help="bind address for --serve (default: 127.0.0.1)")
    args = parser.parse_args()

    local_dir = Path(args.data_dir).resolve()
    lerobot_dir = Path(args.lerobot_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    repo_id = None if args.no_remote else (args.repo_id or infer_repo_id(local_dir))
    lerobot_repo_id = None
    if not args.no_remote:
        lerobot_repo_id = (
            args.lerobot_repo_id
            or os.environ.get("MS_LEROBOT_REPO_ID")
            or (f"{repo_id}-Lerobot" if repo_id else None)
        )
    refresh = args.watch if args.watch > 0 else 0
    if refresh and refresh < 5:
        parser.error("--watch must be at least 5 seconds")

    build_lock = threading.Lock()

    def rebuild() -> dict:
        with build_lock:
            snapshot = build_snapshot(
                local_dir, lerobot_dir, repo_id, args.repo_type, args.prefix, lerobot_repo_id
            )
            write_outputs(snapshot, out_dir, refresh)
            return snapshot

    server = None
    if args.serve:
        StatusRequestHandler.refresh_callback = rebuild
        try:
            server = ThreadingHTTPServer(
                (args.host, args.serve),
                partial(StatusRequestHandler, directory=str(out_dir)),
            )
        except OSError as error:
            parser.error(f"cannot serve on {args.host}:{args.serve}: {error}")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[sync_status] serving {out_dir} at http://{args.host}:{args.serve}/SYNC_STATUS.html",
              flush=True)

    while True:
        rebuild()
        if not refresh and not server:
            return 0
        try:
            time.sleep(refresh if refresh else 3600)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
