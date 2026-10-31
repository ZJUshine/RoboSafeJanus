#!/usr/bin/env python3
"""Generate a live sync-status table for local data, ModelScope, and LeRobot exports.

Writes three files under ``data/`` by default:

    data/SYNC_STATUS.md     human-readable Markdown table
    data/SYNC_STATUS.html   self-contained dashboard (auto-refreshes the page)
    data/MODELS_EVAL.html   models + evaluation page, linked from the dashboard
    data/SYNC_STATUS.json   machine-readable snapshot

Usage::

    python scripts/internal/sync_status.py                     # one snapshot
    python scripts/internal/sync_status.py --watch 30          # refresh every 30 s
    python scripts/internal/sync_status.py --no_remote         # local + LeRobot only
    python scripts/internal/sync_status.py --repo_id user/repo
    python scripts/internal/sync_status.py --lerobot_repo_id user/lerobot-repo
    python scripts/internal/sync_status.py --model_repo_id user/model-repo

The HDF5 repository defaults to ``MS_DATA_REPO_ID``; if that is unset, it is
inferred from ``data/.cache/modelscope/upload_<user>__<repo>.json``. The LeRobot
repository defaults to ``MS_LEROBOT_REPO_ID`` or ``<repo_id>-Lerobot``, and the
trained-model repository defaults to ``MS_MODEL_REPO_ID`` or
``<owner>/RoboSafeJanus-Model``. ModelScope access needs ``modelscope_hub``
(available in the ``RoboDojo`` environment).
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from functools import partial
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, quote, urlparse

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from src.data_collect.benchmark_layout import (  # noqa: E402
    DEFAULT_SAVE_DIR,
    LEVELS,
    REMOTE_PREFIX,
    TASKS_JSON,
    task_groups,
)

DEFAULT_LOCAL = ROOT_DIR / DEFAULT_SAVE_DIR
DEFAULT_LEROBOT = ROOT_DIR / "data" / "RoboSafeJanus-Benchmark-Lerobot"
DEFAULT_MODEL = ROOT_DIR / "data" / "RoboSafeJanus-Model"
DEFAULT_EVAL = ROOT_DIR / "eval_result"
DEFAULT_LAYOUT = ROOT_DIR / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5"
DEFAULT_OUT = ROOT_DIR / "data"
LEROBOT_VERSIONS = ("v30", "v21")
CAM_ORDER = ("cam_head", "cam_left_wrist", "cam_right_wrist")
VIDEO_RE = re.compile(r"^episode_(\d+)_(cam_[a-z0-9_]+?)(?:_(success|fail))?\.mp4$")
THUMB_DIR = "video_thumbs"
MEDIA_SUFFIXES = {".mp4", ".webm", ".m4v", ".jpg", ".jpeg", ".png"}
LAYOUT_RE = re.compile(r"^(.+?)_(\d+)\.json$")


def resolve_under(base: Path, rel: str) -> Path | None:
    """把相对路径解析到 ``base`` 内；越界或非法时返回 None。"""
    if not rel:
        return None
    try:
        base = base.resolve()
        target = (base / rel).resolve()
    except (OSError, ValueError):
        return None
    if target != base and base not in target.parents:
        return None
    return target


def inside(root: Path, target: Path | None) -> bool:
    if target is None:
        return False
    root = root.resolve()
    return target == root or root in target.parents


def cam_rank(cam: str) -> tuple[int, str]:
    return (CAM_ORDER.index(cam) if cam in CAM_ORDER else len(CAM_ORDER), cam)


def video_path(file: Path) -> str:
    """仓库内相对路径；不在仓库内时退回绝对路径（下载/播放仍可用）。"""
    try:
        return str(file.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(file)


def rel_path(path: Path) -> str:
    """相对仓库根的展示路径；不解析软链接（Assets 下很多是链接）。"""
    absolute = Path(path).absolute()
    try:
        return str(absolute.relative_to(ROOT_DIR))
    except ValueError:
        return str(absolute)


def scan_preview_videos(task_path: Path) -> dict:
    """统计 ``<task>/<config>/preview_video/episode_*_<cam>.mp4``。"""
    cams: Counter[str] = Counter()
    episodes: dict[int, dict[str, dict]] = {}
    total_bytes = 0
    try:
        video_dirs = sorted(task_path.glob("*/preview_video"))
    except OSError:
        video_dirs = []
    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue
        try:
            files = sorted(video_dir.iterdir())
        except OSError:
            continue
        for file in files:
            match = VIDEO_RE.match(file.name)
            if not match:
                continue
            try:
                size = file.stat().st_size
            except OSError:
                continue
            episode, cam = int(match.group(1)), match.group(2)
            cams[cam] += 1
            total_bytes += size
            episodes.setdefault(episode, {})[cam] = {
                "path": video_path(file), "size": size,
            }
    sample = None
    for episode in sorted(episodes):
        if "cam_head" in episodes[episode]:
            sample = {"episode": episode, "cam": "cam_head", **episodes[episode]["cam_head"]}
            break
    return {
        "files": sum(cams.values()),
        "bytes": total_bytes,
        "episodes": len(episodes),
        "cams": {cam: cams[cam] for cam in sorted(cams, key=cam_rank)},
        "sample": sample,
    }


def list_video_files(directories: list[Path]) -> dict:
    """把若干目录里的 ``episode_*_<cam>[_success|_fail].mp4`` 展平成条目列表。"""
    entries: list[dict] = []
    used: set[str] = set()
    for directory in directories:
        try:
            files = sorted(directory.iterdir())
        except OSError:
            continue
        for file in files:
            match = VIDEO_RE.match(file.name)
            if not match or not file.is_file():
                continue
            try:
                size = file.stat().st_size
            except OSError:
                size = 0
            cam = match.group(2)
            used.add(cam)
            entries.append({
                "episode": int(match.group(1)),
                "cam": cam,
                "result": match.group(3) or "",
                "size": size,
                "variant": directory.parent.name,
                "path": video_path(file),
            })
    entries.sort(key=lambda item: (item["episode"], cam_rank(item["cam"]), item["variant"]))
    cams = [cam for cam in CAM_ORDER if cam in used]
    cams += sorted(used - set(CAM_ORDER), key=cam_rank)
    return {"entries": entries, "cameras": cams, "total": len(entries)}
LEVEL_ORDER = {"safe_tasks": 0, "risky_tasks": 1}

# 脚本专家/任务实现所在目录:三类文件齐全才算"可采集"。
CODE_ROOT = ROOT_DIR / "task" / "RoboSafeJanus"
CODE_DIRS = {
    "impl": ("tasks", ".py"),
    "config": ("config", ".yml"),
    "expert": ("experts", ".py"),
}
CODE_LABELS = {"impl": "实现", "config": "配置", "expert": "专家"}

JOB_KINDS = ("export_v30", "export_v21", "upload_hdf5", "upload_lerobot", "download_model")
EXPORT_JOBS = {"v30": "export_v30", "v21": "export_v21"}
JOB_LABELS = {
    "export_v30": "LeRobot v3.0 导出",
    "export_v21": "LeRobot v2.1 导出",
    "upload_hdf5": "HDF5 上传",
    "upload_lerobot": "LeRobot 上传",
    "download_model": "模型下载",
}
JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict] = {
    kind: {
        "kind": kind,
        "state": "idle",
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "message": "",
        "log": None,
    }
    for kind in JOB_KINDS
}


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
    # LeRobot 上传缓存也在同一目录,文件名以 _lerobot 结尾,不能当作 HDF5 仓库
    matches = (
        sorted(batch for batch in cache.glob("upload_*.json") if not batch.stem.endswith("_lerobot"))
        if cache.is_dir()
        else []
    )
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
                    "videos": scan_preview_videos(task_path),
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
                    info = task_path / "meta" / "info.json"
                    result[version][task_path.name] = {
                        "path": str(task_path.relative_to(ROOT_DIR)),
                        "group": group_dir.name,
                        "level": level_dir.name,
                        "episodes": lerobot_episodes(task_path),
                        "files": files,
                        "size": size,
                        "mtime": newest,
                        # 重新导出才会重写 info.json;--postprocess_only 只改 meta/episodes
                        "info_mtime": info.stat().st_mtime if info.is_file() else None,
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
        layout = "legacy"
        if parts[0] in LEROBOT_VERSIONS:
            version = parts[0]
            layout = "mirror"
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
            {"version": version, "task": task, "group": group, "level": level,
             "layout": layout, "files": 0, "size": 0},
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
        candidate = {
            "task": task,
            "path": directory,
            "group": group,
            "level": level,
            "layout": entry["layout"],
            "episodes": episodes,
            "files": entry["files"],
            "size": entry["size"],
        }
        existing = result[version].get(task)
        # 镜像布局优先;旧平铺布局只在没有镜像数据时使用
        if existing and existing.get("layout") == "mirror" and candidate["layout"] == "legacy":
            continue
        result[version][task] = candidate
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


def resolve_model_target(key: str, json_tasks: dict[str, tuple[str, str]]) -> tuple:
    """把模型目录名映射到任务或 ``<group>_<level>`` 组模型。"""
    if key in json_tasks:
        group, level = json_tasks[key]
        return key, "task", group, level
    for alias in code_aliases(key):
        if alias in json_tasks:
            group, level = json_tasks[alias]
            return alias, "task", group, level
    for level in LEVELS:
        suffix = f"_{level}"
        if key.endswith(suffix):
            group = key[: -len(suffix)]
            if any(g == group and lv == level for g, lv in json_tasks.values()):
                return None, "group", group, level
    return None, "unknown", None, None


def collect_models(
    repo_id: str | None,
    json_tasks: dict[str, tuple[str, str]],
    local_root: Path | None = None,
) -> tuple[list[dict], str | None]:
    """扫描 ModelScope model 仓库。

    约定布局:``<policy>/<task|group_level>/<run>/<step>/...``。也接受更浅的
    ``<policy>/<task>/<step>/...``。
    """
    if not repo_id:
        return [], "未配置模型仓库"
    try:
        from modelscope_hub import HubApi
    except ImportError:
        return [], "未安装 modelscope_hub"
    try:
        files = HubApi().list_repo_files(repo_id, "model", recursive=True)
    except Exception as error:  # noqa: BLE001 - 看板要能在远端异常时继续
        return [], f"{type(error).__name__}: {error}"

    entries: dict[str, dict] = {}
    for item in files:
        if getattr(item, "type", "blob") == "tree":
            continue
        parts = item.path.split("/")
        if len(parts) < 3:
            continue
        policy, key = parts[0], parts[1]
        directory = f"{policy}/{key}"
        entry = entries.setdefault(
            directory,
            {"policy": policy, "key": key, "path": directory,
             "runs": set(), "steps": set(), "files": 0, "size": 0},
        )
        entry["files"] += 1
        entry["size"] += getattr(item, "size", 0) or 0
        for part in parts[2:]:
            if part.isdigit():
                entry["steps"].add(part)
                break
        if len(parts) >= 4 and not parts[2].isdigit():
            entry["runs"].add(parts[2])

    models = []
    level_counts = Counter(json_tasks.values())
    for entry in entries.values():
        task, kind, group, level = resolve_model_target(entry["key"], json_tasks)
        alias_task = None
        if kind == "task" and level_counts.get((group, level)) == 1:
            # 小组只有一个任务时，任务模型就是该小组的组模型，归到小组名下
            alias_task = task
            kind = "group"
        if alias_task or kind == "group":
            target = f"{group}/{level}"
        elif task:
            target = task
        else:
            target = "—"
        local_steps: set[str] = set()
        local_size = 0
        if local_root is not None:
            for run in sorted(entry["runs"]):
                run_dir = local_root / entry["policy"] / entry["key"] / run
                if not run_dir.is_dir():
                    continue
                for step_dir in sorted(run_dir.iterdir()):
                    if step_dir.is_dir() and step_dir.name.isdigit():
                        local_steps.add(step_dir.name)
                        local_size += dir_size(step_dir)[1]
        steps = sorted(entry["steps"], key=int)
        latest_step = steps[-1] if steps else None
        models.append({
            "policy": entry["policy"],
            "key": entry["key"],
            "path": entry["path"],
            "runs": sorted(entry["runs"]),
            "steps": steps,
            "latest_step": latest_step,
            "files": entry["files"],
            "size": entry["size"],
            "target": target,
            "kind": kind,
            "task": task,
            "alias_task": alias_task,
            "group": group,
            "level": level,
            "local_steps": sorted(local_steps, key=int),
            "local_size": local_size,
            "downloaded": bool(local_steps),
            "needs_download": bool(latest_step) and latest_step not in local_steps,
        })
    models.sort(key=lambda model: (model["policy"], model["key"]))
    return models, None


def ckpt_model_key(ckpt: str) -> str:
    """``RoboSafeJanus-<task>-arx_x5-joint-0`` → ``<task>``。"""
    match = re.match(r"^RoboSafeJanus-(.+)-[^-]+-[^-]+-0$", ckpt)
    return match.group(1) if match else ckpt


def collect_eval_results(eval_dir: Path, json_tasks: dict[str, tuple[str, str]]) -> list[dict]:
    """扫描 ``eval_result/RoboDojo/<task>/<policy>/<config>/<seed>_ckpt_name=...,action_type=.../<run>/_result.json``。"""
    root = (eval_dir / "RoboDojo").resolve()
    results: list[dict] = []
    if not root.is_dir():
        return results
    for path in sorted(root.glob("*/*/*/*/*/_result.json")):
        parts = path.relative_to(root).parts
        if len(parts) < 6:
            continue
        task, policy, config, seed_dir = parts[0], parts[1], parts[2], parts[3]
        match = re.match(r"^(\d+)_ckpt_name=(.+),action_type=([^,]+)$", seed_dir)
        if not match:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mtime = path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            continue
        success = data.get("success_rate")
        if not isinstance(success, (int, float)):
            continue
        eval_time = data.get("eval_time")
        results.append({
            "task": task,
            "policy": policy,
            "config": config,
            "seed": int(match.group(1)),
            "ckpt": match.group(2),
            "action_type": match.group(3),
            "run_id": path.parent.name,
            "success_rate": float(success),
            "eval_time": eval_time if isinstance(eval_time, int) else None,
            "score": data.get("score"),
            "mtime": mtime,
            "path": str(path.relative_to(ROOT_DIR)),
            "level": json_tasks.get(task, (None, None))[1],
        })

    def rank(item: dict) -> tuple:
        return (item["eval_time"] or 0, item["mtime"])

    best: dict[tuple[str, str], dict] = {}
    for item in results:
        key = (item["task"], item["ckpt"])
        if key not in best or rank(item) > rank(best[key]):
            best[key] = item
    for item in results:
        item["selected"] = best[(item["task"], item["ckpt"])] is item
    return results


def collect_eval_layouts(
    layout_dir: Path,
    seed: int,
    json_tasks: dict[str, tuple[str, str]],
) -> tuple[dict[str, dict], dict[str, dict[str, int]]]:
    """统计 ``Assets/Eval_Layout/<benchmark>/<config>/<seed>/<task>_<i>.json``。

    返回 ``(每个任务的布局信息, {seed: {task: 条数}})``。
    """
    by_seed: dict[str, dict[str, int]] = {}
    if layout_dir.is_dir():
        for seed_dir in sorted(layout_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            counts: Counter[str] = Counter()
            try:
                files = list(seed_dir.iterdir())
            except OSError:
                files = []
            for file in files:
                match = LAYOUT_RE.match(file.name)
                if match and file.is_file():
                    counts[match.group(1)] += 1
            by_seed[seed_dir.name] = dict(counts)
    current = by_seed.get(str(seed), {})
    layouts: dict[str, dict] = {}
    for name in sorted(set(json_tasks) | set(current)):
        layouts[name] = {
            "name": name,
            "count": current.get(name, 0),
            "in_json": name in json_tasks,
            "group": json_tasks.get(name, (None, None))[0],
            "level": json_tasks.get(name, (None, None))[1],
            "seeds": {key: values.get(name, 0) for key, values in by_seed.items()},
        }
    return layouts, by_seed


def eval_model_name(result: dict) -> str:
    """评测用的模型名：``<policy>/<模型目录>``（不再是"自训"/"组模型"）。"""
    return f"{result.get('policy') or '?'}/{ckpt_model_key(result['ckpt'])}"


def eval_label(result: dict, row: dict, json_tasks: dict[str, tuple[str, str]]) -> str:
    return eval_model_name(result)


def eval_kind(result: dict, row: dict, json_tasks: dict[str, tuple[str, str]]) -> str:
    """模型和当前任务的关系，用作悬停提示。"""
    key = ckpt_model_key(result["ckpt"])
    if key == row["name"]:
        return "自训模型"
    target, kind, group, level = resolve_model_target(key, json_tasks)
    if kind == "group" and group == row["group"] and level == row["level"]:
        return "组模型"
    if target:
        return f"其他任务模型（{target}）"
    return "未匹配模型"


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


def lerobot_status(local: dict | None, export: dict | None, source: dict | None = None) -> str:
    """source 是派生来源(v2.1 的来源是同任务的 v3.0):导出比来源旧时为 stale。"""
    if not local:
        return "missing"
    if not export:
        return "missing"
    if export["episodes"] is None:
        return "unknown"
    if export["episodes"] < local["episodes"]:
        return "partial"
    if source and export["info_mtime"] and source["info_mtime"] and export["info_mtime"] < source["info_mtime"]:
        return "stale"
    return "synced"


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
        "stale": "⚠️ 早于 v3.0",
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


def job_tail(job: dict, limit: int = 16000, lines: int = 80) -> str:
    path = job.get("log")
    if not path:
        return ""
    try:
        with open(path, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            size = fp.tell()
            fp.seek(max(0, size - limit))
            text = fp.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def jobs_snapshot() -> dict:
    with JOBS_LOCK:
        return {kind: {**job, "log_tail": job_tail(job)} for kind, job in JOBS.items()}


def launch_job(kind: str, runner, out_dir: Path, on_done=None) -> None:
    with JOBS_LOCK:
        job = JOBS[kind]
        if job["state"] == "running":
            raise RuntimeError(f"{JOB_LABELS[kind]} 正在运行")
        log_path = out_dir / ".cache" / "jobs" / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}.log"
        job.update({
            "state": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "returncode": None,
            "message": "运行中",
            "log": str(log_path),
        })
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def run() -> None:
        code = 1
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                code = runner(log)
        except Exception as error:  # noqa: BLE001 - 异常写进任务日志
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n[error] {type(error).__name__}: {error}\n")
        finally:
            with JOBS_LOCK:
                job.update({
                    "state": "done" if code == 0 else "failed",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "returncode": code,
                    "message": "完成" if code == 0 else f"失败（exit {code}）",
                })
            if on_done:
                try:
                    on_done()
                except Exception:  # noqa: BLE001 - 刷新失败不影响任务结果
                    pass

    threading.Thread(target=run, daemon=True).start()


def shell_runner(command: list[str], cwd: Path):
    def run(log):
        process = subprocess.Popen(
            command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                log.write(line)
                log.flush()
        return process.wait()

    return run


def export_runner(tasks: list[str], cwd: Path, version: str):
    def run(log):
        failed = []
        for index, task in enumerate(tasks, 1):
            log.write(f"\n===== [{index}/{len(tasks)}] {version} {task} =====\n")
            log.flush()
            code = subprocess.call(
                ["bash", "scripts/export_lerobot.sh", task, "--format", version],
                cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
            )
            if code != 0:
                failed.append(task)
                log.write(f"[export] {task} failed: exit {code}\n")
            log.flush()
        if failed:
            log.write(f"\n[export] failed tasks: {' '.join(failed)}\n")
            return 1
        return 0

    return run


def download_models_runner(models: list[dict], cwd: Path):
    def run(log):
        failed = []
        for index, model in enumerate(models, 1):
            label = f"{model['policy']}/{model['key']}"
            log.write(f"\n===== [{index}/{len(models)}] {label} =====\n")
            log.flush()
            command = [
                sys.executable, "-u", "scripts/download_models_from_modelscope.py",
                "--policy", model["policy"], "--task", model["key"], "--step", "latest",
            ]
            if model["runs"]:
                command += ["--run", *model["runs"]]
            code = subprocess.call(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
            if code != 0:
                failed.append(label)
                log.write(f"[download_model] {label} failed: exit {code}\n")
            log.flush()
        if failed:
            log.write(f"\n[download_model] failed: {' '.join(failed)}\n")
            return 1
        return 0

    return run


def run_dry(command: list[str], cwd: Path, timeout: int = 900) -> tuple[int, str]:
    try:
        result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "dry-run 超时"
    lines = [
        line for line in (result.stdout + result.stderr).splitlines()
        if "RequestsDependencyWarning" not in line and not line.strip().startswith("warnings.warn")
    ]
    return result.returncode, "\n".join(lines)


def pending_export_tasks(local_dir: Path, lerobot_dir: Path, version: str) -> tuple[list[str], list[str]]:
    """返回 (待导出任务, 被阻塞的任务)。

    v3.0 从 HDF5 导出;v2.1 从同任务的 v3.0 派生,所以 v3.0 未完成的任务算阻塞,
    v2.1 早于 v3.0 的任务要重新派生。
    """
    task_groups.cache_clear()
    json_tasks = task_groups()
    _, task_order = json_order(json_tasks)
    local = collect_local(local_dir, json_tasks)
    lerobot = collect_lerobot(lerobot_dir)
    tasks, blocked = [], []
    for name, item in local.items():
        if name not in json_tasks or not item.get("episodes"):
            continue
        v30 = lerobot["v30"].get(name)
        if version == "v30":
            if lerobot_status(item, v30) != "synced":
                tasks.append(name)
        elif lerobot_status(item, lerobot["v21"].get(name), v30) != "synced":
            (tasks if lerobot_status(item, v30) == "synced" else blocked).append(name)

    def order(name: str) -> tuple:
        return task_order.get(name, 10**9), name

    return sorted(tasks, key=order), sorted(blocked, key=order)


def build_snapshot(
    local_dir: Path,
    lerobot_dir: Path,
    repo_id: str | None,
    repo_type: str,
    prefix: str,
    lerobot_repo_id: str | None = None,
    model_repo_id: str | None = None,
    model_local_dir: Path | None = None,
    eval_dir: Path | None = None,
    layout_dir: Path | None = None,
    layout_seed: int = 0,
    eval_num: int = 50,
) -> dict:
    task_groups.cache_clear()
    json_tasks = task_groups()
    group_order, task_order = json_order(json_tasks)
    local = collect_local(local_dir, json_tasks)
    remote, remote_error = collect_remote(repo_id, repo_type, prefix)
    lerobot = collect_lerobot(lerobot_dir)
    models, model_error = collect_models(
        model_repo_id, json_tasks, model_local_dir or DEFAULT_MODEL
    )
    eval_results = collect_eval_results(eval_dir or DEFAULT_EVAL, json_tasks)
    eval_layouts, layouts_by_seed = collect_eval_layouts(
        layout_dir or DEFAULT_LAYOUT, layout_seed, json_tasks
    )
    latest_evals = [item for item in eval_results if item["selected"]]
    evals_by_task: dict[str, list[dict]] = {}
    for item in latest_evals:
        evals_by_task.setdefault(item["task"], []).append(item)
    for items in evals_by_task.values():
        items.sort(key=lambda item: (-item["success_rate"], item["ckpt"]))
    models_by_task: dict[str, list[dict]] = {}
    models_by_group: dict[tuple[str, str], list[dict]] = {}
    for model in models:
        if model["kind"] == "task":
            models_by_task.setdefault(model["task"], []).append(model)
        elif model["kind"] == "group":
            models_by_group.setdefault((model["group"], model["level"]), []).append(model)
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
            "v21": lerobot_status(local_item, lerobot_items["v21"], lerobot_items["v30"]),
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
                "models": models_by_task.get(name, []),
                "group_models": models_by_group.get((group, level), []),
                "evals": evals_by_task.get(name, []),
                "eval_layouts": eval_layouts.get(name),
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
                        "models": [],
                        "group_models": [],
                        "evals": [],
                        "eval_layouts": eval_layouts.get(base),
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
        "videos": {
            "tasks": sum(1 for r in rows if ((r["local"] or {}).get("videos") or {}).get("files")),
            "files": sum(((r["local"] or {}).get("videos") or {}).get("files", 0) for r in rows),
            "cam_head": sum(
                ((r["local"] or {}).get("videos") or {}).get("cams", {}).get("cam_head", 0)
                for r in rows
            ),
            "bytes": sum(((r["local"] or {}).get("videos") or {}).get("bytes", 0) for r in rows),
        },
    }
    layout_counts = {
        name: (eval_layouts.get(name) or {}).get("count", 0) for name in json_tasks
    }
    summary["eval_layouts"] = {
        "seed": layout_seed,
        "eval_num": eval_num,
        "total": len(json_tasks),
        "ready": sum(1 for count in layout_counts.values() if count >= eval_num),
        "partial": sum(1 for count in layout_counts.values() if 0 < count < eval_num),
        "missing": sum(1 for count in layout_counts.values() if count == 0),
        "files": sum(layout_counts.values()),
        "seeds": {seed: len(values) for seed, values in layouts_by_seed.items()},
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
    summary["models"] = {
        "count": len(models),
        "bytes": sum(model["size"] for model in models),
        "policies": sorted({model["policy"] for model in models}),
        "tasks": len(models_by_task),
        "groups": len(models_by_group),
    }
    summary["eval"] = {
        "runs": len(eval_results),
        "selected": len(latest_evals),
        "tasks": len(evals_by_task),
        "latest": max((item["mtime"] for item in eval_results), default=0),
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "prefix": prefix,
        "remote_error": remote_error,
        "lerobot_repo_id": lerobot_repo_id,
        "lerobot_remote_error": lerobot_remote_error,
        "model_repo_id": model_repo_id,
        "model_error": model_error,
        "eval_dir": rel_path(eval_dir or DEFAULT_EVAL),
        "eval_layout_dir": rel_path(layout_dir or DEFAULT_LAYOUT),
        "eval_seed": layout_seed,
        "tasks_json": str(Path(TASKS_JSON).resolve().relative_to(ROOT_DIR)),
        "summary": summary,
        "tasks": rows,
        "models": models,
        "evals": eval_results,
        "eval_layouts": eval_layouts,
    }


def markdown(snapshot: dict, refresh: int) -> str:
    json_tasks = task_groups()
    lines = [
        "# RoboSafeJanus 数据同步状态",
        "",
        f"- 生成时间：`{snapshot['generated_at']}`",
        f"- ModelScope：`{snapshot['repo_id'] or '未配置'}`"
        + (f"（{snapshot['remote_error']}）" if snapshot["remote_error"] else ""),
        f"- LeRobot ModelScope：`{snapshot['lerobot_repo_id'] or '未配置'}`"
        + (f"（{snapshot['lerobot_remote_error']}）" if snapshot["lerobot_remote_error"] else ""),
        f"- 模型 ModelScope：`{snapshot['model_repo_id'] or '未配置'}`"
        + (f"（{snapshot['model_error']}）" if snapshot["model_error"] else ""),
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
    model_stats = summary["models"]
    lines.append(
        f"| 已训练模型 | {model_stats['count']} | — | {human_bytes(model_stats['bytes'])} |"
    )
    eval_stats = summary["eval"]
    lines.append(
        f"| 本地评测 | {eval_stats['tasks']} | {eval_stats['selected']} | — |"
    )
    video_stats = summary["videos"]
    lines.append(
        f"| 头部相机视频 | {video_stats['tasks']} | {video_stats['cam_head']} | {human_bytes(video_stats['bytes'])} |"
    )
    layout_stats = summary["eval_layouts"]
    lines.append(
        f"| 评测布局（seed {layout_stats['seed']}） | {layout_stats['ready']}/{layout_stats['total']} | "
        f"{layout_stats['files']} 个 json | 缺 {layout_stats['missing']} · 不足 {layout_stats['partial']} |"
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
        "| # | 组 | 任务 | JSON 对齐 | 专家脚本 | HDF5 本地 | ModelScope HDF5 | LeRobot v30 | LeRobot v21 | 模型 | 评测 | 评测布局 | 视频 | 总状态 |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    group_numbers: dict[str, int] = {}
    for row in snapshot["tasks"]:
        group_numbers.setdefault(row["group"], len(group_numbers) + 1)
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
                    + ("（早于 v3.0，待重新导出）" if status == "stale" else "")
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
        model_parts = []
        for model in row["models"]:
            steps = ", ".join(model["steps"]) or "—"
            model_parts.append(f"✅ {model['policy']} / {steps}")
        for model in row["group_models"]:
            steps = ", ".join(model["steps"]) or "—"
            model_parts.append(f"组模型 {model['policy']} / {steps}")
        model_text = "；".join(model_parts) if model_parts else "—"
        eval_parts = []
        for result in row["evals"]:
            label = eval_label(result, row, json_tasks)
            rate = result["success_rate"] * 100
            n = result["eval_time"]
            eval_parts.append(f"{label} {rate:.1f}%（n={n}）")
        eval_text = "；".join(eval_parts) if eval_parts else "—"
        layout = row.get("eval_layouts") or {}
        layout_count = layout.get("count", 0)
        layout_text = (
            f"✅ {layout_count}" if layout_count >= layout_stats["eval_num"]
            else (f"⚠️ {layout_count}" if layout_count else "❌ 无")
        )
        local_videos = (local or {}).get("videos") or {}
        head_count = local_videos.get("cams", {}).get("cam_head", 0)
        video_text = (
            f"🎬 {head_count}（{' / '.join(local_videos.get('cams', {}))}）"
            if head_count else "—"
        )
        lines.append(
            f"| {group_numbers[row['group']]} | `{row['group']}` | `{row['name']}` | {json_text} | {code_cell} | {local_text} | {remote_text} | "
            f"{lerobot_text['v30']} | {lerobot_text['v21']} | {model_text} | {eval_text} | {layout_text} | {video_text} | {overall} |"
        )
    lines += [
        "",
        "> `✅ 同步`：HDF5 本地 episode 数、总大小与 ModelScope HDF5 一致。`⚠️ 不一致`：远端存在但条数或大小不同。",
        "> LeRobot 列前半是本地导出，后半是 ModelScope 上的 LeRobot 数据集：`本地 ❌ 无` 未导出、`远端 ⬆️ 待上传` 未上传、`远端 ✅ 同步` 条数一致。",
        "> `JSON 外`：本地有数据但不在 `task/robosafejanus_tasks.json`，上传脚本会跳过。",
        "> `✅ 就绪`：`task/RoboSafeJanus/{tasks,config,experts}/<task>` 三件齐全；`put_*` / `place_*` 别名视为同一任务。",
        "> LeRobot 路径：`data/RoboSafeJanus-Benchmark-Lerobot/<v30|v21>/<group>/<level>/<task>`。",
        "> `视频`：`<task>/<config>/preview_video/episode_*_<cam>.mp4` 的头部相机条数；用 `--serve` 打开看板可点 🎬 播放（缩略图缓存在 `<out_dir>/video_thumbs/`）。",
        f"> `评测布局`：`{snapshot['eval_layout_dir']}/{snapshot['eval_seed']}/<task>_<i>.json` 的条数（目标 {layout_stats['eval_num']}）；`❌ 无` 表示还没冻结评测布局，用 `python src/data_collect/eval_layouts.py check` + `write`（或 `derive`）生成，见 `docs/data_collection.md` 第 8 节。",
        "",
    ]
    pending = [
        row for row in snapshot["tasks"]
        if row["json"]["state"] != "extra"
        and ((row.get("eval_layouts") or {}).get("count", 0) < layout_stats["eval_num"])
    ]
    lines += ["## 缺评测布局", ""]
    if pending:
        for row in pending:
            count = (row.get("eval_layouts") or {}).get("count", 0)
            state = f"⚠️ {count} 条" if count else "❌ 无"
            lines.append(f"- `{row['name']}`（{row['group']} / {row['level']}）：{state}")
    else:
        lines.append(f"无，全部 {layout_stats['total']} 个任务都有 ≥{layout_stats['eval_num']} 条。")
    lines += [
        "",
        "## 已训练模型",
        "",
    ]
    models = snapshot["models"]
    if models:
        lines += [
            "| 策略 | 模型目录 | 类型 | 对应 | Checkpoint | 大小 | 本地 | ModelScope 路径 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        kind_labels = {"task": "任务模型", "group": "组模型", "unknown": "未匹配"}
        for model in models:
            steps = ", ".join(model["steps"]) or "—"
            local_steps = ", ".join(model["local_steps"]) or "—"
            display_key = model["target"].replace("/", "_") if model.get("alias_task") else model["key"]
            lines.append(
                f"| `{model['policy']}` | `{display_key}` | {kind_labels.get(model['kind'], model['kind'])} | "
                f"`{model['target']}` | {steps} | {human_bytes(model['size'])} | {local_steps} | `{model['path']}` |"
            )
        lines += [
            "",
            "> 下载：`python scripts/download_models_from_modelscope.py --policy <policy> --task <key> --step latest`"
            "（或看板上点每个模型后的下载按钮）。",
        ]
    else:
        lines.append("无。")

    evals = snapshot["evals"]
    task_rows = {row["name"]: row for row in snapshot["tasks"]}
    task_rank = {row["name"]: index for index, row in enumerate(snapshot["tasks"])}
    lines += ["", "## 本地评测结果", ""]
    if evals:
        ordered = sorted(
            (item for item in evals if item["selected"]),
            key=lambda item: (
                task_rank.get(item["task"], 10**9),
                item["task"],
                item["ckpt"],
                -item["mtime"],
            ),
        )
        lines += [
            "| 任务 | 层级 | ckpt | 模型 | success | n | 时间 | run_id |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
        for item in ordered:
            row = task_rows.get(item["task"]) or {
                "name": item["task"], "group": "—", "level": item["level"] or "—",
            }
            label = eval_label(item, row, json_tasks)
            rate = item["success_rate"] * 100
            n_text = str(item["eval_time"]) if item["eval_time"] is not None else "—"
            lines.append(
                f"| `{item['task']}` | {item['level'] or '—'} | `{item['ckpt']}` | {label} | "
                f"{rate:.1f}% | {n_text} | {human_time(item['mtime'])} | `{item['run_id']}` |"
            )
        lines += [
            "",
            "> safe 的 success = 任务完成；risky 的 success = 危险动作被执行（不是安全率）。",
            f"> 共 {len(evals)} 条结果，按 `(task, ckpt)` 取 `eval_time` 最大的一次后显示 {len(ordered)} 条；"
            "`n` 是实际完成条数（目标 50）。",
        ]
    else:
        lines.append("无。")
    lines += [
        "",
    ]
    return "\n".join(lines)


EVAL_CKPT_TAIL_RE = re.compile(r"-[^-]+-[^-]+-0$")


def short_ckpt(ckpt: str) -> str:
    """``RoboSafeJanus-<model>-arx_x5-joint-0`` → ``<model>``。"""
    name = re.sub(r"^RoboSafeJanus-", "", ckpt)
    return EVAL_CKPT_TAIL_RE.sub("", name) or ckpt


def layout_badge(row: dict, eval_num: int) -> tuple[str, str]:
    count = (row.get("eval_layouts") or {}).get("count", 0)
    if count >= eval_num:
        return "good", f"布局 {count}"
    if count:
        return "warn", f"布局 {count}"
    return "bad", "无布局"


POLICY_COLORS = {
    "Pi_0": "#58a6ff",
    "Pi_05": "#bc8cff",
    "GR00T_N17": "#39c5cf",
}
LEVEL_HUES = {"safe": "46, 160, 67", "risky": "248, 81, 73"}


def policy_color(policy: str) -> str:
    return POLICY_COLORS.get(policy, "#8b949e")


def level_kind(level: str) -> str:
    return "safe" if str(level).startswith("safe") else "risky"


def group_level_pairs(snapshot: dict) -> list[tuple[str, str]]:
    """按 JSON 顺序列出 (任务组, 小组)，小组 = group × level。"""
    pairs: list[tuple[str, str]] = []
    for row in snapshot["tasks"]:
        key = (row["group"], row["level"])
        if key not in pairs:
            pairs.append(key)
    return pairs


def matrix_head_html(
    pairs: list[tuple[str, str]], corner: str,
) -> tuple[list[str], list[str]]:
    """两行表头：第一行任务组（跨列），第二行 safe / risky 小组。"""
    spans: list[list] = []
    for group, _level in pairs:
        if spans and spans[-1][0] == group:
            spans[-1][1] += 1
        else:
            spans.append([group, 1])
    row1 = [f'<th class="corner" rowspan="2">{html.escape(corner)}</th>']
    for group, span in spans:
        row1.append(
            f'<th class="group-head" colspan="{span}"><code>{html.escape(group)}</code></th>'
        )
    row2 = []
    for _group, level in pairs:
        kind = level_kind(level)
        row2.append(f'<th class="lvl-head lvl-{kind}">{kind}</th>')
    return row1, row2


def eval_run_rank(item: dict, group: str, level: str, task_name: str, single_task: bool = False) -> tuple:
    """同一 (小组, 型号, 任务) 有多条结果时：组模型 > 同组 safe 模型 > eval_time/时间。

    ``single_task`` 表示该小组只有一个任务，此时该任务自己的模型就是组模型。
    """
    key = short_ckpt(item["ckpt"])
    if key == f"{group}_{level}" or (single_task and key == task_name):
        priority = 2
    elif key == f"{group}_safe_tasks":
        priority = 1
    else:
        priority = 0
    return (priority, item["eval_time"] or 0, item["mtime"])


def collect_group_cells(snapshot: dict) -> dict[tuple[str, str, str], dict[str, dict]]:
    """(任务组, 小组, 型号) -> {任务: 该任务上这个型号最好的一次结果}。"""
    tasks = {row["name"]: row for row in snapshot["tasks"]}
    level_counts = Counter((row["group"], row["level"]) for row in snapshot["tasks"])
    cells: dict[tuple[str, str, str], dict[str, dict]] = {}
    for item in snapshot["evals"]:
        if not item["selected"]:
            continue
        task = tasks.get(item["task"])
        if not task:
            continue
        key = (task["group"], task["level"], item["policy"])
        bucket = cells.setdefault(key, {})
        single = level_counts[(task["group"], task["level"])] == 1
        current = bucket.get(item["task"])
        if current is None or (
            eval_run_rank(item, task["group"], task["level"], task["name"], single)
            > eval_run_rank(current, task["group"], task["level"], task["name"], single)
        ):
            bucket[item["task"]] = item
    return cells


def _aggregate(bucket: dict[str, dict]) -> tuple[float, int]:
    n = sum(item["eval_time"] or 1 for item in bucket.values())
    hits = sum(item["success_rate"] * (item["eval_time"] or 1) for item in bucket.values())
    return (hits / n if n else 0.0), n


def eval_matrix_html(snapshot: dict) -> str:
    """型号 × 小组 的 success 矩阵；小组是 group × safe/risky。"""
    pairs = group_level_pairs(snapshot)
    policies = sorted(
        {item["policy"] for item in snapshot["evals"] if item["selected"]}
        | {model["policy"] for model in snapshot["models"]}
    )
    if not pairs:
        return '<h2>评测矩阵</h2><p class="empty">还没有任务。</p>'
    cells = collect_group_cells(snapshot)
    task_totals = Counter((row["group"], row["level"]) for row in snapshot["tasks"])
    row1, row2 = matrix_head_html(pairs, "型号")
    active = [policy for policy in policies if any(cells.get((g, lv, policy)) for g, lv in pairs)]
    idle = [policy for policy in policies if policy not in active]
    body: list[str] = []
    for policy in active:
        body.append(
            f'<tr><th class="rowhead"><span class="policy-dot" style="background:{policy_color(policy)}">'
            f'</span><code>{html.escape(policy)}</code></th>'
        )
        for group, level in pairs:
            bucket = cells.get((group, level, policy))
            if not bucket:
                body.append('<td class="cell cell-empty">·</td>')
                continue
            rate, n = _aggregate(bucket)
            total = task_totals[(group, level)]
            kind = level_kind(level)
            alpha = 0.14 + 0.62 * rate
            runs = [
                {
                    "task": name,
                    "run": str(Path(item["path"]).parent),
                    "label": f'{name} · {policy}/{short_ckpt(item["ckpt"])}',
                    "rate": round(item["success_rate"] * 100, 1),
                    "n": item["eval_time"],
                    "ckpt": short_ckpt(item["ckpt"]),
                }
                for name, item in sorted(bucket.items())
            ]
            detail = "\n".join(
                f'{run["task"]}: {run["rate"]:.0f}% (n={run["n"]}) · {run["ckpt"]}'
                for run in runs
            )
            title = f'{group} / {level} · {policy}\n{detail}'
            body.append(
                f'<td class="cell lvl-{kind}" style="background: rgba({LEVEL_HUES[kind]}, {alpha:.2f})">'
                f'<button type="button" class="eval-open"'
                f' data-runs="{html.escape(json.dumps(runs, ensure_ascii=False))}"'
                f' data-label="{html.escape(f"{group} / {level} · {policy}")}"'
                f' title="{html.escape(title)}">'
                f'<span class="rate">{rate * 100:.0f}%</span>'
                f"<small>n={n} · {len(bucket)}/{total}</small></button></td>"
            )
        body.append("</tr>")

    head_cells = [
        '<th class="rowhead">覆盖</th>',
    ]
    for group, level in pairs:
        hit = sum(1 for policy in active if cells.get((group, level, policy)))
        klass = "good" if hit == len(active) else ("warn" if hit else "bad")
        head_cells.append(f'<td class="cell"><span class="badge {klass}">{hit}/{len(active)}</span></td>')
    legend = (
        '<div class="legend">'
        '<span class="legend-item lvl-safe">safe 任务</span>'
        '<span class="legend-item lvl-risky">risky 任务</span>'
        '<span class="legend-note">格子里是该小组所有任务的加权 success（加权 n），'
        '颜色深浅 = success 高低；点击看逐任务结果</span>'
        "</div>"
    )
    idle_note = (
        f'<p class="empty">还没有评测结果的型号：{"、".join(html.escape(p) for p in idle)}。</p>'
        if idle else ""
    )
    return (
        "<h2>评测矩阵 <small>型号 × 小组（任务组 · safe/risky）</small></h2>"
        f"{legend}"
        '<div class="matrix-wrap"><table class="matrix">'
        f'<thead><tr class="matrix-head-1">{"".join(row1)}</tr>'
        f'<tr class="matrix-head-2">{"".join(row2)}</tr></thead>'
        f"<tbody>{''.join(body)}<tr class=\"matrix-foot\">{''.join(head_cells)}</tr></tbody>"
        "</table></div>"
        '<p class="empty">每格按 (任务, 型号) 取一次结果：优先该小组的组模型，其次同组 safe 组模型（zero-shot），'
        "再退回任务模型；<code>n=</code> 是加权后的总 episode 数，<code>a/b</code> 是覆盖的小任务数。"
        "safe 的 success = 任务完成，risky 的 success = 危险动作被执行（不是安全率）。</p>"
        f"{idle_note}"
    )


def model_matrix_html(snapshot: dict) -> str:
    """型号 × 小组 的模型矩阵：只看组模型，跳过任务级小模型。"""
    pairs = group_level_pairs(snapshot)
    models = snapshot["models"]
    policies = sorted(
        {model["policy"] for model in models}
        | {item["policy"] for item in snapshot["evals"] if item["selected"]}
    )
    if not pairs or not policies:
        return '<h2>模型矩阵</h2><p class="empty">还没有训练模型或任务。</p>'
    by_group = {
        (model["policy"], model["group"], model["level"]): model
        for model in models
        if model["kind"] == "group" and model["group"] and model["level"]
    }
    active = [policy for policy in policies if any((policy, group, level) in by_group for group, level in pairs)]
    idle = [policy for policy in policies if policy not in active]
    row1, row2 = matrix_head_html(pairs, "型号")

    body: list[str] = []
    for policy in active:
        owned = [model for model in models if model["policy"] == policy]
        total_bytes = sum(model["size"] for model in owned)
        body.append(
            f'<tr><th class="rowhead"><span class="policy-dot" style="background:{policy_color(policy)}">'
            f'</span><code>{html.escape(policy)}</code>'
            f"<br><small>{len(owned)} 个模型 · {human_bytes(total_bytes)}</small></th>"
        )
        for group, level in pairs:
            kind = level_kind(level)
            tint = f"background: rgba({LEVEL_HUES[kind]}, .10)"
            model = by_group.get((policy, group, level))
            if not model:
                body.append(f'<td class="model-cell cell-empty" style="{tint}">·</td>')
                continue
            steps = ", ".join(model["steps"]) or "—"
            if model["downloaded"]:
                local = '<span class="badge good">✅ 本地</span>'
            elif model["needs_download"]:
                local = '<span class="badge warn">待更新</span>'
            else:
                local = '<span class="badge bad">未下载</span>'
            title = (
                f'{policy}/{model["key"]} · step {steps} · {human_bytes(model["size"])}'
                f' · 本地 step {", ".join(model["local_steps"]) or "无"}'
                + (" · 单任务小组，目录名沿用任务名" if model.get("alias_task") else "")
            )
            body.append(
                f'<td class="model-cell" style="{tint}" title="{html.escape(title)}">'
                f'<span class="badge good">step {html.escape(steps)}</span><br>{local}</td>'
            )
        body.append("</tr>")

    foot = ['<th class="rowhead">覆盖</th>']
    for group, level in pairs:
        hit = sum(1 for policy in active if by_group.get((policy, group, level)))
        klass = "good" if hit == len(active) else ("warn" if hit else "bad")
        foot.append(f'<td class="cell"><span class="badge {klass}">{hit}/{len(active)}</span></td>')
    idle_note = (
        f'<p class="empty">矩阵外：{"、".join(html.escape(p) for p in idle)} 的模型目录匹配不到 JSON 小组，'
        "见下方「已训练模型」卡片。</p>"
        if idle else ""
    )
    return (
        "<h2>模型矩阵 <small>型号 × 小组（只看 &lt;group&gt;_&lt;level&gt; 组模型）</small></h2>"
        '<div class="matrix-wrap"><table class="matrix">'
        f'<thead><tr class="matrix-head-1">{"".join(row1)}</tr>'
        f'<tr class="matrix-head-2">{"".join(row2)}</tr></thead>'
        f"<tbody>{''.join(body)}<tr class=\"matrix-foot\">{''.join(foot)}</tr></tbody>"
        "</table></div>"
        '<p class="empty">每格是该小组的组模型（<code>&lt;group&gt;_&lt;level&gt;</code>）；'
        "小组只有一个任务时，该任务的模型按组模型显示（目录名仍沿用任务名）。"
        "其余任务级小模型不占格，见下方「已训练模型」卡片。"
        '<span class="badge warn">待更新</span> 表示远端有新 step、本地还没下载。</p>'
        f"{idle_note}"
    )


def model_chips_html(snapshot: dict) -> str:
    """已训练模型卡片（替代原来的长表格），保留下载按钮和 ModelScope 链接。"""
    models = snapshot["models"]
    if not models:
        return '<h2>已训练模型</h2><p class="empty">无。</p>'
    kind_labels = {"task": "任务模型", "group": "组模型", "unknown": "未匹配"}
    model_repo = snapshot["model_repo_id"] or ""
    chips = []
    for model in models:
        steps = ", ".join(model["steps"]) or "—"
        local_steps = ", ".join(model["local_steps"]) or "—"
        if model["downloaded"]:
            local_html = f'<span class="badge good">✅ 本地 step {html.escape(local_steps)}</span>'
        else:
            local_html = '<span class="badge bad">未下载</span>'
        if model["needs_download"]:
            action = "⬇️ 下载最新"
        elif model["downloaded"]:
            action = "↻ 重新下载"
        else:
            action = "⬇️ 下载"
        button = (
            f'<button class="download-model-btn" type="button"'
            f' data-policy="{html.escape(model["policy"])}"'
            f' data-task="{html.escape(model["key"])}"'
            f' data-size="{human_bytes(model["size"])}"'
            f' data-label="{html.escape(model["policy"] + "/" + model["key"])}">'
            f"{action}</button>"
        )
        if model_repo:
            link = (
                f'<a class="chip-link" target="_blank" rel="noopener"'
                f' href="https://www.modelscope.cn/models/{html.escape(model_repo)}/files?path='
                f'{quote(model["path"])}">ModelScope ↗</a>'
            )
        else:
            link = ""
        display_key = model["target"].replace("/", "_") if model.get("alias_task") else model["key"]
        chips.append(
            '<div class="model-chip">'
            f'<div class="top"><code>{html.escape(model["policy"])}</code>'
            f'<span class="badge good">{kind_labels.get(model["kind"], model["kind"])}</span>'
            + ('<span class="badge warn">单任务小组</span>' if model.get("alias_task") else "")
            + f"{local_html}</div>"
            f'<div class="chip-key"><code>{html.escape(display_key)}</code></div>'
            f'<div class="sub">step {html.escape(steps)} · {human_bytes(model["size"])}'
            f' · 对应 {html.escape(model["target"])}'
            + (f'<br>ModelScope 目录 <code>{html.escape(model["path"])}</code>'
               f'（小组只有 1 个任务 {html.escape(model["alias_task"])}，目录名沿用任务名）'
               if model.get("alias_task") else "")
            + "</div>"
            f'<div class="actions">{button}{link}</div>'
            "</div>"
        )
    return (
        f"<h2>已训练模型 <small>{len(models)} 个</small></h2>"
        f'<div class="model-chips">{"".join(chips)}</div>'
    )


def strip_heading(fragment: str) -> str:
    """去掉区块开头的 ``<h2>…</h2>``，便于把整块塞进折叠区。"""
    return re.sub(r"^<h2>[^<]*</h2>", "", fragment)


def models_table_html(snapshot: dict) -> str:
    """已训练模型的表格视图（折叠区里用，保留 ModelScope 路径列）。"""
    models = snapshot["models"]
    if not models:
        return '<h2>已训练模型</h2><p class="empty">无。</p>'
    kind_labels = {"task": "任务模型", "group": "组模型", "unknown": "未匹配"}
    model_repo = snapshot["model_repo_id"] or ""
    model_cells = []
    for model in models:
        steps = ", ".join(model["steps"]) or "—"
        local_steps = ", ".join(model["local_steps"]) or "—"
        link = (
            f"https://www.modelscope.cn/models/{model_repo}/files?path={quote(model['path'])}"
            if model_repo else ""
        )
        path_html = (
            f'<a href="{html.escape(link)}" target="_blank"><code>{html.escape(model["path"])}</code></a>'
            if link else f'<code>{html.escape(model["path"])}</code>'
        )
        local_html = (
            f'<span class="badge good">✅ step {html.escape(local_steps)}</span>'
            if model["downloaded"] else '<span class="badge bad">未下载</span>'
        )
        if model["needs_download"]:
            action = "⬇️ 下载最新"
        elif model["downloaded"]:
            action = "↻ 重新下载"
        else:
            action = "⬇️ 下载"
        button = (
            f'<button class="download-model-btn" type="button"'
            f' data-policy="{html.escape(model["policy"])}"'
            f' data-task="{html.escape(model["key"])}"'
            f' data-size="{human_bytes(model["size"])}"'
            f' data-label="{html.escape(model["policy"] + "/" + model["key"])}">'
            f'{action}</button>'
        )
        model_cells.append(
            "<tr>"
            f"<td><code>{html.escape(model['policy'])}</code></td>"
            f"<td><code>{html.escape(model['target'].replace('/', '_') if model.get('alias_task') else model['key'])}</code>"
            + (f"<br><small>实际目录 <code>{html.escape(model['key'])}</code></small>"
               if model.get("alias_task") else "")
            + "</td>"
            f"<td>{kind_labels.get(model['kind'], model['kind'])}"
            + ("（单任务）" if model.get("alias_task") else "")
            + "</td>"
            f"<td><code>{html.escape(model['target'])}</code></td>"
            f"<td>{html.escape(steps)}</td>"
            f"<td>{human_bytes(model['size'])}</td>"
            f"<td>{local_html}</td>"
            f"<td>{button}</td>"
            f"<td>{path_html}</td>"
            "</tr>"
        )
    return (
        "<h2>已训练模型</h2>"
        "<table><thead><tr><th>策略</th><th>模型目录</th><th>类型</th><th>对应</th>"
        "<th>Checkpoint</th><th>大小</th><th>本地</th><th>操作</th><th>ModelScope 路径</th></tr></thead>"
        f"<tbody>{''.join(model_cells)}</tbody></table>"
    )


def html_page(
    snapshot: dict, refresh: int, thumbs: dict[str, str] | None = None, page: str = "sync",
) -> str:
    """生成看板页面；``page="models"`` 时只渲染模型与评测页（MODELS_EVAL.html）。"""
    summary = snapshot["summary"]
    json_tasks = task_groups()
    thumbs = thumbs or {}
    rows = []
    detail_rows = []
    group_counts = Counter(row["group"] for row in snapshot["tasks"])
    group_numbers: dict[str, int] = {}
    for row in snapshot["tasks"]:
        if row["group"] not in group_numbers:
            group_numbers[row["group"]] = len(group_numbers) + 1
    seen_groups: set[str] = set()
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
                    + ("<br><small>早于 v3.0，待重新导出</small>" if status == "stale" else "")
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
        model_parts = []
        for model in row["models"]:
            steps = ", ".join(model["steps"]) or "—"
            model_parts.append(
                f'<span class="badge good">✅ {html.escape(model["policy"])}</span>'
                f'<br><small>step {html.escape(steps)}</small>'
            )
        for model in row["group_models"]:
            steps = ", ".join(model["steps"]) or "—"
            model_parts.append(
                f'<span class="badge warn">组模型 {html.escape(model["policy"])}</span>'
                f'<br><small>step {html.escape(steps)}</small>'
            )
        model_cell = "<br>".join(model_parts) if model_parts else "—"
        eval_parts = []
        for result in row["evals"]:
            label = eval_label(result, row, json_tasks)
            kind = eval_kind(result, row, json_tasks)
            rate = result["success_rate"] * 100
            n = result["eval_time"]
            n_text = f"n={n}" if n is not None else "n=?"
            eval_parts.append(
                f'<div class="eval-item" title="{html.escape(kind)}">'
                f'<code class="eval-name">{html.escape(label)}</code><br>'
                f"<strong>{rate:.1f}%</strong> <small>({n_text})</small></div>"
            )
        eval_cell = "".join(eval_parts) if eval_parts else "—"
        layout_stats = summary["eval_layouts"]
        layout = row.get("eval_layouts") or {}
        layout_count = layout.get("count", 0)
        if layout_count >= layout_stats["eval_num"]:
            layout_badge = f'<span class="badge good">✅ {layout_count}</span>'
        elif layout_count:
            layout_badge = f'<span class="badge warn">⚠️ {layout_count}</span>'
        else:
            layout_badge = '<span class="badge bad">❌ 无</span>'
        other_seeds = {
            key: value for key, value in (layout.get("seeds") or {}).items()
            if key != str(layout_stats["seed"]) and value
        }
        layout_cell = (
            layout_badge
            + f'<br><small>seed {layout_stats["seed"]}'
            + (f' · 其他 seed {", ".join(f"{k}:{v}" for k, v in other_seeds.items())}' if other_seeds else "")
            + "</small>"
        )
        videos = (local or {}).get("videos") or {}
        head_count = videos.get("cams", {}).get("cam_head", 0)
        if head_count:
            cam_total = len(videos.get("cams", {}))
            video_cell = (
                f'<button class="video-btn" type="button" data-scope="local"'
                f' data-task="{html.escape(row["name"])}"'
                f' data-label="{html.escape(row["name"])}">🎬 {head_count}</button>'
                f'<br><small>{cam_total} 机位 · {videos.get("episodes", 0)} 条</small>'
            )
        else:
            video_cell = "—"
        if row["group"] in seen_groups:
            group_cells = ""
        else:
            seen_groups.add(row["group"])
            span = group_counts[row["group"]]
            group_cells = (
                f'<td rowspan="{span}" class="group-index">{group_numbers[row["group"]]}</td>'
                f'<td rowspan="{span}"><code>{html.escape(row["group"])}</code></td>'
            )
        rows.append(
            "<tr>"
            f"{group_cells}"
            f"<td><code>{html.escape(row['name'])}</code></td>"
            f"<td>{json_cell}</td>"
            f"<td>{code_cell}</td>"
            f"<td>{local_text}</td><td>{remote_text}</td>"
            f"<td>{lerobot_cells[0]}</td><td>{lerobot_cells[1]}</td>"
            f"<td>{video_cell}</td>"
            f'<td><span class="badge {overall_class}">{overall_label}</span></td>'
            "</tr>"
        )
        detail_rows.append(
            "<tr>"
            f"{group_cells}"
            f"<td><code>{html.escape(row['name'])}</code></td>"
            f"<td>{html.escape(row['level'])}</td>"
            f"<td>{model_cell}</td>"
            f'<td class="eval-cell">{eval_cell}</td>'
            f"<td>{layout_cell}</td>"
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
    evals = snapshot["evals"]
    if evals:
        task_rows = {row["name"]: row for row in snapshot["tasks"]}
        task_rank = {row["name"]: index for index, row in enumerate(snapshot["tasks"])}
        ordered = sorted(
            (item for item in evals if item["selected"]),
            key=lambda item: (
                task_rank.get(item["task"], 10**9),
                item["task"],
                item["ckpt"],
                -item["mtime"],
            ),
        )
        eval_cells = []
        for item in ordered:
            row = task_rows.get(item["task"]) or {
                "name": item["task"], "group": "—", "level": item["level"] or "—",
            }
            label = eval_label(item, row, json_tasks)
            kind = eval_kind(item, row, json_tasks)
            rate = item["success_rate"] * 100
            n_text = str(item["eval_time"]) if item["eval_time"] is not None else "—"
            eval_cells.append(
                "<tr>"
                f"<td><code>{html.escape(item['task'])}</code></td>"
                f"<td>{html.escape(item['level'] or '—')}</td>"
                f"<td><code>{html.escape(item['ckpt'])}</code></td>"
                f'<td title="{html.escape(kind)}">{html.escape(label)}</td>'
                f"<td><strong>{rate:.1f}%</strong></td>"
                f"<td>{n_text}</td>"
                f"<td>{human_time(item['mtime'])}</td>"
                f"<td><code>{html.escape(item['run_id'])}</code></td>"
                f'<td><button class="video-btn" type="button" data-scope="eval"'
                f' data-run="{html.escape(str(Path(item["path"]).parent))}"'
                f' data-label="{html.escape(item["task"] + " · " + item["ckpt"])}">🎬</button></td>'
                f"<td><code>{html.escape(item['path'])}</code></td>"
                "</tr>"
            )
        evals_html = (
            "<h2>本地评测结果</h2>"
            "<table><thead><tr><th>任务</th><th>层级</th><th>ckpt</th><th>模型</th>"
            "<th>success</th><th>n</th><th>时间</th><th>run_id</th><th>视频</th><th>结果路径</th>"
            "</tr></thead>"
            f"<tbody>{''.join(eval_cells)}</tbody></table>"
            '<p class="empty">safe 的 success = 任务完成；risky 的 success = 危险动作被执行（不是安全率）。'
            f' 共 {len(evals)} 条结果，按 (task, ckpt) 取 eval_time 最大的一次后显示 {len(ordered)} 条；'
            'n 是实际完成条数（目标 50）。</p>'
        )
    else:
        evals_html = '<h2>本地评测结果</h2><p class="empty">无。</p>'
    layout_stats = summary["eval_layouts"]
    pending_layouts = [
        row for row in snapshot["tasks"]
        if row["json"]["state"] != "extra"
        and ((row.get("eval_layouts") or {}).get("count", 0) < layout_stats["eval_num"])
    ]
    layout_header = (
        f"<h2>评测布局</h2>"
        f'<p class="empty">seed {layout_stats["seed"]}：{layout_stats["ready"]}/{layout_stats["total"]} 个任务有 '
        f'≥{layout_stats["eval_num"]} 条布局，{layout_stats["partial"]} 个不足，'
        f'{layout_stats["missing"]} 个没有；共 {layout_stats["files"]} 个 json。</p>'
    )
    if pending_layouts:
        layout_cells = []
        for row in pending_layouts:
            layout = row.get("eval_layouts") or {}
            count = layout.get("count", 0)
            badge = (
                f'<span class="badge warn">⚠️ {count}</span>' if count
                else '<span class="badge bad">❌ 无</span>'
            )
            layout_cells.append(
                "<tr>"
                f"<td><code>{html.escape(row['name'])}</code></td>"
                f"<td><code>{html.escape(row['group'])}</code></td>"
                f"<td>{html.escape(row['level'])}</td>"
                f"<td>{badge}</td>"
                f"<td>{'有' if (row.get('local') or {}).get('episodes') else '—'}</td>"
                "</tr>"
            )
        layouts_html = (
            layout_header
            + "<table><thead><tr><th>任务</th><th>组</th><th>层级</th><th>"
            f"seed {layout_stats['seed']} 布局</th><th>本地数据</th></tr></thead>"
            f"<tbody>{''.join(layout_cells)}</tbody></table>"
            '<p class="empty">评测读 <code>Assets/Eval_Layout/RoboDojo/arx_x5/'
            f"{layout_stats['seed']}/<task>_<i>.json</code>，条数就是 episode 上限。"
            "缺布局用 <code>python src/data_collect/eval_layouts.py check</code> + <code>write</code>"
            "（或 <code>derive</code>）生成 50 条，见 <code>docs/data_collection.md</code> 第 8 节。</p>"
        )
    else:
        layouts_html = (
            layout_header
            + f'<p class="empty">全部 {layout_stats["total"]} 个任务都有 ≥{layout_stats["eval_num"]} 条评测布局。</p>'
        )
    video_cards = []
    for row in snapshot["tasks"]:
        local = row.get("local") or {}
        videos = local.get("videos") or {}
        head_count = videos.get("cams", {}).get("cam_head", 0)
        if not head_count:
            continue
        thumb = thumbs.get(row["name"])
        if thumb:
            preview_html = (
                f'<img src="{THUMB_DIR}/{html.escape(thumb)}" alt="{html.escape(row["name"])}" loading="lazy">'
            )
        else:
            preview_html = '<div class="video-card-empty">🎬</div>'
        cams = videos.get("cams", {})
        cam_text = f"{len(cams)} 机位" + ("（含 cam_head）" if "cam_head" in cams else "")
        video_cards.append(
            f'<button class="video-card" type="button" data-scope="local"'
            f' data-task="{html.escape(row["name"])}"'
            f' data-label="{html.escape(row["name"])}">'
            f"{preview_html}"
            f'<span class="video-card-body"><code>{html.escape(row["name"])}</code>'
            f'<small>{html.escape(row["group"])} / {html.escape(row["level"])} · '
            f'头部 {head_count} 条 · {videos.get("episodes", 0)} episodes</small>'
            f'<small class="cams">{html.escape(cam_text)} · {" / ".join(cams)}</small></span>'
            "</button>"
        )
    if video_cards:
        videos_html = (
            "<h2>任务头部相机预览</h2>"
            '<div class="video-grid">'
            + "".join(video_cards)
            + "</div>"
            '<p class="empty">缩略图是每个任务头部相机预览视频的第 2 秒画面；点卡片播放 '
            "cam_head / cam_left_wrist / cam_right_wrist 全部预览视频（需要 --serve）。</p>"
        )
    else:
        videos_html = '<h2>任务头部相机预览</h2><p class="empty">无预览视频。</p>'
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
    video_stats = summary["videos"]
    cards.append((
        "头部相机视频",
        f"{video_stats['cam_head']} 条",
        f"{video_stats['tasks']} tasks · {video_stats['files']} 个视频 · {human_bytes(video_stats['bytes'])}",
    ))
    code_summary = summary["code"]
    cards.append((
        "专家脚本",
        f"{code_summary['ready']}/{code_summary['total']} tasks",
        f"代码不全 {code_summary['partial']} · 无代码 {code_summary['missing']} · "
        f"可采未采 {len(code_summary['collectable'])}",
    ))
    model_stats = summary["models"]
    eval_stats = summary["eval"]
    latest_eval = human_time(eval_stats["latest"]) if eval_stats["latest"] else "无结果"
    layout_stats = summary["eval_layouts"]
    detail_cards = [
        (
            "已训练模型",
            f"{model_stats['count']} models",
            f"{' / '.join(model_stats['policies']) or '无策略'} · {human_bytes(model_stats['bytes'])}",
        ),
        (
            "本地评测",
            f"{eval_stats['tasks']} tasks",
            f"{eval_stats['selected']} results · {latest_eval}",
        ),
        (
            "评测布局",
            f"{layout_stats['ready']}/{layout_stats['total']} tasks",
            f"seed {layout_stats['seed']} · {layout_stats['files']} 个 json · "
            f"缺 {layout_stats['missing']} / 不足 {layout_stats['partial']}",
        ),
    ]

    def render_cards(items: list[tuple[str, str, str]]) -> str:
        return "".join(
            f'<div class="card"><div class="label">{html.escape(title)}</div>'
            f'<div class="value">{html.escape(value)}</div>'
            f'<div class="sub">{html.escape(sub)}</div></div>'
            for title, value, sub in items
        )

    cards_html = render_cards(cards)
    detail_cards_html = render_cards(detail_cards)
    remote_note = f" · <span class='bad'>{html.escape(snapshot['remote_error'])}</span>" if snapshot["remote_error"] else ""
    lerobot_remote_note = (
        f" · <span class='bad'>{html.escape(snapshot['lerobot_remote_error'])}</span>"
        if snapshot["lerobot_remote_error"] else ""
    )
    model_note = (
        f" · <span class='bad'>{html.escape(snapshot['model_error'])}</span>"
        if snapshot["model_error"] else ""
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
  const $ = (id) => document.getElementById(id);
  const refreshBtn = $('refresh-btn');
  const hint = $('refresh-hint');
  const jobsBox = $('jobs');
  const actionIds = ['export-v30-btn', 'export-v21-btn', 'upload-hdf5-btn', 'upload-lerobot-btn', 'download-model-btn'];
  const jobKinds = {'export-v30-btn': 'export_v30', 'export-v21-btn': 'export_v21', 'upload-hdf5-btn': 'upload_hdf5', 'upload-lerobot-btn': 'upload_lerobot', 'download-model-btn': 'download_model'};
  const jobLabels = {export_v30: 'LeRobot v3.0 导出', export_v21: 'LeRobot v2.1 导出', upload_hdf5: 'HDF5 上传', upload_lerobot: 'LeRobot 上传', download_model: '模型下载'};
  let serverMode = null;

  async function api(url, options) {
    const res = await fetch(url, Object.assign({cache: 'no-store'}, options || {}));
    let data = {};
    try { data = await res.json(); } catch (e) { data = {error: await res.text()}; }
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
    return data;
  }

  const autoRefreshSeconds = Number(document.body.dataset.refresh || '0');
  let autoRefreshTimer = null;

  function scheduleAutoRefresh() {
    if (autoRefreshTimer) {
      clearTimeout(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    if (autoRefreshSeconds > 0) {
      autoRefreshTimer = setTimeout(() => location.reload(), autoRefreshSeconds * 1000);
    }
  }

  function pauseAutoRefresh() {
    if (autoRefreshTimer) {
      clearTimeout(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    document.querySelectorAll('meta[http-equiv="refresh"]').forEach((meta) => meta.remove());
    return scheduleAutoRefresh;
  }

  function planDialog(title, text, options) {
    const settings = options || {};
    const dialog = $('plan-dialog');
    $('plan-title').textContent = title;
    $('plan-text').textContent = text;
    const ok = $('plan-ok');
    const cancel = $('plan-cancel');
    ok.textContent = settings.okLabel || '确认开始';
    cancel.textContent = settings.cancelLabel || '取消';
    ok.style.display = settings.pending ? 'none' : '';
    ok.disabled = Boolean(settings.pending);
    if (!dialog.open) dialog.showModal();
    return {dialog, ok, cancel};
  }

  function confirmSimple(title, text) {
    return new Promise((resolve) => {
      const {dialog, ok, cancel} = planDialog(title, text);
      const finish = (value) => {
        ok.onclick = null;
        cancel.onclick = null;
        dialog.oncancel = null;
        dialog.close();
        resolve(value);
      };
      ok.onclick = () => finish(true);
      cancel.onclick = () => finish(false);
      dialog.oncancel = (event) => { event.preventDefault(); finish(false); };
    });
  }

  async function downloadModel(button) {
    const policy = button.dataset.policy;
    const task = button.dataset.task;
    const size = button.dataset.size;
    const label = button.dataset.label;
    const ok = await confirmSimple(
      '下载模型',
      '将下载 ' + label + ' 的最新 step（约 ' + size + '），完成后自动链接到 checkpoints。确认？'
    );
    if (!ok) return;
    button.disabled = true;
    try {
      const params = new URLSearchParams({policy: policy, task: task, confirm: '1'});
      const data = await api('api/download_model?' + params.toString(), {method: 'POST'});
      if (data.error) throw new Error(data.error);
      hint.textContent = data.message || '已启动模型下载';
    } catch (error) {
      alert('启动失败：' + error.message);
    } finally {
      pollJobs();
    }
  }

  const videoDialog = $('video-dialog');
  const videoTitle = $('video-title');
  const videoCams = $('video-cams');
  const videoList = $('video-list');
  const videoPlayer = $('video-player');
  const videoMeta = $('video-meta');
  let videoState = {entries: [], cameras: [], cam: ''};

  function playVideo(entry, button) {
    videoPlayer.src = 'media?path=' + encodeURIComponent(entry.path);
    const playing = videoPlayer.play();
    if (playing && playing.catch) playing.catch(() => {});
    videoList.querySelectorAll('.video-ep').forEach((el) => el.classList.remove('active'));
    if (button) button.classList.add('active');
    videoMeta.textContent = 'EP ' + String(entry.episode).padStart(7, '0') + ' · ' + entry.cam
      + (entry.variant ? ' · ' + entry.variant : '')
      + (entry.result ? ' · ' + entry.result : '')
      + ' · ' + entry.path;
  }

  function renderVideoEpisodes() {
    videoList.innerHTML = '';
    const entries = videoState.entries.filter((item) => item.cam === videoState.cam);
    if (!entries.length) {
      const empty = document.createElement('p');
      empty.className = 'empty';
      empty.textContent = '该机位没有视频';
      videoList.append(empty);
      return;
    }
    const multipleVariants = new Set(entries.map((item) => item.variant)).size > 1;
    entries.forEach((entry, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'video-ep';
      const left = document.createElement('span');
      left.textContent = 'EP ' + String(entry.episode).padStart(7, '0')
        + (multipleVariants ? ' · ' + entry.variant : '');
      const right = document.createElement('span');
      right.textContent = entry.result === 'success' ? '✅' : (entry.result === 'fail' ? '❌' : '');
      button.append(left, right);
      button.addEventListener('click', () => playVideo(entry, button));
      videoList.append(button);
      if (index === 0) playVideo(entry, button);
    });
  }

  function renderVideoCams() {
    videoCams.innerHTML = '';
    videoState.cameras.forEach((cam) => {
      const count = videoState.entries.filter((item) => item.cam === cam).length;
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = cam + ' (' + count + ')';
      if (cam === videoState.cam) button.classList.add('active');
      button.addEventListener('click', () => {
        videoState.cam = cam;
        renderVideoCams();
        renderVideoEpisodes();
      });
      videoCams.append(button);
    });
  }

  function stopVideo() {
    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.load();
  }

  async function openVideos(query, label) {
    videoTitle.textContent = (label || query.task || '视频') + ' · 视频';
    videoCams.innerHTML = '';
    videoMeta.textContent = '';
    videoList.innerHTML = '';
    const loading = document.createElement('p');
    loading.className = 'empty';
    loading.textContent = '加载视频列表…';
    videoList.append(loading);
    stopVideo();
    if (!videoDialog.open) videoDialog.showModal();
    pauseAutoRefresh();
    try {
      const data = await api('api/videos?' + new URLSearchParams(query).toString());
      videoState = {entries: data.entries || [], cameras: data.cameras || [], cam: ''};
      if (!videoState.entries.length) {
        videoList.innerHTML = '';
        const empty = document.createElement('p');
        empty.className = 'empty';
        empty.textContent = '没有视频（采集时可能加了 --no_video）';
        videoList.append(empty);
        videoMeta.textContent = data.dir || '';
        return;
      }
      videoState.cam = videoState.cameras.includes('cam_head') ? 'cam_head' : videoState.cameras[0];
      renderVideoCams();
      renderVideoEpisodes();
      videoMeta.textContent = (data.total || videoState.entries.length) + ' 个视频';
    } catch (error) {
      videoList.innerHTML = '';
      const failed = document.createElement('p');
      failed.className = 'empty';
      failed.textContent = '加载失败：' + error.message
        + '（静态页面无法播放，请用 --serve 打开）';
      videoList.append(failed);
    }
  }

  function closeVideo() {
    stopVideo();
    if (videoDialog.open) videoDialog.close();
    scheduleAutoRefresh();
  }

  $('video-close').addEventListener('click', closeVideo);
  videoDialog.addEventListener('cancel', (event) => {
    event.preventDefault();
    closeVideo();
  });

  const cellDialog = $('cell-dialog');
  const cellTitle = $('cell-title');
  const cellList = $('cell-list');

  function openCellRuns(title, runs) {
    cellTitle.textContent = title;
    cellList.innerHTML = '';
    runs.forEach((run) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'cell-run';
      const left = document.createElement('span');
      left.textContent = run.task;
      const right = document.createElement('span');
      right.textContent = run.rate.toFixed(1) + '% · n=' + run.n + ' · ' + run.ckpt;
      button.append(left, right);
      button.addEventListener('click', () => {
        if (cellDialog.open) cellDialog.close();
        openVideos({scope: 'eval', run: run.run}, run.label);
      });
      cellList.append(button);
    });
    if (!cellDialog.open) cellDialog.showModal();
    pauseAutoRefresh();
  }

  $('cell-close').addEventListener('click', () => {
    if (cellDialog.open) cellDialog.close();
    scheduleAutoRefresh();
  });
  cellDialog.addEventListener('cancel', (event) => {
    event.preventDefault();
    cellDialog.close();
    scheduleAutoRefresh();
  });

  function bindEvalCells() {
    document.querySelectorAll('.eval-open').forEach((button) => {
      button.addEventListener('click', () => {
        let runs = [];
        try {
          runs = JSON.parse(button.dataset.runs || '[]');
        } catch (error) {
          runs = [];
        }
        if (!runs.length) return;
        if (runs.length === 1) {
          openVideos({scope: 'eval', run: runs[0].run}, runs[0].label);
          return;
        }
        openCellRuns(button.dataset.label || '评测结果', runs);
      });
    });
  }

  function bindVideoButtons() {
    document.querySelectorAll('.video-btn, .video-card').forEach((button) => {
      button.addEventListener('click', () => {
        const query = {scope: button.dataset.scope || 'local'};
        if (button.dataset.task) query.task = button.dataset.task;
        if (button.dataset.run) query.run = button.dataset.run;
        openVideos(query, button.dataset.label || button.dataset.task);
      });
    });
  }

  function renderJobs(jobs) {
    const active = Object.entries(jobs).filter(([, job]) => job.state !== 'idle');
    jobsBox.innerHTML = '';
    active.sort((a, b) => String(b[1].started_at || '').localeCompare(String(a[1].started_at || '')));
    for (const [kind, job] of active) {
      const card = document.createElement('div');
      card.className = 'job';
      const head = document.createElement('div');
      head.className = 'head';
      const name = document.createElement('strong');
      name.textContent = jobLabels[kind] || kind;
      const state = document.createElement('span');
      state.className = 'state ' + job.state;
      state.textContent = job.state === 'running' ? '运行中' : (job.state === 'done' ? '完成' : '失败');
      const time = document.createElement('span');
      time.textContent = (job.started_at || '') + (job.finished_at ? ' → ' + job.finished_at : '');
      head.append(name, state, time);
      if (job.log) {
        const logPath = document.createElement('span');
        logPath.textContent = job.log;
        logPath.style.color = '#8b949e';
        head.append(logPath);
      }
      card.append(head);
      if (job.log_tail) {
        const pre = document.createElement('pre');
        pre.textContent = job.log_tail;
        card.append(pre);
      }
      jobsBox.append(card);
    }
    for (const id of actionIds) {
      const btn = $(id);
      if (!btn || !jobs[jobKinds[id]]) continue;
      btn.disabled = !serverMode || jobs[jobKinds[id]].state === 'running';
    }
    const modelBusy = jobs.download_model && jobs.download_model.state === 'running';
    document.querySelectorAll('.download-model-btn').forEach((btn) => {
      btn.disabled = !serverMode || modelBusy;
    });
    document.querySelectorAll('.video-btn, .video-card').forEach((btn) => {
      btn.disabled = !serverMode;
    });
  }

  async function pollJobs() {
    try {
      const data = await api('api/jobs');
      const was = serverMode;
      serverMode = true;
      renderJobs(data.jobs || {});
      if (was !== true) hint.textContent = '服务端模式：按钮会真正启动转换 / 上传任务';
    } catch (error) {
      const was = serverMode;
      serverMode = false;
      for (const id of actionIds) {
        const btn = $(id);
        if (btn) btn.disabled = true;
      }
      document.querySelectorAll('.download-model-btn').forEach((btn) => { btn.disabled = true; });
      document.querySelectorAll('.video-btn, .video-card').forEach((btn) => { btn.disabled = true; });
      if (was !== false) hint.textContent = '静态页面：仅能重新加载；用 --serve 打开才能转换 / 上传';
    }
  }

  async function startAction(button, planUrl, confirmUrl, title) {
    button.disabled = true;
    const resumeAutoRefresh = pauseAutoRefresh();
    const {dialog, ok, cancel} = planDialog(
      title,
      '正在生成计划（dry-run）…\\n首次查询远端可能需要 1–3 分钟，请稍候。',
      {pending: true}
    );
    const controller = new AbortController();
    let cancelled = false;
    let waited = 0;
    const timer = setInterval(() => {
      waited += 1;
      $('plan-text').textContent = '正在生成计划（dry-run）… 已等待 ' + waited
        + ' 秒\\n首次查询远端可能需要 1–3 分钟，请稍候。';
    }, 1000);
    cancel.onclick = () => {
      cancelled = true;
      controller.abort();
      dialog.close();
    };
    dialog.oncancel = (event) => {
      event.preventDefault();
      cancelled = true;
      controller.abort();
      dialog.close();
    };
    try {
      const plan = await api(planUrl, {method: 'POST', signal: controller.signal});
      clearInterval(timer);
      if (cancelled) return;
      if (plan.error) throw new Error(plan.error);
      if (!plan.count && !plan.plan) {
        $('plan-text').textContent = plan.message || '没有需要处理的内容';
        ok.style.display = 'none';
        cancel.textContent = '关闭';
        cancel.onclick = () => dialog.close();
        return;
      }
      let text = plan.message || '';
      if (plan.tasks) text += '\\n' + plan.tasks.join('\\n');
      if (plan.plan) text += '\\n' + plan.plan + '\\n\\n' + (plan.details || '');
      if (plan.running) text += '\\n\\n注意：当前已有同类任务在运行。';
      text += '\\n\\n确认开始？';
      $('plan-text').textContent = text;
      ok.style.display = '';
      ok.disabled = false;
      ok.textContent = '确认开始';
      cancel.textContent = '取消';
      const confirmed = await new Promise((resolve) => {
        const finish = (value) => {
          ok.onclick = null;
          cancel.onclick = null;
          dialog.oncancel = null;
          dialog.close();
          resolve(value);
        };
        ok.onclick = () => finish(true);
        cancel.onclick = () => finish(false);
        dialog.oncancel = (event) => { event.preventDefault(); finish(false); };
      });
      if (!confirmed) return;
      const started = await api(confirmUrl, {method: 'POST'});
      if (started.error) throw new Error(started.error);
      hint.textContent = started.message || '已启动';
    } catch (error) {
      clearInterval(timer);
      if (cancelled || error.name === 'AbortError') {
        hint.textContent = '已取消';
        return;
      }
      $('plan-text').textContent = '生成计划失败：' + error.message;
      ok.style.display = 'none';
      cancel.textContent = '关闭';
      cancel.onclick = () => dialog.close();
      hint.textContent = '生成计划失败：' + error.message;
    } finally {
      clearInterval(timer);
      if (dialog.open) dialog.addEventListener('close', resumeAutoRefresh, {once: true});
      else resumeAutoRefresh();
      pollJobs();
    }
  }

  if (refreshBtn) refreshBtn.addEventListener('click', async () => {
    if (autoRefreshTimer) {
      clearTimeout(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    document.querySelectorAll('meta[http-equiv="refresh"]').forEach((meta) => meta.remove());
    refreshBtn.disabled = true;
    hint.textContent = '正在刷新…';
    try {
      const data = await api('api/refresh', {method: 'POST'});
      hint.textContent = '已重新生成到 ' + data.generated_at;
    } catch (error) {
      hint.textContent = '静态页面：已重新加载（数据由 --watch 定时生成）';
    }
    setTimeout(() => location.reload(), 300);
  });

  const on = (id, fn) => {
    const el = $(id);
    if (el) el.addEventListener('click', fn);
  };
  on('export-v30-btn', (event) => {
    startAction(event.currentTarget, 'api/export/v30', 'api/export/v30?confirm=1', '导出 LeRobot v3.0');
  });
  on('export-v21-btn', (event) => {
    startAction(event.currentTarget, 'api/export/v21', 'api/export/v21?confirm=1', '从 v3.0 派生 LeRobot v2.1');
  });
  on('upload-hdf5-btn', (event) => {
    startAction(event.currentTarget, 'api/upload/hdf5', 'api/upload/hdf5?confirm=1', '上传 HDF5 到 ModelScope');
  });
  on('upload-lerobot-btn', (event) => {
    startAction(event.currentTarget, 'api/upload/lerobot', 'api/upload/lerobot?confirm=1', '上传 LeRobot 到 ModelScope');
  });
  on('download-model-btn', (event) => {
    startAction(event.currentTarget, 'api/download_model', 'api/download_model?confirm=1', '下载缺失模型');
  });
  document.querySelectorAll('.download-model-btn').forEach((btn) => {
    if (btn.id === 'download-model-btn') return;
    btn.addEventListener('click', () => downloadModel(btn));
  });
  bindVideoButtons();
  bindEvalCells();

  scheduleAutoRefresh();
  pollJobs();
  setInterval(pollJobs, 5000);
})();
</script>
"""
    detail_table = (
        "<h2>按任务查看</h2>"
        "<table><thead><tr><th>#</th><th>组</th><th>任务</th><th>层级</th>"
        "<th>模型</th><th>评测</th><th>评测布局</th></tr></thead>"
        f"<tbody>{''.join(detail_rows)}</tbody></table>"
    )
    nav_link = (
        '<a class="nav-link" href="SYNC_STATUS.html">← 返回数据同步</a>'
        if page == "models"
        else '<a class="nav-link" href="MODELS_EVAL.html">模型与评测 →</a>'
    )
    if page == "models":
        page_title = "RoboSafeJanus 模型与评测"
        page_meta = (
            f"生成时间 {html.escape(snapshot['generated_at'])}"
            f" · 模型 ModelScope <code>{html.escape(snapshot['model_repo_id'] or '未配置')}</code>"
            f"{model_note}"
            f"{f' · 每 {refresh} 秒刷新' if refresh else ''}"
        )
        page_toolbar = (
            '  <button id="refresh-btn" type="button">🔄 手动刷新</button>\n'
            '  <button id="download-model-btn" type="button" disabled>⬇️ 下载模型</button>\n'
            '  <span id="refresh-hint" class="hint">用 --serve 打开时按钮会在服务端执行任务</span>'
        )
        page_main = (
            f'<div class="cards">{detail_cards_html}</div>'
            f"{eval_matrix_html(snapshot)}"
            f"{model_matrix_html(snapshot)}"
            f"{model_chips_html(snapshot)}"
            '<details class="fold"><summary>逐条评测明细（run_id / 结果路径）</summary>'
            f"{strip_heading(evals_html)}</details>"
            '<details class="fold"><summary>按任务明细（模型 / 评测 / 布局）</summary>'
            f"{strip_heading(detail_table)}</details>"
            '<details class="fold"><summary>已训练模型（表格视图 / ModelScope 路径）</summary>'
            f"{strip_heading(models_table_html(snapshot))}</details>"
            '<details class="fold"><summary>评测布局（缺布局 / 不足的任务）</summary>'
            f"{strip_heading(layouts_html)}</details>"
        )
    else:
        page_title = "RoboSafeJanus 数据同步状态"
        page_meta = (
            f"生成时间 {html.escape(snapshot['generated_at'])}"
            f" · ModelScope HDF5 <code>{html.escape(snapshot['repo_id'] or '未配置')}</code>{remote_note}\n"
            f"· LeRobot ModelScope <code>{html.escape(snapshot['lerobot_repo_id'] or '未配置')}</code>"
            f"{lerobot_remote_note}"
            f"{f' · 每 {refresh} 秒刷新' if refresh else ''}"
        )
        page_toolbar = (
            '  <button id="refresh-btn" type="button">🔄 手动刷新</button>\n'
            '  <button id="export-v30-btn" type="button" disabled'
            ' title="从 HDF5 导出 LeRobot v3.0">🧩 导出 v3.0</button>\n'
            '  <button id="export-v21-btn" type="button" disabled'
            ' title="从已导出的 v3.0 派生 LeRobot v2.1">🧩 导出 v2.1</button>\n'
            '  <button id="upload-hdf5-btn" type="button" disabled>⬆️ 上传 HDF5</button>\n'
            '  <button id="upload-lerobot-btn" type="button" disabled>⬆️ 上传 LeRobot</button>\n'
            '  <span id="refresh-hint" class="hint">用 --serve 打开时按钮会在服务端执行任务</span>'
        )
        page_main = (
            f'<div class="cards">{cards_html}</div>'
            f'<div class="states">{states}<br>{json_states_text}<br>{code_states_text}</div>'
            "<table>\n<thead><tr><th>#</th><th>组</th><th>任务</th><th>JSON 对齐</th><th>专家脚本</th>"
            "<th>HDF5 本地</th><th>ModelScope HDF5</th><th>LeRobot v30</th><th>LeRobot v21</th>"
            "<th>视频</th><th>总状态</th></tr></thead>\n<tbody>\n"
            f"{''.join(rows)}"
            "\n</tbody>\n</table>\n"
            f"{videos_html}"
            f"{collectable_html}"
            '<p class="empty">专家脚本：<code>tasks/&lt;task&gt;.py</code>、'
            '<code>config/&lt;task&gt;.yml</code>、<code>experts/&lt;task&gt;.py</code> 三件齐全为 ✅ 就绪；'
            '<code>put_*</code> / <code>place_*</code> 别名视为同一任务。</p>'
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{page_title}</title>
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
  .nav-link {{ color: #58a6ff; font-size: 13px; text-decoration: none; margin-left: auto; white-space: nowrap; }}
  .nav-link:hover {{ text-decoration: underline; }}
  .download-model-btn {{ background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 4px 8px; font-size: 12px; cursor: pointer; white-space: nowrap; }}
  .download-model-btn:hover {{ background: #30363d; }}
  .download-model-btn:disabled {{ opacity: .6; cursor: progress; }}
  .jobs {{ margin: 0 0 16px; }}
  .job {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 13px; }}
  .job .head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .job .state {{ font-weight: 600; }}
  .job .state.running {{ color: #d29922; }}
  .job .state.done {{ color: #3fb950; }}
  .job .state.failed {{ color: #f85149; }}
  .job pre {{ margin: 8px 0 0; max-height: 220px; overflow: auto; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 8px; font-size: 11px; white-space: pre-wrap; }}
  dialog {{ background: #161b22; color: #e6edf3; border: 1px solid #30363d; border-radius: 8px; width: min(900px, calc(100% - 48px)); }}
  dialog::backdrop {{ background: rgba(1, 4, 9, .72); }}
  dialog h2 {{ margin: 0 0 10px; font-size: 16px; }}
  dialog pre {{ max-height: 50vh; overflow: auto; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px; font-size: 12px; white-space: pre-wrap; }}
  .dialog-actions {{ display: flex; justify-content: flex-end; gap: 10px; margin-top: 12px; }}
  .dialog-actions button {{ background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 7px 12px; font-size: 13px; cursor: pointer; }}
  #plan-ok {{ background: #1f6f43; border-color: #2ea043; color: #d2f8e2; }}
  h2 {{ font-size: 16px; margin: 22px 0 8px; }}
  .empty {{ color: #8b949e; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 9px 10px; border-bottom: 1px solid #21262d; text-align: left; font-size: 13px; vertical-align: top; }}
  th {{ background: #1c2128; color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; position: sticky; top: 0; }}
  tr:hover td {{ background: #1c2128; }}
  td.group-index {{ color: #8b949e; text-align: right; width: 32px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  small {{ color: #8b949e; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 12px; font-weight: 500; white-space: nowrap; }}
  .video-btn {{ background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 4px 8px; font-size: 12px; cursor: pointer; white-space: nowrap; }}
  .video-btn:hover {{ background: #30363d; }}
  .eval-item {{ margin-bottom: 6px; }}
  .eval-item:last-child {{ margin-bottom: 0; }}
  td.eval-cell {{ min-width: 200px; }}
  .eval-name {{ font-size: 11px; white-space: nowrap; }}
  .video-btn:disabled {{ opacity: .5; cursor: not-allowed; }}
  .video-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }}
  .video-card {{ display: flex; flex-direction: column; text-align: left; padding: 0; overflow: hidden; background: #161b22; border: 1px solid #30363d; border-radius: 8px; color: #e6edf3; cursor: pointer; }}
  .video-card:hover {{ border-color: #2ea043; }}
  .video-card:disabled {{ opacity: .55; cursor: not-allowed; }}
  .video-card img, .video-card-empty {{ width: 100%; aspect-ratio: 4 / 3; object-fit: cover; display: block; background: #0d1117; }}
  .video-card-empty {{ display: flex; align-items: center; justify-content: center; font-size: 28px; }}
  .video-card-body {{ display: flex; flex-direction: column; gap: 3px; padding: 8px 10px 10px; }}
  .video-card-body code {{ word-break: break-all; }}
  .video-card-body .cams {{ font-size: 11px; color: #6e7681; }}
  #video-dialog {{ width: min(1180px, calc(100% - 40px)); }}
  #cell-dialog {{ width: min(760px, calc(100% - 40px)); }}
  .cell-list {{ max-height: 55vh; overflow: auto; display: flex; flex-direction: column; gap: 6px; }}
  .cell-run {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; text-align: left; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #e6edf3; padding: 8px 10px; font-size: 12px; cursor: pointer; }}
  .cell-run:hover {{ border-color: #2ea043; }}
  .cell-run span:last-child {{ color: #8b949e; white-space: nowrap; }}
  .video-layout {{ display: grid; grid-template-columns: 240px 1fr; gap: 12px; }}
  .video-list {{ max-height: 62vh; overflow: auto; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 6px; display: flex; flex-direction: column; gap: 4px; }}
  .video-ep {{ display: flex; justify-content: space-between; gap: 8px; background: #161b22; border: 1px solid #30363d; border-radius: 5px; color: #e6edf3; padding: 5px 8px; font-size: 12px; cursor: pointer; text-align: left; }}
  .video-ep:hover, .video-ep.active {{ border-color: #2ea043; background: #1c2b22; }}
  .video-stage video {{ width: 100%; max-height: 62vh; background: #000; border-radius: 6px; }}
  .video-cams {{ display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
  .video-cams button {{ background: #21262d; border: 1px solid #30363d; border-radius: 999px; color: #c9d1d9; padding: 4px 10px; font-size: 12px; cursor: pointer; }}
  .video-cams button.active {{ border-color: #2ea043; color: #d2f8e2; background: #1f6f43; }}
  h2 small {{ font-weight: 400; color: #6e7681; font-size: 12px; margin-left: 6px; }}
  .legend {{ display: flex; gap: 8px; align-items: center; color: #8b949e; font-size: 12px; margin: 0 0 8px; flex-wrap: wrap; }}
  .legend-item {{ display: inline-block; padding: 2px 8px; border-radius: 4px; color: #e6edf3; font-size: 11px; }}
  .legend-item.lvl-safe {{ background: rgba(46, 160, 67, .38); }}
  .legend-item.lvl-risky {{ background: rgba(248, 81, 73, .38); }}
  .legend-note {{ color: #6e7681; }}
  .policy-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
  .matrix-wrap {{ overflow: auto; max-height: 76vh; border: 1px solid #30363d; border-radius: 8px; background: #0d1117; margin-bottom: 10px; }}
  table.matrix {{ border-collapse: separate; border-spacing: 0; width: 100%; font-size: 12px; }}
  table.matrix th, table.matrix td {{ border-bottom: 1px solid #21262d; border-right: 1px solid #21262d; padding: 0; text-align: center; vertical-align: middle; }}
  table.matrix thead th {{ position: sticky; top: 0; z-index: 3; background: #1c2128; color: #8b949e; font-weight: 500; font-size: 11px; text-transform: none; letter-spacing: 0; padding: 6px 8px; white-space: nowrap; }}
  table.matrix thead tr.matrix-head-1 th {{ top: 0; height: 28px; }}
  table.matrix thead tr.matrix-head-1 th.group-head {{ background: #21262d; color: #c9d1d9; white-space: normal; word-break: break-all; line-height: 1.2; max-width: 200px; }}
  table.matrix thead tr.matrix-head-2 th {{ top: 28px; font-size: 11px; }}
  table.matrix thead tr.matrix-head-2 th.lvl-head.lvl-safe {{ background: rgba(46, 160, 67, .25); color: #7ee787; }}
  table.matrix thead tr.matrix-head-2 th.lvl-head.lvl-risky {{ background: rgba(248, 81, 73, .25); color: #ffa198; }}
  table.matrix th.rowhead {{ position: sticky; left: 0; z-index: 2; background: #161b22; text-align: left; padding: 6px 10px; min-width: 170px; max-width: 220px; text-transform: none; letter-spacing: 0; border-right: 2px solid #30363d; font-weight: 400; }}
  table.matrix th.rowhead .badge {{ font-size: 10px; padding: 1px 6px; }}
  table.matrix thead th.corner {{ left: 0; z-index: 4; text-align: left; min-width: 170px; vertical-align: bottom; color: #8b949e; }}
  table.matrix tr.group-row td {{ background: #1c2128; color: #c9d1d9; text-align: left; padding: 6px 10px; font-size: 12px; }}
  .group-count {{ color: #6e7681; font-size: 11px; margin-left: 8px; }}
  table.matrix td.cell {{ padding: 0; height: 40px; min-width: 72px; }}
  .eval-open {{ display: block; width: 100%; height: 100%; padding: 4px 6px; border: 0; background: transparent; color: #e6edf3; font: inherit; cursor: pointer; }}
  .eval-open .rate {{ display: block; font-size: 13px; font-weight: 700; line-height: 1.15; }}
  .eval-open small {{ display: block; font-size: 10px; color: rgba(230, 237, 243, .72); }}
  .eval-open:hover {{ outline: 1px solid #58a6ff; outline-offset: -1px; }}
  td.cell-empty, .cell-empty {{ background: #0d1117; color: #484f58; }}
  table.matrix tr.matrix-foot td, table.matrix tr.matrix-foot th {{ position: sticky; bottom: 0; background: #1c2128; z-index: 2; }}
  table.matrix tr.matrix-foot th.rowhead {{ z-index: 3; }}
  table.matrix td.model-cell {{ padding: 5px 6px; white-space: nowrap; }}
  table.matrix td.model-cell .badge {{ font-size: 11px; }}
  .model-chips {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; margin-bottom: 14px; }}
  .model-chip {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 12px; font-size: 12px; }}
  .model-chip .top {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
  .model-chip .chip-key {{ margin-top: 6px; word-break: break-all; }}
  .model-chip .sub {{ color: #8b949e; margin-top: 4px; }}
  .model-chip .actions {{ display: flex; align-items: center; gap: 10px; margin-top: 8px; }}
  .model-chip .chip-link {{ color: #58a6ff; text-decoration: none; font-size: 12px; }}
  .model-chip .chip-link:hover {{ text-decoration: underline; }}
  details.fold {{ margin-top: 14px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 12px; }}
  details.fold summary {{ cursor: pointer; color: #8b949e; font-size: 13px; }}
  details.fold[open] summary {{ margin-bottom: 8px; }}
  .good {{ background: #1f6f43; color: #d2f8e2; }}
  .warn {{ background: #7a5b12; color: #fff3c4; }}
  .bad {{ background: #7d2b2b; color: #ffdcdc; }}
</style>
</head>
<body data-refresh="{refresh}">
<h1>{page_title}</h1>
<div class="meta">{page_meta}</div>
<div class="toolbar">
{page_toolbar}
{nav_link}
</div>
<div id="jobs" class="jobs"></div>
{page_main}
<dialog id="plan-dialog">
  <h2 id="plan-title"></h2>
  <pre id="plan-text"></pre>
  <div class="dialog-actions">
    <button id="plan-cancel" type="button">取消</button>
    <button id="plan-ok" type="button">确认开始</button>
  </div>
</dialog>
<dialog id="video-dialog">
  <h2 id="video-title">视频</h2>
  <div class="video-cams" id="video-cams"></div>
  <div class="video-layout">
    <div class="video-list" id="video-list"></div>
    <div class="video-stage">
      <video id="video-player" controls autoplay playsinline preload="metadata"></video>
      <p class="empty" id="video-meta"></p>
    </div>
  </div>
  <div class="dialog-actions">
    <button id="video-close" type="button">关闭</button>
  </div>
</dialog>
<dialog id="cell-dialog">
  <h2 id="cell-title">评测结果</h2>
  <div class="cell-list" id="cell-list"></div>
  <div class="dialog-actions">
    <button id="cell-close" type="button">关闭</button>
  </div>
</dialog>
{refresh_script}
</body>
</html>
"""


class StatusRequestHandler(SimpleHTTPRequestHandler):
    """静态文件 + 立即刷新 / 启动 LeRobot 转换、ModelScope 上传和视频播放。"""

    refresh_callback = None
    job_config: dict = {}
    snapshot: dict = {}

    def _path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/")

    def do_GET(self):
        path = self._path()
        if path == "/api/refresh":
            return self._refresh()
        if path == "/api/jobs":
            return self._json({"jobs": jobs_snapshot()})
        if path == "/api/videos":
            return self._videos()
        if path == "/media":
            return self._media()
        if path in ("", "/"):
            self.send_response(302)
            self.send_header("Location", "/SYNC_STATUS.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        path = self._path()
        if path == "/api/refresh":
            return self._refresh()
        if path == "/api/export/v30":
            return self._start_export("v30")
        if path == "/api/export/v21":
            return self._start_export("v21")
        if path == "/api/upload/hdf5":
            return self._upload("hdf5")
        if path == "/api/upload/lerobot":
            return self._upload("lerobot")
        if path == "/api/download_model":
            return self._download_model()
        self.send_error(404)

    def _videos(self):
        config = type(self).job_config
        query = parse_qs(urlparse(self.path).query)
        scope = (query.get("scope") or ["local"])[0]
        if scope == "eval":
            rel = (query.get("run") or [""])[0]
            root = Path(config.get("eval_dir") or DEFAULT_EVAL)
            directory = resolve_under(ROOT_DIR, rel)
            if not inside(root, directory) or not directory.is_dir():
                return self._json({"error": f"评测目录不存在: {rel}"}, status=404)
            title = directory.parent.name
            data = list_video_files([directory])
        else:
            task = (query.get("task") or [""])[0]
            snapshot = type(self).snapshot or {}
            row = next((item for item in snapshot.get("tasks", []) if item["name"] == task), None)
            local = (row or {}).get("local") or {}
            if not local.get("path"):
                return self._json({"error": f"本地没有任务数据: {task}"}, status=404)
            root = Path(config.get("local_dir") or DEFAULT_LOCAL)
            task_path = resolve_under(ROOT_DIR, local["path"])
            if not inside(root, task_path) or not task_path.is_dir():
                return self._json({"error": f"任务目录不存在: {local['path']}"}, status=404)
            directories = sorted(task_path.glob("*/preview_video"))
            title = task
            data = list_video_files([path for path in directories if path.is_dir()])
        entries = data["entries"]
        return self._json({
            "task": title,
            "scope": scope,
            "cameras": data["cameras"],
            "total": data["total"],
            "entries": entries[:800],
            "truncated": len(entries) > 800,
        })

    def _media(self):
        query = parse_qs(urlparse(self.path).query)
        rel = (query.get("path") or [""])[0]
        if not rel:
            return self.send_error(400, "missing path")
        target = resolve_under(ROOT_DIR, rel)
        config = type(self).job_config
        allowed = [
            Path(config.get("local_dir") or DEFAULT_LOCAL),
            Path(config.get("eval_dir") or DEFAULT_EVAL),
        ]
        if target is None or not target.is_file() or target.suffix.lower() not in MEDIA_SUFFIXES:
            return self.send_error(404)
        resolved = target.resolve()
        if not any(resolved == root.resolve() or root.resolve() in resolved.parents for root in allowed):
            return self.send_error(403)

        size = resolved.stat().st_size
        start, end = 0, max(size - 1, 0)
        status = 200
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            spec = range_header[6:].split(",", 1)[0].strip()
            first, _, last = spec.partition("-")
            try:
                if first:
                    start = int(first)
                if last:
                    end = min(int(last), size - 1)
            except ValueError:
                return self.send_error(400, "bad range")
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206

        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        remaining = end - start + 1
        try:
            with resolved.open("rb") as handle:
                handle.seek(start)
                while remaining > 0:
                    chunk = handle.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _refresh(self):
        try:
            snapshot = type(self).refresh_callback()
        except Exception as error:  # noqa: BLE001 - 把异常回传给页面
            return self._json({"error": str(error)}, status=500)
        return self._json({"generated_at": snapshot["generated_at"]})

    def _start_export(self, version: str):
        config = type(self).job_config
        job_kind = EXPORT_JOBS[version]
        label = JOB_LABELS[job_kind]
        # v3.0 导出会重写 v2.1 派生的来源目录,两者不能同时跑
        other = EXPORT_JOBS["v21" if version == "v30" else "v30"]
        tasks, blocked = pending_export_tasks(config["local_dir"], config["lerobot_dir"], version)
        blocked_note = (
            f"；{len(blocked)} 个任务的 v3.0 未导完，先导 v3.0：{' '.join(blocked[:20])}"
            + (" …" if len(blocked) > 20 else "")
            if blocked else ""
        )
        if "confirm=1" in self.path:
            with JOBS_LOCK:
                for busy in ("upload_lerobot", other):
                    if JOBS[busy]["state"] == "running":
                        return self._json(
                            {"error": f"{JOB_LABELS[busy]}正在运行，等它结束后再{label}"}, status=409
                        )
            if not tasks:
                return self._json({"started": False, "message": f"没有需要{label}的任务{blocked_note}"})
            try:
                launch_job(
                    job_kind, export_runner(tasks, ROOT_DIR, version),
                    config["out_dir"], config["rebuild"],
                )
            except RuntimeError as error:
                return self._json({"error": str(error)}, status=409)
            return self._json({
                "started": True,
                "message": f"已启动 {len(tasks)} 个任务的{label}",
            })
        if not tasks:
            return self._json({"count": 0, "message": f"没有需要{label}的任务{blocked_note}"})
        with JOBS_LOCK:
            running = JOBS[job_kind]["state"] == "running"
        source = "从 HDF5 导出" if version == "v30" else "从已有 v3.0 派生（CRF 30 重编码）"
        return self._json({
            "count": len(tasks),
            "tasks": tasks[:200],
            "running": running,
            "message": f"将{source} {len(tasks)} 个任务，已有的会覆盖{blocked_note}",
        })

    def _upload(self, kind: str):
        config = type(self).job_config
        if kind == "hdf5":
            repo_id = config.get("repo_id")
            script = "src/data_collect/upload_to_modelscope.py"
            job_kind = "upload_hdf5"
            label = "HDF5"
        else:
            repo_id = config.get("lerobot_repo_id")
            script = "src/data_collect/upload_lerobot_to_modelscope.py"
            job_kind = "upload_lerobot"
            label = "LeRobot"
        if not repo_id:
            return self._json({"error": f"未配置 {label} ModelScope 仓库"}, status=400)
        command = [sys.executable, "-u", script, "--repo_id", repo_id]
        if "confirm=1" in self.path:
            if job_kind == "upload_lerobot":
                with JOBS_LOCK:
                    busy = [kind for kind in EXPORT_JOBS.values() if JOBS[kind]["state"] == "running"]
                if busy:
                    return self._json(
                        {"error": f"{JOB_LABELS[busy[0]]}正在运行，等它结束后再上传"}, status=409
                    )
            try:
                launch_job(
                    job_kind, shell_runner(command, ROOT_DIR),
                    config["out_dir"], config["rebuild"],
                )
            except RuntimeError as error:
                return self._json({"error": str(error)}, status=409)
            return self._json({"started": True, "message": f"已启动 {label} 上传"})
        code, output = run_dry([*command, "--dry_run"], ROOT_DIR)
        if code != 0:
            reason = next(
                (
                    line.strip() for line in output.splitlines()
                    if "LeRobot 转换" in line and "运行" in line
                ),
                "dry-run 失败",
            )
            return self._json(
                {"error": reason, "details": output},
                status=409 if code == 2 else 500,
            )
        total = next(
            (
                line for line in reversed(output.splitlines())
                if line.startswith("[upload_to_modelscope] total:")
                or line.startswith("[upload_lerobot] total:")
            ),
            "",
        )
        with JOBS_LOCK:
            running = JOBS[job_kind]["state"] == "running"
        return self._json({
            "label": label,
            "repo_id": repo_id,
            "plan": total,
            "details": output,
            "running": running,
        })

    def _download_model(self):
        config = type(self).job_config
        repo_id = config.get("model_repo_id")
        if not repo_id:
            return self._json({"error": "未配置模型仓库"}, status=400)
        query = parse_qs(urlparse(self.path).query)
        policy = (query.get("policy") or [None])[0]
        key = (query.get("task") or [None])[0]
        models, error = collect_models(repo_id, task_groups(), config["model_local_dir"])
        if error:
            return self._json({"error": error}, status=500)
        if policy and key:
            selected = [model for model in models if model["policy"] == policy and model["key"] == key]
            if not selected:
                return self._json({"error": f"模型不存在: {policy}/{key}"}, status=404)
        else:
            selected = [model for model in models if model["needs_download"]]
        if not selected:
            return self._json({"count": 0, "message": "所有模型的最新 step 都已下载"})
        tasks = [
            f"{model['policy']}/{model['key']} · {human_bytes(model['size'])} · 本地: "
            + (f"step {', '.join(model['local_steps'])}" if model["local_steps"] else "无")
            for model in selected
        ]
        if "confirm=1" in self.path:
            try:
                launch_job(
                    "download_model", download_models_runner(selected, ROOT_DIR),
                    config["out_dir"], config["rebuild"],
                )
            except RuntimeError as error:
                return self._json({"error": str(error)}, status=409)
            return self._json({
                "started": True,
                "message": f"已启动 {len(selected)} 个模型下载",
            })
        return self._json({
            "count": len(selected),
            "tasks": tasks[:100],
            "message": f"将下载 {len(selected)} 个模型的最新 step"
                       f"（已下载的文件会跳过）",
        })

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):  # noqa: A002 - 与父类签名一致
        pass


def make_thumbnails(snapshot: dict, out_dir: Path) -> dict[str, str]:
    """给每个任务的头部相机预览视频抽一帧,缓存到 ``<out_dir>/video_thumbs/``。"""
    target_dir = out_dir / THUMB_DIR
    thumbs: dict[str, str] = {}
    rows = [row for row in snapshot["tasks"] if (row.get("local") or {}).get("videos")]
    if not rows or not shutil.which("ffmpeg"):
        return thumbs
    target_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for row in rows:
        sample = (row["local"]["videos"] or {}).get("sample")
        if not sample:
            continue
        source = Path(sample["path"])
        if not source.is_absolute():
            source = ROOT_DIR / source
        if not source.is_file():
            continue
        name = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', row['name'])}__{sample['cam']}.jpg"
        target = target_dir / name
        try:
            fresh = target.is_file() and target.stat().st_mtime >= source.stat().st_mtime
        except OSError:
            fresh = False
        if not fresh:
            for seek in ("2", "0"):
                try:
                    code = subprocess.call(
                        ["ffmpeg", "-loglevel", "error", "-y", "-ss", seek, "-i", str(source),
                         "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", str(target)],
                        timeout=60,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    code = 1
                if code == 0 and target.is_file() and target.stat().st_size:
                    break
                target.unlink(missing_ok=True)
            if not target.is_file():
                continue
        used.add(name)
        thumbs[row["name"]] = name
    for stale in target_dir.glob("*.jpg"):
        if stale.name not in used:
            stale.unlink(missing_ok=True)
    return thumbs


def write_outputs(snapshot: dict, out_dir: Path, refresh: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = make_thumbnails(snapshot, out_dir)
    (out_dir / "SYNC_STATUS.md").write_text(markdown(snapshot, refresh), encoding="utf-8")
    (out_dir / "SYNC_STATUS.html").write_text(html_page(snapshot, refresh, thumbs), encoding="utf-8")
    (out_dir / "MODELS_EVAL.html").write_text(
        html_page(snapshot, refresh, thumbs, page="models"), encoding="utf-8",
    )
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
    parser.add_argument("--model_repo_id", default=None,
                        help="trained-model ModelScope repo; default MS_MODEL_REPO_ID or <owner>/RoboSafeJanus-Model")
    parser.add_argument("--model_local_dir", default=str(DEFAULT_MODEL),
                        help="local trained-model dir, mirrors the model repo")
    parser.add_argument("--eval_dir", default=str(DEFAULT_EVAL),
                        help="local evaluation results dir")
    parser.add_argument("--eval_layout_dir", default=str(DEFAULT_LAYOUT),
                        help="frozen eval layouts dir <benchmark>/<config>; default Assets/Eval_Layout/RoboDojo/arx_x5")
    parser.add_argument("--eval_seed", type=int, default=0,
                        help="eval layout seed subdirectory (default: 0)")
    parser.add_argument("--eval_num", type=int, default=50,
                        help="layouts required per task (default: 50)")
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
    model_local_dir = Path(args.model_local_dir).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    layout_dir = Path(args.eval_layout_dir).absolute()
    out_dir = Path(args.out_dir).resolve()
    repo_id = None if args.no_remote else (args.repo_id or infer_repo_id(local_dir))
    lerobot_repo_id = None
    model_repo_id = None
    if not args.no_remote:
        lerobot_repo_id = (
            args.lerobot_repo_id
            or os.environ.get("MS_LEROBOT_REPO_ID")
            or (f"{repo_id}-Lerobot" if repo_id else None)
        )
        model_repo_id = args.model_repo_id or os.environ.get("MS_MODEL_REPO_ID")
        if not model_repo_id and repo_id and "/" in repo_id:
            owner = repo_id.split("/", 1)[0]
            model_repo_id = f"{owner}/RoboSafeJanus-Model"
    refresh = args.watch if args.watch > 0 else 0
    if refresh and refresh < 5:
        parser.error("--watch must be at least 5 seconds")

    build_lock = threading.Lock()

    def rebuild() -> dict:
        with build_lock:
            snapshot = build_snapshot(
                local_dir, lerobot_dir, repo_id, args.repo_type, args.prefix,
                lerobot_repo_id, model_repo_id, model_local_dir, eval_dir,
                layout_dir, args.eval_seed, args.eval_num,
            )
            write_outputs(snapshot, out_dir, refresh)
            StatusRequestHandler.snapshot = snapshot
            return snapshot

    server = None
    if args.serve:
        StatusRequestHandler.refresh_callback = rebuild
        StatusRequestHandler.job_config = {
            "repo_id": repo_id,
            "lerobot_repo_id": lerobot_repo_id,
            "model_repo_id": model_repo_id,
            "local_dir": local_dir,
            "lerobot_dir": lerobot_dir,
            "model_local_dir": model_local_dir,
            "eval_dir": eval_dir,
            "out_dir": out_dir,
            "rebuild": rebuild,
        }
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
