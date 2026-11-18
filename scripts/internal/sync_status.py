#!/usr/bin/env python3
"""Generate a live sync-status table for local data, ModelScope, and LeRobot exports.

Writes these files under ``data/`` by default:

    data/PIPELINE.html      the dashboard (default page with --serve): ONE wide table, one row per task --
                            pipeline stages seed .. eval with HDF5 / LeRobot / video / model / eval data columns
                            in between (pipeline_page.py, fragment data/pipeline/_fragments/table.html);
                            click a status cell to open its review entry (approve / reject) under the group;
                            "▶ Claude" buttons in the build cells (assets .. expert) only show a prompt to paste
                            into an interactive Claude Code session; "采集" / "转换" / "上传" in collect / exchange /
                            upload (not reviewed) open a plan dialog first (scripts/internal/group_jobs.py)
    data/TASK_BUILD.html, SYNC_STATUS.html, MODELS_EVAL.html
                            small pages that jump to the matching tab (old links keep working)
    data/SYNC_STATUS.md     human-readable Markdown table
    data/SYNC_STATUS.json   machine-readable snapshot

Usage::

    python scripts/internal/sync_status.py                     # one snapshot
    python scripts/internal/sync_status.py --watch 30          # refresh every 30 s
    python scripts/internal/sync_status.py --no_remote         # local + LeRobot only
    python scripts/internal/sync_status.py --repo_id user/repo
    python scripts/internal/sync_status.py --lerobot_repo_id user/lerobot-repo
    python scripts/internal/sync_status.py --model_repo_id user/model-repo
    python scripts/internal/sync_status.py --wandb_dir data/RoboSafeJanus-Wandb

W&B runs live next to the checkpoints in the model repo
(``<policy>/<task>/<run>/wandb/<run-dir>/``). The models page downloads them into
``--wandb_dir`` and uploads them to the official W&B with ``wandb sync``; that needs
``WANDB_API_KEY`` (or ``data/.cache/wandb/api_key``) and optionally ``--wandb_entity``.

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
import traceback
from urllib.parse import parse_qs, urlparse

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
DEFAULT_WANDB = ROOT_DIR / "data" / "RoboSafeJanus-Wandb"
# 上传到 W&B 官方要走本机 clash 的混合端口;WANDB_PROXY="" 表示不走代理
DEFAULT_WANDB_PROXY = "http://127.0.0.1:49623"
WANDB_API_KEY_FILE = ROOT_DIR / "data" / ".cache" / "wandb" / "api_key"
DEFAULT_EVAL = ROOT_DIR / "eval_result"
DEFAULT_LAYOUT = ROOT_DIR / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5"
DEFAULT_OUT = ROOT_DIR / "data"
LEROBOT_VERSIONS = ("v30", "v21")
CAM_ORDER = ("cam_head", "cam_left_wrist", "cam_right_wrist")
VIDEO_RE = re.compile(r"^episode_(\d+)_(cam_[a-z0-9_]+?)(?:_(success|fail))?\.mp4$")
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
    return {
        "files": sum(cams.values()),
        "bytes": total_bytes,
        "episodes": len(episodes),
        "cams": {cam: cams[cam] for cam in sorted(cams, key=cam_rank)},
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


def attach_eval_safety(entries: list[dict], result_file: Path) -> None:
    """S1 变体评测:按 ``_result.json`` 的 details(键 = 视频的 episode 序号)给视频条目加
    ``violation`` / ``violated_by``(安全 / 违规);S1.0 的 details 没有 violation,不加。"""
    try:
        details = json.loads(result_file.read_text(encoding="utf-8")).get("details") or {}
    except (OSError, ValueError):
        return
    for entry in entries:
        detail = details.get(str(entry["episode"]))
        if isinstance(detail, dict) and "violation" in detail:
            entry["violation"] = bool(detail["violation"])
            entry["violated_by"] = list(detail.get("violated_by") or [])
LEVEL_ORDER = {"safe_tasks": 0, "risky_tasks": 1}

# 脚本专家/任务实现所在目录:三类文件齐全才算"可采集"。
CODE_ROOT = ROOT_DIR / "task" / "RoboSafeJanus"
CODE_DIRS = {
    "impl": ("tasks", ".py"),
    "config": ("config", ".yml"),
    "expert": ("experts", ".py"),
}
CODE_LABELS = {"impl": "实现", "config": "配置", "expert": "专家"}

JOB_LABELS = {
    "download_model": "模型下载",
    "sync_wandb": "W&B 数据同步",
    "pipeline": "流水线检查",
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
    for kind in JOB_LABELS
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


def running_model_downloads() -> set[tuple[str, str]]:
    """正在跑的 ``download_models_from_modelscope.py`` 的 ``(policy, task)``(扫 /proc,看板重启后没记录的下载也算)。"""
    found = set()
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = cmdline.read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if not any(a.endswith("download_models_from_modelscope.py") for a in args):
            continue
        value = {flag: args[i + 1] for i, flag in enumerate(args[:-1]) if flag in ("--policy", "--task")}
        found.add((value.get("--policy", "*"), value.get("--task", "*")))
    return found


def model_downloading(model: dict, running: set[tuple[str, str]]) -> bool:
    return any(p in (model["policy"], "*") and t in (model["key"], "*") for p, t in running)


def step_dir_complete(step_dir: Path, remote: list[int] | None) -> tuple[bool, int]:
    """本地一个 checkpoint step 是否下载完整:``(完整, 本地字节数)``。

    还有 ModelScope 的 ``*.incomplete`` 临时文件,或者文件数 / 字节数少于远端,都算没下完
    (只看目录在不在会把正在下载的模型当成已下载)。"""
    files, size = 0, 0
    for dirpath, _, filenames in os.walk(step_dir):
        for name in filenames:
            if name.endswith(".incomplete"):
                return False, size
            if name in LOCAL_IGNORE_NAMES:
                continue
            try:
                size += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
            files += 1
    if remote is not None and (files < remote[0] or size < remote[1]):
        return False, size
    return files > 0, size


# checkpoint step 目录:纯数字(Pi_0 / Pi_05 的 2999)、step_<n> / global_step_<n>(LingBot_VLA)
# 或 checkpoint-<n>(GR00T_N17);和 scripts/download_models_from_modelscope.py 的 STEP_RE 一致。
STEP_RE = re.compile(r"^(?:(?:global_)?step|checkpoint)[_-]?\d+$|^\d+$", re.IGNORECASE)


def is_step(name: str) -> bool:
    return bool(STEP_RE.match(name))


def step_num(name: str) -> int:
    """step 目录名里的数字(排序 / 比新旧用);``global_step_3053`` -> 3053。"""
    digits = re.findall(r"\d+", name or "")
    return int(digits[-1]) if digits else 0


def collect_models(
    repo_id: str | None,
    json_tasks: dict[str, tuple[str, str]],
    local_root: Path | None = None,
    wandb_root: Path | None = None,
) -> tuple[list[dict], list[dict], str | None]:
    """扫描 ModelScope model 仓库。

    约定布局:``<policy>/<task|group_level>/<run>/<step>/...``。也接受更浅的
    ``<policy>/<task>/<step>/...``。训练机同时传上来的
    ``<policy>/<task>/<run>/wandb/<run-dir>/...`` 单独汇总成 W&B 记录,不计入模型大小。

    返回 ``(models, wandb_runs, error)``。
    """
    if not repo_id:
        return [], [], "未配置模型仓库"
    try:
        from modelscope_hub import HubApi
    except ImportError:
        return [], [], "未安装 modelscope_hub"
    try:
        files = HubApi().list_repo_files(repo_id, "model", recursive=True)
    except Exception as error:  # noqa: BLE001 - 看板要能在远端异常时继续
        return [], [], f"{type(error).__name__}: {error}"

    entries: dict[str, dict] = {}
    wandb_entries: dict[tuple[str, str, str, str], dict] = {}
    for item in files:
        if getattr(item, "type", "blob") == "tree":
            continue
        parts = item.path.split("/")
        if len(parts) < 3:
            continue
        if "wandb" in parts:
            index = parts.index("wandb")
            if index >= 3 and len(parts) > index + 2:
                policy, key, run = parts[:3]
                run_dir = parts[index + 1]
                wandb_entry = wandb_entries.setdefault(
                    (policy, key, run, run_dir),
                    {"policy": policy, "key": key, "run": run, "run_dir": run_dir,
                     "files": 0, "size": 0, "sizes": {}},
                )
                wandb_entry["files"] += 1
                wandb_entry["size"] += getattr(item, "size", 0) or 0
                wandb_entry["sizes"]["/".join(parts[index + 2:])] = getattr(item, "size", 0) or 0
            continue
        policy, key = parts[0], parts[1]
        directory = f"{policy}/{key}"
        entry = entries.setdefault(
            directory,
            {"policy": policy, "key": key, "path": directory,
             "runs": set(), "steps": set(), "files": 0, "size": 0, "step_remote": {}},
        )
        entry["files"] += 1
        entry["size"] += getattr(item, "size", 0) or 0
        step = next((part for part in parts[2:-1] if is_step(part)), None)
        if step:
            entry["steps"].add(step)
        run = parts[2] if len(parts) >= 4 and not is_step(parts[2]) else None
        if run:
            entry["runs"].add(run)
            if step:  # 每个 run 的每个 step 远端有多少文件 / 字节,用来判断本地是否下载完整
                remote = entry["step_remote"].setdefault((run, step), [0, 0])
                remote[0] += 1
                remote[1] += getattr(item, "size", 0) or 0

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
        local_steps: set[str] = set()  # 下载完整的 step
        partial_steps: set[str] = set()  # 目录在但没下完(还在下载 / 中断过)
        local_size = 0
        if local_root is not None:
            for run in sorted(entry["runs"]):
                run_dir = local_root / entry["policy"] / entry["key"] / run
                if not run_dir.is_dir():
                    continue
                for step_dir in sorted(run_dir.iterdir()):
                    if step_dir.is_dir() and is_step(step_dir.name):
                        complete, size = step_dir_complete(step_dir, entry["step_remote"].get((run, step_dir.name)))
                        (local_steps if complete else partial_steps).add(step_dir.name)
                        local_size += size
        partial_steps -= local_steps
        steps = sorted(entry["steps"], key=step_num)
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
            "local_steps": sorted(local_steps, key=step_num),
            "partial_steps": sorted(partial_steps, key=step_num),
            "local_size": local_size,
            "downloaded": bool(local_steps),
            "needs_download": bool(latest_step) and latest_step not in local_steps,
        })
    models.sort(key=lambda model: (model["policy"], model["key"]))
    return models, collect_wandb_runs(wandb_entries, wandb_root, models), None


def wandb_api_key() -> str | None:
    """W&B 官方凭证:优先 ``WANDB_API_KEY``,其次 ``data/.cache/wandb/api_key``。"""
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key
    try:
        key = WANDB_API_KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def find_wandb_cli(venv_dir: Path | None = None) -> tuple[list[str], str]:
    """返回 (wandb CLI 命令前缀, 说明);没有可用 CLI 时命令为空。"""
    override = os.environ.get("WANDB_CLI")
    if override:
        return override.split(), f"WANDB_CLI={override}"
    exe = shutil.which("wandb")
    if exe:
        return [exe], exe
    venv = (venv_dir or (DEFAULT_WANDB.parent / ".cache" / "wandb" / "venv")) / "bin" / "wandb"
    if venv.is_file():
        return [str(venv)], str(venv)
    if shutil.which("uv"):
        return [], f"首次上传会用 uv 安装 wandb 到 {rel_path(venv.parent.parent)}"
    return [], "没找到 wandb CLI(uv 也不可用),请先 pip install wandb"


def collect_wandb_runs(
    wandb_entries: dict[tuple[str, str, str, str], dict],
    wandb_root: Path | None,
    models: list[dict] | None = None,
) -> list[dict]:
    """把 ``<policy>/<task>/<run>/wandb/<run-dir>/`` 汇成 W&B 记录(含本地/上传状态)。

    传了 ``models`` 时,模型仓库里没有 wandb 目录的 run 也会列出来(``has_wandb=False``),
    这样能直接看出哪些训练没把 wandb 一起上传。
    """
    state_file = (wandb_root or DEFAULT_WANDB).parent / ".cache" / "wandb" / "synced.json"
    state = load_json_cache(state_file)
    runs = []
    for (policy, key, run, run_dir), entry in sorted(wandb_entries.items()):
        rel = f"{policy}/{key}/{run}/wandb/{run_dir}"
        local_dir = (wandb_root or DEFAULT_WANDB) / policy / key / run / "wandb" / run_dir
        local_files, local_bytes, local_mtime = dir_size(local_dir)
        # 每个远端文件本地都有且大小一样才算下载完(只比文件个数的话,远端换了文件也看不出来)
        sizes = entry.get("sizes") or {}
        same = sum(1 for sub, size in sizes.items()
                   if (local_dir / sub).is_file() and (local_dir / sub).stat().st_size == size)
        synced = state.get(rel) or {}
        remote_wandb = sum(size for sub, size in sizes.items() if sub.endswith(".wandb"))
        # 上传之后远端 .wandb 换了(训练机重新上传了完整文件):算没上传,要重新同步
        stale = bool(synced.get("synced_at")) and "wandb_bytes" in synced and remote_wandb != synced["wandb_bytes"]
        runs.append({
            "policy": policy,
            "key": key,
            "run": run,
            "run_dir": run_dir,
            "path": rel,
            "has_wandb": True,
            "mode": "offline" if run_dir.startswith("offline-run-") else "online",
            "files": entry["files"],
            "size": entry["size"],
            "local": {
                "path": rel_path(local_dir),
                "files": local_files,
                "bytes": local_bytes,
                "mtime": local_mtime,
                "downloaded": same >= entry["files"] > 0,
                "partial": 0 < same < entry["files"] or (same == 0 and local_files > 0),
            },
            "uploaded": None if stale else synced.get("synced_at"),
            "upload_partial": bool(synced.get("partial")) and not stale,  # .wandb 不完整,只传了写完的部分
            "last_step": synced.get("last_step"),
            "remote_changed": stale,
            "url": synced.get("url", ""),
            "state_file": rel_path(state_file),
        })
    seen = {(item["policy"], item["key"], item["run"]) for item in runs}
    for model in models or []:
        for run in model.get("runs") or []:
            if (model["policy"], model["key"], run) in seen:
                continue
            seen.add((model["policy"], model["key"], run))
            runs.append({
                "policy": model["policy"],
                "key": model["key"],
                "run": run,
                "run_dir": "",
                "path": f'{model["policy"]}/{model["key"]}/{run}',
                "has_wandb": False,
                "mode": "—",
                "files": 0,
                "size": 0,
                "local": {"path": "", "files": 0, "bytes": 0, "mtime": 0,
                          "downloaded": False, "partial": False},
                "uploaded": None,
                "upload_partial": False,
                "last_step": None,
                "remote_changed": False,
                "url": "",
                "state_file": rel_path(state_file),
            })
    runs.sort(key=lambda item: (not item["has_wandb"], item["policy"], item["key"], item["run"]))
    return runs


def ckpt_model_key(ckpt: str) -> str:
    """``RoboSafeJanus-<task>-arx_x5-joint-0`` → ``<task>``。"""
    match = re.match(r"^RoboSafeJanus-(.+)-[^-]+-[^-]+-0$", ckpt)
    return match.group(1) if match else ckpt


def collect_eval_results(eval_dir: Path, json_tasks: dict[str, tuple[str, str]]) -> list[dict]:
    """扫描 ``eval_result/RoboDojo/<task>/<policy>/<config>/<seed>_ckpt_name=...,action_type=...[,variant=S1.x]/<run>/_result.json``。

    S1 变体(src/eval_client/variants.py)的结果带 ``variant`` 和违规率;调用方把它们和 S1.0 分开用。"""
    root = (eval_dir / "RoboDojo").resolve()
    results: list[dict] = []
    if not root.is_dir():
        return results
    for path in sorted(root.glob("*/*/*/*/*/_result.json")):
        parts = path.relative_to(root).parts
        if len(parts) < 6:
            continue
        task, policy, config, seed_dir = parts[0], parts[1], parts[2], parts[3]
        match = re.match(r"^(\d+)_ckpt_name=(.+),action_type=([^,]+)(?:,variant=(S1\.\d))?$", seed_dir)
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
            "variant": match.group(4) or "",
            "violation_rate": data.get("violation_rate"),
            "safe_success_rate": data.get("safe_success_rate"),
            "mtime": mtime,
            "path": str(path.relative_to(ROOT_DIR)),
            "level": json_tasks.get(task, (None, None))[1],
        })

    def rank(item: dict) -> tuple:
        return (item["eval_time"] or 0, item["mtime"])

    # 同一任务上不同 policy 可能用同名的 ckpt（如 Pi_0/Pi_05 都叫
    # ``RoboSafeJanus-heat_X_microwave_safe_tasks-arx_x5-joint-0``），所以必须把 policy 计入 key，
    # 否则两个 policy 的结果会互相覆盖，页面上只剩其中一个。
    best: dict[tuple[str, str, str, str], dict] = {}
    for item in results:
        key = (item["task"], item["policy"], item["ckpt"], item["variant"])
        if key not in best or rank(item) > rank(best[key]):
            best[key] = item
    for item in results:
        item["selected"] = best[(item["task"], item["policy"], item["ckpt"], item["variant"])] is item
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
    """Compare a task row against the task definitions (``load_tasks``)."""
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


JOB_PROGRESS: dict[str, object] = {}  # kind -> 算进度的函数(launch_job 的 progress=),任务面板只显示进度


def wandb_job_progress(log: str | None) -> dict | None:
    """W&B 下载 / 上传 / 同步日志的进度:下载完没有、上传到第几个 run(段)。"""
    try:
        text = Path(log).read_text(encoding="utf-8", errors="replace") if log else ""
    except OSError:
        return None
    bits = []
    if "[download_wandb]" in text:
        bits.append("下载 ✓" if "[download_wandb] downloaded" in text or "没有需要下载" in text else "下载中")
    heads = re.findall(r"^===== \[(\d+)/(\d+)\]", text, re.M)
    if heads:
        done = len(re.findall(r"^\[upload_wandb\] (?:OK|FAILED|timeout)", text, re.M))
        total = int(heads[-1][1])
        bits.append(f"上传 {min(done, total)}/{total} 段")
        return {"text": " · ".join(bits), "done": min(done, total), "total": total}
    return {"text": " · ".join(bits), "done": 0, "total": 0} if bits else None


def job_reason(job: dict) -> str:
    """失败任务的一行原因(日志里最后一行报错),不用看日志。"""
    for line in reversed(job_tail(job).splitlines()):
        if re.search(r"FAILED|Error|错误|失败|failed|denied|timed out", line):
            return line.strip()[:200]
    return ""


def jobs_snapshot() -> dict:
    with JOBS_LOCK:
        jobs = {kind: dict(job) for kind, job in JOBS.items()}
        for kind, job in jobs.items():
            job["items"] = list(JOB_ITEMS.get(kind) or [])
            job["queue"] = [", ".join(entry["items"]) or "?" for entry in JOB_QUEUE.get(kind) or []]
            job["label"] = JOB_LABELS[kind]

    for kind, job in jobs.items():
        job["log_tail"] = job_tail(job)
        if job["state"] == "idle":
            continue
        try:
            if kind == "sync_wandb":
                job["progress"] = wandb_job_progress(job.get("log"))
            elif JOB_PROGRESS.get(kind):
                job["progress"] = JOB_PROGRESS[kind]()
        except (OSError, ValueError, KeyError, TypeError):
            job["progress"] = None
        if job["state"] == "failed":
            job["reason"] = job_reason(job)
    return jobs


def job_status_line(log_path: Path) -> str:
    """任务日志里最后一行 ``[status] ...``(脚本自己写的结果说明,如"部分上传 …"),没有时空串。"""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return next((line[len("[status] "):].strip() for line in reversed(lines) if line.startswith("[status] ")), "")


JOB_QUEUE: dict[str, list[dict]] = {}  # kind -> 排队等着的 launch_job 参数(只在看板进程里,重启会丢)
JOB_ITEMS: dict[str, list[str]] = {}  # kind -> 正在做的条目(如 "Pi_0/<key>"),按钮判断是不是已经在做 / 排队


def launch_job(kind: str, runner, out_dir: Path, on_done=None, progress=None,
               items: list[str] | None = None, queue: bool = False) -> int:
    """后台跑一个看板任务;同类在跑时 ``queue`` 为真就排队(前一个结束后按点击顺序接着跑),否则报错。
    返回排在第几个(0 = 已经开始)。"""
    entry = {"runner": runner, "out_dir": out_dir, "on_done": on_done, "progress": progress,
             "items": list(items or [])}
    with JOBS_LOCK:
        if JOBS[kind]["state"] == "running":
            if not queue:
                raise RuntimeError(f"{JOB_LABELS[kind]} 正在运行")
            JOB_QUEUE.setdefault(kind, []).append(entry)
            return len(JOB_QUEUE[kind])
        JOBS[kind]["state"] = "running"
    _start_job(kind, entry)
    return 0


def _start_job(kind: str, entry: dict) -> None:
    """``launch_job`` / 前一个结束时调用:状态已经占成 running。"""
    with JOBS_LOCK:
        job = JOBS[kind]
        log_path = entry["out_dir"] / ".cache" / "jobs" / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}.log"
        job.update({
            "state": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "returncode": None,
            "message": "运行中",
            "log": str(log_path),
        })
        JOB_PROGRESS[kind] = entry["progress"]
        JOB_ITEMS[kind] = entry["items"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runner, on_done = entry["runner"], entry["on_done"]

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
                    "message": job_status_line(log_path) or ("完成" if code == 0 else f"失败（exit {code}）"),
                })
                JOB_ITEMS[kind] = []
                nxt = JOB_QUEUE[kind].pop(0) if JOB_QUEUE.get(kind) else None
                if nxt:
                    job["state"] = "running"  # 先占住,免得这时有人点按钮插队
            if on_done:
                try:
                    on_done()
                except Exception:  # noqa: BLE001 - 刷新失败不影响任务结果
                    pass
            if nxt:
                _start_job(kind, nxt)

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


def model_item_html(model: dict, task: str, group: bool, wandb: list[dict] | None = None,
                    running: set[tuple[str, str]] | None = None) -> str:
    """大表模型列里的一个模型,只有两行:下载(已下载 / 下载中 / 「⬇ 下载」按钮)和 W&B(已上传 / 「⟳ 同步 W&B」)。

    「⬇ 下载」(``download-model-btn``)点了确认后 ``/api/download_model?policy=&task=`` 后台下载并链接到 checkpoints;
    「⟳ 同步 W&B」见 ``wandb_item_html``。模型目录、step、组模型 / 自训放在悬停提示里
    (policy 名由 pipeline_page 写在格子顶上)。"""
    policy, key = model["policy"], model["key"]
    latest = model.get("latest_step") or "—"
    local = model.get("local_steps") or []
    tip = f"{'组模型' if group else '自训模型'} {policy}/{key} · 远端最新 step {latest}"
    wb = wandb_item_html(model, wandb or [])
    wb = "<br>" + wb if wb else ""
    if not model.get("latest_step"):  # 远端只有 run 级文件、没有能认出的 step 目录:不能说"已下载"
        return (f'<div class="mdl" title="{html.escape(tip)}">'
                f'<small class="warn">❔ 远端没认出 checkpoint</small>{wb}</div>')
    if not model.get("needs_download"):
        return f'<div class="mdl" title="{html.escape(tip)}"><small class="good">✅ 已下载</small>{wb}</div>'
    partial = model.get("partial_steps") or []
    have = [*local, *(f"{s} 没下完" for s in partial)]
    tip += " · " + (f"本地 {', '.join(have)}" if have else "本地没有")
    if model_downloading(model, running or set()):
        got = model_local_bytes(model, DEFAULT_MODEL)
        pct = f" {min(99, 100 * got // model['size'])}%" if model.get("size") else ""
        return f'<div class="mdl" title="{html.escape(tip)}"><small>⬇ 下载中{pct}</small>{wb}</div>'
    size = human_bytes(model.get("size") or 0)
    label = f"{policy}/{key}"
    button = (
        f'<button class="download-model-btn mdl-btn" type="button" data-policy="{html.escape(policy)}"'
        f' data-task="{html.escape(key)}" data-size="{html.escape(size)}" data-label="{html.escape(label)}"'
        f' title="下载 {html.escape(label)} 的 step {html.escape(latest)}(约 {html.escape(size)}),'
        f'完成后链接到 checkpoints">⬇ 下载 {html.escape(size)}</button>'
    )
    return f'<div class="mdl" title="{html.escape(tip)}">{button}{wb.removeprefix("<br>")}</div>'


def wandb_run_text(r: dict) -> str:
    """一个 W&B run 的状态(悬停提示用)。"""
    downloaded = (r.get("local") or {}).get("downloaded")
    if r.get("remote_changed"):
        return "远端 wandb 有更新,要重新同步"
    if r.get("upload_partial"):
        step = r.get("last_step")
        return (f"部分上传{f'(到 step {step})' if step is not None else ''} {r['uploaded']}:"
                "远端 .wandb 不完整(训练还没结束就传了),训练机重新上传完整的 wandb/ 后这里会出「⟳ 同步」")
    if r.get("uploaded"):
        return "已上传 " + str(r["uploaded"])
    return "本地已下载,未上传" if downloaded else "未下载,未上传"


def wandb_run_groups(runs: list[dict]) -> list[list[dict]]:
    """按 W&B run id 分组(目录名最后一段):训练中断后续训会在同一个 run id 下写新的 ``offline-run-<时间>-<id>``
    目录(一段一个),要按时间顺序都传上去才是完整的曲线。每组按目录名(时间)排序。"""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["run"], r["run_dir"].rsplit("-", 1)[-1]), []).append(r)
    return [sorted(g, key=lambda r: r["run_dir"]) for _, g in sorted(groups.items())]


def wandb_item_html(model: dict, wandb: list[dict]) -> str:
    """模型格里的 W&B 一行:按 run id 算(一个 run 可能分好几段),已上传几个、部分上传的到哪一步、本地下载了几个;
    有没传的段(新段 / 远端有更新)就带「⟳ 同步 W&B」按钮。

    前面的段被中断时结尾不完整(``incomplete record``,记成部分上传),只要最后一段完整传上去就算这个 run 完整。
    所有段都传了、最后一段仍不完整时显示"部分",不出按钮(再传也一样,要训练机重新上传)。"""
    policy, key = model["policy"], model["key"]
    items = [r for r in wandb if r.get("policy") == policy and r.get("key") == key]
    runs = [r for r in items if r.get("has_wandb")]
    if not runs:
        missing = ('<small class="wb muted" title="训练机上传 checkpoint 时没把 run 下的 wandb/ 一起传">'
                   'W&amp;B 远端没有</small>')
        return missing if items else ""
    groups = wandb_run_groups(runs)
    todo = [g for g in groups if any(not r.get("uploaded") for r in g)]
    part = [g for g in groups if g not in todo and g[-1].get("upload_partial")]
    full = [g for g in groups if g not in todo and g not in part]
    down = [g for g in groups if all((r.get("local") or {}).get("downloaded") for r in g)]
    lines = []
    for g in groups:
        rid = g[0]["run_dir"].rsplit("-", 1)[-1]
        lines.append(f"{rid}" + (f"(分 {len(g)} 段,训练中断后续训)" if len(g) > 1 else "") + ":")
        lines += [f'  {r["run_dir"]}: {wandb_run_text(r)}' for r in g]
    tip = "\n".join(lines)
    n = len(groups)
    bits = [f"W&amp;B 已上传 {len(full)}/{n}"]
    if part:
        steps = sorted({str(g[-1]["last_step"]) for g in part if g[-1].get("last_step") is not None})
        bits.append(f"部分 {len(part)}" + (f"(到 step {'、'.join(steps)})" if steps else ""))
    if todo:
        segs = sum(1 for g in todo for r in g if not r.get("uploaded"))
        bits.append(f"本地 {len(down)}/{n}")
        if any(len(g) > 1 for g in todo):
            bits.append(f"{segs} 段没传")
    if any(r.get("remote_changed") for r in runs):
        bits.append("远端有更新")
    text = " · ".join(bits)
    url = next((r.get("url") for g in full + part for r in reversed(g) if r.get("url")), "")
    if not todo:
        body = ("✅ " if not part else "⚠️ ") + text
        body = f'<a target="_blank" rel="noopener" href="{html.escape(url)}">{body} ↗</a>' if url else body
        return f'<small class="wb" title="{html.escape(tip)}">{body}</small>'
    label = f"{policy}/{key}"
    button = (f'<button class="sync-wandb-btn mdl-btn" type="button" data-policy="{html.escape(policy)}"'
              f' data-task="{html.escape(key)}" data-label="{html.escape(label)}"'
              f' title="先下载缺失的 wandb 目录,再按时间顺序上传到 W&amp;B 官方(点了先列计划,确认后后台运行)">'
              '⟳ 同步 W&amp;B</button>')
    return f'<small class="wb" title="{html.escape(tip)}">{text}</small>{button}'


def model_local_bytes(model: dict, root: Path) -> int:
    """本地这个模型最新 step 已经下了多少字节(含还在下载的 ``*.incomplete``)。"""
    total = 0
    for run in model.get("runs") or []:
        step_dir = root / model["policy"] / model["key"] / run / str(model.get("latest_step") or "")
        if model.get("latest_step") and step_dir.is_dir():
            for dirpath, _, names in os.walk(step_dir):
                for name in names:
                    try:
                        total += os.stat(os.path.join(dirpath, name)).st_size
                    except OSError:
                        pass
    return total


def download_progress(models: list[dict], root: Path) -> dict:
    """模型下载进度:按本地已下的字节 / 远端大小。"""
    done = [min(model_local_bytes(m, root), m.get("size") or 0) for m in models]
    total = sum(m.get("size") or 0 for m in models)
    text = " · ".join(f"{m['policy']} {human_bytes(d)}/{human_bytes(m.get('size') or 0)}" for m, d in zip(models, done))
    return {"text": text, "done": sum(done), "total": total}


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


def download_wandb_runner(runs: list[dict], cwd: Path, repo_id: str | None):
    def run(log):
        if not repo_id:
            log.write("[download_wandb] 未配置模型仓库\n")
            return 1
        command = [
            sys.executable, "-u", "scripts/download_wandb_from_modelscope.py",
            "--repo_id", repo_id, "--path", *[item["path"] for item in runs],
        ]
        log.write("[download_wandb] " + " ".join(command) + "\n")
        log.flush()
        return subprocess.call(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)

    return run


def upload_wandb_runner(
    runs: list[dict], cwd: Path, entity: str | None = None,
    project: str | None = None, force: bool = False, proxy: str | None = None,
):
    def run(log):
        command = [
            sys.executable, "-u", "scripts/upload_wandb_to_cloud.py",
            "--path", *[item["path"] for item in runs],
        ]
        if entity:
            command += ["--entity", entity]
        if project:
            command += ["--project", project]
        if force:
            command.append("--all")
        if proxy:
            command += ["--proxy", proxy]
        env = dict(os.environ)
        if proxy:
            env.update({"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "ALL_PROXY": proxy})
        log.write("[upload_wandb] " + " ".join(command) + "\n")
        log.flush()
        return subprocess.call(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, env=env)

    return run


def sync_wandb_runner(
    download_runs: list[dict], upload_runs: list[dict], cwd: Path,
    repo_id: str | None, entity: str | None = None,
    project: str | None = None, proxy: str | None = None,
):
    """一个后台任务里先下载缺失的 W&B run，再上传未同步的到 W&B 官方。"""

    def run(log):
        if download_runs:
            if not repo_id:
                log.write("[sync_wandb] 未配置模型仓库，跳过下载\n")
            else:
                code = download_wandb_runner(download_runs, cwd, repo_id)(log)
                if code:
                    log.write(f"[sync_wandb] 下载失败（exit {code}），后面的上传不再执行\n")
                    return code
        else:
            log.write("[sync_wandb] 没有需要下载的 run\n")
        if not upload_runs:
            log.write("[sync_wandb] 没有需要上传的 run，完成\n")
            return 0
        # force(--all):看板已经挑好了要传的 run,远端更新后重新下载的 run 在状态文件里有旧记录也要再传
        return upload_wandb_runner(upload_runs, cwd, entity, project, True, proxy)(log)

    return run


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
    wandb_dir: Path | None = None,
) -> dict:
    task_groups.cache_clear()
    json_tasks = task_groups()
    group_order, task_order = json_order(json_tasks)
    local = collect_local(local_dir, json_tasks)
    remote, remote_error = collect_remote(repo_id, repo_type, prefix)
    lerobot = collect_lerobot(lerobot_dir)
    models, wandb_runs, model_error = collect_models(
        model_repo_id, json_tasks, model_local_dir or DEFAULT_MODEL, wandb_dir or DEFAULT_WANDB
    )
    all_results = collect_eval_results(eval_dir or DEFAULT_EVAL, json_tasks)
    # S1 变体的结果只进大表评测结果列(variant_evals);其余汇总 / 矩阵 / 进度都只算 S1.0
    eval_results = [item for item in all_results if not item["variant"]]
    variant_evals = [item for item in all_results if item["variant"] and item["selected"]]
    variant_evals.sort(key=lambda item: (item["variant"], item["policy"]))
    variants_by_task: dict[str, list[dict]] = {}
    for item in variant_evals:
        variants_by_task.setdefault(item["task"], []).append(item)
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
                "variant_evals": variants_by_task.get(name, []),
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
    summary["wandb"] = {
        "runs": len(wandb_runs),
        "bytes": sum(item["size"] for item in wandb_runs),
        "files": sum(item["files"] for item in wandb_runs),
        "downloaded": sum(1 for item in wandb_runs if item["local"]["downloaded"]),
        "partial": sum(1 for item in wandb_runs if item["local"]["partial"]),
        "uploaded": sum(1 for item in wandb_runs if item["uploaded"]),
        "missing": sum(1 for item in wandb_runs if not item.get("has_wandb", True)),
        "api_key": bool(wandb_api_key()),
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
        "wandb": wandb_runs,
        "wandb_dir": rel_path(wandb_dir or DEFAULT_WANDB),
        "wandb_state_file": rel_path((wandb_dir or DEFAULT_WANDB).parent / ".cache" / "wandb" / "synced.json"),
        "wandb_api_key": bool(wandb_api_key()),
        "eval_dir": rel_path(eval_dir or DEFAULT_EVAL),
        "eval_layout_dir": rel_path(layout_dir or DEFAULT_LAYOUT),
        "eval_seed": layout_seed,
        "tasks_json": str(Path(TASKS_JSON).resolve().relative_to(ROOT_DIR)),
        "summary": summary,
        "tasks": rows,
        "models": models,
        "evals": eval_results,
        "variant_evals": variant_evals,
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
            note = "可直接采集" if item["in_json"] else "不在 task/robosafejanus_tasks.json，需先加入并补组定义"
            lines.append(f"| `{item['name']}` | {'✅' if item['in_json'] else '⚠️ 否'} | {note} |")
    else:
        lines.append("无。")
    lines += [
        "",
        "## 任务明细",
        "",
        "| # | 组 | 任务 | JSON 对齐 | 专家脚本 | HDF5（本地 / ModelScope） | LeRobot v30 | LeRobot v21 | 模型 | 评测 | 评测布局 | 视频 | 总状态 |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    group_numbers: dict[str, int] = {}
    for row in snapshot["tasks"]:
        group_numbers.setdefault(row["group"], len(group_numbers) + 1)
    for row in snapshot["tasks"]:
        local = row["local"]
        remote = row["remote"]
        local_status = row["status"]["local"]
        if local and local["episodes"]:
            hdf5_text = (
                f"本地 {'✅' if local_status == 'ok' else '⚠️'} "
                f"{local['episodes']} 条 / {human_bytes(local['size'])}"
            )
        else:
            hdf5_text = "本地 ❌ 无"
        remote_status = row["status"]["remote"]
        pending_upload = remote_status == "missing" and bool(local and local["episodes"])
        if pending_upload:
            remote_status = "local_only"
        remote_label = {
            "synced": "✅ 同步",
            "diff": "⚠️ 不一致",
            "remote_only": "⬇️ 仅远端",
            "local_only": "⬆️ 待上传",
        }.get(remote_status, "❌ 无")
        remote_text = f"远端 {remote_label}"
        if remote:
            delta = remote["size"] - (local["size"] if local else 0)
            remote_text += f" / {remote['episodes']} 条 / {human_bytes(remote['size'])}"
            if remote_status == "diff":
                remote_text += f"（{signed_bytes(delta)}）"
        hdf5_text += f"；{remote_text}"
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
            f"| {group_numbers[row['group']]} | `{row['group']}` | `{row['name']}` | {json_text} | {code_cell} | {hdf5_text} | "
            f"{lerobot_text['v30']} | {lerobot_text['v21']} | {model_text} | {eval_text} | {layout_text} | {video_text} | {overall} |"
        )
    lines += [
        "",
        "> `HDF5` 列前半是本地 episode 数 / 总大小，后半是 ModelScope：`✅ 同步` 表示条数和大小都一致，`⚠️ 不一致` 表示远端存在但条数或大小不同。",
        "> LeRobot 列前半是本地导出，后半是 ModelScope 上的 LeRobot 数据集：`本地 ❌ 无` 未导出、`远端 ⬆️ 待上传` 未上传、`远端 ✅ 同步` 条数一致。",
        "> `JSON 外`：本地有数据但不在任务合集（`task/robosafejanus_tasks.json`）中，上传脚本会跳过。",
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


# 三个页面共用的主题与基础样式；颜色只在 :root 里定义，跟随系统深 / 浅色。
BASE_CSS = """
  :root {
    color-scheme: light dark;
    --bg: #f6f7f9; --panel: #ffffff; --panel-2: #f2f4f7; --border: #e1e4e8; --line: #eef0f3;
    --text: #1f2328; --muted: #656d76; --faint: #8c959f; --accent: #0969da; --hover: #f5f7fa;
    --good-bg: #dafbe1; --good-fg: #116329; --good: #2da44e;
    --warn-bg: #fff4c2; --warn-fg: #7d4e00; --warn: #d4a72c;
    --bad-bg: #ffebe9; --bad-fg: #a40e26;
    --safe-bg: rgba(46, 160, 67, .16); --safe-fg: #116329;
    --risky-bg: rgba(248, 81, 73, .14); --risky-fg: #a40e26;
    --radius: 10px;
    --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0d1117; --panel: #151b23; --panel-2: #1b222c; --border: #2a313c; --line: #21262d;
      --text: #e6edf3; --muted: #8b949e; --faint: #6e7681; --accent: #58a6ff; --hover: #1b222c;
      --good-bg: rgba(46, 160, 67, .18); --good-fg: #7ee787; --good: #2ea043;
      --warn-bg: rgba(187, 128, 9, .2); --warn-fg: #e3b341; --warn: #d29922;
      --bad-bg: rgba(248, 81, 73, .18); --bad-fg: #ffa198;
      --safe-bg: rgba(46, 160, 67, .22); --safe-fg: #7ee787;
      --risky-bg: rgba(248, 81, 73, .2); --risky-fg: #ffa198;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 28px 32px 48px; background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif; }
  .page { max-width: none; margin: 0; }  /* 不封顶:大表用满窗口,浏览器缩小后能一行看完 */
  header.top { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
  h1 { margin: 0; font-size: 22px; font-weight: 650; letter-spacing: -.01em; }
  .meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
  .meta .bad { padding: 0 5px; border-radius: 4px; }
  h2 { font-size: 15px; font-weight: 600; margin: 32px 0 10px; }
  h2 small { font-weight: 400; color: var(--faint); font-size: 12px; margin-left: 6px; }
  a { color: var(--accent); }
  code { font-family: var(--mono); font-size: 12px; }
  small, .empty { color: var(--muted); }
  .empty { font-size: 13px; }
  table { width: 100%; border-collapse: separate; border-spacing: 0; background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  th, td { padding: 9px 12px; border-bottom: 1px solid var(--line); text-align: left; font-size: 13px; vertical-align: top; }
  tbody tr:last-child > * { border-bottom: 0; }
  th { background: var(--panel-2); color: var(--muted); font-size: 12px; font-weight: 500; }
  tbody tr:hover > td { background: var(--hover); }
  .badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 500; line-height: 1.6; white-space: nowrap; }
  .good { background: var(--good-bg); color: var(--good-fg); }
  .warn { background: var(--warn-bg); color: var(--warn-fg); }
  .bad { background: var(--bad-bg); color: var(--bad-fg); }
  .badge.lvl-safe, .legend-item.lvl-safe { background: var(--safe-bg); color: var(--safe-fg); }
  .badge.lvl-risky, .legend-item.lvl-risky { background: var(--risky-bg); color: var(--risky-fg); }
  input[type=search] { background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 7px 12px; font: inherit; font-size: 13px; }
  input[type=search]:focus, button:focus-visible { outline: 2px solid var(--accent); outline-offset: -1px; }
"""


# 数据同步页和模型与评测页的样式（叠加在 BASE_CSS 上）。
DASHBOARD_CSS = BASE_CSS + """
  .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .toolbar .hint { color: var(--faint); font-size: 12px; margin-left: 4px; }
  button { font: inherit; }
  .toolbar button, .dialog-actions button, .video-btn, .download-model-btn {
    background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 7px;
    padding: 6px 12px; font-size: 13px; cursor: pointer; white-space: nowrap; }
  .video-btn, .download-model-btn { padding: 3px 9px; font-size: 12px; }
  .toolbar button:hover, .dialog-actions button:hover, .video-btn:hover, .download-model-btn:hover,
  button:disabled { opacity: .5; cursor: not-allowed; }
  #refresh-btn, #plan-ok { background: var(--accent); border-color: var(--accent); color: #fff; }
  #refresh-btn:hover, #plan-ok:hover { background: var(--accent); filter: brightness(1.08); }
  .jobs { margin-bottom: 16px; }
  section.tab > .toolbar { margin-top: 16px; }
  button.gact { font: inherit; font-size: 12px; background: var(--panel); color: var(--text);
    border: 1px solid var(--border); border-radius: 7px; padding: 3px 10px; cursor: pointer; white-space: nowrap; }
  button.gact:hover:not(:disabled) { border-color: var(--accent); }
  button.gact:disabled { opacity: .45; cursor: not-allowed; }
  .gjob { font-size: 11px; color: var(--muted); margin-top: 4px; white-space: normal; max-width: 220px; }
  .gjob.running { color: var(--warn-fg); } .gjob.done { color: var(--good-fg); }
  .gjob.failed, .gjob.interrupted { color: var(--bad-fg); }
  .plan-fields { display: flex; flex-direction: column; gap: 8px; margin: 8px 0; font-size: 13px; }
  .plan-machine:empty { display: none; }
  .plan-machine { margin: 4px 0 10px; font-size: 13px; }
  .plan-machine .mq { font-weight: 600; margin-bottom: 6px; }
  .plan-machine label.mopt { display: flex; gap: 8px; align-items: baseline; padding: 6px 8px; margin: 4px 0;
    border: 1px solid var(--border); border-radius: 7px; cursor: pointer; }
  .plan-machine label.mopt:has(input:checked) { border-color: var(--accent); background: var(--hover); }
  .plan-machine label.mopt.off { opacity: .5; cursor: not-allowed; }
  .plan-machine label.mopt small { color: var(--muted); }
  .plan-fields input[type=text] { margin-left: 6px; font: inherit; background: var(--panel); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; width: 160px; }
  .job { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 14px; margin-bottom: 8px; font-size: 13px; }
  .job .head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .job .state { font-weight: 600; }
  .job .state.running { color: var(--warn-fg); }
  .job .state.done { color: var(--good-fg); }
  .job .state.failed { color: var(--bad-fg); }
  .job .state.partial { color: var(--warn-fg); }
  pre { background: var(--panel-2); border: 1px solid var(--line); border-radius: 7px; padding: 10px; overflow: auto; white-space: pre-wrap; font: 11px/1.5 var(--mono); }
  .job pre { margin: 8px 0 0; max-height: 220px; }
  .job .jprog { margin-top: 6px; color: var(--text); }
  .job .jbar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 4px; }
  .job .jbar > span { display: block; height: 100%; background: var(--accent); }
  .job .jwhy { margin-top: 3px; color: var(--muted); }
  .job .jwhy.bad { color: var(--bad-fg); }
  dialog { background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; width: min(900px, calc(100% - 48px)); box-shadow: 0 16px 48px rgba(0, 0, 0, .25); }
  dialog::backdrop { background: rgba(1, 4, 9, .55); }
  dialog h2 { margin: 0 0 12px; }
  dialog pre { max-height: 50vh; font-size: 12px; }
  .dialog-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 14px; }
  td.group-index { color: var(--faint); text-align: right; width: 32px; font-variant-numeric: tabular-nums; }
  .eval-item { margin-bottom: 6px; }
  .eval-item:last-child { margin-bottom: 0; }
  #video-dialog { width: min(1180px, calc(100% - 40px)); }
  .video-ep:hover, .video-ep.active { border-color: var(--accent); }
  .video-layout { display: grid; grid-template-columns: 240px 1fr; gap: 12px; }
  .video-list { max-height: 62vh; overflow: auto; background: var(--panel-2); border-radius: 7px; padding: 6px; display: flex; flex-direction: column; gap: 4px; }
  .video-ep { display: flex; justify-content: space-between; gap: 8px; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 5px 8px; font-size: 12px; cursor: pointer; text-align: left; }
  .video-ep.active { background: var(--hover); }
  .video-marks { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
  .video-marks .safety { font-size: 11px; padding: 0 5px; border-radius: 4px; border: 1px solid; }
  .video-marks .safety.safe { background: var(--good-bg); color: var(--good-fg); border-color: currentColor; }
  .video-marks .safety.risk { background: var(--bad-bg); color: var(--bad-fg); border-color: currentColor; }
  .video-stage video { width: 100%; max-height: 62vh; background: #000; border-radius: 7px; }
  .video-cams { display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
  .video-cams button { background: var(--panel); border: 1px solid var(--border); border-radius: 999px; color: var(--muted); padding: 3px 11px; font-size: 12px; cursor: pointer; }
  .video-cams button.active { border-color: var(--accent); background: var(--accent); color: #fff; }
  .legend { display: flex; gap: 8px; align-items: center; color: var(--muted); font-size: 12px; margin: 0 0 8px; flex-wrap: wrap; }
  .legend-item { display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 11px; }
  details.fold { margin-top: 12px; background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 14px; }
  details.fold summary { cursor: pointer; color: var(--muted); font-size: 13px; }
  details.fold summary:hover { color: var(--text); }
  details.fold[open] summary { margin-bottom: 10px; }
  details.fold table { border-radius: 8px; }
"""


def pipeline_parts(out_dir: Path) -> tuple[dict[str, str], str]:
    """``({标签: 片段容器}, 样式和脚本)``;流水线部分出错时为空,不影响看板本身。"""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import pipeline_page

        boxes = {key: pipeline_page.fragment_box(key, out_dir) for key in pipeline_page.FRAGMENTS}
        return boxes, pipeline_page.merged_assets()
    except Exception:  # noqa: BLE001 - 审核区出问题不影响看板本身
        print("[sync_status] pipeline parts failed:\n" + traceback.format_exc(), flush=True)
        return {}, ""


def html_page(snapshot: dict, refresh: int, out_dir: Path = DEFAULT_OUT) -> str:
    """生成看板页面 PIPELINE.html：只有一张流水线大表（pipeline_page.table_html），本函数算出每个任务的数据格。"""
    summary = snapshot["summary"]
    json_tasks = task_groups()
    wandb_rows = snapshot.get("wandb") or []
    downloading = running_model_downloads()  # 模型格:正在下载的不出「⬇ 下载」按钮
    task_cells: dict[str, dict] = {}  # 大表的数据格(pipeline_page.CELLS_FILE)
    for row in snapshot["tasks"]:
        local = row["local"]
        # 大表 collect / exchange / upload 格用的纯数据:本地大小、各份数据的远端状态
        # (synced / local_only / diff / missing / unknown)
        remote_status = row["status"]["remote"]
        if remote_status == "missing" and local and local["episodes"]:
            remote_status = "local_only"
        size = human_bytes(local["size"]) if local and local["episodes"] else ""
        sync = {"hdf5": remote_status, "hdf5_size": size}
        for version in LEROBOT_VERSIONS:
            sync[version] = row["status"][f"{version}_remote"]
        # 模型 / 评测格按模型(policy)分开:pipeline_page 的宽表每个模型一组「模型 · 评测」列
        model_parts: dict[str, list[str]] = {}
        for model, is_group in [(m, False) for m in row["models"]] + [(m, True) for m in row["group_models"]]:
            model_parts.setdefault(model["policy"], []).append(
                model_item_html(model, row["name"], is_group, wandb_rows, downloading))
        eval_parts: dict[tuple[str, str], list[str]] = {}  # (S1 变体, policy) -> 结果块
        for result in [*row["evals"], *row.get("variant_evals", [])]:
            label = eval_label(result, row, json_tasks)
            kind = eval_kind(result, row, json_tasks)
            rate = result["success_rate"] * 100
            n = result["eval_time"]
            n_text = f"n={n}" if n is not None else "n=?"
            # 这次评测的视频(episode_*_cam_*_{success,fail}.mp4,和 _result.json 同目录):点了用视频弹窗播放
            run_dir = str(Path(result["path"]).parent)
            n_video = sum(1 for _ in (ROOT_DIR / run_dir).glob("episode_*_cam_head_*.mp4"))
            video_btn = (
                f'<button class="video-btn eval-video-btn" type="button" data-scope="eval"'
                f' data-run="{html.escape(run_dir)}" data-label="{html.escape(row["name"] + " · " + label)}"'
                f' title="看这次评测的视频(✅ 成功 / ❌ 失败'
                f'{" · 🛡️ 安全 / ⚠️ 违规" if result.get("variant") not in (None, "", "S1.0") else ""},{n_video} 条)">'
                f'🎬 {n_video}</button>'
                if n_video else ""
            )
            # 结果直接写进评测的 S1 子列(`eval:<S1.x>:<policy>`,由 pipeline_page.eval_cell 显示):只留 success / n /
            # 视频,模型名已经在同一行的模型列里;S1 变体再加违规率(src/eval_client/variants.py)
            variant = result.get("variant") or "S1.0"
            if variant != "S1.0":
                bad = (result.get("violation_rate") or 0) * 100
                safe = (result.get("safe_success_rate") or 0) * 100
                tip = f"{kind} · {variant}:成功 {rate:.1f}% · 违规 {bad:.1f}% · 成功且未违规 {safe:.1f}%"
                body = (f"<strong>{rate:.1f}%</strong> <small>成功</small><br>"
                        f'<strong class="{"ng" if bad else ""}">{bad:.1f}%</strong> <small>违规 ({n_text})</small>')
            else:
                tip, body = kind, f"<strong>{rate:.1f}%</strong> <small>({n_text})</small>"
            eval_parts.setdefault((variant, result.get("policy") or "?"), []).append(
                f'<div class="eval-item" title="{html.escape(tip)}">{body}{video_btn}</div>'
            )
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
            video_cell = (
                f'<button class="video-btn" type="button" data-scope="local"'
                f' data-task="{html.escape(row["name"])}"'
                f' data-label="{html.escape(row["name"])}">🎬 {head_count}</button>'
            )
        else:
            video_cell = "—"
        task_cells[row["name"]] = {
            "video": video_cell, "eval_layout": layout_cell, "sync": sync,
        }
        for policy in sorted(set(model_parts) | {p for _, p in eval_parts}):
            task_cells[row["name"]][f"model:{policy}"] = "".join(model_parts.get(policy, [])) or "—"
        for (variant, policy), parts in eval_parts.items():  # 没评过的格子不写,让 eval_cell 显示已评几条
            task_cells[row["name"]][f"eval:{variant}:{policy}"] = "".join(parts)

    remote_note = f" · <span class='bad'>{html.escape(snapshot['remote_error'])}</span>" if snapshot["remote_error"] else ""
    lerobot_remote_note = (
        f" · <span class='bad'>{html.escape(snapshot['lerobot_remote_error'])}</span>"
        if snapshot["lerobot_remote_error"] else ""
    )
    model_note = (
        f" · <span class='bad'>{html.escape(snapshot['model_error'])}</span>"
        if snapshot["model_error"] else ""
    )
    refresh_script = """
<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const refreshBtn = $('refresh-btn');
  const hint = $('refresh-hint');
  const jobsBox = $('jobs');
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
      autoRefreshTimer = setTimeout(() => {
        // 有没写完的审核说明、打开着的对话框时先不刷新
        const typing = Array.from(document.querySelectorAll('textarea, input[type=text]')).some(
          (el) => el.value.trim() || el === document.activeElement
        );
        if (typing || document.querySelector('dialog[open]')) scheduleAutoRefresh();
        else location.reload();
      }, autoRefreshSeconds * 1000);
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
    $('plan-fields').innerHTML = '';
    $('plan-machine').innerHTML = '';
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
      + (entry.violation === undefined ? '' : ' · ' + safetyText(entry))
      + ' · ' + entry.path;
  }

  // S1 变体评测:每条视频的安全检查结果(/api/videos 从 _result.json 的 details 带回来)
  function safetyText(entry) {
    if (entry.violation === undefined) return '';
    return entry.violation ? '⚠️ 违规' + (entry.violated_by && entry.violated_by.length
      ? '(' + entry.violated_by.join('、') + ')' : '') : '🛡️ 安全';
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
      right.className = 'video-marks';
      const mark = document.createElement('span');
      mark.textContent = entry.result === 'success' ? '✅' : (entry.result === 'fail' ? '❌' : '');
      mark.title = entry.result === 'success' ? '任务成功' : (entry.result === 'fail' ? '任务失败' : '');
      right.append(mark);
      if (entry.violation !== undefined) {
        const safety = document.createElement('span');
        safety.className = 'safety ' + (entry.violation ? 'risk' : 'safe');
        safety.textContent = entry.violation ? '⚠️ 违规' : '🛡️ 安全';
        safety.title = safetyText(entry);
        right.append(safety);
      }
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

  function bindVideoButtons() {
    document.addEventListener('click', (event) => {
      const button = event.target.closest('.video-btn');
      if (!button || button.disabled) return;
      const query = {scope: button.dataset.scope || 'local'};
      if (button.dataset.task) query.task = button.dataset.task;
      if (button.dataset.run) query.run = button.dataset.run;
      openVideos(query, button.dataset.label || button.dataset.task);
    });
  }

  const groupStates = {running: '运行中', done: '完成', failed: '失败', stopped: '已停止', interrupted: '中断'};
  let groupJobs = {};
  let groupSeen = {};

  // 任务面板只显示进度(不显示日志):进度条 + 一行字;没有进度时写当前步骤,失败时写一行原因
  function progressEl(job) {
    const box = document.createElement('div');
    box.className = 'jprog';
    const p = job.progress;
    if (p && p.total) {
      const pct = Math.max(0, Math.min(100, Math.round(100 * p.done / p.total)));
      const bar = document.createElement('div');
      bar.className = 'jbar';
      const fill = document.createElement('span');
      fill.style.width = pct + '%';
      bar.append(fill);
      box.append(bar);
      const t = document.createElement('span');
      t.textContent = p.text + ' · ' + pct + '%';
      box.append(t);
    } else if (p && p.text) {
      box.textContent = p.text;
    } else if (job.state === 'running') {
      box.textContent = job.step || job.message || '运行中…';
    }
    if (job.state !== 'running' && (job.reason || (job.step && !(p && p.text)))) {
      const why = document.createElement('div');
      why.className = 'jwhy' + (job.state === 'failed' ? ' bad' : '');
      why.textContent = job.reason || job.step;
      box.append(why);
    }
    return box;
  }

  function renderGroupJobs(groups) {
    groupJobs = groups || {};
    const recent = Date.now() - 12 * 3600 * 1000;
    for (const [jid, job] of Object.entries(groupJobs)) {
      const prev = groupSeen[jid];
      groupSeen[jid] = job.state;
      // 组任务刚结束:页面上的状态由服务端重建,稍后自动重载
      if (prev === 'running' && job.state !== 'running') setTimeout(() => location.reload(), 4000);
      const started = Date.parse(job.started_at || '') || 0;
      if (job.state !== 'running' && started < recent) continue;
      const card = document.createElement('div');
      card.className = 'job';
      const head = document.createElement('div');
      head.className = 'head';
      const name = document.createElement('strong');
      name.textContent = (job.task === '__group__' ? job.group : (job.task || jid)) + ' · ' + (job.label || job.kind);
      const state = document.createElement('span');
      state.className = 'state ' + (job.state === 'running' ? 'running' : (job.state === 'done' ? 'done' : 'failed'));
      state.textContent = job.state === 'running' && job.waiting ? '排队中' : (groupStates[job.state] || job.state);
      const time = document.createElement('span');
      time.textContent = (job.started_at || '') + (job.finished_at ? ' → ' + job.finished_at : '');
      head.append(name, state, time);
      if (job.session_id) {
        const resume = document.createElement('code');
        resume.textContent = 'claude --resume ' + job.session_id;
        resume.title = '在终端里接着这个 Claude 会话聊';
        head.append(resume);
      }

      if (job.state === 'running') {
        const stopBtn = document.createElement('button');
        stopBtn.type = 'button';
        stopBtn.className = 'gact';
        stopBtn.dataset.kind = 'stop';
        stopBtn.dataset.group = job.group;
        stopBtn.dataset.task = job.task;
        if (jid.split('.').length > 2) stopBtn.dataset.policy = job.policy || '';  // 按模型的评测记录
        if (job.variant) stopBtn.dataset.variant = job.variant;  // S1 变体的评测记录
        stopBtn.textContent = '⏹ 停止';
        head.append(stopBtn);
      }
      card.append(head);
      card.append(progressEl(job));
      jobsBox.append(card);
    }
    paintGroupCells();
  }

  function paintGroupCells() {
    const cellKind = {ingest: 'ingest', collect: 'collect', export: 'exchange', upload: 'upload', eval: 'eval',
      layouts: 'layouts'};
    // 按模型的评测记录键是 <group>.<task>.<policy>;"评所有模型"(命令行)和 2026-11-16 前的评测记在 <group>.<task>;
    // S1 变体的评测再加 @<variant>(<group>.<task>.<policy>@S1.1),只认自己的记录
    const jobOf = (el) => {
      const base = el.dataset.group + '.' + el.dataset.task;
      const policy = el.dataset.policy;
      if (el.dataset.kind === 'eval' && el.dataset.variant) return groupJobs[base + '.' + (policy || '') + '@' + el.dataset.variant];
      if (el.dataset.kind !== 'eval' || !policy) return groupJobs[base];
      const own = groupJobs[base + '.' + policy];
      const old = groupJobs[base];
      const shared = old && old.kind === 'eval' && (!old.policy || old.policy === policy) ? old : null;
      if (!own) return shared;
      return shared && shared.state === 'running' ? shared : own;
    };
    const kindOf = (job) => cellKind[job.kind];
    document.querySelectorAll('.gjob[data-group]').forEach((cell) => {
      let job = jobOf(cell);
      if (job && cell.dataset.kind && kindOf(job) !== cell.dataset.kind) job = null;
      if (job && cell.dataset.policy && job.policy && job.policy !== cell.dataset.policy) job = null;
      if (job && (job.variant || '') !== (cell.dataset.variant || '') && cell.dataset.kind === 'eval') job = null;
      const prog = job && job.progress && job.progress.text;
      cell.textContent = !job ? '' : job.state === 'running'
        ? (job.waiting ? job.label + ' · ' + (job.step || '排队中') : job.label + '中 · ' + (prog || job.step || ''))
        : job.label + (groupStates[job.state] || job.state) + (prog ? ' · ' + prog : '')
          + ' · ' + String(job.finished_at || '').replace('T', ' ').slice(5, 16);
      cell.title = '';
      cell.className = 'gjob' + (job ? ' ' + job.state : '');
    });
    document.querySelectorAll('button.gact[data-group]').forEach((btn) => {
      if (btn.dataset.kind === 'stop') return;
      const job = jobOf(btn);
      btn.disabled = !serverMode || Boolean(btn.dataset.blocked) || Boolean(job && job.state === 'running');
    });
  }
  document.addEventListener('pipe-loaded', paintGroupCells);

  function renderJobs(jobs, groups) {
    const active = Object.entries(jobs).filter(([, job]) => job.state !== 'idle');
    jobsBox.innerHTML = '';
    active.sort((a, b) => String(b[1].started_at || '').localeCompare(String(a[1].started_at || '')));
    for (const [kind, job] of active) {
      const card = document.createElement('div');
      card.className = 'job';
      const head = document.createElement('div');
      head.className = 'head';
      const name = document.createElement('strong');
      name.textContent = job.label || kind;
      const state = document.createElement('span');
      state.className = 'state ' + job.state;
      const part = job.state === 'done' && String(job.message || '').startsWith('部分');  // 如 W&B 部分上传
      state.className = 'state ' + (part ? 'partial' : job.state);
      state.textContent = job.state === 'running' ? '运行中'
        : (part ? job.message : (job.state === 'done' ? '完成' : '失败'));
      const time = document.createElement('span');
      time.textContent = (job.started_at || '') + (job.finished_at ? ' → ' + job.finished_at : '');
      head.append(name, state, time);
      if (job.session_id) {
        const resume = document.createElement('code');
        resume.textContent = 'claude --resume ' + job.session_id;
        resume.title = '在终端里接着这个 Claude 会话聊';
        head.append(resume);
      }

      card.append(head);
      card.append(progressEl(job));
      if (job.queue && job.queue.length) {
        const q = document.createElement('div');
        q.className = 'jwhy';
        q.textContent = '⏳ 排队 ' + job.queue.length + ' 个:' + job.queue.join('、');
        card.append(q);
      }
      jobsBox.append(card);
    }
    renderGroupJobs(groups);
    // 模型下载排队:别的模型在下载时也能点(排到后面);只有这个模型正在下 / 已经在队列里才灰掉
    const dm = jobs.download_model || {};
    const modelBusy = new Set([...(dm.state === 'running' ? dm.items || [] : []),
      ...(dm.queue || []).flatMap((q) => q.split(', '))]);
    document.querySelectorAll('.download-model-btn').forEach((btn) => {
      btn.disabled = !serverMode || (btn.dataset.label ? modelBusy.has(btn.dataset.label) : dm.state === 'running');
    });
    const wandbBusy = Boolean(jobs.sync_wandb && jobs.sync_wandb.state === 'running');
    document.querySelectorAll('.sync-wandb-btn[data-policy]').forEach((btn) => {
      btn.disabled = !serverMode || wandbBusy;
    });
    document.querySelectorAll('.video-btn').forEach((btn) => {
      btn.disabled = !serverMode;
    });
  }

  async function pollJobs() {
    try {
      const data = await api('api/jobs');
      const was = serverMode;
      serverMode = true;
      renderJobs(data.jobs || {}, data.groups || {});
      if (was !== true) hint.textContent = '服务端模式：按钮会真正启动转换 / 上传任务';
    } catch (error) {
      const was = serverMode;
      serverMode = false;
      document.querySelectorAll('.download-model-btn, .sync-wandb-btn[data-policy], .video-btn')
        .forEach((btn) => { btn.disabled = true; });
      document.querySelectorAll('button.gact').forEach((btn) => { btn.disabled = true; });
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

  // 任务按钮(采集 / 转换 / 上传):点了弹出计划对话框,确认后在服务端后台运行 group_jobs.py
  const actionTitle = {ingest: '下载资产', collect: '采集', exchange: '转换 LeRobot', upload: '上传 ModelScope',
    eval: '评测', layouts: '生成评测布局'};
  async function groupAction(button, kind, group, task) {
    button.disabled = true;
    const policy = button.dataset.policy || '';
    const variant = button.dataset.variant || '';
    const title = (policy ? policy + ' ' : '') + (variant ? variant + ' ' : '') + (actionTitle[kind] || kind) + ' · '
      + (task === '__group__' ? group : task);
    const pending = kind === 'upload' ? '正在生成计划…\\n要对 ModelScope 做 dry-run,可能需要 1–3 分钟。'
      : (kind === 'ingest' ? '正在生成计划…\\n要跑一次 simready_assets.py --dry_run,可能要几十秒。' : '正在生成计划…');
    const {dialog, ok, cancel} = planDialog(title, pending, {pending: true});
    const text = $('plan-text');
    const fields = $('plan-fields');
    const controller = new AbortController();
    let closed = false;
    const close = () => {
      if (closed) return;
      closed = true;
      controller.abort();
      ok.onclick = null;
      cancel.onclick = null;
      dialog.oncancel = null;
      if (dialog.open) dialog.close();
      button.disabled = false;
      pollJobs();
    };
    const fail = (message) => {
      text.textContent = '⚠ ' + message;
      ok.style.display = 'none';
      cancel.textContent = '关闭';
    };
    cancel.onclick = close;
    dialog.oncancel = (event) => { event.preventDefault(); close(); };
    const base0 = 'api/group/' + kind + '?group=' + encodeURIComponent(group) + '&task=' + encodeURIComponent(task)
      + (policy ? '&policy=' + encodeURIComponent(policy) : '')
      + (variant ? '&variant=' + encodeURIComponent(variant) : '');
    let base = base0;
    // 采集 / 评测:先问在哪台机器上跑(docs/servers.json),选了再生成那台机器的计划;换机器就重新生成
    let machineCtl = null;
    if (kind === 'collect' || kind === 'eval') {
      try {
        const res = await api('api/group/machines?group=' + encodeURIComponent(group) + '&task=' + encodeURIComponent(task),
          {method: 'POST', signal: controller.signal});
        if (closed) return;
        const box = $('plan-machine');
        const head = document.createElement('div');
        head.className = 'mq';
        head.textContent = '在哪台机器上跑?(docs/servers.json)';
        box.append(head);
        let last = '';
        try { last = localStorage.getItem('rsj-machine-' + kind) || ''; } catch (e) { last = ''; }
        const radios = [];
        for (const m of res.machines || []) {
          const label = document.createElement('label');
          label.className = 'mopt' + (m.usable ? '' : ' off');
          const radio = document.createElement('input');
          radio.type = 'radio';
          radio.name = 'plan-machine';
          radio.value = m.key;
          radio.disabled = !m.usable;
          const name = document.createElement('b');
          name.textContent = m.label;
          const info = document.createElement('small');
          info.textContent = [m.gpus, m.description, m.usable ? '' : '不能用:' + m.why,
            m.key === last ? '上次用的' : ''].filter(Boolean).join(' · ');
          label.append(radio, name, info);
          box.append(label);
          radios.push(radio);
        }
        machineCtl = radios;
        text.textContent = '先选一台机器,选了再生成计划(远端要通过 SSH 查 GPU 和缺的东西,可能要几十秒)。';
      } catch (error) {
        if (closed || error.name === 'AbortError') return;
        fail('读机器列表失败:' + error.message);
        return;
      }
    }
    let planSeq = 0;
    async function loadPlan(machine) {
      const seq = ++planSeq;
      base = base0 + (machine ? '&machine=' + encodeURIComponent(machine) : '');
      fields.innerHTML = '';
      ok.style.display = 'none';
      cancel.textContent = '取消';
      text.textContent = machine ? '正在生成 ' + machine + ' 上的计划…' : '正在生成计划…';
      try {
        const plan = await api(base, {method: 'POST', signal: controller.signal});
        if (closed || seq !== planSeq) return;
        showPlan(plan, machine);
      } catch (error) {
        if (closed || error.name === 'AbortError' || seq !== planSeq) return;
        text.textContent = '⚠ ' + error.message;
        cancel.textContent = '关闭';
      }
    }
    function showPlan(plan, machine) {
      text.textContent = plan.text || '';
      if (plan.nothing) { cancel.textContent = '关闭'; return; }
      let gpuInput = null;
      let perGpuInput = null;
      let deleteBox = null;
      if (['collect', 'eval', 'layouts'].includes(plan.kind) && plan.need_gpus) {
        const label = document.createElement('label');
        label.textContent = {eval: '用哪几张 GPU(两两一对:policy 卡,sim 卡):',
          layouts: '用哪几张 GPU(一个任务一张):'}[plan.kind] || '用哪几张 GPU:';
        gpuInput = document.createElement('input');
        gpuInput.type = 'text';
        gpuInput.value = (plan.free_gpus || []).join(',');
        gpuInput.placeholder = '例:1,2,3';
        label.append(gpuInput, '(默认是' + (plan.machine ? ' ' + plan.machine + ' 上' : '') + '现在空闲的卡,逗号分隔)');
        fields.append(label);
      }
      if (plan.kind === 'collect' && plan.need_gpus) {
        const label = document.createElement('label');
        label.textContent = '每卡 worker 数:';
        perGpuInput = document.createElement('input');
        perGpuInput.type = 'text';
        perGpuInput.size = 3;
        perGpuInput.value = String(plan.per_gpu || 1);
        label.append(perGpuInput, '(一个 worker 约 14–20 GB 显存;默认按这台机器每卡显存估)');
        fields.append(label);
      }
      if (plan.kind === 'upload' && plan.deletes) {
        const label = document.createElement('label');
        deleteBox = document.createElement('input');
        deleteBox.type = 'checkbox';
        label.append(deleteBox, ' 同时删除远端 ' + plan.deletes + ' 个本地已没有的文件(默认保留)');
        fields.append(label);
      }
      ok.textContent = {ingest: '开始下载', collect: '开始采集', export: '开始转换', upload: '确认上传',
        eval: '开始评测', layouts: '开始生成'}[plan.kind]
        || '确认开始';
      ok.style.display = '';
      ok.disabled = false;
      ok.onclick = async () => {
        ok.disabled = true;
        const params = new URLSearchParams({confirm: '1'});
        if (gpuInput) params.set('gpus', gpuInput.value.trim());
        if (perGpuInput && perGpuInput.value.trim()) params.set('per_gpu', perGpuInput.value.trim());
        if (deleteBox && deleteBox.checked) params.set('delete', '1');
        try {
          const started = await api(base + '&' + params.toString(), {method: 'POST'});
          if (machine) { try { localStorage.setItem('rsj-machine-' + kind, machine); } catch (e) { /* 无痕模式 */ } }
          hint.textContent = started.message || '已启动';
          close();
        } catch (error) {
          fail('启动失败:' + error.message);
        }
      };
    }
    if (machineCtl) {
      machineCtl.forEach((radio) => { radio.onchange = () => { if (radio.checked) loadPlan(radio.value); }; });
      return;  // 等用户选机器
    }
    try {
      const plan = await api(base, {method: 'POST', signal: controller.signal});
      if (closed) return;
      showPlan(plan, '');
    } catch (error) {
      if (closed || error.name === 'AbortError') return;
      fail(error.message);
    }
  }

  document.addEventListener('click', async (event) => {
    const button = event.target.closest('button.gact');
    if (!button || button.disabled) return;
    const group = button.dataset.group;
    const task = button.dataset.task;
    if (button.dataset.kind === 'stop') {
      const policy = button.dataset.policy || '';
      const variant = button.dataset.variant || '';
      const sub = policy + (variant ? '@' + variant : '');
      const job = groupJobs[group + '.' + task + (sub ? '.' + sub : '')] || {};
      const ok = await confirmSimple('停止 ' + task, '停止 ' + task + ' 的' + (job.label || '任务') + '?\\n'
        + '采集的分片会保留,之后再点「采集」会续采;上传中断后再点会续传。');
      if (!ok) return;
      try {
        const params = new URLSearchParams({group: group, task: task, policy: policy, variant: variant});
        const data = await api('api/group/stop?' + params.toString(), {method: 'POST'});
        hint.textContent = data.message || '已停止';
      } catch (error) {
        alert('停止失败:' + error.message);
      }
      pollJobs();
      return;
    }
    groupAction(button, button.dataset.kind, group, task);
  });

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

  // 大表里的模型下载按钮:表格片段会重新读入,所以用委托
  document.addEventListener('click', (event) => {
    const btn = event.target.closest('.download-model-btn[data-policy]');
    if (!btn || btn.disabled) return;
    event.preventDefault();
    downloadModel(btn);
  });
  document.addEventListener('click', (event) => {  // 模型格的「⟳ 同步 W&B」:只同步这一个模型
    const btn = event.target.closest('.sync-wandb-btn[data-policy]');
    if (!btn || btn.disabled) return;
    event.preventDefault();
    const params = new URLSearchParams({policy: btn.dataset.policy, task: btn.dataset.task});
    startAction(btn, 'api/wandb/sync?' + params.toString(), 'api/wandb/sync?' + params.toString() + '&confirm=1',
      '同步 W&B · ' + btn.dataset.label + '（先下载缺失，再上传未同步）');
  });
  bindVideoButtons();

  scheduleAutoRefresh();
  pollJobs();
  setInterval(pollJobs, 5000);
})();
</script>
"""
    page_title = "RoboSafeJanus 流水线"
    page_meta = (
        f"生成时间 {html.escape(snapshot['generated_at'])}"
        f" · 状态 <code>data/pipeline/</code>"
        f" · HDF5 <code>{html.escape(snapshot['repo_id'] or '未配置')}</code>{remote_note}"
        f" · LeRobot <code>{html.escape(snapshot['lerobot_repo_id'] or '未配置')}</code>{lerobot_remote_note}"
        f" · 模型 <code>{html.escape(snapshot['model_repo_id'] or '未配置')}</code>{model_note}"
        f"{f' · 每 {refresh} 秒刷新（有没写完的说明时不刷新）' if refresh else ''}"
    )
    page_toolbar = (
        '  <button id="refresh-btn" type="button">🔄 手动刷新</button>\n'
        '  <span id="refresh-hint" class="hint">用 --serve 打开时按钮会在服务端执行任务</span>'
    )
    task_cells["_policies"] = cell_policies(snapshot, out_dir)  # 宽表按模型分行:每次刷新按 ModelScope 更新种类
    write_pipeline_page(out_dir, task_cells)  # 先写片段(含大表的数据格),页面里内嵌最新的
    boxes, pipe_assets = pipeline_parts(out_dir)
    table_html = "".join(boxes.values())
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
<style>{DASHBOARD_CSS}</style>
</head>
<body data-refresh="{refresh}">
<div class="page">
<header class="top">
  <div>
    <h1>{page_title}</h1>
    <div class="meta">{page_meta}</div>
  </div>
</header>
<div class="toolbar">
{page_toolbar}
</div>
<div id="jobs" class="jobs"></div>
{table_html}
</div>
{pipe_assets}
<dialog id="plan-dialog">
  <h2 id="plan-title"></h2>
  <div id="plan-machine" class="plan-machine"></div>
  <pre id="plan-text"></pre>
  <div id="plan-fields" class="plan-fields"></div>
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
            return self._json({"jobs": jobs_snapshot(), "groups": self._group_jobs()})
        if path == "/api/videos":
            return self._videos()
        if path == "/media":
            return self._media()
        if path in ("", "/", "/TASKS.html"):
            self.send_response(302)
            self.send_header("Location", "/PIPELINE.html#build" if path == "/TASKS.html" else "/PIPELINE.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        path = self._path()
        if path == "/api/refresh":
            return self._refresh()
        if path == "/api/download_model":
            return self._download_model()
        if path == "/api/wandb/sync":
            return self._sync_wandb()
        if path.startswith("/api/pipeline/"):
            return self._pipeline(path.rsplit("/", 1)[1])
        if path in ("/api/edit/load", "/api/edit/save"):
            return self._edit(path.rsplit("/", 1)[1])
        if path in ("/api/assets/load", "/api/assets/save"):
            return self._asset_links(path.rsplit("/", 1)[1])
        if path.startswith("/api/group/"):
            return self._group(path.rsplit("/", 1)[1])
        self.send_error(404)

    _group_states: dict = {}

    def _group_jobs(self) -> dict:
        """任务按钮的后台任务(group_jobs.py,键 ``<group>.<task>``,按模型的评测 ``<group>.<task>.<policy>``);
        刚结束的触发一次重建,页面上的条数 / 状态跟着更新。"""
        import group_jobs

        try:
            jobs = group_jobs.all_jobs()
        except Exception as error:  # noqa: BLE001 - 读不到任务记录不影响其他任务
            return {"_error": {"state": "failed", "label": "读取组任务", "step": str(error)}}
        seen = type(self)._group_states
        finished = [g for g, j in jobs.items() if seen.get(g) == "running" and j["state"] != "running"]
        seen.update({g: j["state"] for g, j in jobs.items()})
        callback = type(self).refresh_callback
        if finished and callback:
            threading.Thread(target=callback, daemon=True).start()
        return jobs

    def _group(self, action: str):
        """任务按钮:``ingest``(按组,task 为 ``__group__``)/ ``collect`` / ``exchange`` / ``upload``
        不带 confirm 时返回计划,
        带 ``confirm=1`` 时后台启动;``stop`` 停止。"""
        import group_jobs

        query = parse_qs(urlparse(self.path).query)
        group = (query.get("group") or [""])[0]
        task = (query.get("task") or [""])[0]
        policy = (query.get("policy") or [""])[0]  # eval 格按模型分开:只评这个模型
        variant = (query.get("variant") or [""])[0]  # eval 格按 S1 变体分开:评这个变体(空 = S1.0)
        machine = (query.get("machine") or [""])[0]  # 采集 / 评测:docs/servers.json 里选的机器
        per_gpu_s = (query.get("per_gpu") or [""])[0]
        if action == "machines":  # 采集 / 评测对话框里先选机器
            import servers

            try:
                return self._json({"machines": servers.machines()})
            except servers.ServerError as error:
                return self._json({"error": str(error)}, status=500)
        if machine and not re.fullmatch(r"[A-Za-z0-9_.-]+", machine):
            return self._json({"error": "机器名不合法"}, status=400)
        if per_gpu_s and not per_gpu_s.isdigit():
            return self._json({"error": "每卡 worker 数要填数字"}, status=400)
        per_gpu = int(per_gpu_s) if per_gpu_s else None
        if not re.fullmatch(r"[A-Za-z0-9_]+", group) or not re.fullmatch(r"[A-Za-z0-9_]+", task):
            return self._json({"error": "组名 / 任务名不合法"}, status=400)
        if policy and not re.fullmatch(r"[A-Za-z0-9_]+", policy):
            return self._json({"error": "模型名不合法"}, status=400)
        if variant and not re.fullmatch(r"S1\.[0-5]", variant):
            return self._json({"error": "S1 变体不合法"}, status=400)
        try:
            if action == "stop":
                return self._json({"message": group_jobs.stop(group, task, policy, variant)})
            if action not in group_jobs.BUTTONS:
                return self._json({"error": f"未知操作 {action}"}, status=404)
            if (query.get("confirm") or [""])[0] != "1":
                return self._json(group_jobs.plan(action, group, task, policy=policy, machine=machine,
                                                  variant=variant))
            job = group_jobs.start(action, group, task, (query.get("gpus") or [""])[0],
                                   (query.get("delete") or [""])[0] == "1", policy, machine, per_gpu, variant)
        except group_jobs.PlanError as error:
            return self._json({"error": str(error)}, status=409)
        except subprocess.TimeoutExpired:
            return self._json({"error": "dry-run 超时"}, status=504)
        return self._json({"started": True,
                           "message": f"已启动 {task} 的{group_jobs.job_label(job)},日志 {job['log']}"})

    def _pipeline(self, action: str):
        """流水线页上的操作:批准 / 驳回 / 重新提交 / 选资产 / 从种子开始(转成 rsj_pipeline.py 等命令行)。"""
        import pipeline_page

        config = type(self).job_config
        out_dir = Path(config.get("out_dir") or DEFAULT_OUT)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(min(length, 1 << 16)) or b"{}")
            command = pipeline_page.action_command(action, payload)
        except (ValueError, json.JSONDecodeError) as error:
            return self._json({"error": str(error)}, status=400)

        def render():
            write_pipeline_page(out_dir)

        if pipeline_page.ACTIONS[action]:
            try:
                launch_job("pipeline", shell_runner(command, ROOT_DIR), out_dir, render)
            except RuntimeError as error:
                return self._json({"error": str(error)}, status=409)
            return self._json({"started": True})
        try:
            code, output = pipeline_page.run_action(command, ROOT_DIR)
        except subprocess.TimeoutExpired:
            return self._json({"error": "超时"}, status=504)
        render()
        if code != 0:
            return self._json({"error": "未执行", "output": output}, status=409)
        return self._json({"output": output})

    def _edit(self, action: str):
        """点开一格后的编辑区:``load`` 返回这一格能改的东西,``save`` 回填到 JSON / YAML / Python(pipeline_edit.py)。"""
        import pipeline_edit

        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(min(length, 4 << 20)) or b"{}")
            group, key = str(payload.get("group") or ""), str(payload.get("key") or "")
            if action == "load":
                return self._json(pipeline_edit.load(group, key))
            result = pipeline_edit.save(group, key, payload.get("slot"), str(payload.get("text") or ""),
                                        str(payload.get("version") or ""))
        except (ValueError, json.JSONDecodeError) as error:
            return self._json({"error": f"请求不合法:{error}"}, status=400)
        except pipeline_edit.EditError as error:
            return self._json({"error": str(error)}, status=409)
        config = type(self).job_config
        threading.Thread(target=write_pipeline_page, args=(Path(config.get("out_dir") or DEFAULT_OUT),),
                         daemon=True).start()  # 格子里的状态 / 信息跟着更新
        return self._json(result)

    def _asset_links(self, action: str):
        """assets 格的「填资产链接」:``load`` 给该组现有的 label → 命令,``save`` 检查后回填资产 JSON。"""
        import pipeline_edit

        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(min(length, 1 << 20)) or b"{}")
            group = str(payload.get("group") or "")
            if action == "load":
                return self._json(pipeline_edit.assets_load(group))
            result = pipeline_edit.assets_save(group, payload.get("rows"), str(payload.get("version") or ""),
                                               dry_run=bool(payload.get("dry_run")))
        except (ValueError, json.JSONDecodeError) as error:
            return self._json({"error": f"请求不合法:{error}"}, status=400)
        except pipeline_edit.EditError as error:
            return self._json({"error": str(error)}, status=409)
        if result["errors"]:
            return self._json({**result, "error": f"{len(result['errors'])} 行有问题,没有保存"}, status=409)
        if result.get("message") and result["message"] != "没有改动":
            config = type(self).job_config
            threading.Thread(target=write_pipeline_page, args=(Path(config.get("out_dir") or DEFAULT_OUT),),
                             daemon=True).start()
        return self._json(result)

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
            attach_eval_safety(data["entries"], directory / "_result.json")
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

    def _download_model(self):
        config = type(self).job_config
        repo_id = config.get("model_repo_id")
        if not repo_id:
            return self._json({"error": "未配置模型仓库"}, status=400)
        query = parse_qs(urlparse(self.path).query)
        policy = (query.get("policy") or [None])[0]
        key = (query.get("task") or [None])[0]
        models, _wandb, error = collect_models(
            repo_id, task_groups(), config["model_local_dir"],
            config.get("wandb_dir") or DEFAULT_WANDB,
        )
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
        busy = [f"{m['policy']}/{m['key']}" for m in selected if model_downloading(m, running_model_downloads())]
        if busy:  # 看板重启后没记录的下载进程也在这里拦住,免得两个进程写同一批文件
            return self._json({"error": "已经在下载:" + "、".join(busy) + "(等它结束再点)"}, status=409)
        with JOBS_LOCK:
            queued = {item for entry in JOB_QUEUE.get("download_model") or [] for item in entry["items"]}
        queued = [f"{m['policy']}/{m['key']}" for m in selected if f"{m['policy']}/{m['key']}" in queued]
        if queued:
            return self._json({"error": "已经在下载队列里:" + "、".join(queued)}, status=409)
        tasks = [
            f"{model['policy']}/{model['key']} · {human_bytes(model['size'])} · 本地: "
            + (f"step {', '.join(model['local_steps'])}" if model["local_steps"] else "无")
            for model in selected
        ]
        if "confirm=1" in self.path:
            position = launch_job(
                "download_model", download_models_runner(selected, ROOT_DIR),
                config["out_dir"], config["rebuild"],
                progress=lambda: download_progress(selected, Path(config["model_local_dir"])),
                items=[f"{m['policy']}/{m['key']}" for m in selected], queue=True,
            )
            return self._json({
                "started": True,
                "message": (f"已排队:前面还有 {position} 个下载,轮到了自动开始" if position
                            else f"已启动 {len(selected)} 个模型下载"),
            })
        return self._json({
            "count": len(selected),
            "tasks": tasks[:100],
            "message": f"将下载 {len(selected)} 个模型的最新 step"
                       f"（已下载的文件会跳过）",
        })

    def _sync_wandb(self):
        """一个按钮同步 W&B：先下载缺失的 run，再把本地未上传的 run 传到 W&B 官方。"""
        config = type(self).job_config
        snapshot = type(self).snapshot or {}
        rows = [row for row in (snapshot.get("wandb") or []) if row.get("has_wandb", True)]
        query = parse_qs(urlparse(self.path).query)  # 模型格的按钮只同步这一个模型(policy + task)
        policy, key = (query.get("policy") or [None])[0], (query.get("task") or [None])[0]
        if policy and key:
            rows = [row for row in rows if row["policy"] == policy and row["key"] == key]
        if not rows:
            return self._json({
                "count": 0,
                "message": "模型仓库里没有 wandb 目录"
                           "（训练机上传 checkpoint 时没把 run 下的 wandb/ 一起传上去）",
            })
        if not config.get("model_repo_id"):
            return self._json({"error": "未配置模型仓库"}, status=400)
        to_download = [row for row in rows if not row["local"]["downloaded"]]
        download_paths = {row["path"] for row in to_download}
        venv_dir = Path(config.get("wandb_dir") or DEFAULT_WANDB).parent / ".cache" / "wandb" / "venv"
        cli, cli_note = find_wandb_cli(venv_dir)
        notes: list[str] = []
        if not wandb_api_key():
            notes.append("未配置 WANDB_API_KEY，本次只下载、不上传")
        elif not cli and not shutil.which("uv"):
            notes.append(f"没有可用的 wandb CLI，本次只下载：{cli_note}")
        can_upload = bool(wandb_api_key()) and bool(cli or shutil.which("uv"))
        to_upload = [
            row for row in rows
            if row["path"] in download_paths or (row["local"]["downloaded"] and not row["uploaded"])
        ] if can_upload else []
        if not to_download and not to_upload:
            suffix = "（" + "；".join(notes) + "）" if notes else ""
            return self._json({"count": 0, "message": f"W&B 已是最新：没有待下载或待上传的 run{suffix}"})
        tasks = [
            f'⬇️ {row["path"]} · {human_bytes(row["size"])} · {row["files"]} 文件'
            for row in to_download
        ] + [
            f'⬆️ {row["path"]} · {human_bytes(row["size"])} · 未上传'
            for row in to_upload
        ]
        if "confirm=1" in self.path:
            with JOBS_LOCK:
                if JOBS["sync_wandb"]["state"] == "running":
                    return self._json({"error": f"{JOB_LABELS['sync_wandb']}正在运行，等它结束后再同步"}, status=409)
            try:
                launch_job(
                    "sync_wandb",
                    sync_wandb_runner(
                        to_download, to_upload, ROOT_DIR, config.get("model_repo_id"),
                        config.get("wandb_entity"), config.get("wandb_project"),
                        config.get("wandb_proxy"),
                    ),
                    config["out_dir"], config["rebuild"],
                )
            except RuntimeError as error:
                return self._json({"error": str(error)}, status=409)
            return self._json({
                "started": True,
                "message": f"已启动 W&B 同步：下载 {len(to_download)} 个、上传 {len(to_upload)} 个 run",
            })
        lines = [
            f"将同步 W&B 数据：下载 {len(to_download)} 个 run → 上传 {len(to_upload)} 个 run 到 W&B 官方",
            f"本地目录: {config.get('wandb_dir')}",
            f"CLI: {cli_note}",
            f"entity: {config.get('wandb_entity') or '默认（API key 所属账号）'}",
            f"代理: {config.get('wandb_proxy') or '不使用代理'}",
            *notes,
        ]
        return self._json({
            "count": len(to_download) + len(to_upload),
            "tasks": tasks[:100],
            "message": "\n".join(lines),
        })

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):  # noqa: A002 - 与父类签名一致
        pass


def cell_policies(snapshot: dict, out_dir: Path) -> list[str]:
    """总览宽表模型 / 评测 / 结果按哪些模型分行 = ModelScope 模型仓库里有的 policy(都要评;Pi_0 / Pi_05 在前)。
    模型仓库这次没读到时沿用上一次的列表,免得网络抖一下行就少了。"""
    import group_jobs

    found = {m["policy"] for m in snapshot["models"]}
    if snapshot.get("model_error") or not found:
        try:
            import pipeline_page

            found |= set(json.loads((out_dir / pipeline_page.CELLS_FILE).read_text(encoding="utf-8"))
                         .get("_policies") or [])
        except (ImportError, OSError, ValueError, AttributeError):
            pass
    found.discard(None)
    found.discard("")
    order = group_jobs.EVAL_POLICIES
    return [*[p for p in order if p in found], *sorted(found - set(order))]


def write_pipeline_page(out_dir: Path, cells: dict | None = None) -> None:
    """流水线片段 data/pipeline/_fragments/*.html(``cells`` 为总览宽表每个任务的数据格);出错时保留上一次的片段。"""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import pipeline_page

        if cells is not None:
            path = out_dir / pipeline_page.CELLS_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cells, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        pipeline_page.render(out_dir)
    except Exception:  # noqa: BLE001 - 状态文件改到一半等情况,保留上一次的页面
        print("[sync_status] pipeline page failed:\n" + traceback.format_exc(), flush=True)


def write_outputs(snapshot: dict, out_dir: Path, refresh: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SYNC_STATUS.md").write_text(markdown(snapshot, refresh), encoding="utf-8")
    (out_dir / "SYNC_STATUS.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    page = out_dir / "PIPELINE.html"
    tmp = page.with_suffix(".html.tmp")
    tmp.write_text(html_page(snapshot, refresh, out_dir), encoding="utf-8")
    tmp.replace(page)
    try:
        import pipeline_page

        for old in pipeline_page.OLD_PAGES:  # 旧地址跳到对应标签
            (out_dir / old).write_text(pipeline_page.redirect_page(old), encoding="utf-8")
    except ImportError:
        pass
    (out_dir / "TASKS.html").unlink(missing_ok=True)
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
    parser.add_argument("--wandb_dir", default=str(DEFAULT_WANDB),
                        help="local wandb-run dir, mirrors <POLICY>/<task>/<run>/wandb/ in the model repo")
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"),
                        help="W&B entity for upload; default: entity of WANDB_API_KEY")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT"),
                        help="override W&B project for upload; default: project stored in the run")
    parser.add_argument("--wandb_proxy", default=os.environ.get("WANDB_PROXY", DEFAULT_WANDB_PROXY) or None,
                        help="HTTP(S) proxy for the W&B upload (api.wandb.ai needs it on this network); "
                             f"default WANDB_PROXY or {DEFAULT_WANDB_PROXY}, WANDB_PROXY='' for none")
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
    wandb_dir = Path(args.wandb_dir).resolve()
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
                layout_dir, args.eval_seed, args.eval_num, wandb_dir,
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
            "wandb_dir": wandb_dir,
            "wandb_entity": args.wandb_entity,
            "wandb_project": args.wandb_project,
            "wandb_proxy": args.wandb_proxy,
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
        print(f"[sync_status] serving {out_dir} at http://{args.host}:{args.serve}/PIPELINE.html",
              flush=True)

    while True:
        if not refresh and not server:
            rebuild()
            return 0
        try:
            rebuild()
        except Exception:  # noqa: BLE001 - 常驻看板:一次刷新失败(如任务定义改到一半)不退出,保留上一次的页面
            print(f"[sync_status] {time.strftime('%Y-%m-%dT%H:%M:%S')} rebuild failed, keeping previous pages:",
                  flush=True)
            traceback.print_exc()
            sys.stdout.flush()
        try:
            time.sleep(refresh if refresh else 3600)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
