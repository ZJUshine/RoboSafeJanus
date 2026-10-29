"""把本地 LeRobot 导出同步到 ModelScope 数据集仓库。

本地布局(``scripts/export_lerobot.sh`` 的输出)::

    data/RoboSafeJanus-Benchmark-Lerobot/<v30|v21>/<group>/<level>/<task>/{meta,data,videos}

远端沿用同样的镜像布局,即 ``<repo>/<v30|v21>/<group>/<level>/<task>/...``。这样 v3.0 与
v2.1 不会混在一起,也不会覆盖仓库里旧的 ``RoboSafeJanus-<task>-<robot>-<action>/`` 平铺数据集。

行为与 ``upload_to_modelscope.py`` 一致:先建上传视图/断点缓存,上传新增或修改的文件,
最后删除远端 ``v30/`` / ``v21/`` 下本地已不存在的文件(``--no_delete`` 关闭删除)。
只处理带 ``meta/info.json`` 的数据集目录。

用法::

    python src/data_collect/upload_lerobot_to_modelscope.py --repo_id <user>/<repo> --dry_run
    python src/data_collect/upload_lerobot_to_modelscope.py --repo_id <user>/<repo>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from src.data_collect.benchmark_layout import task_groups  # noqa: E402

try:
    from modelscope_hub import HubApi
    from modelscope_hub.constants import Visibility
except ImportError:
    sys.exit('[upload_lerobot] modelscope_hub not found; run: pip install -U "modelscope>=1.40"')

VERSIONS = ("v30", "v21")
DEFAULT_LOCAL = ROOT_DIR / "data" / "RoboSafeJanus-Benchmark-Lerobot"


def human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1000
    return f"{value:.2f} TB"


def iter_datasets(local_dir: Path) -> list[dict]:
    """``[{version, group, level, task, path, files: {相对路径: Path}}]``。"""
    datasets = []
    for version in VERSIONS:
        root = local_dir / version
        if not root.is_dir():
            continue
        for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for level_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                for task_dir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
                    info = task_dir / "meta" / "info.json"
                    if not info.is_file():
                        continue
                    files = {
                        path.relative_to(local_dir).as_posix(): path
                        for path in task_dir.rglob("*")
                        if path.is_file()
                    }
                    datasets.append({
                        "version": version,
                        "group": group_dir.name,
                        "level": level_dir.name,
                        "task": task_dir.name,
                        "path": task_dir.relative_to(local_dir).as_posix(),
                        "files": files,
                    })
    return datasets


def keep_remote(path: str) -> bool:
    return path.split("/", 1)[0] in VERSIONS


def running_exports() -> list[str]:
    """正在运行的 LeRobot 转换进程(导出会重写/清理数据集目录)。"""
    try:
        result = subprocess.run(
            ["pgrep", "-af", r"export_lerobot\.sh|transform_lerobot_v(30|21)_format\.py"],
            capture_output=True, text=True,
        )
    except OSError:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def build_stable_view(local_files: dict[str, Path], view_dir: Path) -> int:
    """把要上传的文件硬链接到稳定视图,避免上传过程中导出进程删除文件。"""
    shutil.rmtree(view_dir, ignore_errors=True)
    view_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for rel, src in local_files.items():
        dst = view_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            try:
                shutil.copy2(src, dst)
            except OSError as error:
                print(f"[upload_lerobot] skip {rel}: {error}")
                continue
        linked += 1
    return linked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", default=os.environ.get("MS_LEROBOT_REPO_ID"),
                        help="<user or org>/<repo>; 也可用环境变量 MS_LEROBOT_REPO_ID")
    parser.add_argument("--local_dir", default=str(DEFAULT_LOCAL))
    parser.add_argument("--repo_type", default="dataset", choices=("dataset", "model"))
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_delete", action="store_true", help="不删除远端多出来的文件")
    parser.add_argument("--include_non_json", action="store_true",
                        help="也上传不在 task/robosafejanus_tasks.json 中的任务")
    parser.add_argument("--dry_run", action="store_true", help="只列出要上传 / 删除的文件")
    parser.add_argument("--force", action="store_true", help="即使有 LeRobot 转换进程在运行也继续")
    args = parser.parse_args()

    if not args.repo_id:
        parser.error("--repo_id is required (or set MS_LEROBOT_REPO_ID)")
    if not args.force:
        blockers = running_exports()
        if blockers:
            print("[upload_lerobot] LeRobot 转换正在运行，等它结束后再上传（或加 --force）：",
                  file=sys.stderr)
            for line in blockers:
                print(f"  {line}", file=sys.stderr)
            return 2
    local_dir = Path(args.local_dir).resolve()
    if not local_dir.is_dir():
        sys.exit(f"[upload_lerobot] local_dir not found: {local_dir}")
    datasets = iter_datasets(local_dir)
    if not datasets:
        sys.exit(f"[upload_lerobot] no LeRobot datasets under {local_dir}; refusing to sync")
    if not args.include_non_json:
        json_tasks = set(task_groups())
        skipped = [dataset for dataset in datasets if dataset["task"] not in json_tasks]
        datasets = [dataset for dataset in datasets if dataset["task"] in json_tasks]
        for dataset in skipped:
            print(f"[upload_lerobot] skip {dataset['path']}: not in robosafejanus_tasks.json")
        if not datasets:
            sys.exit(f"[upload_lerobot] no datasets for JSON tasks under {local_dir}; refusing to sync")

    local_files = {path: src for dataset in datasets for path, src in dataset["files"].items()}
    local_sizes = {path: src.stat().st_size for path, src in local_files.items()}
    total_bytes = sum(local_sizes.values())
    print(f"[upload_lerobot] local: {len(datasets)} datasets, {len(local_files)} files, "
          f"{human_bytes(total_bytes)} under {local_dir}")

    api = HubApi(endpoint=args.endpoint)
    remote_sizes: dict[str, int] = {}
    if api.repo_exists(args.repo_id, args.repo_type):
        for item in api.list_repo_files(args.repo_id, args.repo_type, recursive=True):
            if not getattr(item, "type", "blob") == "tree" and keep_remote(item.path):
                remote_sizes[item.path] = getattr(item, "size", 0) or 0
    remote_tasks = len({"/".join(path.split("/")[:4]) for path in remote_sizes})
    print(f"[upload_lerobot] remote: {remote_tasks} dataset dirs, {len(remote_sizes)} files, "
          f"{human_bytes(sum(remote_sizes.values()))}")
    web_endpoint = (args.endpoint or os.environ.get("MODELSCOPE_ENDPOINT") or "https://www.modelscope.cn").rstrip("/")
    repo_url = f"{web_endpoint}/{'datasets' if args.repo_type == 'dataset' else 'models'}/{args.repo_id}"
    print(f"[upload_lerobot] remote: {repo_url} ({'/'.join(VERSIONS)}/)")

    total_new = total_changed = 0
    for version in VERSIONS:
        for dataset in (d for d in datasets if d["version"] == version):
            new = changed = 0
            size = 0
            for path, local_path in dataset["files"].items():
                remote_size = remote_sizes.get(path)
                if remote_size is None:
                    new += 1
                    size += local_sizes[path]
                elif remote_size != local_sizes[path]:
                    changed += 1
                    size += local_sizes[path]
            total_new += new
            total_changed += changed
            if new or changed:
                print(f"  upload {dataset['path']}  {new} new, {changed} changed ({human_bytes(size)})")
    stale = sorted(
        path for path in remote_sizes
        if path not in local_files
        and any(path.startswith(f"{dataset['path']}/") for dataset in datasets)
    )
    upload_size = sum(
        local_sizes[path] for path in local_files
        if path not in remote_sizes or remote_sizes[path] != local_sizes[path]
    )
    print(f"[upload_lerobot] total: upload {total_new + total_changed} files "
          f"({total_new} new + {total_changed} changed, {human_bytes(upload_size)}), "
          f"delete {len(stale)} remote files")

    if args.dry_run:
        return 0
    if not total_new and not total_changed and not stale:
        print("[upload_lerobot] nothing to do")
        return 0

    user = api.whoami()
    print(f"[upload_lerobot] logged in as {user.username}")
    if not api.repo_exists(args.repo_id, args.repo_type):
        api.create_repo(args.repo_id, args.repo_type, visibility=Visibility.PRIVATE)
        print(f"[upload_lerobot] created private {args.repo_type} {args.repo_id}")
    info = api.get_repo(args.repo_id, args.repo_type)
    if info.private is False or (info.visibility is not None and info.visibility != Visibility.PRIVATE):
        sys.exit(f"[upload_lerobot] {args.repo_id} is not private (visibility={info.visibility}); "
                 f"make it private on the website first")

    tracker_path = (
        local_dir.parent / ".cache" / "modelscope"
        / f"upload_{args.repo_id.replace('/', '__')}_lerobot.json"
    )
    view_dir = local_dir.parent / ".cache" / "lerobot_upload_view"
    linked = build_stable_view(local_files, view_dir)
    print(f"[upload_lerobot] prepared stable view: {linked}/{len(local_files)} files in {view_dir}")
    api.upload_folder(
        args.repo_id,
        args.repo_type,
        str(view_dir),
        allow_patterns=[f"{version}/*" for version in VERSIONS],
        max_workers=args.num_workers,
        use_cache=True,
        tracker_path=str(tracker_path),
        commit_message="Sync LeRobot exports",
    )

    if not args.no_delete and stale:
        api.delete_files(args.repo_id, args.repo_type, stale,
                         commit_message=f"Delete {len(stale)} stale LeRobot files")
        print(f"[upload_lerobot] deleted {len(stale)} stale files")
    print("[upload_lerobot] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
