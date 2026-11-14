"""把本地 LeRobot 导出同步到 ModelScope 数据集仓库。

本地布局(``scripts/export_lerobot.sh`` 的输出)::

    data/RoboSafeJanus-Benchmark-Lerobot/<v30|v21>/<group>/<level>/<task>/{meta,data,videos}

远端沿用同样的镜像布局,即 ``<repo>/<v30|v21>/<group>/<level>/<task>/...``。这样 v3.0 与
v2.1 不会混在一起,也不会覆盖仓库里旧的 ``RoboSafeJanus-<task>-<robot>-<action>/`` 平铺数据集。

行为与 ``upload_to_modelscope.py`` 一致:按 **sha256 内容哈希** 比对远端(远端列表里的 ``sha256``
与本地 ``sha256sum`` 一致),只上传内容不同的文件,所以"同名、同大小、只是内容变了"也会重传;
上传前清掉这些文件在断点缓存里的条目,SDK 不会按 mtime/size 跳过;最后删除远端 ``v30/`` /
``v21/`` 下本地已不存在的文件(``--no_delete`` 关闭删除),再校验一遍大小和 sha256
(``--no_verify`` 关闭校验)。只处理带 ``meta/info.json`` 的数据集目录。

用法::

    python src/data_collect/upload_lerobot_to_modelscope.py --repo_id <user>/<repo> --dry_run
    python src/data_collect/upload_lerobot_to_modelscope.py --repo_id <user>/<repo>
    python src/data_collect/upload_lerobot_to_modelscope.py --repo_id <user>/<repo> --group <group> [--dry_run]
    python src/data_collect/upload_lerobot_to_modelscope.py --repo_id <user>/<repo> --task <task> [--dry_run]

``--group`` / ``--task``:只同步这些组的自有任务 / 这些任务的数据集(v30 和 v21),删除也只在这些数据集目录里;
不带 ``--dry_run`` 时要求每个任务的流水线 expert 阶段已批准
(``rsj_pipeline.py gate <group> expert --task <task> --approved``)。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from src.data_collect.benchmark_layout import task_groups  # noqa: E402
from src.data_collect.upload_common import (  # noqa: E402
    local_sha256_map,
    plan_sync,
    purge_tracker,
    remote_file_map,
    require_pipeline_approval,
    scoped_tasks,
    verify_sync,
)

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
            ["pgrep", "-af", r"export_lerobot\.sh|transform_lerobot_v(30|21)_format\.py|lerobot_v30_to_v21\.py"],
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
    parser.add_argument("--hash_workers", type=int, default=None, help="本地 sha256 计算线程数(默认 8)")
    parser.add_argument("--no_delete", action="store_true", help="不删除远端多出来的文件")
    parser.add_argument("--no_verify", action="store_true", help="上传后不校验远端 sha256")
    parser.add_argument("--include_non_json", action="store_true",
                        help="也上传不在任务定义(task/robosafejanus_tasks)中的任务")
    parser.add_argument("--dry_run", action="store_true", help="只列出要上传 / 删除的文件")
    parser.add_argument("--force", action="store_true", help="即使有 LeRobot 转换进程在运行也继续")
    parser.add_argument("--group", nargs="+", default=[],
                        help="只同步这些组的自有任务;真正上传要求每个任务的流水线 expert 阶段已批准")
    parser.add_argument("--task", nargs="+", default=[],
                        help="只同步这些任务;真正上传要求流水线 expert 阶段已批准")
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
    if args.group or args.task:
        scoped = set(scoped_tasks(args.group, args.task))
        if not args.dry_run:
            require_pipeline_approval(sorted(scoped), "upload_lerobot")
        datasets = [dataset for dataset in datasets if dataset["task"] in scoped]
        scope = " ".join(f"--{k} {' '.join(v)}" for k, v in (("group", args.group), ("task", args.task)) if v)
        print(f"[upload_lerobot] {scope}: "
              f"{len(datasets)} datasets")
    if not datasets:
        sys.exit(f"[upload_lerobot] no LeRobot datasets under {local_dir}; refusing to sync")
    if not args.include_non_json:
        json_tasks = set(task_groups())
        skipped = [dataset for dataset in datasets if dataset["task"] not in json_tasks]
        datasets = [dataset for dataset in datasets if dataset["task"] in json_tasks]
        for dataset in skipped:
            print(f"[upload_lerobot] skip {dataset['path']}: not in task/robosafejanus_tasks")
        if not datasets:
            sys.exit(f"[upload_lerobot] no datasets for JSON tasks under {local_dir}; refusing to sync")

    local_files = {path: src for dataset in datasets for path, src in dataset["files"].items()}
    local_sizes = {path: src.stat().st_size for path, src in local_files.items()}
    total_bytes = sum(local_sizes.values())
    print(f"[upload_lerobot] local: {len(datasets)} datasets, {len(local_files)} files, "
          f"{human_bytes(total_bytes)} under {local_dir}")

    api = HubApi(endpoint=args.endpoint)
    remote_map = remote_file_map(api, args.repo_id, args.repo_type, keep_remote)
    remote_tasks = len({"/".join(path.split("/")[:4]) for path in remote_map})
    print(f"[upload_lerobot] remote: {remote_tasks} dataset dirs, {len(remote_map)} files, "
          f"{human_bytes(sum(item.size for item in remote_map.values()))}")
    web_endpoint = (args.endpoint or os.environ.get("MODELSCOPE_ENDPOINT") or "https://www.modelscope.cn").rstrip("/")
    repo_url = f"{web_endpoint}/{'datasets' if args.repo_type == 'dataset' else 'models'}/{args.repo_id}"
    print(f"[upload_lerobot] remote: {repo_url} ({'/'.join(VERSIONS)}/)")

    cache_dir = local_dir.parent / ".cache" / "modelscope"
    tracker_path = cache_dir / f"upload_{args.repo_id.replace('/', '__')}_lerobot.json"
    hash_cache_path = cache_dir / f"local_sha256_{args.repo_id.replace('/', '__')}_lerobot.json"
    view_dir = local_dir.parent / ".cache" / "lerobot_upload_view"

    started = time.time()
    try:
        local_hashes = local_sha256_map(local_files, hash_cache_path, args.hash_workers)
    except RuntimeError as error:
        sys.exit(f"[upload_lerobot] {error}")
    print(f"[upload_lerobot] hashed {len(local_hashes)} local files in {time.time() - started:.1f}s")
    to_upload, stale = plan_sync(local_files, local_sizes, local_hashes, remote_map)
    dataset_prefixes = tuple(f"{dataset['path']}/" for dataset in datasets)
    remote_in_scope = {path for path in remote_map if path.startswith(dataset_prefixes)}
    stale = sorted(remote_in_scope - set(local_files))

    for version in VERSIONS:
        for dataset in (d for d in datasets if d["version"] == version):
            prefix = f"{dataset['path']}/"
            uploads = [path for path in to_upload if path.startswith(prefix)]
            if uploads:
                print(f"  upload {dataset['path']}  {len(uploads)} files "
                      f"({human_bytes(sum(local_sizes[path] for path in uploads))})")
    new = sum(1 for reason in to_upload.values() if reason in ("new", "unverified"))
    changed = len(to_upload) - new
    upload_size = sum(local_sizes[path] for path in to_upload)
    print(f"[upload_lerobot] total: upload {len(to_upload)} files "
          f"({new} new + {changed} changed, {human_bytes(upload_size)}), "
          f"delete {len(stale)} remote files")

    if args.dry_run:
        return 0

    if to_upload or (stale and not args.no_delete):
        user = api.whoami()
        print(f"[upload_lerobot] logged in as {user.username}")
        if not api.repo_exists(args.repo_id, args.repo_type):
            api.create_repo(args.repo_id, args.repo_type, visibility=Visibility.PRIVATE)
            print(f"[upload_lerobot] created private {args.repo_type} {args.repo_id}")
        info = api.get_repo(args.repo_id, args.repo_type)
        if info.private is False or (info.visibility is not None and info.visibility != Visibility.PRIVATE):
            sys.exit(f"[upload_lerobot] {args.repo_id} is not private (visibility={info.visibility}); "
                     f"make it private on the website first")

    if to_upload:
        linked = build_stable_view({path: local_files[path] for path in to_upload}, view_dir)
        purged = purge_tracker(str(tracker_path), to_upload)
        print(f"[upload_lerobot] uploading {len(to_upload)} changed files "
              f"(stable view {linked}, purged {purged} tracker entries so nothing is skipped)")
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
    else:
        print("[upload_lerobot] remote content already matches local; nothing to upload")

    if args.no_delete:
        print(f"[upload_lerobot] --no_delete: keeping {len(stale)} remote files absent locally")
    elif stale:
        api.delete_files(args.repo_id, args.repo_type, stale,
                         commit_message=f"Delete {len(stale)} stale LeRobot files")
        print(f"[upload_lerobot] deleted {len(stale)} stale files")

    if not args.no_verify:
        problems = []
        for attempt in range(1, 4):
            problems = verify_sync(local_files, local_sizes, local_hashes,
                                   remote_file_map(api, args.repo_id, args.repo_type, keep_remote))
            if not problems:
                break
            print(f"[upload_lerobot] verify attempt {attempt}/3: {len(problems)} mismatch(es)")
            if attempt < 3:
                time.sleep(5)
        if problems:
            for path, reason in problems[:20]:
                print(f"  MISMATCH {path}: {reason}")
            if len(problems) > 20:
                print(f"  ... and {len(problems) - 20} more")
            sys.exit(f"[upload_lerobot] verification failed: {len(problems)} file(s) out of sync")
        print(f"[upload_lerobot] verified: {len(local_files)} files match remote sha256")
    print("[upload_lerobot] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
