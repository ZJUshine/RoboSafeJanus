#!/usr/bin/env python3
"""把 S1 变体的评测结果同步到 ModelScope dataset ``zjushine/RoboSafeJanus-Evaluation``。

远端布局 ``<task>/<variant>/<policy>/<run_id>/``(``run_id`` 如 ``2026-11-18_21-13-57``),传 run 目录
顶层的 ``_result.json``、全部 ``*.mp4`` 和 ``data/*.hdf5``(评测时加了 ``--save-traj`` 才有轨迹),
``_stream/`` 不传。按 (路径, 大小) 和远端比对,只传新增 / 变化的文件;中断后再跑会续传。上传完重新列
一遍远端逐文件核对。

    python scripts/internal/upload_eval_results.py --task put_bread_toaster --policy Pi_0 --variant S1.1 --dry_run
    python scripts/internal/upload_eval_results.py --task put_bread_toaster --policy Pi_0 --variant S1.1
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EVAL_RESULT = ROOT / "eval_result" / "RoboDojo"
CACHE = ROOT / "data" / ".cache" / "modelscope"
DEFAULT_REPO = os.environ.get("MS_EVAL_REPO_ID", "zjushine/RoboSafeJanus-Evaluation")
PAYLOAD_PATTERNS = ["_result.json", "*.mp4", "data/*.hdf5"]  # 结果 + 全部相机视频 + 轨迹(有才传)
IGNORE_PATTERNS = ["_stream/*", "_stream/**"]


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def run_dirs(task: str, policy: str, variant: str) -> list[tuple[str, str, Path]]:
    """``[(policy, run_id, run_dir), ...]``:有 ``_result.json`` 的 run。``policy`` 为空时所有模型。"""
    out = []
    for policy_dir in sorted((EVAL_RESULT / task).glob("*")):
        if not policy_dir.is_dir() or (policy and policy_dir.name != policy):
            continue
        for base in sorted(policy_dir.glob(f"*/*,variant={variant}")):
            for result in sorted(base.glob("*/_result.json")):
                out.append((policy_dir.name, result.parent.name, result.parent))
    return out


def local_files(run_dir: Path) -> list[tuple[str, int]]:
    """要上传的文件 ``[(远端相对路径, 大小), ...]``:``_result.json`` + 顶层视频 + ``data/*.hdf5``。"""
    files = sorted(p for p in run_dir.iterdir()
                   if p.is_file() and (p.name == "_result.json" or p.suffix == ".mp4"))
    files += sorted((run_dir / "data").glob("*.hdf5")) if (run_dir / "data").is_dir() else []
    return [(p.relative_to(run_dir).as_posix(), p.stat().st_size) for p in files]


def remote_sizes(api, repo: str) -> tuple[dict[str, int | None], str | None]:
    """远端 ``{路径: 大小}``;列不出来时返回 ``({}, 原因)``。"""
    try:
        items = api.list_repo_files(repo, "dataset", recursive=True)
    except Exception as error:  # noqa: BLE001 - 远端列不出来时按全部要传处理
        return {}, f"{type(error).__name__}: {error}"
    out: dict[str, int | None] = {}
    for item in items:
        path = getattr(item, "path", None) or (item.get("path") if isinstance(item, dict) else None)
        if not path or getattr(item, "type", "blob") == "tree":
            continue
        out[path] = getattr(item, "size", None) or (item.get("size") if isinstance(item, dict) else None)
    return out, None


def remote_path(task: str, variant: str, policy: str, run_id: str, name: str) -> str:
    return f"{task}/{variant}/{policy}/{run_id}/{name}"


def dry_run(api, repo: str, task: str, variant: str, runs: list[tuple[str, str, Path]]) -> int:
    """列出要传的文件;最后一行 ``pending=<n> bytes=<n>`` 给看板解析。"""
    remote, error = remote_sizes(api, repo)
    total_files = total_bytes = pending = pending_bytes = 0
    print(f"[upload_eval] {len(runs)} 个 run,远端 {repo}")
    for policy, run_id, run_dir in runs:
        files = local_files(run_dir)
        size = sum(n for _, n in files)
        new = missing = 0
        for name, length in files:
            total_files += 1
            total_bytes += length
            have = remote.get(remote_path(task, variant, policy, run_id, name))
            if have is None:
                new += 1
                pending += 1
                pending_bytes += length
            elif have != length:
                missing += 1
                pending += 1
                pending_bytes += length
        note = f"新增 {new}" if error else f"新增 {new},大小不同 {missing}"
        print(f"  {policy}/{run_id}: {len(files)} 个文件 {human(size)}(远端 {note})")
    if error:
        print(f"[upload_eval] 远端没列出来({error}),按全部要传算")
    print(f"[upload_eval] 合计 {total_files} 个文件 {human(total_bytes)};"
          f"要传 {pending} 个 {human(pending_bytes)}")
    print(f"[upload_eval] pending={pending} bytes={pending_bytes}")
    return 0


def upload(api, repo: str, task: str, variant: str, runs: list[tuple[str, str, Path]]) -> int:
    """逐个 run 上传(同一个仓库的评测上传串行,避免同时 commit 冲突);已传过的跳过。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / ".upload_eval.lock", "w") as lock:  # noqa: SIM115 - 持有到上传结束
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, (policy, run_id, run_dir) in enumerate(runs, 1):
            files = local_files(run_dir)
            print(f"[upload_eval] {policy}/{run_id}:上传 {len(files)} 个文件 {human(sum(n for _, n in files))}",
                  flush=True)

            def on_progress(state: dict, index: int = index, policy: str = policy) -> None:
                done, total = state.get("committed_files"), state.get("total_files")
                if not total or done is None:
                    return
                print(f"[upload_eval] upload {index}/{len(runs)} · {done}/{total} files "
                      f"{human(state.get('committed_bytes') or 0)}", flush=True)

            tracker = CACHE / f"upload_eval_{task}__{variant}__{policy}__{run_id}.json"
            api.upload_folder(
                repo, "dataset", str(run_dir),
                path_in_repo=f"{task}/{variant}/{policy}/{run_id}",
                allow_patterns=PAYLOAD_PATTERNS,
                ignore_patterns=IGNORE_PATTERNS,
                commit_message=f"Upload {task} {variant} {policy} {run_id}",
                use_cache=True,
                disable_tqdm=True,
                tracker_path=str(tracker),
                progress_callback=on_progress,
            )
            print(f"[upload_eval] done {index}/{len(runs)} runs", flush=True)
    return 0


def verify(api, repo: str, task: str, variant: str, runs: list[tuple[str, str, Path]]) -> int:
    """上传后重新列远端,逐文件比对大小;有缺失 / 不一致就返回 1。"""
    remote, error = remote_sizes(api, repo)
    if error:
        print(f"[upload_eval] 核对失败,列不出远端:{error}")
        return 1
    checked = bad = 0
    for policy, run_id, run_dir in runs:
        for name, length in local_files(run_dir):
            checked += 1
            have = remote.get(remote_path(task, variant, policy, run_id, name))
            if have != length:
                bad += 1
                state = "远端没有" if have is None else f"大小不同({have} != {length})"
                print(f"[upload_eval] ✗ {policy}/{run_id}/{name}:{state}")
    if bad:
        print(f"[upload_eval] 核对不通过:{bad}/{checked} 个文件不一致")
        return 1
    print(f"[upload_eval] verified {checked} 个文件已和远端一致")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", default="", help="只传这个模型;默认所有模型")
    parser.add_argument("--variant", required=True, help="S1 变体(S1.1 …)")
    parser.add_argument("--repo_id", default=DEFAULT_REPO,
                        help="默认 $MS_EVAL_REPO_ID 或 zjushine/RoboSafeJanus-Evaluation")
    parser.add_argument("--dry_run", action="store_true", help="只列出要传的文件,不上传")
    args = parser.parse_args()

    try:
        from modelscope_hub import HubApi
    except ImportError:
        print('[upload_eval] modelscope_hub not found; run: pip install -U "modelscope>=1.40"')
        return 2

    runs = run_dirs(args.task, args.policy, args.variant)
    if not runs:
        print(f"[upload_eval] {args.task} {args.policy} 没有 {args.variant} 的评测结果")
        return 1
    api = HubApi()
    if args.dry_run:
        return dry_run(api, args.repo_id, args.task, args.variant, runs)
    code = upload(api, args.repo_id, args.task, args.variant, runs)
    return verify(api, args.repo_id, args.task, args.variant, runs) if code == 0 else code


if __name__ == "__main__":
    sys.exit(main())
