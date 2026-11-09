"""把本地资产目录 ``data/RoboSafeJanus-Asset``(``Assets/`` + ``ext_assets/``)同步到 ModelScope 私有 dataset
``zjushine/RoboSafeJanus-Asset``,目录结构与远端一致(``scripts/init_assets.sh`` 下载到同一个位置)。

只传 ``Assets/``、``ext_assets/robodojo_format/``、``ext_assets/integrate/``(包装物体引用的原件和生成脚本),只传新增 /
修改过的文件,不删除远端多出来的文件。已提交的文件记在 ``.cache/modelscope/upload_RoboSafeJanus-Asset.json``
(key 是"仓库内路径|mtime|大小"),中断后重跑即续传::

    python scripts/internal/upload_assets.py --dry_run   # 列出要上传的文件,不上传
    python scripts/internal/upload_assets.py
"""

import argparse
import collections
import fnmatch
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACKER = os.path.join(ROOT, ".cache", "modelscope", "upload_RoboSafeJanus-Asset.json")
ALLOW = ["Assets/*", "ext_assets/robodojo_format/*", "ext_assets/integrate/*"]
IGNORE = ["*/__pycache__/*", "*.pyc", "*/.ms_upload_cache"]


def _local_files(asset_dir):
    """``asset_dir`` 下要上传的文件:``[(仓库内路径, 本地路径), ...]``。"""
    out = []
    for top, _, files in os.walk(asset_dir):
        for name in files:
            path = os.path.join(top, name)
            rel = os.path.relpath(path, asset_dir).replace(os.sep, "/")
            if any(fnmatch.fnmatch(rel, p) for p in ALLOW) and not any(fnmatch.fnmatch(rel, p) for p in IGNORE):
                out.append((rel, path))
    return sorted(out)


def _pending(files, tracker_path):
    """缓存里没有记为已提交(按 路径 + mtime + 大小)的文件。"""
    committed = set()
    if os.path.exists(tracker_path):
        for key, entry in json.load(open(tracker_path))["files"].items():
            if entry.get("status") == "c":
                committed.add(key)
    pending = []
    for rel, path in files:
        st = os.stat(path)
        if f"{rel}|{st.st_mtime}|{st.st_size}" not in committed:
            pending.append((rel, st.st_size))
    return pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="zjushine/RoboSafeJanus-Asset")
    parser.add_argument("--asset_dir", default=os.path.join(ROOT, "data", "RoboSafeJanus-Asset"))
    parser.add_argument("--tracker", default=TRACKER)
    parser.add_argument("--dry_run", action="store_true", help="list the files that would be uploaded and exit")
    parser.add_argument("--message", default="Update RoboSafeJanus assets")
    args = parser.parse_args()

    asset_dir = os.path.realpath(args.asset_dir)
    for sub in ("Assets", "ext_assets"):
        if not os.path.isdir(os.path.join(asset_dir, sub)):
            raise SystemExit(f"[upload_assets] {asset_dir}/{sub} not found")
    files = _local_files(asset_dir)
    pending = _pending(files, args.tracker)
    print(
        f"[upload_assets] {len(files)} files under {asset_dir}; {len(pending)} new or changed "
        f"({sum(s for _, s in pending) / 1e6:.1f} MB)"
    )
    by_dir = collections.Counter("/".join(rel.split("/")[:4]) for rel, _ in pending)
    for d, n in sorted(by_dir.items()):
        print(f"  {n:6d}  {d}")
    if args.dry_run or not pending:
        return

    from modelscope_hub import HubApi

    os.makedirs(os.path.dirname(args.tracker), exist_ok=True)
    HubApi().upload_folder(
        args.repo_id,
        "dataset",
        asset_dir,
        commit_message=args.message,
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
        use_cache=True,
        tracker_path=args.tracker,
    )
    print(f"[upload_assets] done: {args.repo_id}")


if __name__ == "__main__":
    main()
