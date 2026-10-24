"""把采集数据目录(默认 ``data/RoboSafeJanus``)同步到 Hugging Face 私有 dataset 仓库。

远端布局与本地一致:``data/RoboSafeJanus/<task>/<config>/...`` → ``<repo>/RoboSafeJanus/<task>/<config>/...``。

每次运行做两步,可以反复执行:
    1. 上传:``HfApi.upload_large_folder`` 多线程、可断点续传,只传新增 / 修改过的文件
       (哈希缓存在 ``<local_dir 的父目录>/.cache/huggingface/``,中断后重跑即可续传)。
    2. 删除:远端 ``RoboSafeJanus/`` 下本地已不存在的文件,分批提交删除(``--no_delete`` 关闭)。
       先上传后删除,上传失败时不会删任何远端文件;本地目录为空时直接报错退出,防止误删整个远端。

默认排除 ``_shards/``(并行采集的中间分片,合并后会被删除)。正在采集的任务请等它结束再上传,
否则可能传上去写了一半的 hdf5(下次运行会被覆盖为完整版本)。

hf-mirror.com 只能下载,不支持上传,所以默认 endpoint 是 https://huggingface.co(忽略 ``HF_ENDPOINT``),
需要能直连或经代理(``HTTPS_PROXY``)访问。token 用 ``hf auth login`` 登录或设 ``HF_TOKEN``,需要 write 权限。

用法:
    python src/data_collect/upload_to_hf.py --repo_id <user>/<repo> --dry_run   # 只看要删哪些文件
    python src/data_collect/upload_to_hf.py --repo_id <user>/<repo>
    python src/data_collect/upload_to_hf.py --repo_id <user>/<repo> --exclude "RoboSafeJanus/*_failed/*"
"""

import argparse
import os
import sys

from huggingface_hub import CommitOperationDelete, HfApi
from huggingface_hub.utils import filter_repo_objects

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default=os.environ.get("HF_DATA_REPO_ID"),
                    help="<user or org>/<repo>; 也可用环境变量 HF_DATA_REPO_ID")
parser.add_argument("--local_dir", type=str, default="data/RoboSafeJanus")
parser.add_argument("--repo_type", type=str, default="dataset", choices=["dataset", "model"])
parser.add_argument("--endpoint", type=str, default=os.environ.get("HF_UPLOAD_ENDPOINT", "https://huggingface.co"))
parser.add_argument("--exclude", type=str, nargs="*", default=[],
                    help="额外排除的 fnmatch 模式(相对 local_dir 的父目录,如 'RoboSafeJanus/*_failed/*')")
parser.add_argument("--num_workers", type=int, default=None, help="上传线程数(默认由 huggingface_hub 决定)")
parser.add_argument("--no_delete", action="store_true", help="不删除远端多出来的文件")
parser.add_argument("--delete_batch", type=int, default=1000, help="每个删除 commit 包含的文件数")
parser.add_argument("--dry_run", action="store_true", help="只统计和列出待删除文件,不上传、不删除")
args = parser.parse_args()

if not args.repo_id:
    parser.error("--repo_id is required (or set HF_DATA_REPO_ID)")

local_dir = os.path.abspath(args.local_dir)
if not os.path.isdir(local_dir):
    sys.exit(f"[upload_to_hf] local_dir not found: {local_dir}")
# upload_large_folder 只能上传到仓库根目录,所以上传父目录并只放行 <name>/**,远端即为 <name>/...
root_dir, prefix = os.path.split(local_dir)
allow_patterns = [f"{prefix}/*"]
# 与 upload_large_folder 的默认忽略项保持一致,本地 / 远端文件列表才能正确对比
ignore_patterns = [".git/*", "*/.git/*", ".cache/huggingface/*", f"{prefix}/_shards/*", *args.exclude]


def list_local_files():
    """``root_dir`` 下经 allow / ignore 过滤后的相对路径集合(与上传时的过滤规则相同)。"""
    paths = []
    for dirpath, _, filenames in os.walk(local_dir):
        for name in filenames:
            paths.append(os.path.relpath(os.path.join(dirpath, name), root_dir).replace(os.sep, "/"))
    return set(filter_repo_objects(paths, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns))


def list_remote_files(api):
    """远端 ``<prefix>/`` 下、且不在 ignore 范围内的文件(被 ignore 的远端文件不参与删除)。"""
    if not api.repo_exists(args.repo_id, repo_type=args.repo_type):
        return set()
    files = api.list_repo_files(args.repo_id, repo_type=args.repo_type)
    return set(filter_repo_objects(files, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns))


api = HfApi(endpoint=args.endpoint)
local_files = list_local_files()
if not local_files:
    sys.exit(f"[upload_to_hf] no files under {local_dir} after filtering; refusing to sync (would wipe remote)")
local_bytes = sum(os.path.getsize(os.path.join(root_dir, p)) for p in local_files)
print(f"[upload_to_hf] local: {len(local_files)} files, {local_bytes / 1e9:.1f} GB under {local_dir}")
repo_url = f"{args.endpoint}/{'datasets/' if args.repo_type == 'dataset' else ''}{args.repo_id}"
print(f"[upload_to_hf] remote: {repo_url}/{prefix}/")

if args.dry_run:
    remote_files = list_remote_files(api)
    stale = sorted(remote_files - local_files)
    print(f"[upload_to_hf] remote has {len(remote_files)} files; "
          f"{len(local_files - remote_files)} new local paths; {len(stale)} remote files would be deleted")
    for path in stale:
        print(f"  delete: {path}")
    sys.exit(0)

# 1. 建仓库(已存在则复用,但必须是私有的)
api.create_repo(args.repo_id, repo_type=args.repo_type, private=True, exist_ok=True)
if not api.repo_info(args.repo_id, repo_type=args.repo_type).private:
    sys.exit(f"[upload_to_hf] {args.repo_id} exists but is PUBLIC; make it private on the website first")

# 2. 上传新增 / 修改的文件
api.upload_large_folder(
    repo_id=args.repo_id,
    folder_path=root_dir,
    repo_type=args.repo_type,
    allow_patterns=allow_patterns,
    ignore_patterns=ignore_patterns,
    num_workers=args.num_workers,
)

# 3. 删除远端多出来的文件
if args.no_delete:
    sys.exit(0)
stale = sorted(list_remote_files(api) - local_files)
print(f"[upload_to_hf] {len(stale)} remote files no longer exist locally")
for start in range(0, len(stale), args.delete_batch):
    batch = stale[start:start + args.delete_batch]
    api.create_commit(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        operations=[CommitOperationDelete(path_in_repo=p) for p in batch],
        commit_message=f"Delete {len(batch)} files removed from {prefix}/",
    )
    print(f"[upload_to_hf] deleted {start + len(batch)}/{len(stale)}")
print("[upload_to_hf] done")
