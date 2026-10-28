"""把采集数据目录(默认 ``data/RoboSafeJanus-Benchmark``)同步到 Hugging Face 私有 dataset 仓库。

远端沿用扁平布局:``data/RoboSafeJanus-Benchmark/<group>/<level>/<task>[_failed]/<config>/...``
→ ``<repo>/RoboSafeJanus/<task>[_failed]/<config>/...``(``--prefix`` 可改顶层目录名)。只上传
``task/robosafejanus_tasks.json`` 中的任务;映射见 ``benchmark_layout.upload_files``。

每次运行做两步,可以反复执行:
    1. 上传:在 ``<local_dir 的父目录>/.cache/upload_view/`` 下按远端路径建硬链接,再用
       ``HfApi.upload_large_folder`` 多线程、可断点续传,只传新增 / 修改过的文件
       (哈希缓存在 ``.cache/upload_view/.cache/huggingface/``,中断后重跑即可续传)。
    2. 删除:远端 ``RoboSafeJanus/`` 下本地已不存在的文件,分批提交删除(``--no_delete`` 关闭)。
       先上传后删除,上传失败时不会删任何远端文件;本地没有可上传文件时直接报错退出,防止误删整个远端。

``_shards/``(并行采集的中间分片)和 JSON 之外的目录不上传。正在采集的任务请等它结束再上传,
否则可能传上去写了一半的 hdf5(下次运行会被覆盖为完整版本)。

hf-mirror.com 只能下载,不支持上传,所以默认 endpoint 是 https://huggingface.co(忽略 ``HF_ENDPOINT``),
需要能直连或经代理(``HTTPS_PROXY``)访问。token 用 ``hf auth login`` 登录或设 ``HF_TOKEN``,需要 write 权限。

用法:
    python src/data_collect/upload_to_hf.py --repo_id <user>/<repo> --dry_run   # 按任务列出要传 / 要删的文件
    python src/data_collect/upload_to_hf.py --repo_id <user>/<repo>
    python src/data_collect/upload_to_hf.py --repo_id <user>/<repo> --exclude "RoboSafeJanus/*_failed/*"
"""

import argparse
import os
import sys

from huggingface_hub import CommitOperationDelete, HfApi
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import filter_repo_objects

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from src.data_collect.benchmark_layout import REMOTE_PREFIX, build_upload_view, upload_view_dir  # noqa: E402
from src.data_collect.upload_common import local_upload_files, print_plan, unmapped_dirs  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default=os.environ.get("HF_DATA_REPO_ID"),
                    help="<user or org>/<repo>; 也可用环境变量 HF_DATA_REPO_ID")
parser.add_argument("--local_dir", type=str, default="data/RoboSafeJanus-Benchmark")
parser.add_argument("--prefix", type=str, default=REMOTE_PREFIX, help="远端顶层目录,布局为 <prefix>/<task>/...")
parser.add_argument("--repo_type", type=str, default="dataset", choices=["dataset", "model"])
parser.add_argument("--endpoint", type=str, default=os.environ.get("HF_UPLOAD_ENDPOINT", "https://huggingface.co"))
parser.add_argument("--exclude", type=str, nargs="*", default=[],
                    help="额外排除的 fnmatch 模式(远端路径,如 'RoboSafeJanus/*_failed/*')")
parser.add_argument("--num_workers", type=int, default=None, help="上传线程数(默认由 huggingface_hub 决定)")
parser.add_argument("--no_delete", action="store_true", help="不删除远端多出来的文件")
parser.add_argument("--delete_batch", type=int, default=1000, help="每个删除 commit 包含的文件数")
parser.add_argument("--dry_run", action="store_true", help="只按任务列出要上传 / 删除的文件,不上传、不删除")
args = parser.parse_args()

if not args.repo_id:
    parser.error("--repo_id is required (or set HF_DATA_REPO_ID)")

local_dir = os.path.abspath(args.local_dir)
if not os.path.isdir(local_dir):
    sys.exit(f"[upload_to_hf] local_dir not found: {local_dir}")
prefix = args.prefix.strip("/")
# upload_large_folder 只能上传到仓库根目录,所以上传硬链接视图并只放行 <prefix>/**
allow_patterns = [f"{prefix}/*"]
# 与 upload_large_folder 的默认忽略项保持一致,本地 / 远端文件列表才能正确对比
ignore_patterns = [".git/*", "*/.git/*", ".cache/huggingface/*", *args.exclude]
view_dir = upload_view_dir(local_dir)


def keep(path):
    return bool(list(filter_repo_objects([path], allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)))


def list_remote_files(api):
    """远端 ``<prefix>/`` 下、且不在 ignore 范围内的文件 ``{路径: 字节数}``(被 ignore 的远端文件不参与删除)。"""
    if not api.repo_exists(args.repo_id, repo_type=args.repo_type):
        return {}
    tree = api.list_repo_tree(args.repo_id, repo_type=args.repo_type, recursive=True)
    return {f.path: f.size for f in tree if isinstance(f, RepoFile) and keep(f.path)}


api = HfApi(endpoint=args.endpoint)
local_files = local_upload_files(local_dir, prefix, keep)
if not local_files:
    sys.exit(f"[upload_to_hf] no files under {local_dir} after filtering; refusing to sync (would wipe remote)")
local_sizes = {path: os.path.getsize(src) for path, src in local_files.items()}
print(f"[upload_to_hf] local: {len(local_files)} files, {sum(local_sizes.values()) / 1e9:.1f} GB under {local_dir}")
for name in unmapped_dirs(local_dir):
    print(f"[upload_to_hf] skip {name}/: not a group in task/robosafejanus_tasks.json")
repo_url = f"{args.endpoint}/{'datasets/' if args.repo_type == 'dataset' else ''}{args.repo_id}"
print(f"[upload_to_hf] remote: {repo_url}/{prefix}/")

if args.dry_run:
    print_plan("upload_to_hf", local_sizes, list_remote_files(api))
    sys.exit(0)

# 1. 建仓库(已存在则复用,但必须是私有的)
api.create_repo(args.repo_id, repo_type=args.repo_type, private=True, exist_ok=True)
if not api.repo_info(args.repo_id, repo_type=args.repo_type).private:
    sys.exit(f"[upload_to_hf] {args.repo_id} exists but is PUBLIC; make it private on the website first")

# 2. 按远端路径建硬链接视图,上传新增 / 修改的文件
build_upload_view(local_files, view_dir)
api.upload_large_folder(
    repo_id=args.repo_id,
    folder_path=view_dir,
    repo_type=args.repo_type,
    allow_patterns=allow_patterns,
    ignore_patterns=ignore_patterns,
    num_workers=args.num_workers,
)

# 3. 删除远端多出来的文件
if args.no_delete:
    sys.exit(0)
stale = sorted(set(list_remote_files(api)) - set(local_files))
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
