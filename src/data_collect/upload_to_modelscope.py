"""把采集数据目录(默认 ``data/RoboSafeJanus``)同步到 ModelScope 私有 dataset 仓库。

远端布局与本地一致:``data/RoboSafeJanus/<task>/<config>/...`` → ``<repo>/RoboSafeJanus/<task>/<config>/...``。
用法和行为与 ``upload_to_hf.py`` 对齐,每次运行做两步,可以反复执行:
    1. 上传:``HubApi.upload_folder`` 多线程、分批 commit、可断点续传。已提交文件按 (路径, mtime, size)
       记在 ``<local_dir 的父目录>/.cache/modelscope/`` 下的进度缓存里,重跑直接跳过;
       缓存丢失时会重新算 sha256,内容相同的 LFS 文件服务端复用,不会重复传输。
    2. 删除:远端 ``RoboSafeJanus/`` 下本地已不存在的文件(``--no_delete`` 关闭)。
       先上传后删除,上传有任何失败时不会删远端文件;本地目录为空时直接报错退出,防止误删整个远端。

默认排除 ``_shards/``(并行采集的中间分片,合并后会被删除)。正在采集的任务请等它结束再上传,
否则可能传上去写了一半的 hdf5(下次运行会被覆盖为完整版本)。

依赖 ``pip install -U "modelscope>=1.40"``(会带上 ``modelscope-hub``)。token 在
https://www.modelscope.cn/my/myaccesstoken 生成,用 ``modelscope login --token <token>`` 登录
或设 ``MODELSCOPE_API_TOKEN``。endpoint 默认 https://www.modelscope.cn,国际站用
``--endpoint https://www.modelscope.ai``(或环境变量 ``MODELSCOPE_ENDPOINT``)。

用法:
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo> --dry_run   # 只看要删哪些文件
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo>
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo> --exclude "RoboSafeJanus/*_failed/*"
"""

import argparse
import fnmatch
import os
import sys

try:
    from modelscope_hub import HubApi
    from modelscope_hub.constants import DEFAULT_IGNORE_PATTERNS, Visibility
except ImportError:
    sys.exit('[upload_to_modelscope] modelscope_hub not found; run: pip install -U "modelscope>=1.40"')

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default=os.environ.get("MS_DATA_REPO_ID"),
                    help="<user or org>/<repo>; 也可用环境变量 MS_DATA_REPO_ID")
parser.add_argument("--local_dir", type=str, default="data/RoboSafeJanus")
parser.add_argument("--repo_type", type=str, default="dataset", choices=["dataset", "model"])
parser.add_argument("--endpoint", type=str, default=None,
                    help="默认用 SDK 配置(MODELSCOPE_ENDPOINT,否则 https://www.modelscope.cn)")
parser.add_argument("--exclude", type=str, nargs="*", default=[],
                    help="额外排除的 fnmatch 模式(相对 local_dir 的父目录,如 'RoboSafeJanus/*_failed/*')")
parser.add_argument("--num_workers", type=int, default=None, help="上传线程数(默认由 SDK 决定)")
parser.add_argument("--no_delete", action="store_true", help="不删除远端多出来的文件")
parser.add_argument("--dry_run", action="store_true", help="只统计和列出待删除文件,不上传、不删除")
args = parser.parse_args()

if not args.repo_id:
    parser.error("--repo_id is required (or set MS_DATA_REPO_ID)")

local_dir = os.path.abspath(args.local_dir)
if not os.path.isdir(local_dir):
    sys.exit(f"[upload_to_modelscope] local_dir not found: {local_dir}")
# 上传父目录并只放行 <name>/**,远端即为 <name>/...,--exclude 的写法与 upload_to_hf.py 相同
root_dir, prefix = os.path.split(local_dir)
allow_patterns = [f"{prefix}/*"]
# upload_folder 会再追加 DEFAULT_IGNORE_PATTERNS,这里也带上,本地 / 远端文件列表才能正确对比
ignore_patterns = [f"{prefix}/_shards/*", *args.exclude]
filter_ignore = [*ignore_patterns, *DEFAULT_IGNORE_PATTERNS]
# 进度缓存放在 .cache 下(被默认忽略,不会被上传),按仓库区分
tracker_path = os.path.join(root_dir, ".cache", "modelscope", f"upload_{args.repo_id.replace('/', '__')}.json")


def filter_paths(paths):
    """与 upload_folder 相同的过滤规则:匹配任一 allow 且不匹配任何 ignore(fnmatch,``*`` 可跨 ``/``)。"""
    return {
        p for p in paths
        if any(fnmatch.fnmatch(p, pat) for pat in allow_patterns)
        and not any(fnmatch.fnmatch(p, pat) for pat in filter_ignore)
    }


def list_local_files():
    """``root_dir`` 下经 allow / ignore 过滤后的相对路径集合。"""
    paths = []
    for dirpath, _, filenames in os.walk(local_dir):
        for name in filenames:
            paths.append(os.path.relpath(os.path.join(dirpath, name), root_dir).replace(os.sep, "/"))
    return filter_paths(paths)


def list_remote_files(api):
    """远端 ``<prefix>/`` 下、且不在 ignore 范围内的文件(被 ignore 的远端文件不参与删除)。"""
    if not api.repo_exists(args.repo_id, args.repo_type):
        return set()
    files = api.list_repo_files(args.repo_id, args.repo_type, recursive=True)
    return filter_paths(f.path for f in files if not f.is_dir)


api = HubApi(endpoint=args.endpoint)
local_files = list_local_files()
if not local_files:
    sys.exit(f"[upload_to_modelscope] no files under {local_dir} after filtering; refusing to sync (would wipe remote)")
local_bytes = sum(os.path.getsize(os.path.join(root_dir, p)) for p in local_files)
print(f"[upload_to_modelscope] local: {len(local_files)} files, {local_bytes / 1e9:.1f} GB under {local_dir}")
web_endpoint = (args.endpoint or os.environ.get("MODELSCOPE_ENDPOINT") or "https://www.modelscope.cn").rstrip("/")
repo_url = f"{web_endpoint}/{'datasets' if args.repo_type == 'dataset' else 'models'}/{args.repo_id}"
print(f"[upload_to_modelscope] remote: {repo_url} ({prefix}/)")

if args.dry_run:
    remote_files = list_remote_files(api)
    stale = sorted(remote_files - local_files)
    print(f"[upload_to_modelscope] remote has {len(remote_files)} files; "
          f"{len(local_files - remote_files)} new local paths; {len(stale)} remote files would be deleted")
    for path in stale:
        print(f"  delete: {path}")
    sys.exit(0)

# 1. 建仓库(已存在则复用,但必须是私有的)
user = api.whoami()
print(f"[upload_to_modelscope] logged in as {user.username}")
if not api.repo_exists(args.repo_id, args.repo_type):
    api.create_repo(args.repo_id, args.repo_type, visibility=Visibility.PRIVATE)
    print(f"[upload_to_modelscope] created private {args.repo_type} {args.repo_id}")
info = api.get_repo(args.repo_id, args.repo_type)
if info.private is False or (info.visibility is not None and info.visibility != Visibility.PRIVATE):
    sys.exit(f"[upload_to_modelscope] {args.repo_id} is not private (visibility={info.visibility}); "
             f"make it private on the website first")

# 2. 上传新增 / 修改的文件(有失败会抛 StorageError,不会走到删除;重跑即续传)
api.upload_folder(
    args.repo_id,
    args.repo_type,
    root_dir,
    allow_patterns=allow_patterns,
    ignore_patterns=ignore_patterns,
    max_workers=args.num_workers,
    use_cache=True,
    tracker_path=tracker_path,
    commit_message=f"Sync {prefix}/",
)

# 3. 删除远端多出来的文件(SDK 内部按每 commit 2000 个分批)
if args.no_delete:
    sys.exit(0)
stale = sorted(list_remote_files(api) - local_files)
print(f"[upload_to_modelscope] {len(stale)} remote files no longer exist locally")
if stale:
    api.delete_files(args.repo_id, args.repo_type, stale,
                     commit_message=f"Delete {len(stale)} files removed from {prefix}/")
    print(f"[upload_to_modelscope] deleted {len(stale)} files")
print("[upload_to_modelscope] done")
