"""把采集数据目录(默认 ``data/RoboSafeJanus-Benchmark``)同步到 ModelScope 私有 dataset 仓库。

远端沿用扁平布局:``data/RoboSafeJanus-Benchmark/<group>/<level>/<task>[_failed]/<config>/...``
→ ``<repo>/RoboSafeJanus/<task>[_failed]/<config>/...``(``--prefix`` 可改顶层目录名)。只上传
``task/robosafejanus_tasks.json`` 中的任务;映射见 ``benchmark_layout.upload_files``。
用法和行为与 ``upload_to_hf.py`` 对齐,每次运行做两步,可以反复执行:
    1. 上传:在 ``<local_dir 的父目录>/.cache/upload_view/`` 下按远端路径建硬链接,再用
       ``HubApi.upload_folder`` 多线程、分批 commit、可断点续传。已提交文件按 (路径, mtime, size)
       记在 ``<local_dir 的父目录>/.cache/modelscope/`` 下的进度缓存里,重跑直接跳过;
       缓存丢失时会重新算 sha256,内容相同的 LFS 文件服务端复用,不会重复传输。
    2. 删除:远端 ``RoboSafeJanus/`` 下本地已不存在的文件(``--no_delete`` 关闭),如改名前的旧任务。
       先上传后删除,上传有任何失败时不会删远端文件;本地没有可上传文件时直接报错退出,防止误删整个远端。

``_shards/``(并行采集的中间分片)和 JSON 之外的目录不上传。正在采集的任务请等它结束再上传,
否则可能传上去写了一半的 hdf5(下次运行会被覆盖为完整版本)。

依赖 ``pip install -U "modelscope>=1.40"``(会带上 ``modelscope-hub``)。token 在
https://www.modelscope.cn/my/myaccesstoken 生成,用 ``modelscope login --token <token>`` 登录
或设 ``MODELSCOPE_API_TOKEN``。endpoint 默认 https://www.modelscope.cn,国际站用
``--endpoint https://www.modelscope.ai``(或环境变量 ``MODELSCOPE_ENDPOINT``)。

用法:
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo> --dry_run   # 按任务列出要传 / 要删的文件
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo>
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo> --exclude "RoboSafeJanus/*_failed/*"
"""

import argparse
import fnmatch
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from src.data_collect.benchmark_layout import REMOTE_PREFIX, build_upload_view, upload_view_dir  # noqa: E402
from src.data_collect.upload_common import local_upload_files, print_plan, unmapped_dirs  # noqa: E402

try:
    from modelscope_hub import HubApi
    from modelscope_hub.constants import DEFAULT_IGNORE_PATTERNS, Visibility
except ImportError:
    sys.exit('[upload_to_modelscope] modelscope_hub not found; run: pip install -U "modelscope>=1.40"')

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default=os.environ.get("MS_DATA_REPO_ID"),
                    help="<user or org>/<repo>; 也可用环境变量 MS_DATA_REPO_ID")
parser.add_argument("--local_dir", type=str, default="data/RoboSafeJanus-Benchmark")
parser.add_argument("--prefix", type=str, default=REMOTE_PREFIX, help="远端顶层目录,布局为 <prefix>/<task>/...")
parser.add_argument("--repo_type", type=str, default="dataset", choices=["dataset", "model"])
parser.add_argument("--endpoint", type=str, default=None,
                    help="默认用 SDK 配置(MODELSCOPE_ENDPOINT,否则 https://www.modelscope.cn)")
parser.add_argument("--exclude", type=str, nargs="*", default=[],
                    help="额外排除的 fnmatch 模式(远端路径,如 'RoboSafeJanus/*_failed/*')")
parser.add_argument("--num_workers", type=int, default=None, help="上传线程数(默认由 SDK 决定)")
parser.add_argument("--no_delete", action="store_true", help="不删除远端多出来的文件")
parser.add_argument("--dry_run", action="store_true", help="只按任务列出要上传 / 删除的文件,不上传、不删除")
args = parser.parse_args()

if not args.repo_id:
    parser.error("--repo_id is required (or set MS_DATA_REPO_ID)")

local_dir = os.path.abspath(args.local_dir)
if not os.path.isdir(local_dir):
    sys.exit(f"[upload_to_modelscope] local_dir not found: {local_dir}")
prefix = args.prefix.strip("/")
allow_patterns = [f"{prefix}/*"]
# upload_folder 会再追加 DEFAULT_IGNORE_PATTERNS,这里也带上,本地 / 远端文件列表才能正确对比
ignore_patterns = [*args.exclude]
filter_ignore = [*ignore_patterns, *DEFAULT_IGNORE_PATTERNS]
view_dir = upload_view_dir(local_dir)
# 进度缓存放在 .cache 下(被默认忽略,不会被上传),按仓库区分
tracker_path = os.path.join(os.path.dirname(local_dir), ".cache", "modelscope",
                            f"upload_{args.repo_id.replace('/', '__')}.json")


def keep(path):
    """与 upload_folder 相同的过滤规则:匹配任一 allow 且不匹配任何 ignore(fnmatch,``*`` 可跨 ``/``)。"""
    return (any(fnmatch.fnmatch(path, pat) for pat in allow_patterns)
            and not any(fnmatch.fnmatch(path, pat) for pat in filter_ignore))


def list_remote_files(api):
    """远端 ``<prefix>/`` 下、且不在 ignore 范围内的文件 ``{路径: 字节数}``(被 ignore 的远端文件不参与删除)。"""
    if not api.repo_exists(args.repo_id, args.repo_type):
        return {}
    files = api.list_repo_files(args.repo_id, args.repo_type, recursive=True)
    return {f.path: f.size for f in files if not f.is_dir and keep(f.path)}


api = HubApi(endpoint=args.endpoint)
local_files = local_upload_files(local_dir, prefix, keep)
if not local_files:
    sys.exit(f"[upload_to_modelscope] no files under {local_dir} after filtering; refusing to sync (would wipe remote)")
local_sizes = {path: os.path.getsize(src) for path, src in local_files.items()}
print(f"[upload_to_modelscope] local: {len(local_files)} files, {sum(local_sizes.values()) / 1e9:.1f} GB "
      f"under {local_dir}")
for name in unmapped_dirs(local_dir):
    print(f"[upload_to_modelscope] skip {name}/: not a group in task/robosafejanus_tasks.json")
web_endpoint = (args.endpoint or os.environ.get("MODELSCOPE_ENDPOINT") or "https://www.modelscope.cn").rstrip("/")
repo_url = f"{web_endpoint}/{'datasets' if args.repo_type == 'dataset' else 'models'}/{args.repo_id}"
print(f"[upload_to_modelscope] remote: {repo_url} ({prefix}/)")

if args.dry_run:
    print_plan("upload_to_modelscope", local_sizes, list_remote_files(api))
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

# 2. 按远端路径建硬链接视图,上传新增 / 修改的文件(有失败会抛 StorageError,不会走到删除;重跑即续传)
build_upload_view(local_files, view_dir)
api.upload_folder(
    args.repo_id,
    args.repo_type,
    view_dir,
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
stale = sorted(set(list_remote_files(api)) - set(local_files))
print(f"[upload_to_modelscope] {len(stale)} remote files no longer exist locally")
if stale:
    api.delete_files(args.repo_id, args.repo_type, stale,
                     commit_message=f"Delete {len(stale)} files removed from {prefix}/")
    print(f"[upload_to_modelscope] deleted {len(stale)} files")
print("[upload_to_modelscope] done")
