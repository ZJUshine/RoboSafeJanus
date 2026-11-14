"""把采集数据目录(默认 ``data/RoboSafeJanus-Benchmark``)同步到 ModelScope 私有 dataset 仓库。

远端沿用扁平布局:``data/RoboSafeJanus-Benchmark/<group>/<level>/<task>[_failed]/<config>/...``
→ ``<repo>/RoboSafeJanus/<task>[_failed]/<config>/...``(``--prefix`` 可改顶层目录名)。只上传
任务定义(``task/robosafejanus_tasks`` + ``task/robosafejanus_sim``)中的任务;映射见 ``benchmark_layout.upload_files``。

判定是否要传以 **sha256 内容哈希** 为准:本地算一遍,和远端 ``list_repo_files`` 返回的 ``sha256``
逐个比对。文件名、大小、mtime 都没变但内容变了,也会重传——不会因为 Cache 命中而跳过。
本地哈希按 (size, mtime_ns, ctime_ns) 缓存在 ``.cache/modelscope/local_sha256_*.json``,
只省读盘,不参与判定。

每次运行做三步,可以反复执行:
    1. 上传:在 ``<local_dir 的父目录>/.cache/upload_view/`` 下只给**内容不同**的文件建硬链接,
       再用 ``HubApi.upload_folder`` 多线程、分批 commit。上传前会清掉这些文件在
       ``.cache/modelscope/upload_*.json`` 断点缓存里的条目,确保 SDK 不会按 mtime/size 跳过。
    2. 删除:远端 ``RoboSafeJanus/`` 下本地已不存在的文件(``--no_delete`` 关闭),如改名前的旧任务。
       先上传后删除,上传有任何失败时不会删远端文件;本地没有可上传文件时直接报错退出,防止误删整个远端。
    3. 校验:重新拉取远端列表,逐个比对大小和 sha256,有任何不一致就以非 0 退出(``--no_verify`` 关闭)。

``_shards/``(并行采集的中间分片)和 JSON 之外的目录不上传。正在采集的任务请等它结束再上传,
否则可能传上去写了一半的 hdf5(下次运行会因为内容不同而被覆盖为完整版本)。

依赖 ``pip install -U "modelscope>=1.40"``(会带上 ``modelscope-hub``)。token 在
https://www.modelscope.cn/my/myaccesstoken 生成,用 ``modelscope login --token <token>`` 登录
或设 ``MODELSCOPE_API_TOKEN``。endpoint 默认 https://www.modelscope.cn,国际站用
``--endpoint https://www.modelscope.ai``(或环境变量 ``MODELSCOPE_ENDPOINT``)。

用法:
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo> --dry_run   # 按任务列出要传 / 要删的文件
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo>
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo> --exclude "RoboSafeJanus/*_failed/*"
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo> --group <group> [--dry_run]
    python src/data_collect/upload_to_modelscope.py --repo_id <user>/<repo> --task <task> [--dry_run]

``--group`` / ``--task``:只同步这些组的自有任务 / 这些任务(含 ``_failed``),远端的删除 / 校验也只在这些任务目录里;
不带 ``--dry_run`` 时要求每个任务的流水线 expert 阶段已批准
(``rsj_pipeline.py gate <group> expert --task <task> --approved``)。
"""

import argparse
import fnmatch
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from src.data_collect.benchmark_layout import REMOTE_PREFIX, build_upload_view, upload_view_dir  # noqa: E402
from src.data_collect.upload_common import (  # noqa: E402
    local_sha256_map,
    local_upload_files,
    plan_sync,
    print_sync_plan,
    purge_tracker,
    remote_file_map,
    require_pipeline_approval,
    scoped_tasks,
    unmapped_dirs,
    verify_sync,
)

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
parser.add_argument("--hash_workers", type=int, default=None, help="本地 sha256 计算线程数(默认 8)")
parser.add_argument("--no_delete", action="store_true", help="不删除远端多出来的文件")
parser.add_argument("--no_verify", action="store_true", help="上传后不校验远端 sha256")
parser.add_argument("--dry_run", action="store_true", help="只按任务列出要上传 / 删除的文件,不上传、不删除")
parser.add_argument("--group", nargs="+", default=[],
                    help="只同步这些组的自有任务;真正上传要求每个任务的流水线 expert 阶段已批准")
parser.add_argument("--task", nargs="+", default=[],
                    help="只同步这些任务;真正上传要求流水线 expert 阶段已批准")
args = parser.parse_args()

if not args.repo_id:
    parser.error("--repo_id is required (or set MS_DATA_REPO_ID)")

local_dir = os.path.abspath(args.local_dir)
if not os.path.isdir(local_dir):
    sys.exit(f"[upload_to_modelscope] local_dir not found: {local_dir}")
prefix = args.prefix.strip("/")
if args.group or args.task:
    scoped = scoped_tasks(args.group, args.task)
    if not scoped:
        sys.exit(f"[upload_to_modelscope] groups {args.group} have no own tasks")
    if not args.dry_run:
        require_pipeline_approval(scoped, "upload_to_modelscope")
    allow_patterns = [pat for task in scoped for pat in (f"{prefix}/{task}/*", f"{prefix}/{task}_failed/*")]
    scope = " ".join(f"--{k} {' '.join(v)}" for k, v in (("group", args.group), ("task", args.task)) if v)
    print(f"[upload_to_modelscope] {scope}: "
          f"{len(scoped)} tasks {scoped}")
else:
    allow_patterns = [f"{prefix}/*"]
# upload_folder 会再追加 DEFAULT_IGNORE_PATTERNS,这里也带上,本地 / 远端文件列表才能正确对比
ignore_patterns = [*args.exclude]
filter_ignore = [*ignore_patterns, *DEFAULT_IGNORE_PATTERNS]
view_dir = upload_view_dir(local_dir)
cache_dir = os.path.join(os.path.dirname(local_dir), ".cache")
# 断点缓存放在 .cache 下(被默认忽略,不会被上传),按仓库区分
tracker_path = os.path.join(cache_dir, "modelscope", f"upload_{args.repo_id.replace('/', '__')}.json")
# 本地 sha256 缓存,名字不能以 upload_ 开头(sync_status 用它反推仓库名)
hash_cache_path = os.path.join(cache_dir, "modelscope",
                               f"local_sha256_{args.repo_id.replace('/', '__')}.json")


def keep(path):
    """与 upload_folder 相同的过滤规则:匹配任一 allow 且不匹配任何 ignore(fnmatch,``*`` 可跨 ``/``)。"""
    return (any(fnmatch.fnmatch(path, pat) for pat in allow_patterns)
            and not any(fnmatch.fnmatch(path, pat) for pat in filter_ignore))


api = HubApi(endpoint=args.endpoint)
local_files = local_upload_files(local_dir, prefix, keep)
if not local_files:
    sys.exit(f"[upload_to_modelscope] no files under {local_dir} after filtering; refusing to sync (would wipe remote)")
local_sizes = {path: os.path.getsize(src) for path, src in local_files.items()}
print(f"[upload_to_modelscope] local: {len(local_files)} files, {sum(local_sizes.values()) / 1e9:.1f} GB "
      f"under {local_dir}")
for name in unmapped_dirs(local_dir):
    print(f"[upload_to_modelscope] skip {name}/: not a group in task/robosafejanus_tasks")
web_endpoint = (args.endpoint or os.environ.get("MODELSCOPE_ENDPOINT") or "https://www.modelscope.cn").rstrip("/")
repo_url = f"{web_endpoint}/{'datasets' if args.repo_type == 'dataset' else 'models'}/{args.repo_id}"
print(f"[upload_to_modelscope] remote: {repo_url} ({prefix}/)")

# 1. 内容级比对:本地 sha256 vs 远端 sha256
remote_map = remote_file_map(api, args.repo_id, args.repo_type, keep)
started = time.time()
try:
    local_hashes = local_sha256_map(local_files, hash_cache_path, args.hash_workers)
except RuntimeError as error:
    sys.exit(f"[upload_to_modelscope] {error}")
print(f"[upload_to_modelscope] hashed {len(local_hashes)} local files in {time.time() - started:.1f}s")
to_upload, stale = plan_sync(local_files, local_sizes, local_hashes, remote_map)
print_sync_plan("upload_to_modelscope", to_upload, local_sizes, stale, remote_map, local_files)

if args.dry_run:
    sys.exit(0)

# 2. 建仓库(已存在则复用,但必须是私有的);只在真要写远端时登录
if to_upload or (stale and not args.no_delete):
    user = api.whoami()
    print(f"[upload_to_modelscope] logged in as {user.username}")
    if not api.repo_exists(args.repo_id, args.repo_type):
        api.create_repo(args.repo_id, args.repo_type, visibility=Visibility.PRIVATE)
        print(f"[upload_to_modelscope] created private {args.repo_type} {args.repo_id}")
    info = api.get_repo(args.repo_id, args.repo_type)
    if info.private is False or (info.visibility is not None and info.visibility != Visibility.PRIVATE):
        sys.exit(f"[upload_to_modelscope] {args.repo_id} is not private (visibility={info.visibility}); "
                 f"make it private on the website first")

# 3. 只给内容不同的文件建视图并上传(有失败会抛 StorageError,不会走到删除;重跑按内容差异续传)
if to_upload:
    build_upload_view({path: local_files[path] for path in to_upload}, view_dir)
    purged = purge_tracker(tracker_path, to_upload)
    print(f"[upload_to_modelscope] uploading {len(to_upload)} changed files "
          f"(purged {purged} tracker entries so nothing is skipped)")
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
else:
    print("[upload_to_modelscope] remote content already matches local; nothing to upload")

# 4. 删除远端多出来的文件(SDK 内部按每 commit 2000 个分批)
if args.no_delete:
    print(f"[upload_to_modelscope] --no_delete: keeping {len(stale)} remote files absent locally")
elif stale:
    print(f"[upload_to_modelscope] {len(stale)} remote files no longer exist locally")
    api.delete_files(args.repo_id, args.repo_type, stale,
                     commit_message=f"Delete {len(stale)} files removed from {prefix}/")

# 5. 校验:远端每个文件的大小和 sha256 都必须和本地一致
if not args.no_verify:
    problems = []
    for attempt in range(1, 4):
        problems = verify_sync(local_files, local_sizes, local_hashes,
                               remote_file_map(api, args.repo_id, args.repo_type, keep))
        if not problems:
            break
        print(f"[upload_to_modelscope] verify attempt {attempt}/3: {len(problems)} mismatch(es)")
        if attempt < 3:
            time.sleep(5)
    if problems:
        for path, reason in problems[:20]:
            print(f"  MISMATCH {path}: {reason}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        sys.exit(f"[upload_to_modelscope] verification failed: {len(problems)} file(s) out of sync")
    print(f"[upload_to_modelscope] verified: {len(local_files)} files match remote sha256")
print("[upload_to_modelscope] done")
