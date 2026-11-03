"""从 ModelScope 私有 model 仓库下载训练 run 的 wandb 记录。

训练机把 checkpoint 和 ``wandb/`` 目录一起传到模型仓库,远端布局::

    <POLICY>/<task>/<run>/wandb/<run-dir>/...
    Pi_05/drop_X_table_safe_tasks/RoboSafeJanus-drop_X_table_safe_tasks-arx_x5-joint-0/
        wandb/run-20261030_110956-jb44twmh/files/...

本地按原样镜像到 ``--local_dir``(默认 ``data/RoboSafeJanus-Wandb``),再用
``scripts/upload_wandb_to_cloud.py`` 上传到 W&B 官方。可反复运行,已下载的文件会跳过。

用法::

    python scripts/download_wandb_from_modelscope.py --list             # 只列出远端有哪些 wandb run
    python scripts/download_wandb_from_modelscope.py --policy Pi_05
    python scripts/download_wandb_from_modelscope.py --path Pi_05/drop_X_table_safe_tasks/<run>/wandb/<run-dir>
    python scripts/download_wandb_from_modelscope.py                    # 仓库里全部 wandb 目录

``--policy`` / ``--task`` / ``--run`` 是 fnmatch 模式,可传多个。私有仓库需先
``modelscope login --token <token>`` 或设 ``MODELSCOPE_API_TOKEN``。
默认强制走 IPv4(``--ipv6`` 关闭),原因见 ``download_models_from_modelscope.py``。
"""

import argparse
from collections import defaultdict
import fnmatch
import os
import sys

try:
    from modelscope_hub import HubApi
except ImportError:
    sys.exit('[download_wandb] modelscope_hub not found; run: pip install -U "modelscope>=1.40"')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default=os.environ.get("MS_MODEL_REPO_ID", "zjushine/RoboSafeJanus-Model"),
                    help="<user or org>/<repo>; 也可用环境变量 MS_MODEL_REPO_ID")
parser.add_argument("--local_dir", type=str, default=os.path.join(REPO_ROOT, "data", "RoboSafeJanus-Wandb"))
parser.add_argument("--path", type=str, nargs="*", default=None,
                    help="wandb run 的完整前缀(相对仓库根,可多个,不用带 wandb/ 之后的部分)")
parser.add_argument("--policy", type=str, nargs="*", default=["*"], help="策略名 fnmatch 模式,如 Pi_05")
parser.add_argument("--task", type=str, nargs="*", default=["*"], help="任务/组模型目录名 fnmatch 模式")
parser.add_argument("--run", type=str, nargs="*", default=["*"], help="run 目录名 fnmatch 模式")
parser.add_argument("--revision", type=str, default=None)
parser.add_argument("--endpoint", type=str, default=None,
                    help="默认用 SDK 配置(MODELSCOPE_ENDPOINT,否则 https://www.modelscope.cn)")
parser.add_argument("--num_workers", type=int, default=8, help="下载线程数")
parser.add_argument("--ipv6", action="store_true",
                    help="用系统默认地址族(通常优先 IPv6);默认强制 IPv4")
parser.add_argument("--list", action="store_true", help="只列出匹配的 wandb run 和大小,不下载")
args = parser.parse_args()

if not args.ipv6:
    import socket

    import urllib3.util.connection as urllib3_connection

    urllib3_connection.allowed_gai_family = lambda: socket.AF_INET


def matches(name, patterns):
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


api = HubApi(endpoint=args.endpoint)
files = api.list_repo_files(args.repo_id, "model", revision=args.revision, recursive=True)

# wandb 前缀 -> [(path, size)]; 前缀本身形如 <POLICY>/<task>/<run>/wandb/<run-dir>
runs = defaultdict(list)
for item in files:
    parts = item.path.split("/")
    if getattr(item, "type", "blob") == "tree" or "wandb" not in parts:
        continue
    index = parts.index("wandb")
    if index < 3 or len(parts) <= index + 1:
        continue
    policy, task, run = parts[:3]
    prefix = "/".join(parts[: index + 2])
    if args.path and not any(prefix == p.rstrip("/") or prefix.startswith(p.rstrip("/") + "/") for p in args.path):
        continue
    if not (matches(policy, args.policy) and matches(task, args.task) and matches(run, args.run)):
        continue
    runs[prefix].append((item.path, getattr(item, "size", 0) or 0))

if not runs:
    sys.exit(f"[download_wandb] no wandb runs in {args.repo_id} match "
             f"path={args.path} policy={args.policy} task={args.task} run={args.run}")

total_files = total_bytes = 0
for prefix, items in sorted(runs.items()):
    size = sum(entry_size for _, entry_size in items)
    total_files += len(items)
    total_bytes += size
    print(f"  {prefix}  ({len(items)} files, {size / 1e6:.1f} MB)")
print(f"[download_wandb] {len(runs)} runs, {total_files} files, {total_bytes / 1e6:.1f} MB from {args.repo_id}")
if args.list:
    sys.exit(0)

local_dir = os.path.abspath(args.local_dir)
os.makedirs(local_dir, exist_ok=True)
paths = [path for items in runs.values() for path, _ in items]
api.download_repo(
    args.repo_id,
    "model",
    revision=args.revision,
    local_dir=local_dir,
    allow_patterns=paths,
    max_workers=args.num_workers,
)
missing = [path for path in paths if not os.path.isfile(os.path.join(local_dir, path))]
if missing:
    sys.exit(f"[download_wandb] {len(missing)} files missing after download (e.g. {missing[0]}); rerun to resume")
print(f"[download_wandb] downloaded {len(runs)} runs to {local_dir}")
