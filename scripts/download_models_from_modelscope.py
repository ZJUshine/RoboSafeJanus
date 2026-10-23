"""从 ModelScope 私有 model 仓库(默认 ``zjushine/RoboJanus-Models``)下载训练好的策略 checkpoint。

远端布局:``<POLICY>/<task>/<run>/<step>/...``,例如
``Pi_05/heat_sandwich_microwave/RoboJanus-heat_sandwich_microwave-arx_x5-joint-0/1999/params/...``。
本地按原样镜像到 ``--local_dir``(默认 ``data/RoboJanus-Models``),再把每个 run 软链接到
``XPolicyLab/policy/<POLICY>/checkpoints/<run>``,这样 ``--ckpt <run>`` 就能直接用于 eval
(``--no_link`` 关闭)。已存在且不是软链接的 ``checkpoints/<run>`` 不会被覆盖。

``--policy`` / ``--task`` / ``--run`` 都是 fnmatch 模式,可传多个;``--step latest`` 只下载每个 run
最大的 step,``--step 1999 2999`` 只下载指定 step,默认全部下载。可反复运行,已下载的文件会跳过。

依赖 ``pip install -U "modelscope>=1.40"``(会带上 ``modelscope-hub``)。私有仓库需先
``modelscope login --token <token>`` 或设 ``MODELSCOPE_API_TOKEN``
(token 在 https://www.modelscope.cn/my/myaccesstoken 生成)。

用法:
    python scripts/download_models_from_modelscope.py --list                       # 只列出远端有哪些 run
    python scripts/download_models_from_modelscope.py --policy Pi_05 --task heat_sandwich_microwave
    python scripts/download_models_from_modelscope.py --policy Pi_05 --step latest
    python scripts/download_models_from_modelscope.py                              # 整个仓库
"""

import argparse
from collections import defaultdict
import fnmatch
import os
import sys

try:
    from modelscope_hub import HubApi
except ImportError:
    sys.exit('[download_models] modelscope_hub not found; run: pip install -U "modelscope>=1.40"')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default=os.environ.get("MS_MODEL_REPO_ID", "zjushine/RoboJanus-Models"),
                    help="<user or org>/<repo>; 也可用环境变量 MS_MODEL_REPO_ID")
parser.add_argument("--local_dir", type=str, default=os.path.join(REPO_ROOT, "data", "RoboJanus-Models"))
parser.add_argument("--policy", type=str, nargs="*", default=["*"], help="策略名 fnmatch 模式,如 Pi_05")
parser.add_argument("--task", type=str, nargs="*", default=["*"], help="任务名 fnmatch 模式")
parser.add_argument("--run", type=str, nargs="*", default=["*"], help="run 目录名 fnmatch 模式")
parser.add_argument("--step", type=str, nargs="*", default=None,
                    help="只下载这些 step;'latest' 表示每个 run 最大的 step;默认全部")
parser.add_argument("--revision", type=str, default=None)
parser.add_argument("--endpoint", type=str, default=None,
                    help="默认用 SDK 配置(MODELSCOPE_ENDPOINT,否则 https://www.modelscope.cn)")
parser.add_argument("--num_workers", type=int, default=8, help="下载线程数")
parser.add_argument("--no_link", action="store_true", help="不在 XPolicyLab/policy/<POLICY>/checkpoints/ 下建软链接")
parser.add_argument("--list", action="store_true", help="只列出匹配的 run / step 和大小,不下载")
args = parser.parse_args()


def matches(name, patterns):
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def step_key(step):
    return (0, int(step)) if step.isdigit() else (1, step)


api = HubApi(endpoint=args.endpoint)
files = api.list_repo_files(args.repo_id, "model", revision=args.revision, recursive=True)

# (policy, task, run) -> step -> [(path, size)];顶层 README / .gitattributes 等不是 checkpoint,忽略
runs = defaultdict(lambda: defaultdict(list))
for f in files:
    parts = f.path.split("/")
    if f.is_dir or len(parts) < 5:
        continue
    policy, task, run, step = parts[:4]
    if matches(policy, args.policy) and matches(task, args.task) and matches(run, args.run):
        runs[(policy, task, run)][step].append((f.path, f.size or 0))

selected = {}
for key, steps in sorted(runs.items()):
    if args.step is None:
        keep = list(steps)
    else:
        keep = [s for s in steps if s in args.step]
        if "latest" in args.step:
            keep.append(max(steps, key=step_key))
    keep = sorted(set(keep), key=step_key)
    if keep:
        selected[key] = {s: steps[s] for s in keep}

if not selected:
    sys.exit(f"[download_models] no checkpoints in {args.repo_id} match "
             f"policy={args.policy} task={args.task} run={args.run} step={args.step}")

total_files, total_bytes = 0, 0
for (policy, task, run), steps in selected.items():
    for step, items in steps.items():
        size = sum(s for _, s in items)
        total_files += len(items)
        total_bytes += size
        print(f"  {policy}/{task}/{run}/{step}  ({len(items)} files, {size / 1e9:.2f} GB)")
print(f"[download_models] {len(selected)} runs, {total_files} files, {total_bytes / 1e9:.2f} GB from {args.repo_id}")
if args.list:
    sys.exit(0)

local_dir = os.path.abspath(args.local_dir)
os.makedirs(local_dir, exist_ok=True)
paths = [p for steps in selected.values() for items in steps.values() for p, _ in items]
api.download_repo(
    args.repo_id,
    "model",
    revision=args.revision,
    local_dir=local_dir,
    allow_patterns=paths,
    max_workers=args.num_workers,
)
missing = [p for p in paths if not os.path.isfile(os.path.join(local_dir, p))]
if missing:
    sys.exit(f"[download_models] {len(missing)} files missing after download (e.g. {missing[0]}); rerun to resume")
print(f"[download_models] downloaded to {local_dir}")

if args.no_link:
    sys.exit(0)
for policy, task, run in selected:
    policy_dir = os.path.join(REPO_ROOT, "XPolicyLab", "policy", policy)
    if not os.path.isdir(policy_dir):
        print(f"[download_models] skip link: policy dir not found: {policy_dir}")
        continue
    src = os.path.join(local_dir, policy, task, run)
    dst = os.path.join(policy_dir, "checkpoints", run)
    if os.path.islink(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            continue
        os.unlink(dst)
    elif os.path.exists(dst):
        print(f"[download_models] skip link: {dst} already exists and is not a symlink")
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    print(f"[download_models] linked {os.path.relpath(dst, REPO_ROOT)} -> {os.path.relpath(src, REPO_ROOT)}")
print("[download_models] done")
