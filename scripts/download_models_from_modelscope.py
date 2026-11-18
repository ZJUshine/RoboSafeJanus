"""从 ModelScope 私有 model 仓库(默认 ``zjushine/RoboSafeJanus-Model``)下载训练好的策略 checkpoint。

远端布局:``<POLICY>/<task>/<run>/<step>/...``,例如
``Pi_05/heat_sandwich_microwave/RoboSafeJanus-heat_sandwich_microwave-arx_x5-joint-0/1999/params/...``。
本地按原样镜像到 ``--local_dir``(默认 ``data/RoboSafeJanus-Model``),再把每个 run 软链接到
``XPolicyLab/policy/<POLICY>/checkpoints/<run>``,这样 ``--ckpt <run>`` 就能直接用于 eval
(``--no_link`` 关闭)。已存在且不是软链接的 ``checkpoints/<run>`` 不会被覆盖。
GR00T_N17 的 step 目录叫 ``checkpoint-<n>``,其中 json 的 ``model_name`` 是训练机 Cosmos 路径:
``checkpoints/<run>`` 会建成逐文件链接的真目录，这几个 json 复制后改成本机路径(镜像不改)。

``--policy`` / ``--task`` / ``--run`` 都是 fnmatch 模式,可传多个;``--step latest`` 只下载每个 run
最大的 step,``--step 1999 2999`` 只下载指定 step,默认全部下载。可反复运行,已下载的文件会跳过。

依赖 ``pip install -U "modelscope>=1.40"``(会带上 ``modelscope-hub``)。私有仓库需先
``modelscope login --token <token>`` 或设 ``MODELSCOPE_API_TOKEN``
(token 在 https://www.modelscope.cn/my/myaccesstoken 生成)。

默认强制走 IPv4(``--ipv6`` 关闭):文件 CDN ``cdn-lfs-cn-1.modelscope.cn`` 同时有 A/AAAA
记录,解析时优先 IPv6,但本机到该 CDN 的 IPv6 路由丢包严重(实测 47%,吞吐
0.1–0.2 MB/s),IPv4 0% 丢包、单流 8–10 MB/s。API 域名只有 IPv4,所以 ``--list`` 看起来正常。

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
import re
import shutil
import sys

try:
    from modelscope_hub import HubApi
except ImportError:
    sys.exit('[download_models] modelscope_hub not found; run: pip install -U "modelscope>=1.40"')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default=os.environ.get("MS_MODEL_REPO_ID", "zjushine/RoboSafeJanus-Model"),
                    help="<user or org>/<repo>; 也可用环境变量 MS_MODEL_REPO_ID")
parser.add_argument("--local_dir", type=str, default=os.path.join(REPO_ROOT, "data", "RoboSafeJanus-Model"))
parser.add_argument("--policy", type=str, nargs="*", default=["*"], help="策略名 fnmatch 模式,如 Pi_05")
parser.add_argument("--task", type=str, nargs="*", default=["*"], help="任务名 fnmatch 模式")
parser.add_argument("--run", type=str, nargs="*", default=["*"], help="run 目录名 fnmatch 模式")
parser.add_argument("--step", type=str, nargs="*", default=None,
                    help="只下载这些 step;'latest' 表示每个 run 最大的 step;默认全部")
parser.add_argument("--revision", type=str, default=None)
parser.add_argument("--endpoint", type=str, default=None,
                    help="默认用 SDK 配置(MODELSCOPE_ENDPOINT,否则 https://www.modelscope.cn)")
parser.add_argument("--num_workers", type=int, default=8, help="下载线程数")
parser.add_argument("--ipv6", action="store_true",
                    help="用系统默认地址族(通常优先 IPv6);默认强制 IPv4,见文件头说明")
parser.add_argument("--no_link", action="store_true", help="不在 XPolicyLab/policy/<POLICY>/checkpoints/ 下建软链接")
parser.add_argument("--list", action="store_true", help="只列出匹配的 run / step 和大小,不下载")
args = parser.parse_args()

if not args.ipv6:
    # modelscope_hub 用 requests/urllib3,而 urllib3 默认按 getaddrinfo 顺序(IPv6 优先)建连;
    # 这里限制成 AF_INET,避免走丢包严重的 IPv6 路由。
    import socket

    import urllib3.util.connection as urllib3_connection

    urllib3_connection.allowed_gai_family = lambda: socket.AF_INET


def matches(name, patterns):
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def step_key(step):
    """按 step 数字排序:``checkpoint-6000`` < ``checkpoint-12000``(字符串比较会排反)。"""
    digits = re.findall(r"\d+", step)
    return (0, int(digits[-1])) if digits else (1, step)


api = HubApi(endpoint=args.endpoint)
files = api.list_repo_files(args.repo_id, "model", revision=args.revision, recursive=True)

# 只有纯数字(Pi_0 / Pi_05)、step_<n> / global_step_<n>(LingBot_VLA)或 checkpoint-<n>(GR00T_N17,
# HF Trainer 的目录名)才是 checkpoint step;assets/、wandb/ 等是 run 级附属内容。
STEP_RE = re.compile(r"^(?:(?:global_)?step|checkpoint)[_-]?\d+$|^\d+$", re.IGNORECASE)

# (policy, task, run) -> step -> [(path, size)];顶层 README / .gitattributes 等不是 checkpoint,忽略
runs = defaultdict(lambda: defaultdict(list))
# (policy, task, run) -> [(path, size)];run 级附属文件,例如
# LingBot_VLA/<task>/<run>/lingbotvla_cli.yaml(评测适配器需要)和
# LingBot_VLA/<task>/<run>/assets/norm_stats/*.json,它们不属于任何 step。
run_files = defaultdict(list)
for f in files:
    parts = f.path.split("/")
    if f.is_dir or len(parts) < 4:
        continue
    policy, task, run = parts[:3]
    if not (matches(policy, args.policy) and matches(task, args.task) and matches(run, args.run)):
        continue
    if len(parts) == 4:
        run_files[(policy, task, run)].append((f.path, f.size or 0))
        continue
    step = parts[3]
    # wandb/ 是训练日志目录,由 download_wandb_from_modelscope.py 单独处理,
    # 不能当成一个 step,否则 --step latest 会误选它。
    if step == "wandb":
        continue
    if STEP_RE.match(step):
        runs[(policy, task, run)][step].append((f.path, f.size or 0))
    else:
        # assets/ 等 run 级附属子目录:不跟 --step 走。
        run_files[(policy, task, run)].append((f.path, f.size or 0))

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

# run 根目录的附属文件不跟 --step 走:只要这个 run 被选中,就一并下载。
selected_run_files = {key: run_files[key] for key in selected if run_files.get(key)}

total_files, total_bytes = 0, 0
for (policy, task, run), steps in selected.items():
    for step, items in steps.items():
        size = sum(s for _, s in items)
        total_files += len(items)
        total_bytes += size
        print(f"  {policy}/{task}/{run}/{step}  ({len(items)} files, {size / 1e9:.2f} GB)")
    items = selected_run_files.get((policy, task, run), [])
    if items:
        size = sum(s for _, s in items)
        total_files += len(items)
        total_bytes += size
        names = ", ".join(os.path.basename(p) for p, _ in items[:5])
        if len(items) > 5:
            names += f", ... (+{len(items) - 5})"
        print(f"  {policy}/{task}/{run}/  (run 级附属 {len(items)} files: {names})")
print(f"[download_models] {len(selected)} runs, {total_files} files, {total_bytes / 1e9:.2f} GB from {args.repo_id}")
if args.list:
    sys.exit(0)

local_dir = os.path.abspath(args.local_dir)
os.makedirs(local_dir, exist_ok=True)
paths = [p for steps in selected.values() for items in steps.values() for p, _ in items]
paths += [p for items in selected_run_files.values() for p, _ in items]
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

SHIM_MARK = ".rsj_ckpt_shim"


def needs_shim(src):
    """LingBot_VLA 的评测脚本找 ``<root>/checkpoints/global_step_*/hf_ckpt``,而 ModelScope 上是
    ``<run>/global_step_*/``(训练机上传时去掉了 checkpoints/ 这一层)。"""
    return (os.path.isfile(os.path.join(src, "lingbotvla_cli.yaml"))
            and not os.path.exists(os.path.join(src, "checkpoints"))
            and any(STEP_RE.match(n) for n in os.listdir(src)))


# ``lingbotvla_cli.yaml`` 里 ``data.norm_stats_file`` 是训练机上的绝对路径(如
# ``/mnt/xuancun/RoboDojo/XPolicyLab/.../norm_stats/x.json``),换机器评测时必然找不到。
# 上传时该 JSON 会放在 run 级的 ``assets/norm_stats/`` 下,评测适配器则按
# ``policy/<POLICY>/lingbot_vla/<norm_stats_file>`` 解析相对路径。
NORM_STATS_RE = re.compile(r"^(?P<indent>[ \t]*)norm_stats_file:[ \t]*(?P<value>\S.*?)[ \t]*$", re.MULTILINE)


def norm_stats_relpath(run_dir, recorded, policy_dir):
    """把训练机写死的绝对路径换成本机可解析的相对路径;换不了就返回 None。"""
    if not os.path.isabs(recorded) or os.path.exists(recorded):
        return None  # 相对路径按适配器规则自己就能解析,绝对路径存在就不用动
    assets = os.path.join(run_dir, "assets", "norm_stats")
    cand = os.path.join(assets, os.path.basename(recorded))
    if not os.path.isfile(cand) and os.path.isdir(assets):
        jsons = sorted(n for n in os.listdir(assets) if n.endswith(".json"))
        if len(jsons) == 1:
            cand = os.path.join(assets, jsons[0])
    if not os.path.isfile(cand):
        print(f"[download_models] warn: {os.path.basename(recorded)} not under "
              f"{os.path.relpath(assets, REPO_ROOT)}/ and {recorded} missing; "
              f"copy it into {os.path.relpath(assets, REPO_ROOT)}/ then rerun")
        return None
    # 适配器把相对路径拼到 policy/<POLICY>/lingbot_vla/ 后面,这里算的就是相对它的路径。
    base = os.path.join(os.path.realpath(policy_dir), "lingbot_vla")
    return os.path.relpath(os.path.realpath(cand), base)


def write_patched_cli(src_cli, dst_cli, policy_dir):
    """``lingbotvla_cli.yaml`` 复制成真文件(而非软链),顺带修掉 norm_stats_file 的失效绝对路径。"""
    with open(src_cli, encoding="utf-8") as f:
        text = f.read()
    run_dir = os.path.dirname(src_cli)
    fixed = []

    def sub(m):
        rel = norm_stats_relpath(run_dir, m.group("value").strip("\"'"), policy_dir)
        if rel is None:
            return m.group(0)
        fixed.append((m.group("value"), rel))
        return f"{m.group('indent')}norm_stats_file: {rel}"

    text = NORM_STATS_RE.sub(sub, text)
    with open(dst_cli, "w", encoding="utf-8") as f:
        f.write(text)
    for old, new in fixed:
        print(f"[download_models] patched norm_stats_file: {old} -> {new}")


def link_shim(src, dst):
    """``checkpoints/<run>`` 建成真目录:run 级文件逐个链接,step 目录链接到 ``checkpoints/`` 下。
    只重建带标记的目录,别人放的真目录不动。"""
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.isdir(dst):
        if not os.path.exists(os.path.join(dst, SHIM_MARK)):
            print(f"[download_models] skip link: {dst} already exists and is not ours")
            return
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "checkpoints"))
    open(os.path.join(dst, SHIM_MARK), "w").close()
    policy_dir = os.path.dirname(os.path.dirname(dst))
    for name in sorted(os.listdir(src)):
        here = os.path.join(dst, "checkpoints", name) if STEP_RE.match(name) else os.path.join(dst, name)
        if name == "lingbotvla_cli.yaml":
            write_patched_cli(os.path.join(src, name), here, policy_dir)
            continue
        os.symlink(os.path.join(src, name), here)
    print(f"[download_models] linked {os.path.relpath(dst, REPO_ROOT)}/ -> {os.path.relpath(src, REPO_ROOT)}"
          "(step 目录放在 checkpoints/ 下)")

# GR00T_N17:``checkpoint-<n>/config.json`` 和 ``processor_config.json`` 里的 ``model_name`` 是训练机上
# Cosmos 骨干的绝对路径(如 ``/mnt/xuancun/.../GR00T_N17/shared/Cosmos-Reason2-2B``)。评测适配器只在加载时
# 临时改 processor_config.json,``AutoModel.from_pretrained`` 仍按 config.json 去加载骨干,换机器就找不到。
GR00T_STEP_RE = re.compile(r"^checkpoint-\d+$")
MODEL_NAME_RE = re.compile(r'"model_name"\s*:\s*"(?P<value>[^"]+)"')
COSMOS_FALLBACK = "nvidia/Cosmos-Reason2-2B"


def stale_model_names(step_dir):
    """step 目录顶层 json 里本机不存在的绝对 ``model_name`` -> {文件名: {旧路径}}。"""
    stale = {}
    for name in sorted(os.listdir(step_dir)):
        path = os.path.join(step_dir, name)
        if not (name.endswith(".json") and os.path.isfile(path)):
            continue
        with open(path, encoding="utf-8") as f:
            values = {m.group("value") for m in MODEL_NAME_RE.finditer(f.read())}
        values = {v for v in values if os.path.isabs(v) and not os.path.exists(v)}
        if values:
            stale[name] = values
    return stale


def needs_gr00t_shim(src):
    return any(GR00T_STEP_RE.match(n) and os.path.isdir(os.path.join(src, n))
               and stale_model_names(os.path.join(src, n)) for n in os.listdir(src))


def local_cosmos_model(policy_dir):
    """按 deploy.yml 的 ``cosmos_model_path``(和 model.py 同样的查找顺序)找本机 Cosmos 目录。"""
    raw = "shared/Cosmos-Reason2-2B"
    deploy = os.path.join(policy_dir, "deploy.yml")
    if os.path.isfile(deploy):
        with open(deploy, encoding="utf-8") as f:
            m = re.search(r"^cosmos_model_path:[ \t]*[\"']?([^\s\"'#]+)", f.read(), re.MULTILINE)
        if m:
            raw = m.group(1)
    for cand in (os.path.join(policy_dir, raw), os.path.join(policy_dir, "checkpoints", raw)):
        if os.path.isfile(os.path.join(cand, "config.json")):
            return os.path.realpath(cand)  # 和 model.py 的 resolve() 一致,加载时就不会再改写
    print(f"[download_models] warn: Cosmos backbone not found at {os.path.join(policy_dir, raw)}; "
          f"model_name 改成 HF id {COSMOS_FALLBACK}(需要能联网、有 gated 权限)")
    return COSMOS_FALLBACK


def link_gr00t_shim(src, dst, policy_dir):
    """``checkpoints/<run>`` 建成真目录:``checkpoint-<n>/`` 里权重等文件软链回镜像,
    含失效 ``model_name`` 的 json 复制一份改成本机 Cosmos 路径;镜像本身不改(看板按大小判断下载完整)。
    只重建带标记的目录,别人放的真目录不动。"""
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.isdir(dst):
        if not os.path.exists(os.path.join(dst, SHIM_MARK)):
            print(f"[download_models] skip link: {dst} already exists and is not ours")
            return
        shutil.rmtree(dst)
    os.makedirs(dst)
    open(os.path.join(dst, SHIM_MARK), "w").close()
    cosmos = None
    for name in sorted(os.listdir(src)):
        here, there = os.path.join(dst, name), os.path.join(src, name)
        stale = stale_model_names(there) if GR00T_STEP_RE.match(name) and os.path.isdir(there) else {}
        if not stale:
            os.symlink(there, here)
            continue
        cosmos = cosmos or local_cosmos_model(policy_dir)
        os.makedirs(here)
        for child in sorted(os.listdir(there)):
            if child not in stale:
                os.symlink(os.path.join(there, child), os.path.join(here, child))
                continue
            # 复制成真文件:model.py 加载时会就地改写 processor_config.json,不能写穿到镜像里
            with open(os.path.join(there, child), encoding="utf-8") as f:
                text = f.read()
            for old in stale[child]:
                text = text.replace(f'"{old}"', f'"{cosmos}"')
            with open(os.path.join(here, child), "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[download_models] patched {name}/{child} model_name: {', '.join(sorted(stale[child]))} -> {cosmos}")
    print(f"[download_models] linked {os.path.relpath(dst, REPO_ROOT)}/ -> {os.path.relpath(src, REPO_ROOT)}"
          "(checkpoint 目录逐文件链接)")


for policy, task, run in selected:
    policy_dir = os.path.join(REPO_ROOT, "XPolicyLab", "policy", policy)
    if not os.path.isdir(policy_dir):
        print(f"[download_models] skip link: policy dir not found: {policy_dir}")
        continue
    src = os.path.join(local_dir, policy, task, run)
    dst = os.path.join(policy_dir, "checkpoints", run)
    if needs_shim(src):
        link_shim(src, dst)
        continue
    if needs_gr00t_shim(src):
        link_gr00t_shim(src, dst, policy_dir)
        continue
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
